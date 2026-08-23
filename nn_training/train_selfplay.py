"""Fine-tune Model C with PPO against a frozen, mixed opponent pool."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from gamemanager import TichuGame
from gym_agents.baseline_b import BaselineBAgent
from gym_agents.model_c import ModelCAgent
from gym_agents.model_c_rl import ModelCPPOAgent
from gym_agents.model_c_selfplay import ModelCSelfPlayAgent, SelfPlayTrajectory
from nn_training.imitation_features import FeatureSchema
from nn_training.model_c import ModelCEncoder
from nn_training.selfplay import ModelCActorCritic, ppo_update, save_selfplay_checkpoint


def make_opponent(name, *, model_c_checkpoint, baseline_b_checkpoint, snapshot_checkpoint=None):
    if name == "model-c":
        return ModelCAgent(model_c_checkpoint, device="cpu")
    if name == "baseline-b":
        return BaselineBAgent(baseline_b_checkpoint, device="cpu")
    if name == "snapshot":
        if snapshot_checkpoint is None:
            raise ValueError("snapshot opponent requires a checkpoint")
        return ModelCPPOAgent(snapshot_checkpoint, device="cpu")
    raise ValueError("unknown self-play opponent: {}".format(name))


def collect_game(
    *,
    actor_critic,
    encoder,
    device,
    opponent_name,
    model_c_checkpoint,
    baseline_b_checkpoint,
    snapshot_checkpoint=None,
    seed,
    target_points,
    learner_on_even_team,
    gamma,
    gae_lambda,
):
    """Play one game and return learner actions with round-level team rewards."""
    trajectory = SelfPlayTrajectory()
    learner = ModelCSelfPlayAgent(actor_critic, encoder, trajectory, device=device)
    opponent = make_opponent(
        opponent_name,
        model_c_checkpoint=model_c_checkpoint,
        baseline_b_checkpoint=baseline_b_checkpoint,
        snapshot_checkpoint=snapshot_checkpoint,
    )
    agents = (
        (learner, opponent, learner, opponent)
        if learner_on_even_team
        else (opponent, learner, opponent, learner)
    )
    game = TichuGame(*agents)
    points = (0, 0)
    round_number = 0
    actor_critic.eval()
    try:
        while points[0] < target_points and points[1] < target_points:
            start_index = len(trajectory.transitions)
            round_points, _ = game._start_round(seed=seed + round_number)
            trajectory.reward_round(start_index, round_points)
            points = (
                points[0] + round_points[0],
                points[1] + round_points[1],
            )
            round_number += 1
    finally:
        game.env.close()
    learner_points, opponent_points = (
        points if learner_on_even_team else (points[1], points[0])
    )
    transitions = trajectory.finish(gamma=gamma, gae_lambda=gae_lambda)
    return transitions, {
        "seed": seed,
        "opponent": opponent_name,
        "learner_on_even_team": learner_on_even_team,
        "learner_points": learner_points,
        "opponent_points": opponent_points,
        "point_diff": learner_points - opponent_points,
        "rounds": round_number,
        "transitions": len(trajectory.transitions),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", default="models/model-c-v2/model-c.pt")
    parser.add_argument("--baseline-b-checkpoint", default="models/archive/baseline-b-v2/refine/wide.pt")
    parser.add_argument("--output", default="models/experiments/model-c-ppo.pt")
    parser.add_argument(
        "--resume-checkpoint",
        help="resume actor/value weights from a saved PPO checkpoint; optimizer state is reset",
    )
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--target", type=int, default=200, dest="target_points")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--kl-coefficient", type=float, default=0.02)
    parser.add_argument("--opponents", default="model-c,baseline-b")
    parser.add_argument("--snapshot-dir", default="models/experiments/selfplay-snapshots")
    parser.add_argument("--snapshot-interval", type=int, default=8)
    parser.add_argument("--evaluation-interval", type=int, default=8)
    parser.add_argument("--evaluation-games", type=int, default=30)
    parser.add_argument("--evaluation-seed", type=int, default=83000)
    parser.add_argument("--evaluation-output-dir", default="results/benchmarks/smoke/selfplay")
    parser.add_argument(
        "--stop-on-significant-evaluation",
        action="store_true",
        help="stop when the paired 95%% confidence interval excludes zero",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=80000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet-game-log", action="store_true")
    parser.add_argument("--verbose-episodes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def select_opponent(episode_index, snapshot_paths):
    """Use C v2 50%, Baseline B 25%, and historical PPO snapshots 25%."""
    slot = episode_index % 4
    if slot in (0, 1):
        return "model-c", None
    if slot == 2 or not snapshot_paths:
        return "baseline-b", None
    return "snapshot", snapshot_paths[episode_index % len(snapshot_paths)]


def paired_confidence_interval(results):
    pairs = {}
    for result in results:
        pairs.setdefault(result["seed"], []).append(result["point_diff"])
    paired_diffs = [mean(values) for values in pairs.values()]
    if len(paired_diffs) < 2:
        return mean(paired_diffs), None, None
    margin = 2.045 * stdev(paired_diffs) / len(paired_diffs) ** 0.5
    return mean(paired_diffs), mean(paired_diffs) - margin, mean(paired_diffs) + margin


def evaluate_checkpoint(args, checkpoint, update):
    """Evaluate a greedy PPO checkpoint on seeds disjoint from self-play."""
    from benchmark import run_benchmark, summarize, write_csv

    results = run_benchmark(
        "model-c-rl",
        "model-c",
        games=args.evaluation_games,
        seed=args.evaluation_seed,
        target_points=args.target_points,
        model_c_rl_checkpoint=checkpoint,
        model_c_rl_device="cpu",
        model_c_checkpoint=args.base_checkpoint,
        model_c_device="cpu",
        show_progress=False,
    )
    output = Path(args.evaluation_output_dir) / "{}-update-{:03d}-vs-model-c-v2-{}seeds.csv".format(
        Path(args.output).stem, update, args.evaluation_games
    )
    write_csv(results, output)
    paired_mean, lower, upper = paired_confidence_interval(results)
    return {
        "event": "evaluation",
        "update": update,
        "output": str(output),
        **summarize(results),
        "paired_mean_point_diff": paired_mean,
        "paired_ci95_lower": lower,
        "paired_ci95_upper": upper,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.updates < 1 or args.episodes_per_update < 1:
        raise SystemExit("--updates and --episodes-per-update must be positive")
    if args.snapshot_interval < 1 or args.evaluation_interval < 1:
        raise SystemExit("snapshot and evaluation intervals must be positive")
    if args.dry_run:
        args.updates = 1
        args.episodes_per_update = 1
        args.ppo_epochs = 1
        args.target_points = min(args.target_points, 100)
    opponents = tuple(name.strip() for name in args.opponents.split(",") if name.strip())
    if not opponents:
        raise SystemExit("--opponents must contain at least one agent")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if args.quiet_game_log:
        logging.getLogger("console_logger").setLevel(logging.CRITICAL + 1)

    if args.resume_checkpoint:
        actor_critic, base_saved = ModelCActorCritic.from_selfplay_checkpoint(
            args.resume_checkpoint, map_location="cpu"
        )
        starting_update = base_saved["update"]
    else:
        actor_critic, base_saved = ModelCActorCritic.from_model_c_checkpoint(
            args.base_checkpoint, map_location="cpu"
        )
        starting_update = 0
    schema = FeatureSchema.from_dict(base_saved["feature_schema"])
    encoder = ModelCEncoder(schema)
    actor_critic = actor_critic.to(device)
    reference_policy = copy.deepcopy(actor_critic.policy).to(device)
    reference_policy.eval()
    for parameter in reference_policy.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=args.learning_rate)
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_paths = sorted(snapshot_dir.glob("*.pt"))

    print(json.dumps({
        "event": "start",
        "base_checkpoint": args.base_checkpoint,
        "device": str(device),
        "updates": args.updates,
        "episodes_per_update": args.episodes_per_update,
        "opponents": opponents,
        "snapshot_count": len(snapshot_paths),
        "starting_update": starting_update,
        "parameters": sum(parameter.numel() for parameter in actor_critic.parameters()),
        "dry_run": args.dry_run,
    }, sort_keys=True), flush=True)

    started = time.monotonic()
    for update in range(starting_update + 1, args.updates + 1):
        transitions = []
        games = []
        for episode in range(args.episodes_per_update):
            game_seed = args.seed + (update - 1) * args.episodes_per_update + episode
            episode_index = (update - 1) * args.episodes_per_update + episode
            opponent, snapshot_checkpoint = select_opponent(episode_index, snapshot_paths)
            game_transitions, game = collect_game(
                actor_critic=actor_critic,
                encoder=encoder,
                device=device,
                opponent_name=opponent,
                model_c_checkpoint=args.base_checkpoint,
                baseline_b_checkpoint=args.baseline_b_checkpoint,
                snapshot_checkpoint=snapshot_checkpoint,
                seed=game_seed,
                target_points=args.target_points,
                learner_on_even_team=(episode % 2 == 0),
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            transitions.extend(game_transitions)
            games.append(game)
            if args.verbose_episodes:
                print(json.dumps({"event": "episode", "update": update, **game}), flush=True)

        metrics = ppo_update(
            actor_critic=actor_critic,
            reference_policy=reference_policy,
            encoder=encoder,
            transitions=transitions,
            optimizer=optimizer,
            device=device,
            ppo_epochs=args.ppo_epochs,
            clip_ratio=args.clip_ratio,
            value_coefficient=args.value_coefficient,
            entropy_coefficient=args.entropy_coefficient,
            kl_coefficient=args.kl_coefficient,
            minibatch_size=args.minibatch_size,
        )
        metrics.update({
            "event": "update",
            "update": update,
            "updates": args.updates,
            "games": len(games),
            "mean_point_diff": float(np.mean([game["point_diff"] for game in games])),
            "mean_rounds": float(np.mean([game["rounds"] for game in games])),
        })
        elapsed_seconds = time.monotonic() - started
        metrics["progress_percent"] = round(100.0 * update / args.updates, 1)
        metrics["elapsed_seconds"] = round(elapsed_seconds, 2)
        metrics["eta_seconds"] = round(
            elapsed_seconds * (args.updates - update) / update, 2
        )
        save_selfplay_checkpoint(
            args.output,
            actor_critic=actor_critic,
            feature_schema=schema,
            base_checkpoint=args.base_checkpoint,
            update=update,
            metrics=metrics,
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if update % args.snapshot_interval == 0:
            snapshot_path = snapshot_dir / "{}-update-{:03d}.pt".format(
                Path(args.output).stem, update
            )
            save_selfplay_checkpoint(
                snapshot_path,
                actor_critic=actor_critic,
                feature_schema=schema,
                base_checkpoint=args.base_checkpoint,
                update=update,
                metrics=metrics,
            )
            snapshot_paths.append(snapshot_path)
        if update % args.evaluation_interval == 0:
            evaluation = evaluate_checkpoint(args, args.output, update)
            print(json.dumps(evaluation, sort_keys=True), flush=True)
            lower = evaluation["paired_ci95_lower"]
            upper = evaluation["paired_ci95_upper"]
            if args.stop_on_significant_evaluation and (
                (lower is not None and lower > 0) or (upper is not None and upper < 0)
            ):
                print(json.dumps({
                    "event": "early_stop",
                    "reason": "paired_ci95_excludes_zero",
                    "update": update,
                    "paired_ci95_lower": lower,
                    "paired_ci95_upper": upper,
                }, sort_keys=True), flush=True)
                break
    print(json.dumps({"event": "complete", "output": str(Path(args.output))}, sort_keys=True))


if __name__ == "__main__":
    main()
