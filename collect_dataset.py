"""Collect privacy-safe decision records from reproducible Tichu games."""

import argparse
import json
from pathlib import Path

from benchmark import AGENT_NAMES, make_agent
from gamemanager import TichuGame
from gym_agents.training_data import (
    SCHEMA_VERSION,
    DecisionRecorder,
    RecordingAgent,
    split_for_seed,
    write_jsonl,
)


def collect_games(
    team_a,
    team_b,
    *,
    games,
    seed,
    target_points,
    iterations,
    max_time,
    record_agent,
    validation_fraction,
):
    records = {"train": [], "validation": []}
    game_count = 0
    for game_number in range(games):
        game_seed = seed + game_number
        split = split_for_seed(game_seed, validation_fraction)
        for swapped in (False, True):
            even_team, odd_team = (team_b, team_a) if swapped else (team_a, team_b)
            team_names = (even_team, odd_team, even_team, odd_team)
            recorder = DecisionRecorder(
                game_id="seed-{}-swapped-{}".format(game_seed, int(swapped)),
                seed=game_seed,
                swapped=swapped,
            )
            agents = []
            for agent_name in team_names:
                agent = make_agent(
                    agent_name,
                    iterations=iterations,
                    max_time=max_time,
                )
                agents.append(
                    RecordingAgent(agent, recorder)
                    if record_agent == "all" or agent_name == record_agent
                    else agent
                )

            game = TichuGame(*agents)
            try:
                outcome = game.start_game(
                    target_points=target_points,
                    seed=game_seed,
                    max_round_errors=0,
                )
            finally:
                game.env.close()
            records[split].extend(recorder.finalize(outcome.points))
            game_count += 1
    return records, game_count


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--team-a", choices=AGENT_NAMES, default="fuegi-mcts")
    parser.add_argument("--team-b", choices=AGENT_NAMES, default="fuegi")
    parser.add_argument("--record-agent", choices=AGENT_NAMES + ("all",), default="fuegi-mcts")
    parser.add_argument("--games", type=int, default=10, help="Seeds; each seed is played twice.")
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--target", type=int, default=100, dest="target_points")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-time", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--output-dir", default="datasets/tichu-decisions-v2")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    records, game_count = collect_games(
        args.team_a,
        args.team_b,
        games=args.games,
        seed=args.seed,
        target_points=args.target_points,
        iterations=args.iterations,
        max_time=args.max_time,
        record_agent=args.record_agent,
        validation_fraction=args.validation_fraction,
    )
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", records["train"])
    write_jsonl(output_dir / "validation.jsonl", records["validation"])
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "team_a": args.team_a,
        "team_b": args.team_b,
        "record_agent": args.record_agent,
        "seed_start": args.seed,
        "seed_count": args.games,
        "games": game_count,
        "target_points": args.target_points,
        "iterations": args.iterations,
        "max_time": args.max_time,
        "validation_fraction": args.validation_fraction,
        "train_records": len(records["train"]),
        "validation_records": len(records["validation"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Dataset collection complete")
    for key, value in metadata.items():
        print("{}: {}".format(key, value))
    print("output: {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
