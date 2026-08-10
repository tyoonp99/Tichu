"""Run reproducible, seat-swapped Tichu agent benchmarks."""

import argparse
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter

from gamemanager import TichuGame
from gym_agents import (
    BalancedRandomAgent,
    BaseMonteCarloAgent,
    FuegiHeuristicAgent,
    RandomAgent,
)
from gym_agents.mcts import make_default_ismctsearch


AGENT_NAMES = ("random", "balanced-random", "fuegi", "mcts")


def make_agent(name, *, iterations=10, max_time=0.2):
    if name == "random":
        return RandomAgent()
    if name == "balanced-random":
        return BalancedRandomAgent()
    if name == "fuegi":
        return FuegiHeuristicAgent()
    if name == "mcts":
        search = make_default_ismctsearch(name="BenchmarkMcts")
        return BaseMonteCarloAgent(
            search,
            iterations=iterations,
            max_time=max_time,
            cheat=False,
        )
    raise ValueError("Unknown agent: {}".format(name))


def play_game(
    team_a,
    team_b,
    *,
    seed,
    target_points,
    swap_seats=False,
    iterations=10,
    max_time=0.2,
):
    even_team, odd_team = (team_b, team_a) if swap_seats else (team_a, team_b)
    agents = [
        make_agent(even_team, iterations=iterations, max_time=max_time),
        make_agent(odd_team, iterations=iterations, max_time=max_time),
        make_agent(even_team, iterations=iterations, max_time=max_time),
        make_agent(odd_team, iterations=iterations, max_time=max_time),
    ]
    game = TichuGame(*agents)
    started = perf_counter()
    try:
        outcome = game.start_game(
            target_points=target_points,
            seed=seed,
            max_round_errors=0,
        )
    finally:
        game.env.close()

    even_points, odd_points = outcome.points
    team_a_points, team_b_points = (
        (odd_points, even_points) if swap_seats else (even_points, odd_points)
    )
    return {
        "seed": seed,
        "swapped": swap_seats,
        "team_a": team_a,
        "team_b": team_b,
        "team_a_points": team_a_points,
        "team_b_points": team_b_points,
        "point_diff": team_a_points - team_b_points,
        "winner": "A" if team_a_points > team_b_points else "B" if team_b_points > team_a_points else "draw",
        "rounds": len(outcome.history),
        "elapsed_seconds": round(perf_counter() - started, 6),
    }


def run_benchmark(
    team_a,
    team_b,
    *,
    games,
    seed,
    target_points,
    iterations=10,
    max_time=0.2,
):
    results = []
    for game_number in range(games):
        game_seed = seed + game_number
        for swap_seats in (False, True):
            result = play_game(
                team_a,
                team_b,
                seed=game_seed,
                target_points=target_points,
                swap_seats=swap_seats,
                iterations=iterations,
                max_time=max_time,
            )
            result["game"] = game_number
            results.append(result)
    return results


def summarize(results):
    return {
        "games": len(results),
        "team_a_wins": sum(result["winner"] == "A" for result in results),
        "team_b_wins": sum(result["winner"] == "B" for result in results),
        "draws": sum(result["winner"] == "draw" for result in results),
        "average_point_diff": mean(result["point_diff"] for result in results),
        "elapsed_seconds": round(sum(result["elapsed_seconds"] for result in results), 6),
    }


def write_csv(results, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--team-a", choices=AGENT_NAMES, default="mcts")
    parser.add_argument("--team-b", choices=AGENT_NAMES, default="balanced-random")
    parser.add_argument("--games", type=int, default=10, help="Number of seeds; each seed is played twice.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=int, default=100, dest="target_points")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-time", type=float, default=0.2)
    parser.add_argument("--output", default="benchmark-results.csv")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    results = run_benchmark(
        args.team_a,
        args.team_b,
        games=args.games,
        seed=args.seed,
        target_points=args.target_points,
        iterations=args.iterations,
        max_time=args.max_time,
    )
    write_csv(results, args.output)
    summary = summarize(results)
    print("Benchmark complete: {} vs {}".format(args.team_a, args.team_b))
    for key, value in summary.items():
        print("{}: {}".format(key, value))
    print("results: {}".format(Path(args.output).resolve()))


if __name__ == "__main__":
    main()
