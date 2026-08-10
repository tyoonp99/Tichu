"""
Starts a game against the computer
"""
import argparse
import datetime
import logging
from pathlib import Path

from gym_agents import BaseMonteCarloAgent, HumanInputAgent
from gym_agents.mcts import make_default_ismctsearch

from gamemanager import TichuGame
import logginginit

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent


def make_ismcts_agent(iterations: int=100, max_time: float=1):
    return BaseMonteCarloAgent(
            make_default_ismctsearch(name='ISMCTS'),
            iterations=iterations, max_time=max_time, cheat=False
    )


def print_game_outcome(outcome):
    assert len(outcome) == 2
    print("Final Result: {}".format(outcome[0]))
    rounds = outcome[1]
    for round in rounds:
        # round is a History object
        print("====================  New Round  ===================")
        print(round)

    print("Final Result: {}".format(outcome[0]))


def create_agent_against_agent(type1, type2)->TichuGame:
    agents = [type1(), type2(),
              type1(), type2()]
    return TichuGame(*agents)


def human_against_ismcts(target_points: int, iterations: int=100, max_time: float=1):
    agents = [
        HumanInputAgent(position=0),
        *(make_ismcts_agent(iterations, max_time) for _ in range(3)),
    ]
    game = TichuGame(*agents)

    res = game.start_game(target_points=target_points)
    return res


def ismcts_against_ismcts(target_points: int, iterations: int=100, max_time: float=1):
    agents = [make_ismcts_agent(iterations, max_time) for _ in range(4)]
    game = TichuGame(*agents)

    res = game.start_game(target_points=target_points)
    return res


def build_parser():
    parser = argparse.ArgumentParser(description='Play', allow_abbrev=False)

    parser.add_argument('--target', dest='target_points', type=int, required=False, default=1000,
                        help='The number of points to play for')

    parser.add_argument('--lazy', dest='lazy', required=False, action='store_true',
                        help='When this flag is present, a game between 4 ISMCTS agents is started.')

    parser.add_argument('--cheat', dest='cheat', required=False, action='store_true',
                        help='When this flag is present, then you can see the handcards of the other players.')

    parser.add_argument('--iterations', type=int, default=100,
                        help='Maximum MCTS iterations for each decision (default: 100).')

    parser.add_argument('--max-time', type=float, default=1,
                        help='Maximum MCTS search time per decision in seconds (default: 1).')

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    start_ftime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_folder_name = PROJECT_DIR / "logs" / "game_starter_{}".format(start_ftime)

    logging_mode = logginginit.HumanplayCheatMode if args.cheat or args.lazy else logginginit.HumanplayMode

    logginginit.initialize_loggers(output_dir=str(log_folder_name), logging_mode=logging_mode, min_loglevel=logging.DEBUG)

    if args.lazy:
        res = ismcts_against_ismcts(
            target_points=args.target_points,
            iterations=args.iterations,
            max_time=args.max_time,
        )
    else:
        res = human_against_ismcts(
            target_points=args.target_points,
            iterations=args.iterations,
            max_time=args.max_time,
        )
    print_game_outcome(res)


if __name__ == "__main__":
    main()
