from gym_agents import BaseMonteCarloAgent
from play import build_parser, make_ismcts_agent


def test_play_parser_exposes_mcts_limits():
    args = build_parser().parse_args(
        ["--lazy", "--target", "50", "--iterations", "3", "--max-time", "0.25"]
    )

    assert args.lazy
    assert args.target_points == 50
    assert args.iterations == 3
    assert args.max_time == 0.25


def test_play_uses_core_mcts_without_legacy_dqn_dependencies():
    agent = make_ismcts_agent(iterations=2, max_time=0.1)

    assert isinstance(agent, BaseMonteCarloAgent)
    assert agent.iterations == 2
    assert agent.max_time == 0.1
