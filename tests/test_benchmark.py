from benchmark import build_parser, make_agent, summarize
from gym_agents import FuegiHeuristicAgent, FuegiMctsAgent


def test_benchmark_parser_configures_paired_matches():
    args = build_parser().parse_args(
        [
            "--team-a",
            "mcts",
            "--team-b",
            "random",
            "--games",
            "3",
            "--seed",
            "42",
        ]
    )

    assert args.team_a == "mcts"
    assert args.team_b == "random"
    assert args.games == 3
    assert args.seed == 42


def test_benchmark_can_create_fuegi_agent():
    assert isinstance(make_agent("fuegi"), FuegiHeuristicAgent)


def test_benchmark_can_create_fuegi_mcts_agent():
    assert isinstance(make_agent("fuegi-mcts"), FuegiMctsAgent)


def test_benchmark_summary_uses_team_a_perspective():
    results = [
        {"winner": "A", "point_diff": 50, "elapsed_seconds": 1.25},
        {"winner": "B", "point_diff": -25, "elapsed_seconds": 0.75},
        {"winner": "draw", "point_diff": 0, "elapsed_seconds": 1.0},
    ]

    summary = summarize(results)

    assert summary == {
        "games": 3,
        "team_a_wins": 1,
        "team_b_wins": 1,
        "draws": 1,
        "average_point_diff": 25 / 3,
        "elapsed_seconds": 3.0,
    }


def test_setup_round_reproduces_deal_and_trades_with_seed():
    from gamemanager import TichuGame
    from gym_agents import RandomAgent

    first_game = TichuGame(*(RandomAgent() for _ in range(4)))
    second_game = TichuGame(*(RandomAgent() for _ in range(4)))
    try:
        first_state, _, _, _ = first_game._setup_round(seed=1234)
        second_state, _, _, _ = second_game._setup_round(seed=1234)
    finally:
        first_game.env.close()
        second_game.env.close()

    assert first_state.handcards == second_state.handcards
    assert first_state.player_pos == second_state.player_pos
