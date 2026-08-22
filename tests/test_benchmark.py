from benchmark import build_parser, make_agent, play_game, run_benchmark, summarize
from gym_agents import FuegiHeuristicAgent, FuegiMctsAgent, ModelCAgent, ModelCGuidedMctsAgent


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
    assert args.bc_checkpoint == "models/behavior-cloning-v1.pt"
    assert args.bc_device == "cpu"
    assert args.baseline_b_checkpoint == "models/archive/baseline-b-v2/refine/wide.pt"
    assert args.baseline_b_device == "cpu"
    assert args.model_c_checkpoint == "models/model-c-v2/model-c.pt"
    assert args.model_c_device == "cpu"
    assert args.puct_constant == 1.25
    assert args.quiet_game_log is False


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


def test_benchmark_summary_reports_actual_mcts_work():
    results = [
        {
            "winner": "A",
            "point_diff": 50,
            "elapsed_seconds": 1.0,
            "mcts_search_calls": 2,
            "mcts_iterations_requested": 20,
            "mcts_iterations_completed": 15,
            "mcts_time_limit_hits": 1,
            "mcts_iteration_limit_hits": 1,
        },
        {
            "winner": "B",
            "point_diff": -10,
            "elapsed_seconds": 2.0,
            "mcts_search_calls": 1,
            "mcts_iterations_requested": 10,
            "mcts_iterations_completed": 10,
            "mcts_time_limit_hits": 0,
            "mcts_iteration_limit_hits": 1,
        },
    ]

    summary = summarize(results)

    assert summary["mcts_search_calls"] == 3
    assert summary["mcts_average_requested_iterations"] == 10
    assert summary["mcts_average_iterations"] == 8.333333
    assert summary["mcts_time_limit_hits"] == 1
    assert summary["mcts_iteration_limit_hits"] == 2


def test_command_line_benchmark_progress_is_visible(monkeypatch, capsys):
    def fake_play_game(team_a, team_b, **kwargs):
        return {
            "winner": "A",
            "point_diff": 100,
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr("benchmark.play_game", fake_play_game)

    results = run_benchmark(
        "behavior-cloning",
        "fuegi",
        games=2,
        seed=42,
        target_points=100,
        show_progress=True,
    )

    output = capsys.readouterr().out
    assert len(results) == 4
    assert "Progress: 4/4 (100.0%)" in output


def test_fixed_iteration_benchmark_is_reproducible():
    arguments = {
        "seed": 4321,
        "target_points": 100,
        "iterations": 2,
        "max_time": float("inf"),
    }

    first = play_game("mcts", "fuegi", **arguments)
    second = play_game("mcts", "fuegi", **arguments)
    stable_fields = (
        "team_a_points",
        "team_b_points",
        "point_diff",
        "winner",
        "rounds",
        "mcts_search_calls",
        "mcts_iterations_completed",
        "mcts_average_iterations",
        "mcts_time_limit_hits",
        "mcts_iteration_limit_hits",
    )

    assert {field: first[field] for field in stable_fields} == {
        field: second[field] for field in stable_fields
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
