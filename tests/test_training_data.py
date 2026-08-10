import json

import pytest

from collect_dataset import build_parser
from gym_agents import BaseMonteCarloAgent, DefaultGymAgent
from gym_agents.mcts import make_fuegi_ismctsearch
from gym_agents.training_data import (
    DecisionRecorder,
    RecordingAgent,
    encode_observation,
    split_for_seed,
    write_jsonl,
)
from gym_tichu.envs.internals import Card


def test_observation_contains_only_the_acting_players_cards(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.THREE_JADE},
            {Card.DRAGON},
            {Card.PHOENIX},
            {Card.MAHJONG},
        ],
        player_pos=0,
    )

    observation = encode_observation(state, observer=0)
    serialized = json.dumps(observation)

    assert observation["hand"] == ["TWO_JADE", "THREE_JADE"]
    assert observation["hand_sizes"] == [2, 1, 1, 1]
    assert "DRAGON" not in serialized
    assert "PHOENIX" not in serialized
    assert "MAHJONG" not in serialized
    assert "history" not in observation


def test_recording_agent_records_legal_choice_and_final_team_result(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ],
        player_pos=0,
    )
    recorder = DecisionRecorder(game_id="game-1", seed=42, swapped=False)
    agent = RecordingAgent(DefaultGymAgent(), recorder)

    chosen = agent.action(state)
    records = recorder.finalize((120, 80))

    assert len(records) == 1
    assert records[0]["chosen_action"] in records[0]["legal_actions"]
    assert records[0]["outcome"] == {
        "team_points": 120,
        "opponent_points": 80,
        "point_diff": 40,
        "won": True,
    }
    assert chosen in state.possible_actions_set


def test_recording_agent_captures_root_mcts_statistics(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ],
        player_pos=0,
    )
    search = make_fuegi_ismctsearch(name="DatasetStatisticsTest")
    teacher = BaseMonteCarloAgent(
        search,
        iterations=2,
        max_time=1,
        cheat=True,
    )
    recorder = DecisionRecorder(game_id="game-2", seed=43, swapped=True)

    RecordingAgent(teacher, recorder).action(state)

    statistics = recorder.records[0]["search_statistics"]
    assert statistics
    assert all(item["visits"] >= 0 for item in statistics)
    assert all(item["availability"] >= 0 for item in statistics)


def test_seed_split_is_stable_and_rejects_invalid_fraction():
    assert split_for_seed(1234, 0.2) == split_for_seed(1234, 0.2)
    with pytest.raises(ValueError, match="validation_fraction"):
        split_for_seed(1234, 1)


def test_jsonl_writer_produces_one_valid_object_per_line(tmp_path):
    output = tmp_path / "records.jsonl"
    records = [{"seed": 1}, {"seed": 2}]

    write_jsonl(output, records)

    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == records


def test_dataset_parser_defaults_to_fuegi_mcts_teacher():
    args = build_parser().parse_args([])

    assert args.team_a == "fuegi-mcts"
    assert args.team_b == "fuegi"
    assert args.record_agent == "fuegi-mcts"
    assert args.output_dir == "datasets/tichu-decisions-v1"
