import gzip
import json

import pytest

torch = pytest.importorskip("torch")

from nn_training.behavior_cloning import BaselineBPolicy
from nn_training.decision_dataset import make_decision_dataloader
from nn_training.evaluate_baseline_b import (
    build_parser,
    chosen_action_groups,
    evaluate,
    write_report,
)
from nn_training.imitation_features import FeatureEncoder, FeatureSchema


def _record(action):
    return {
        "observation": {
            "hand": ["PHOENIX"],
            "hand_sizes": [1, 2, 3, 4],
            "won_trick_counts": [0, 0, 0, 0],
            "won_trick_points": [0, 0, 0, 0],
            "trick": [],
            "wish": None,
            "ranking": [],
            "announced_tichu": [],
            "announced_grand_tichu": [],
        },
        "legal_actions": [action],
        "chosen_action": action,
    }


def test_chosen_action_groups_cover_passes_combinations_bombs_and_phoenix():
    bomb_pass = _record({"type": "PassBombAction", "player": 0})
    phoenix_single = _record(
        {
            "type": "PlayCombination",
            "player": 0,
            "combination": "Single",
            "cards": ["PHOENIX"],
            "height": 1.5,
        }
    )
    bomb = _record(
        {
            "type": "PlayBomb",
            "player": 0,
            "combination": "SquareBomb",
            "cards": ["PHOENIX"],
            "height": 10,
        }
    )

    assert ("pass_context", "bomb_response_pass") in set(chosen_action_groups(bomb_pass))
    assert ("phoenix_availability", "phoenix_held") in set(
        chosen_action_groups(bomb_pass)
    )
    assert ("combination", "Single") in set(chosen_action_groups(phoenix_single))
    assert ("phoenix", "includes_phoenix") in set(chosen_action_groups(phoenix_single))
    assert ("bomb", "bomb") in set(chosen_action_groups(bomb))


def test_evaluate_and_write_report(tmp_path):
    records = [
        _record({"type": "PassAction", "player": 0}),
        _record({"type": "PassBombAction", "player": 0}),
        _record(
            {
                "type": "PlayCombination",
                "player": 0,
                "combination": "Single",
                "cards": ["PHOENIX"],
                "height": 1.5,
            }
        ),
    ]
    path = tmp_path / "test.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    schema = FeatureSchema.from_records(records)
    encoder = FeatureEncoder(schema)
    model = BaselineBPolicy(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        hidden_size=8,
        dropout=0,
    )
    metrics = evaluate(
        model=model,
        encoder=encoder,
        loader=make_decision_dataloader(path, batch_size=2),
        device=torch.device("cpu"),
        log_every=0,
    )
    report = {"overall": metrics["overall"], "breakdowns": metrics["breakdowns"]}
    json_path, csv_path = write_report(tmp_path / "report", report)

    assert metrics["overall"]["records"] == 3
    assert metrics["overall"]["top1"] == 1.0
    assert metrics["overall"]["top3"] == 1.0
    assert "bomb_response_pass" in metrics["breakdowns"]["pass_context"]
    assert "Single" in metrics["breakdowns"]["combination"]
    assert "phoenix_held" in metrics["breakdowns"]["phoenix_availability"]
    assert json_path.exists()
    assert csv_path.exists()


def test_evaluation_split_defaults_to_test_but_can_be_validation():
    parser = build_parser()
    assert parser.parse_args(["--checkpoint", "model.pt"]).split == "test"
    assert (
        parser.parse_args(["--checkpoint", "model.pt", "--split", "validation"]).split
        == "validation"
    )
