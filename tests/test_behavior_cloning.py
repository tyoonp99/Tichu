import json

import pytest

torch = pytest.importorskip("torch")

from nn_training.behavior_cloning import (  # noqa: E402
    CandidatePolicy,
    batch_loss_and_rankings,
    encode_batch,
)
from nn_training.imitation_features import FeatureEncoder, FeatureSchema  # noqa: E402
from nn_training.train_behavior_cloning import run_epoch  # noqa: E402


def sample_record():
    pass_action = {"type": "PassAction", "player": 0}
    play_action = {
        "type": "PlayCombination",
        "player": 0,
        "combination": "Single",
        "cards": ["A_JADE"],
        "height": 14,
    }
    return {
        "observation": {
            "hand": ["A_JADE", "PHOENIX"],
            "hand_sizes": [2, 5, 4, 3],
            "won_trick_counts": [1, 0, 2, 1],
            "won_trick_points": [15, 0, 25, 10],
            "trick": [{"type": "PassAction", "player": 3}],
            "wish": None,
            "ranking": [2],
            "announced_tichu": [1],
            "announced_grand_tichu": [],
        },
        "legal_actions": [pass_action, play_action],
        "chosen_action": play_action,
    }


def test_candidate_policy_scores_every_legal_action_and_backpropagates():
    records = [sample_record(), sample_record()]
    encoder = FeatureEncoder(FeatureSchema.from_records(records))
    batch = encode_batch(records, encoder)
    model = CandidatePolicy(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        hidden_size=16,
        dropout=0,
    )

    scores = model(batch)
    loss, predictions, top3_matches, targets = batch_loss_and_rankings(
        scores, batch.targets
    )
    loss.backward()

    assert [len(group) for group in scores] == [2, 2]
    assert loss.item() > 0
    assert len(predictions) == 2
    assert len(targets) == 2
    assert sum(top3_matches) == 2
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_evaluation_separates_non_forced_and_non_pass_metrics(tmp_path):
    records = [sample_record(), sample_record()]
    records[1]["chosen_action"] = records[1]["legal_actions"][0]
    path = tmp_path / "validation.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    encoder = FeatureEncoder(FeatureSchema.from_records(records))
    model = CandidatePolicy(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        hidden_size=8,
        dropout=0,
    )

    metrics = run_epoch(
        model=model,
        encoder=encoder,
        path=path,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert metrics["records"] == 2
    assert metrics["non_forced"]["records"] == 2
    assert metrics["non_pass"]["records"] == 1
    assert metrics["pass_when_available"]["human_rate"] == 0.5
    assert metrics["uniform_random_top1"] == 0.5
    assert metrics["pass_else_first_top1"] == 0.5
    assert set(metrics["by_action"]) == {"PassAction", "PlayCombination/Single"}
