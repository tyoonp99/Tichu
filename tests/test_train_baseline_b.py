import gzip
import json

import pytest

torch = pytest.importorskip("torch")

from nn_training.behavior_cloning import BaselineBPolicy
from nn_training.decision_dataset import make_tensor_dataloader
from nn_training.imitation_features import FeatureEncoder, FeatureSchema
from nn_training.train_baseline_b import (
    checkpoint_is_better,
    run_epoch,
    train,
)


def _record(index):
    pass_action = {"type": "PassAction", "player": 0}
    play_action = {
        "type": "PlayCombination",
        "player": 0,
        "combination": "Single",
        "cards": ["A_JADE"],
        "height": 14,
    }
    chosen_action = play_action if index % 2 else pass_action
    return {
        "observation": {
            "hand": ["A_JADE"],
            "hand_sizes": [1, 2, 3, 4],
            "won_trick_counts": [0, 0, 0, 0],
            "won_trick_points": [0, 0, 0, 0],
            "trick": [],
            "wish": None,
            "ranking": [],
            "announced_tichu": [],
            "announced_grand_tichu": [],
        },
        "legal_actions": [pass_action, play_action],
        "chosen_action": chosen_action,
    }


def _dataset(tmp_path, count=8):
    path = tmp_path / "records.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for index in range(count):
            output.write(json.dumps(_record(index)) + "\n")
    records = [_record(index) for index in range(count)]
    return path, FeatureEncoder(FeatureSchema.from_records(records))


def _model(encoder):
    return BaselineBPolicy(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        hidden_size=8,
        dropout=0,
    )


def test_run_epoch_reports_cross_entropy_and_top1(tmp_path):
    path, encoder = _dataset(tmp_path)
    loader = make_tensor_dataloader(path, encoder=encoder, batch_size=4)
    model = _model(encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    metrics = run_epoch(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        phase="train",
        log_every_batches=0,
    )

    assert metrics["records"] == 8
    assert metrics["batches"] == 2
    assert metrics["loss"] > 0
    assert 0 <= metrics["top1"] <= 1


def test_train_saves_best_validation_checkpoint(tmp_path):
    path, encoder = _dataset(tmp_path)
    model = _model(encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    output = tmp_path / "baseline-b.pt"
    train_loader = make_tensor_dataloader(path, encoder=encoder, batch_size=4)
    validation_loader = make_tensor_dataloader(path, encoder=encoder, batch_size=4)

    best = train(
        model=model,
        schema=encoder.schema,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=torch.device("cpu"),
        epochs=2,
        optimizer=optimizer,
        output=output,
        log_every_batches=0,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)

    assert output.exists()
    assert checkpoint["model"] == "baseline-b-candidate-mlp"
    assert checkpoint["validation"] == best
    assert checkpoint["selection_metric"] == "validation.top1_then_loss"


def test_checkpoint_prefers_top1_then_lower_loss():
    assert checkpoint_is_better({"top1": 0.6, "loss": 1.0}, None)
    assert checkpoint_is_better(
        {"top1": 0.6, "loss": 0.9}, {"top1": 0.5, "loss": 0.1}
    )
    assert checkpoint_is_better(
        {"top1": 0.6, "loss": 0.9}, {"top1": 0.6, "loss": 1.0}
    )
    assert not checkpoint_is_better(
        {"top1": 0.5, "loss": 0.1}, {"top1": 0.6, "loss": 1.0}
    )


def test_train_stops_after_configured_validation_patience(monkeypatch, capsys):
    import nn_training.train_baseline_b as training

    metrics = iter(
        [
            {"loss": 1.0, "top1": 0.1, "records": 1, "batches": 1},
            {"loss": 1.0, "top1": 0.7, "records": 1, "batches": 1},
            {"loss": 1.0, "top1": 0.1, "records": 1, "batches": 1},
            {"loss": 1.1, "top1": 0.7, "records": 1, "batches": 1},
            {"loss": 1.0, "top1": 0.1, "records": 1, "batches": 1},
            {"loss": 1.2, "top1": 0.7, "records": 1, "batches": 1},
        ]
    )
    saved = []
    monkeypatch.setattr(training, "run_epoch", lambda **_: next(metrics))
    monkeypatch.setattr(
        training, "save_checkpoint", lambda *args, **kwargs: saved.append(kwargs["epoch"])
    )

    best = training.train(
        model=object(),
        schema=None,
        train_loader=None,
        validation_loader=None,
        device=torch.device("cpu"),
        epochs=5,
        optimizer=object(),
        output="unused.pt",
        early_stopping_patience=2,
    )

    assert best["top1"] == 0.7
    assert saved == [1]
    assert '"event": "early_stopping"' in capsys.readouterr().out
