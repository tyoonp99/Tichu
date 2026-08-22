import csv
import json

from nn_training.tune_baseline_b import (
    Candidate,
    DEFAULT_CANDIDATES,
    LiveTuningProgress,
    candidates_by_name,
    format_duration,
    select_winner,
    write_reports,
)
from nn_training.tune_baseline_b_refine import REFINEMENT_CANDIDATES


def _result(name, *, top1, loss, status="completed"):
    return {
        "candidate": name,
        "hidden_size": 64,
        "learning_rate": 0.001,
        "dropout": 0.0,
        "status": status,
        "best_epoch": 2,
        "validation_top1": top1,
        "validation_loss": loss,
        "validation_records": 100,
        "elapsed_seconds": 1.0,
        "checkpoint": "models/{}.pt".format(name),
        "error": None,
    }


def test_default_candidates_cover_four_distinct_hyperparameter_combinations():
    assert len(DEFAULT_CANDIDATES) == 4
    assert len(
        {
            (candidate.hidden_size, candidate.learning_rate, candidate.dropout)
            for candidate in DEFAULT_CANDIDATES
        }
    ) == 4
    assert candidates_by_name(("base", "wide")) == (
        DEFAULT_CANDIDATES[1],
        DEFAULT_CANDIDATES[2],
    )


def test_refinement_candidates_hold_architecture_constant_and_bracket_learning_rate():
    assert [candidate.name for candidate in REFINEMENT_CANDIDATES] == [
        "wide-lr-low",
        "wide",
        "wide-lr-high",
    ]
    assert {candidate.hidden_size for candidate in REFINEMENT_CANDIDATES} == {256}
    assert {candidate.dropout for candidate in REFINEMENT_CANDIDATES} == {0.1}
    assert [candidate.learning_rate for candidate in REFINEMENT_CANDIDATES] == [
        1e-4,
        3e-4,
        6e-4,
    ]


def test_winner_uses_top1_then_validation_loss():
    results = [
        _result("higher-loss", top1=0.7, loss=0.5),
        _result("lower-loss", top1=0.7, loss=0.4),
        _result("failed", top1=None, loss=None, status="failed"),
    ]

    assert select_winner(results)["candidate"] == "lower-loss"


def test_reports_write_json_csv_and_current_winner(tmp_path):
    results = [
        _result("first", top1=0.6, loss=0.7),
        _result("second", top1=0.7, loss=0.8),
    ]

    json_path, csv_path, winner = write_reports(
        tmp_path, config={"epochs": 2}, results=results
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert winner["candidate"] == "second"
    assert report["winner"]["candidate"] == "second"
    assert [row["candidate"] for row in rows] == ["first", "second"]


def test_live_progress_renders_only_requested_monitoring_fields(capsys):
    progress = LiveTuningProgress(
        candidate=Candidate("wide", hidden_size=256, learning_rate=3e-4, dropout=0.1),
        candidate_index=2,
        candidate_count=4,
        epochs=5,
        train_batches=100,
        validation_batches=10,
    )

    progress(
        2,
        {
            "phase": "train",
            "batches": 50,
            "batches_per_second": 25.0,
            "loss": 0.482,
            "top1": 0.823,
        },
    )
    progress.finish()

    output = capsys.readouterr().out
    assert "\r\x1b[2K[wide] Epoch 2/5 | 25.0 batch/s" in output
    assert "Epoch ETA:" in output
    assert "Total ETA:" in output
    assert "Loss: 0.482" in output
    assert "Top-1: 82.3%" in output
    assert output.endswith("\n")


def test_format_duration_is_compact():
    assert format_duration(14) == "14s"
    assert format_duration(14 * 60 + 20) == "14m 20s"
    assert format_duration(3 * 3600 + 10 * 60) == "3h 10m"
