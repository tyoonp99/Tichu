"""Train comparable Baseline B candidates and write a validation summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from nn_training.behavior_cloning import BaselineBPolicy
from nn_training.decision_dataset import make_tensor_dataloader
from nn_training.imitation_features import FeatureEncoder, resolve_jsonl_path
from nn_training.train_baseline_b import build_schema, train


@dataclass(frozen=True)
class Candidate:
    name: str
    hidden_size: int
    learning_rate: float
    dropout: float


DEFAULT_CANDIDATES = (
    Candidate("small-fast", hidden_size=64, learning_rate=1e-3, dropout=0.0),
    Candidate("base", hidden_size=128, learning_rate=3e-4, dropout=0.1),
    Candidate("wide", hidden_size=256, learning_rate=3e-4, dropout=0.1),
    Candidate("regularized", hidden_size=128, learning_rate=1e-3, dropout=0.2),
)


def format_duration(seconds):
    """Format an approximate duration compactly for a terminal status line."""
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return "{}h {}m".format(hours, minutes)
    if minutes:
        return "{}m {}s".format(minutes, seconds)
    return "{}s".format(seconds)


def batches_for_records(records, batch_size, maximum=None):
    batches = int(math.ceil(records / batch_size))
    return min(batches, maximum) if maximum is not None else batches


def split_record_counts(data_dir):
    """Read deterministic split sizes from the v2 dataset manifest."""
    metadata_path = Path(data_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    counts = metadata["counts"]
    return counts["train_decisions"], counts["validation_decisions"]


class LiveTuningProgress:
    """Render one carriage-return-updated status line during a tuning run."""

    def __init__(
        self,
        *,
        candidate,
        candidate_index,
        candidate_count,
        epochs,
        train_batches,
        validation_batches,
    ):
        self.candidate = candidate
        self.candidate_index = candidate_index
        self.candidate_count = candidate_count
        self.epochs = epochs
        self.train_batches = train_batches
        self.validation_batches = validation_batches
        self.batches_per_epoch = train_batches + validation_batches
        self.total_batches = candidate_count * epochs * self.batches_per_epoch
        self.rendered = False

    def __call__(self, epoch, progress):
        speed = progress["batches_per_second"]
        if not speed:
            return
        phase_batches = (
            self.train_batches
            if progress["phase"] == "train"
            else self.validation_batches
        )
        remaining_epoch_batches = (
            phase_batches - progress["batches"]
            + (self.validation_batches if progress["phase"] == "train" else 0)
        )
        completed_candidate_batches = (
            (epoch - 1) * self.batches_per_epoch
            + (progress["batches"] if progress["phase"] == "train" else self.train_batches + progress["batches"])
        )
        completed_total_batches = (
            self.candidate_index * self.epochs * self.batches_per_epoch
            + completed_candidate_batches
        )
        line = (
            "[{candidate}] Epoch {epoch}/{epochs} | {speed:.1f} batch/s | "
            "Epoch ETA: {epoch_eta} | Total ETA: {total_eta} | "
            "Loss: {loss:.3f} | Top-1: {top1:.1%}"
        ).format(
            candidate=self.candidate.name,
            epoch=epoch,
            epochs=self.epochs,
            speed=speed,
            epoch_eta=format_duration(remaining_epoch_batches / speed),
            total_eta=format_duration(
                (self.total_batches - completed_total_batches) / speed
            ),
            loss=progress["loss"],
            top1=progress["top1"],
        )
        print("\r\033[2K" + line, end="", flush=True)
        self.rendered = True

    def finish(self):
        if self.rendered:
            print()
            self.rendered = False


def candidates_by_name(names):
    known = {candidate.name: candidate for candidate in DEFAULT_CANDIDATES}
    return tuple(known[name] for name in names)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def result_sort_key(result):
    """Sort successful candidates by Top-1, then validation loss."""
    return (result["validation_top1"], -result["validation_loss"])


def select_winner(results):
    successful = [result for result in results if result["status"] == "completed"]
    return max(successful, key=result_sort_key) if successful else None


def _atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_reports(output_dir, *, config, results):
    """Persist a machine-readable JSON report and a spreadsheet-friendly CSV."""
    output_dir = Path(output_dir)
    winner = select_winner(results)
    report = {
        "format_version": 1,
        "config": config,
        "candidates": results,
        "winner": winner,
    }
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    _atomic_write(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    fieldnames = (
        "candidate",
        "hidden_size",
        "learning_rate",
        "dropout",
        "status",
        "best_epoch",
        "validation_top1",
        "validation_loss",
        "validation_records",
        "elapsed_seconds",
        "checkpoint",
        "error",
    )
    rows = [{field: result.get(field) for field in fieldnames} for result in results]
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return json_path, csv_path, winner


def _make_loaders(args, *, train_path, validation_path, encoder, candidate_index):
    candidate_seed = args.seed + candidate_index
    return (
        make_tensor_dataloader(
            train_path,
            encoder=encoder,
            batch_size=args.batch_size,
            shuffle_buffer=args.shuffle_buffer,
            seed=candidate_seed,
            num_workers=args.train_workers,
        ),
        make_tensor_dataloader(
            validation_path,
            encoder=encoder,
            batch_size=args.batch_size,
            shuffle_buffer=1,
            seed=candidate_seed,
            num_workers=args.validation_workers,
        ),
    )


def tune(args, *, candidates=None):
    selected = tuple(candidates) if candidates is not None else candidates_by_name(args.candidates)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train_path = resolve_jsonl_path(args.data_dir, "train")
    validation_path = resolve_jsonl_path(args.data_dir, "validation")
    schema = build_schema(
        train_path,
        max_trick_actions=args.max_trick_actions,
        maximum=args.schema_records,
        log_every=args.schema_log_every,
    )
    encoder = FeatureEncoder(schema)
    train_records, validation_records = split_record_counts(args.data_dir)
    train_batches = batches_for_records(
        train_records, args.batch_size, args.max_train_batches
    )
    validation_batches = batches_for_records(
        validation_records, args.batch_size, args.max_validation_batches
    )
    config = {
        "data_dir": str(args.data_dir),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "schema_records": args.schema_records,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "early_stopping_patience": args.early_stopping_patience,
        "candidates": [asdict(candidate) for candidate in selected],
    }
    results = []
    for candidate_index, candidate in enumerate(selected):
        torch.manual_seed(args.seed + candidate_index)
        checkpoint = Path(args.model_dir) / "{}.pt".format(candidate.name)
        started = time.monotonic()
        print(
            json.dumps(
                {
                    "event": "candidate_start",
                    "candidate": asdict(candidate),
                    "device": str(device),
                },
                sort_keys=True,
            )
        )
        live_progress = LiveTuningProgress(
            candidate=candidate,
            candidate_index=candidate_index,
            candidate_count=len(selected),
            epochs=args.epochs,
            train_batches=train_batches,
            validation_batches=validation_batches,
        )
        try:
            model = BaselineBPolicy(
                state_size=encoder.state_size,
                action_size=encoder.action_size,
                hidden_size=candidate.hidden_size,
                dropout=candidate.dropout,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=candidate.learning_rate, weight_decay=1e-4
            )
            train_loader, validation_loader = _make_loaders(
                args,
                train_path=train_path,
                validation_path=validation_path,
                encoder=encoder,
                candidate_index=candidate_index,
            )
            train(
                model=model,
                schema=schema,
                train_loader=train_loader,
                validation_loader=validation_loader,
                device=device,
                epochs=args.epochs,
                optimizer=optimizer,
                output=checkpoint,
                max_train_batches=args.max_train_batches,
                max_validation_batches=args.max_validation_batches,
                log_every_batches=args.log_every_batches,
                progress_callback=live_progress,
                early_stopping_patience=args.early_stopping_patience,
            )
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            validation = saved["validation"]
            result = {
                "candidate": candidate.name,
                "hidden_size": candidate.hidden_size,
                "learning_rate": candidate.learning_rate,
                "dropout": candidate.dropout,
                "status": "completed",
                "best_epoch": saved["epoch"],
                "validation_top1": validation["top1"],
                "validation_loss": validation["loss"],
                "validation_records": validation["records"],
                "elapsed_seconds": time.monotonic() - started,
                "checkpoint": str(checkpoint),
                "error": None,
            }
            print(json.dumps({"event": "candidate_complete", **result}, sort_keys=True))
        except Exception as error:
            live_progress.finish()
            result = {
                "candidate": candidate.name,
                "hidden_size": candidate.hidden_size,
                "learning_rate": candidate.learning_rate,
                "dropout": candidate.dropout,
                "status": "failed",
                "best_epoch": None,
                "validation_top1": None,
                "validation_loss": None,
                "validation_records": None,
                "elapsed_seconds": time.monotonic() - started,
                "checkpoint": str(checkpoint),
                "error": "{}: {}".format(type(error).__name__, error),
            }
            print(json.dumps({"event": "candidate_failed", **result}, sort_keys=True))
        results.append(result)
        json_path, csv_path, winner = write_reports(
            args.output_dir, config=config, results=results
        )
        print(
            json.dumps(
                {
                    "event": "report_updated",
                    "json": str(json_path),
                    "csv": str(csv_path),
                    "winner": winner["candidate"] if winner else None,
                },
                sort_keys=True,
            )
        )
    return results


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="datasets/processed/brettspielwelt-decisions-v2"
    )
    parser.add_argument("--model-dir", default="models/baseline-b-v2-tuning")
    parser.add_argument("--output-dir", default="results/tuning/baseline-b-v2")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(candidate.name for candidate in DEFAULT_CANDIDATES),
        default=tuple(candidate.name for candidate in DEFAULT_CANDIDATES),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-workers", type=int, default=2)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=50000)
    parser.add_argument("--max-trick-actions", type=int, default=12)
    parser.add_argument("--schema-records", type=int)
    parser.add_argument("--schema-log-every", type=int, default=100000)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    if args.dry_run:
        args.epochs = min(args.epochs, 2)
        args.schema_records = args.schema_records or 4096
        args.max_train_batches = args.max_train_batches or 10
        args.max_validation_batches = args.max_validation_batches or 10
        args.log_every_batches = min(args.log_every_batches, 1)
    results = tune(args)
    if not select_winner(results):
        raise SystemExit("all tuning candidates failed")


if __name__ == "__main__":
    main()
