"""Evaluate a Baseline B checkpoint once on the held-out Tichu test split."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from nn_training.behavior_cloning import BaselineBPolicy, encode_padded_batch
from nn_training.decision_dataset import make_decision_dataloader
from nn_training.imitation_features import FeatureEncoder, FeatureSchema, resolve_jsonl_path
from nn_training.model_c import ModelCEncoder, ModelCPolicy


@dataclass
class MetricAccumulator:
    records: int = 0
    loss_sum: float = 0.0
    top1_matches: int = 0
    top3_matches: int = 0

    def add(self, *, loss, top1, top3) -> None:
        self.records += 1
        self.loss_sum += float(loss)
        self.top1_matches += int(top1)
        self.top3_matches += int(top3)

    def as_dict(self) -> dict:
        if not self.records:
            raise ValueError("cannot summarize an empty metric group")
        return {
            "records": self.records,
            "loss": self.loss_sum / self.records,
            "top1": self.top1_matches / self.records,
            "top3": self.top3_matches / self.records,
        }


def chosen_action_groups(record):
    """Return overlapping, human-readable slices for one labelled decision."""
    action = record["chosen_action"]
    action_type = action.get("type", "UNKNOWN")
    combination = action.get("combination")
    cards = action.get("cards", ())
    hand = record["observation"].get("hand", ())

    if action_type == "PassBombAction":
        yield "pass_context", "bomb_response_pass"
    elif action_type == "PassAction":
        yield "pass_context", "general_pass"

    if combination is not None:
        yield "combination", combination

    is_bomb = action_type == "PlayBomb" or str(combination).endswith("Bomb")
    yield "bomb", "bomb" if is_bomb else "not_bomb"
    yield "phoenix", "includes_phoenix" if "PHOENIX" in cards else "no_phoenix"
    if "PHOENIX" in hand and "PHOENIX" not in cards:
        yield "phoenix_availability", "phoenix_held"


def load_checkpoint(path, *, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    schema = FeatureSchema.from_dict(checkpoint["feature_schema"])
    if checkpoint.get("model") == "baseline-b-candidate-mlp":
        encoder = FeatureEncoder(schema)
        model = BaselineBPolicy(**checkpoint["model_config"])
    elif checkpoint.get("model") == "model-c-set-attention-bigru":
        encoder = ModelCEncoder(schema)
        model = ModelCPolicy(**checkpoint["model_config"])
    else:
        raise ValueError("unsupported checkpoint model")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return checkpoint, encoder, model


def evaluate(*, model, encoder, loader, device, max_batches=None, log_every=100):
    """Score one held-out split, retaining aggregate and per-rule metrics only."""
    overall = MetricAccumulator()
    breakdowns = defaultdict(lambda: defaultdict(MetricAccumulator))
    started = time.monotonic()

    with torch.no_grad():
        for batch_number, records in enumerate(loader, start=1):
            if max_batches is not None and batch_number > max_batches:
                break
            batch = (
                encoder.encode_batch(records)
                if isinstance(encoder, ModelCEncoder)
                else encode_padded_batch(records, encoder)
            ).to(device)
            logits = model(batch)
            losses = F.cross_entropy(logits, batch.targets, reduction="none")
            top1 = logits.argmax(dim=1).eq(batch.targets)
            top_k = min(3, logits.shape[1])
            top3 = torch.topk(logits, k=top_k, dim=1).indices.eq(
                batch.targets[:, None]
            ).any(dim=1)

            losses = losses.detach().cpu().tolist()
            top1 = top1.detach().cpu().tolist()
            top3 = top3.detach().cpu().tolist()
            for record, loss, top1_match, top3_match in zip(
                records, losses, top1, top3
            ):
                overall.add(loss=loss, top1=top1_match, top3=top3_match)
                for dimension, bucket in chosen_action_groups(record):
                    breakdowns[dimension][bucket].add(
                        loss=loss, top1=top1_match, top3=top3_match
                    )

            if log_every and batch_number % log_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "evaluation_progress",
                            "batches": batch_number,
                            "records": overall.records,
                            "elapsed_seconds": time.monotonic() - started,
                            **overall.as_dict(),
                        },
                        sort_keys=True,
                    )
                )

    return {
        "overall": overall.as_dict(),
        "breakdowns": {
            dimension: {
                bucket: accumulator.as_dict()
                for bucket, accumulator in sorted(groups.items())
            }
            for dimension, groups in sorted(breakdowns.items())
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def write_report(output_dir, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "test-evaluation.json"
    csv_path = output_dir / "test-evaluation-breakdowns.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [{"dimension": "overall", "bucket": "all", **report["overall"]}]
    for dimension, groups in report["breakdowns"].items():
        rows.extend(
            {"dimension": dimension, "bucket": bucket, **metrics}
            for bucket, metrics in groups.items()
        )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=("dimension", "bucket", "records", "loss", "top1", "top3")
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-dir", default="datasets/processed/brettspielwelt-decisions-v2"
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Evaluation split. Keep test for the single final, post-selection run.",
    )
    parser.add_argument("--output-dir", default="results/evaluations/baseline-b-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    checkpoint, encoder, model = load_checkpoint(args.checkpoint, device=device)
    evaluation_path = resolve_jsonl_path(args.data_dir, args.split)
    loader = make_decision_dataloader(
        evaluation_path,
        batch_size=args.batch_size,
        shuffle_buffer=1,
        num_workers=args.workers,
    )
    metrics = evaluate(
        model=model,
        encoder=encoder,
        loader=loader,
        device=device,
        max_batches=args.max_batches,
        log_every=args.log_every,
    )
    report = {
        "format_version": 1,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": checkpoint["epoch"],
        "model_config": checkpoint["model_config"],
        **metrics,
    }
    json_path, csv_path = write_report(args.output_dir, report)
    print(
        json.dumps(
            {
                "event": "evaluation_complete",
                "overall": report["overall"],
                "json": str(json_path),
                "csv": str(csv_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
