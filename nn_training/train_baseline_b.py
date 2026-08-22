"""Train and validate the mask-aware Baseline B behavior-cloning policy."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from nn_training.behavior_cloning import BaselineBPolicy
from nn_training.decision_dataset import make_tensor_dataloader
from nn_training.imitation_features import (
    FeatureEncoder,
    FeatureSchema,
    iter_jsonl,
    resolve_jsonl_path,
)


def records_for_schema(path, *, maximum=None, log_every=100000):
    """Yield a bounded schema sample while reporting progress for large splits."""
    for index, record in enumerate(iter_jsonl(path), start=1):
        if maximum is not None and index > maximum:
            return
        yield record
        if log_every and index % log_every == 0:
            print(
                json.dumps(
                    {"event": "schema_progress", "records": index},
                    sort_keys=True,
                )
            )


def build_schema(path, *, max_trick_actions, maximum=None, log_every=100000):
    return FeatureSchema.from_records(
        records_for_schema(path, maximum=maximum, log_every=log_every),
        max_trick_actions=max_trick_actions,
    )


def run_epoch(
    *,
    model,
    loader,
    device,
    optimizer=None,
    max_batches=None,
    phase,
    log_every_batches=100,
    progress_callback=None,
):
    """Run one train or validation pass and return loss and Top-1 metrics."""
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    top1 = 0
    records = 0
    batches = 0
    started = time.monotonic()

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(batch)
            per_sample_loss = F.cross_entropy(logits, batch.targets, reduction="none")
            target_weights = getattr(batch, "target_weights", None) if training else None
            loss = (
                (per_sample_loss * target_weights).sum() / target_weights.sum()
                if target_weights is not None
                else per_sample_loss.mean()
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        size = batch.targets.numel()
        loss_sum += float(loss.detach()) * size
        top1 += int((logits.argmax(dim=1) == batch.targets).sum().detach())
        records += size
        batches += 1
        if log_every_batches and batches % log_every_batches == 0:
            elapsed = time.monotonic() - started
            progress = {
                "event": "batch_progress",
                "phase": phase,
                "batches": batches,
                "records": records,
                "loss": loss_sum / records,
                "top1": top1 / records,
                "batches_per_second": batches / elapsed,
            }
            if progress_callback is not None:
                progress_callback(progress)
            else:
                print(json.dumps(progress, sort_keys=True))

    if not records:
        raise ValueError("no records were read for {} evaluation".format(phase))
    return {
        "loss": loss_sum / records,
        "top1": top1 / records,
        "records": records,
        "batches": batches,
    }


def checkpoint_is_better(candidate, best):
    """Prefer validation Top-1; use lower validation loss as a deterministic tie-break."""
    if best is None:
        return True
    if candidate["top1"] != best["top1"]:
        return candidate["top1"] > best["top1"]
    return candidate["loss"] < best["loss"]


def save_checkpoint(path, *, model, schema, epoch, train_metrics, validation_metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model": getattr(model, "checkpoint_name", "baseline-b-candidate-mlp"),
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "feature_schema": schema.to_dict(),
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_metric": "validation.top1_then_loss",
        },
        path,
    )


def train(
    *,
    model,
    schema,
    train_loader,
    validation_loader,
    device,
    epochs,
    optimizer,
    output,
    max_train_batches=None,
    max_validation_batches=None,
    log_every_batches=100,
    progress_callback=None,
    early_stopping_patience=None,
):
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive when provided")
    best_validation = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        started = time.monotonic()

        def report_progress(progress):
            if progress_callback is not None:
                progress_callback(epoch, progress)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_batches=max_train_batches,
            phase="train",
            log_every_batches=log_every_batches,
            progress_callback=report_progress if progress_callback else None,
        )
        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            max_batches=max_validation_batches,
            phase="validation",
            log_every_batches=log_every_batches,
            progress_callback=report_progress if progress_callback else None,
        )
        finish_progress = getattr(progress_callback, "finish", None)
        if finish_progress is not None:
            finish_progress()
        event = {
            "event": "epoch",
            "epoch": epoch,
            "elapsed_seconds": time.monotonic() - started,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        print(json.dumps(event, sort_keys=True))
        if checkpoint_is_better(validation_metrics, best_validation):
            best_validation = validation_metrics
            epochs_without_improvement = 0
            save_checkpoint(
                output,
                model=model,
                schema=schema,
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "epoch": epoch,
                        "path": str(output),
                        "validation": validation_metrics,
                    },
                    sort_keys=True,
                )
            )
        else:
            epochs_without_improvement += 1
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                            "patience": early_stopping_patience,
                            "best_validation": best_validation,
                        },
                        sort_keys=True,
                    )
                )
                break
    return best_validation


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="datasets/processed/brettspielwelt-decisions-v2"
    )
    parser.add_argument("--output", default="models/baseline-b-v2.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
        args.max_train_batches = args.max_train_batches or 10
        args.max_validation_batches = args.max_validation_batches or 10
        args.schema_records = args.schema_records or 4096
        args.log_every_batches = min(args.log_every_batches, 1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_path = resolve_jsonl_path(data_dir, "train")
    validation_path = resolve_jsonl_path(data_dir, "validation")
    schema = build_schema(
        train_path,
        max_trick_actions=args.max_trick_actions,
        maximum=args.schema_records,
        log_every=args.schema_log_every,
    )
    encoder = FeatureEncoder(schema)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = BaselineBPolicy(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    train_loader = make_tensor_dataloader(
        train_path,
        encoder=encoder,
        batch_size=args.batch_size,
        max_records=None,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        num_workers=args.train_workers,
    )
    validation_loader = make_tensor_dataloader(
        validation_path,
        encoder=encoder,
        batch_size=args.batch_size,
        shuffle_buffer=1,
        seed=args.seed,
        num_workers=args.validation_workers,
    )
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "state_size": encoder.state_size,
                "action_size": encoder.action_size,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    best_validation = train(
        model=model,
        schema=schema,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=args.epochs,
        optimizer=optimizer,
        output=args.output,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        log_every_batches=args.log_every_batches,
        early_stopping_patience=args.early_stopping_patience,
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "best_validation": best_validation,
                "output": args.output,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
