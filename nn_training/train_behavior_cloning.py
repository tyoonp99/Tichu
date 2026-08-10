"""Train and evaluate a supervised candidate-ranking Tichu policy."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from nn_training.behavior_cloning import (
    CandidatePolicy,
    batch_loss_and_rankings,
    encode_batch,
)
from nn_training.imitation_features import (
    FeatureEncoder,
    FeatureSchema,
    batches,
    buffered_shuffle,
    iter_jsonl,
)


def limited(records, maximum):
    for index, record in enumerate(records):
        if maximum is not None and index >= maximum:
            return
        yield record


def run_epoch(
    *,
    model,
    encoder,
    path,
    batch_size,
    device,
    optimizer=None,
    shuffle_buffer=1,
    seed=0,
    max_records=None,
    phase=None,
    log_every=10000,
):
    training = optimizer is not None
    model.train(training)
    records = iter_jsonl(path)
    if training:
        records = buffered_shuffle(
            records,
            buffer_size=shuffle_buffer,
            rng=np.random.default_rng(seed),
        )
    records = limited(records, max_records)

    loss_sum = 0.0
    top1 = 0
    top3 = 0
    count = 0
    legal_action_sum = 0
    uniform_top1_sum = 0.0
    forced_count = 0
    non_forced_count = 0
    non_forced_top1 = 0
    non_forced_top3 = 0
    non_pass_count = 0
    non_pass_top1 = 0
    pass_available_count = 0
    pass_chosen_count = 0
    pass_else_first_top1 = 0
    action_counts = defaultdict(int)
    action_top1 = defaultdict(int)
    started = time.monotonic()
    next_log = log_every
    for raw_batch in batches(records, batch_size):
        batch = encode_batch(raw_batch, encoder).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            scores = model(batch)
            loss, predictions, top3_matches, target_indices = batch_loss_and_rankings(
                scores, batch.targets
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        size = len(raw_batch)
        loss_sum += float(loss.detach()) * size
        top1 += sum(
            prediction == target
            for prediction, target in zip(predictions, target_indices)
        )
        top3 += sum(top3_matches)
        count += size
        for record, candidate_scores, prediction, correct_top3, target_index in zip(
            raw_batch, scores, predictions, top3_matches, target_indices
        ):
            candidate_count = len(candidate_scores)
            correct_top1 = prediction == target_index
            legal_action_sum += candidate_count
            uniform_top1_sum += 1.0 / candidate_count

            if candidate_count == 1:
                forced_count += 1
            else:
                non_forced_count += 1
                non_forced_top1 += int(correct_top1)
                non_forced_top3 += int(correct_top3)

            chosen = record["chosen_action"]
            action_label = chosen["type"]
            if chosen["type"] == "PlayCombination":
                action_label += "/{}".format(chosen.get("combination", "UNKNOWN"))
            action_counts[action_label] += 1
            action_top1[action_label] += int(correct_top1)

            if chosen["type"] != "PassAction":
                non_pass_count += 1
                non_pass_top1 += int(correct_top1)

            if any(
                action["type"] == "PassAction"
                for action in record["legal_actions"]
            ):
                pass_available_count += 1
                pass_chosen_count += int(chosen["type"] == "PassAction")
            pass_index = next(
                (
                    index
                    for index, action in enumerate(record["legal_actions"])
                    if action["type"] == "PassAction"
                ),
                0,
            )
            pass_else_first_top1 += int(pass_index == target_index)
        if phase and log_every > 0 and count >= next_log:
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "phase": phase,
                        "records": count,
                        "records_per_second": count / elapsed,
                    },
                    sort_keys=True,
                )
            )
            while next_log <= count:
                next_log += log_every
    if not count:
        raise ValueError(f"no records found in {path}")
    metrics = {
        "loss": loss_sum / count,
        "top1": top1 / count,
        "top3": top3 / count,
        "records": count,
        "mean_legal_actions": legal_action_sum / count,
        "uniform_random_top1": uniform_top1_sum / count,
        "pass_else_first_top1": pass_else_first_top1 / count,
        "forced": {
            "records": forced_count,
            "fraction": forced_count / count,
        },
        "non_forced": {
            "records": non_forced_count,
            "top1": non_forced_top1 / non_forced_count
            if non_forced_count
            else None,
            "top3": non_forced_top3 / non_forced_count
            if non_forced_count
            else None,
        },
        "non_pass": {
            "records": non_pass_count,
            "top1": non_pass_top1 / non_pass_count if non_pass_count else None,
        },
        "pass_when_available": {
            "records": pass_available_count,
            "human_rate": pass_chosen_count / pass_available_count
            if pass_available_count
            else None,
        },
        "by_action": {
            label: {
                "records": action_counts[label],
                "top1": action_top1[label] / action_counts[label],
            }
            for label in sorted(action_counts)
        },
    }
    return metrics


def save_checkpoint(path, *, model, schema, epoch, validation_metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model": "candidate-policy-mlp",
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "feature_schema": schema.to_dict(),
            "epoch": epoch,
            "validation": validation_metrics,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="datasets/processed/brettspielwelt-decisions-v1",
    )
    parser.add_argument("--output", default="models/behavior-cloning-v1.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--max-trick-actions", type=int, default=12)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-validation-records", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    validation_path = data_dir / "validation.jsonl"
    schema = FeatureSchema.from_jsonl(
        train_path, max_trick_actions=args.max_trick_actions
    )
    encoder = FeatureEncoder(schema)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = CandidatePolicy(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    best_loss = float("inf")
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "state_size": encoder.state_size,
                "action_size": encoder.action_size,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            sort_keys=True,
        )
    )
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        train_metrics = run_epoch(
            model=model,
            encoder=encoder,
            path=train_path,
            batch_size=args.batch_size,
            device=device,
            optimizer=optimizer,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed + epoch,
            max_records=args.max_train_records,
            phase="train",
            log_every=args.log_every,
        )
        validation_metrics = run_epoch(
            model=model,
            encoder=encoder,
            path=validation_path,
            batch_size=args.batch_size,
            device=device,
            max_records=args.max_validation_records,
            phase="validation",
            log_every=args.log_every,
        )
        event = {
            "event": "epoch",
            "epoch": epoch,
            "elapsed_seconds": time.monotonic() - started,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        print(json.dumps(event, sort_keys=True))
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            save_checkpoint(
                args.output,
                model=model,
                schema=schema,
                epoch=epoch,
                validation_metrics=validation_metrics,
            )
            print(json.dumps({"event": "checkpoint", "path": args.output}))


if __name__ == "__main__":
    main()
