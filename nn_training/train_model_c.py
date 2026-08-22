"""Train Model C (Set Attention hand encoder + Bi-GRU trick encoder) on v2 data."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from nn_training.imitation_features import resolve_jsonl_path
from nn_training.model_c import ModelCEncoder, ModelCPolicy, make_model_c_dataloader
from nn_training.train_baseline_b import build_schema, train
from nn_training.tune_baseline_b import Candidate, LiveTuningProgress, batches_for_records, split_record_counts

LOSS_PROFILES = {
    "uniform": {},
    "rare-combination": {
        "SquareBomb": 3.0,
        "StraightBomb": 4.0,
        "FullHouse": 1.75,
        "Pair": 1.25,
    },
}

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/processed/brettspielwelt-decisions-v2")
    parser.add_argument("--output", default="models/model-c-v1/model-c.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--loss-profile", choices=tuple(LOSS_PROFILES), default="uniform")
    parser.add_argument("--train-workers", type=int, default=2)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=50000)
    parser.add_argument("--max-trick-actions", type=int, default=12)
    parser.add_argument("--schema-records", type=int)
    parser.add_argument("--schema-log-every", type=int, default=500000)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--log-every-batches", type=int, default=100)
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
        args.max_train_batches = args.max_train_batches or 2
        args.max_validation_batches = args.max_validation_batches or 2
        args.log_every_batches = min(args.log_every_batches, 1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    train_path = resolve_jsonl_path(args.data_dir, "train")
    validation_path = resolve_jsonl_path(args.data_dir, "validation")
    schema = build_schema(train_path, max_trick_actions=args.max_trick_actions, maximum=args.schema_records, log_every=args.schema_log_every)
    encoder = ModelCEncoder(schema, target_weights=LOSS_PROFILES[args.loss_profile])
    model = ModelCPolicy(**encoder.model_token_config(), width=args.width, heads=args.heads, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_loader = make_model_c_dataloader(train_path, encoder=encoder, batch_size=args.batch_size, shuffle_buffer=args.shuffle_buffer, seed=args.seed, num_workers=args.train_workers)
    validation_loader = make_model_c_dataloader(validation_path, encoder=encoder, batch_size=args.batch_size, shuffle_buffer=1, seed=args.seed, num_workers=args.validation_workers)
    if args.dry_run:
        sample = next(iter(train_loader))
        print(json.dumps({
            "event": "dry_run_batch_shapes",
            "hand_cards": list(sample.hand_cards.shape),
            "public": list(sample.public.shape),
            "trick_type": list(sample.trick_type.shape),
            "candidate_cards": list(sample.candidate_cards.shape),
            "legal_action_mask": list(sample.legal_action_mask.shape),
        }, sort_keys=True))
    train_records, validation_records = split_record_counts(args.data_dir)
    progress = LiveTuningProgress(candidate=Candidate("model-c", args.width, args.learning_rate, args.dropout), candidate_index=0, candidate_count=1, epochs=args.epochs, train_batches=batches_for_records(train_records, args.batch_size, args.max_train_batches), validation_batches=batches_for_records(validation_records, args.batch_size, args.max_validation_batches))
    print(json.dumps({"event": "start", "model": model.checkpoint_name, "device": str(device), "parameters": sum(p.numel() for p in model.parameters()), "loss_profile": args.loss_profile, "target_weights": LOSS_PROFILES[args.loss_profile], "dry_run": args.dry_run}, sort_keys=True))
    best = train(model=model, schema=schema, train_loader=train_loader, validation_loader=validation_loader, device=device, epochs=args.epochs, optimizer=optimizer, output=args.output, max_train_batches=args.max_train_batches, max_validation_batches=args.max_validation_batches, log_every_batches=args.log_every_batches, progress_callback=progress, early_stopping_patience=args.early_stopping_patience)
    print(json.dumps({"event": "complete", "best_validation": best, "output": args.output}, sort_keys=True))


if __name__ == "__main__":
    main()
