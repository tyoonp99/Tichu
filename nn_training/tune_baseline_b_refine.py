"""Second-stage, validation-only tuning around the first-stage Wide MLP winner."""

from __future__ import annotations

from nn_training.tune_baseline_b import Candidate, build_parser, select_winner, tune


REFINEMENT_CANDIDATES = (
    Candidate("wide-lr-low", hidden_size=256, learning_rate=1e-4, dropout=0.1),
    Candidate("wide", hidden_size=256, learning_rate=3e-4, dropout=0.1),
    Candidate("wide-lr-high", hidden_size=256, learning_rate=6e-4, dropout=0.1),
)


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(
        model_dir="models/baseline-b-v2-refine",
        output_dir="results/tuning/baseline-b-v2-refine",
        epochs=12,
        early_stopping_patience=2,
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    if args.dry_run:
        args.epochs = min(args.epochs, 2)
        args.schema_records = args.schema_records or 4096
        args.max_train_batches = args.max_train_batches or 10
        args.max_validation_batches = args.max_validation_batches or 10
        args.log_every_batches = min(args.log_every_batches, 1)
    results = tune(args, candidates=REFINEMENT_CANDIDATES)
    if not select_winner(results):
        raise SystemExit("all refinement candidates failed")


if __name__ == "__main__":
    main()
