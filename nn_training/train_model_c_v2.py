"""Train Model C v2 with targeted rare-combination weighted cross-entropy."""

from __future__ import annotations

from nn_training.train_model_c import build_parser, main as train_main


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output="models/model-c-v2/model-c.pt",
        epochs=12,
        early_stopping_patience=2,
        loss_profile="rare-combination",
    )
    args = parser.parse_args(argv)
    forwarded = []
    for key, value in vars(args).items():
        if isinstance(value, bool):
            if value:
                forwarded.append("--{}".format(key.replace("_", "-")))
        elif value is not None:
            forwarded.extend(("--{}".format(key.replace("_", "-")), str(value)))
    train_main(forwarded)


if __name__ == "__main__":
    main()
