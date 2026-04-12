"""Train a baseline recommendation model.

Usage:
    uv run python scripts/train_baseline.py --model svd
    uv run python scripts/train_baseline.py --model als
    uv run python scripts/train_baseline.py --model content_based
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline LRS model")
    parser.add_argument(
        "--model",
        choices=["svd", "als", "content_based"],
        required=True,
        help="Which baseline model to train",
    )
    args = parser.parse_args()

    # TODO: Load data, instantiate model, call fit(), serialize artifact
    print(f"Training baseline model: {args.model}")
    raise NotImplementedError("Implement after src/lrs/models/baseline/ is complete")


if __name__ == "__main__":
    main()
