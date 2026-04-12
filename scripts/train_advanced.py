"""Train an advanced recommendation model.

Usage:
    uv run python scripts/train_advanced.py --model ncf
    uv run python scripts/train_advanced.py --model lightgcn
    uv run python scripts/train_advanced.py --model xgboost

Note: NCF and LightGCN require the [advanced] dependency group.
Install with: uv sync --group advanced
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an advanced LRS model")
    parser.add_argument(
        "--model",
        choices=["ncf", "lightgcn", "xgboost"],
        required=True,
        help="Which advanced model to train",
    )
    args = parser.parse_args()

    # TODO: Load data, instantiate model, call fit(), serialize artifact
    print(f"Training advanced model: {args.model}")
    raise NotImplementedError("Implement after src/lrs/models/advanced/ is complete")


if __name__ == "__main__":
    main()
