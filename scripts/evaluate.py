"""Run the offline evaluation suite against a trained model.

Usage:
    uv run python scripts/evaluate.py --model svd
    uv run python scripts/evaluate.py --model ncf --n-test-contests 5
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for LRS models")
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument(
        "--n-test-contests",
        type=int,
        default=5,
        help="Number of most-recent contests to hold out for testing",
    )
    args = parser.parse_args()

    # TODO: Load model artifact, run backtest, print metrics table
    print(f"Evaluating model: {args.model} | test contests: {args.n_test_contests}")
    raise NotImplementedError("Implement after src/lrs/evaluation/ is complete")


if __name__ == "__main__":
    main()
