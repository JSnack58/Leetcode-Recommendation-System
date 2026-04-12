"""Generate recommendations for one or all users.

Usage:
    uv run python scripts/generate_recommendations.py --user alice123
    uv run python scripts/generate_recommendations.py --user all
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LRS recommendations")
    parser.add_argument(
        "--user",
        default="all",
        help="User ID to generate recommendations for, or 'all' for batch",
    )
    parser.add_argument("--model", default="svd", help="Model to use for scoring")
    parser.add_argument("--tier-size", type=int, default=5, help="Problems per tier")
    args = parser.parse_args()

    # TODO: Load model, run recommendation pipeline, output results
    print(f"Generating recommendations | user={args.user} | model={args.model}")
    raise NotImplementedError("Implement after recommendation/ layer is complete")


if __name__ == "__main__":
    main()
