"""Train an advanced recommendation model.

Usage:
    py scripts/train_advanced.py --model lightgbm
    py scripts/train_advanced.py --model xgboost

Requires data/processed/interactions.parquet to exist.
Run the preprocessor first if it doesn't:
    py -c "from lrs.data.preprocessor import preprocess; preprocess()"
"""

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an advanced LRS model")
    parser.add_argument(
        "--model",
        choices=["lightgbm", "xgboost", "ncf", "lightgcn"],
        required=True,
        help="Which advanced model to train",
    )
    args = parser.parse_args()

    if args.model in ("lightgbm", "xgboost"):
        from lrs.data.loader import load_interactions
        from lrs.models.advanced.xgboost_ranker import LightGBMRecommender

        print("Loading interactions...")
        t0 = time.time()
        interactions = load_interactions()
        print(f"Loaded {len(interactions):,} rows in {time.time() - t0:.1f}s")

        print("\nTraining LightGBM model...")
        model = LightGBMRecommender()
        model.fit(interactions)

        model_path = model.save()
        print(f"\nDone! Model saved to {model_path}")

    else:
        print(f"Model '{args.model}' is not yet implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()
