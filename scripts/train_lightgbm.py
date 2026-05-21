"""Train the LightGBM recommendation model.

Usage (from project root):
    py scripts/train_lightgbm.py

Reads raw JSONL data from data/raw/contests/, trains the model,
and saves artifacts to models/lightgbm/.
"""

import time
from lrs.models.advanced.lightgbm_model import LightGBMRecommender


def main() -> None:
    print("=" * 60)
    print("LightGBM Recommender - Training")
    print("=" * 60)

    t0 = time.time()
    model = LightGBMRecommender()
    model.fit_from_raw()

    path = model.save()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Model saved to {path}")

    # Quick sanity check: predict for a sample user
    all_problems = list(model._prob_feats["problem_id"])
    sample_user = model._user_feats["user_id"].iloc[0]
    scores = model.predict(sample_user, all_problems[:10])
    print(f"\nSample predictions for user '{sample_user}':")
    for pid, score in zip(all_problems[:10], scores):
        print(f"  Problem {pid}: P(solved) = {score:.3f}")


if __name__ == "__main__":
    main()
