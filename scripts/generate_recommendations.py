#!/usr/bin/env python3
"""Generate tiered recommendations for a user."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from lrs.config import FEATURES_DIR, MODELS_DIR, PROCESSED_DIR
from lrs.models.baseline.als import ALSRecommender
from lrs.models.baseline.content_based import ContentBasedRecommender
from lrs.models.ensemble import EnsembleRecommender
from lrs.recommendation.graph_reranker import SimilarityGraphReranker
from lrs.recommendation.pipeline import (
    generate_for_user,
    save_recommendations,
    slug_to_user_id,
)


def load_ensemble() -> EnsembleRecommender:
    ensemble_path = MODELS_DIR / "ensemble.pkl"
    if ensemble_path.exists():
        with open(ensemble_path, "rb") as f:
            return pickle.load(f)

    als = ALSRecommender.load(MODELS_DIR / "als")
    content = ContentBasedRecommender.load(MODELS_DIR / "content")
    graph = SimilarityGraphReranker()
    ensemble = EnsembleRecommender()
    ensemble.add_models(als, content, graph)
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LRS recommendations")
    parser.add_argument("--user-slug", default=None, help="LeetCode user slug")
    parser.add_argument("--user-id", default=None, help="Hashed user id")
    parser.add_argument("--output", default="recommendations.json")
    parser.add_argument("--tier-size", type=int, default=5)
    args = parser.parse_args()

    if not args.user_slug and not args.user_id:
        parser.error("Provide --user-slug or --user-id")

    user_id = args.user_id or slug_to_user_id(args.user_slug)

    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    problems = pd.read_parquet(PROCESSED_DIR / "problems_clean.parquet")
    user_features_path = FEATURES_DIR / "user_features.parquet"
    user_features = (
        pd.read_parquet(user_features_path)
        if user_features_path.exists()
        else pd.DataFrame()
    )

    if user_id not in interactions["user_id"].values:
        print(f"Warning: user_id {user_id} not found in interactions")

    ensemble = load_ensemble()
    ensemble.set_user_history(interactions)

    from lrs.recommendation.tiers import TierAssigner

    assigner = TierAssigner(tier_size=args.tier_size)
    result = generate_for_user(
        user_id,
        ensemble,
        interactions,
        problems,
        user_features,
        tier_assigner=assigner,
    )
    if args.user_slug:
        result["user_slug"] = args.user_slug

    save_recommendations(result, Path(args.output))
    print(f"Wrote recommendations to {args.output}")
    for tier in ("edge_of_competence", "blind_spots", "confidence_builders"):
        print(f"  {tier}: {len(result[tier])} problems")


if __name__ == "__main__":
    main()
