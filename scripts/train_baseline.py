#!/usr/bin/env python3
"""Train baseline recommendation models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd

from lrs.config import FEATURES_DIR, MODELS_DIR, PROCESSED_DIR
from lrs.evaluation.backtest import fit_calibrator, temporal_split
from lrs.models.baseline.als import ALSRecommender
from lrs.models.baseline.content_based import ContentBasedRecommender
from lrs.models.calibration import ScoreCalibrator
from lrs.models.ensemble import EnsembleRecommender
from lrs.recommendation.graph_reranker import SimilarityGraphReranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LRS baseline models")
    parser.add_argument(
        "--model",
        choices=["als", "content", "ensemble", "all"],
        default="all",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()

    interactions_path = PROCESSED_DIR / "interactions.parquet"
    if not interactions_path.exists():
        raise FileNotFoundError(
            f"{interactions_path} not found. Run scripts/build_dataset.py first."
        )

    interactions = pd.read_parquet(interactions_path)
    problems = pd.read_parquet(PROCESSED_DIR / "problems_clean.parquet")

    train, val = temporal_split(interactions, n_test_contests=10)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    als_model = ALSRecommender(latent_dim=args.latent_dim, iterations=args.iterations)
    content_model = ContentBasedRecommender()

    if args.model in ("als", "all", "ensemble"):
        print("Training ALS...")
        als_model.fit(train)
        cal_als = fit_calibrator(als_model, val)
        als_model.calibrator = cal_als
        als_model.save(MODELS_DIR / "als")
        cal_als.save(MODELS_DIR / "calibration_als.pkl")
        print("Saved ALS to", MODELS_DIR / "als")

    if args.model in ("content", "all", "ensemble"):
        print("Training content-based...")
        content_model.fit(train, problems)
        # Calibrate on sample
        sample = val.head(min(20000, len(val)))
        raw, labels = [], []
        for _, row in sample.iterrows():
            raw.append(content_model._raw_scores(row["user_id"], [row["problem_id"]])[0])
            labels.append(float(row["solved"]))
        cal_c = ScoreCalibrator(method="isotonic")
        cal_c.fit(np.array(raw), np.array(labels))
        content_model.calibrator = cal_c
        content_model.save(MODELS_DIR / "content")
        cal_c.save(MODELS_DIR / "calibration_content.pkl")
        print("Saved content model to", MODELS_DIR / "content")

    if args.model in ("ensemble", "all"):
        print("Building ensemble...")
        als_model = ALSRecommender.load(MODELS_DIR / "als")
        content_model = ContentBasedRecommender.load(MODELS_DIR / "content")
        graph = SimilarityGraphReranker()
        ensemble = EnsembleRecommender()
        ensemble.add_models(als_model, content_model, graph)
        ensemble.set_user_history(interactions)
        import pickle

        with open(MODELS_DIR / "ensemble.pkl", "wb") as f:
            pickle.dump(ensemble, f)
        print("Saved ensemble to", MODELS_DIR / "ensemble.pkl")

    print("Training complete.")


if __name__ == "__main__":
    main()
