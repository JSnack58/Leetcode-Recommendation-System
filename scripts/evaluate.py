#!/usr/bin/env python3
"""Run offline evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from lrs.config import PROCESSED_DIR
from lrs.evaluation.backtest import run_backtest
from lrs.models.baseline.als import ALSRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LRS models")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-test-contests", type=int, default=10)
    parser.add_argument("--output", default=None, help="JSON metrics output path")
    args = parser.parse_args()

    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    metrics = run_backtest(
        interactions,
        model=ALSRecommender(latent_dim=32, iterations=10),
        n_test_contests=args.n_test_contests,
        k=args.k,
    )

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
