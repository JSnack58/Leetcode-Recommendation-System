"""LightGBM binary classifier for problem-solve prediction.

Self-contained module that loads raw contest JSONL data, engineers features,
and trains a LightGBM model to predict P(solved) for (user, problem) pairs.

This model incorporates side features that collaborative filtering ignores:
penalty counts, solve times, user skill stats, and problem difficulty.

Usage:
    from lrs.models.advanced.lightgbm_model import LightGBMRecommender

    model = LightGBMRecommender()
    model.fit_from_raw()           # loads JSONL, preprocesses, trains
    probs = model.predict("user123", ["2663", "2664", "2665"])
    recs  = model.recommend("user123", all_problem_ids, k=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lrs.config import RAW_DIR, MODELS_DIR, RANDOM_SEED
from lrs.models.base import BaseRecommender


# ── Feature column definitions ───────────────────────────────────────────

USER_FEATURES = [
    "total_contests",
    "total_solved",
    "total_attempted",
    "overall_solve_rate",
    "avg_rank",
    "avg_penalty_count",
    "avg_solve_time",
    "num_languages_used",
]

PROBLEM_FEATURES = [
    "prob_total_attempts",
    "prob_total_solves",
    "prob_solve_rate",
    "prob_avg_penalty_count",
    "prob_avg_solve_time",
    "prob_num_contests",
]

ALL_FEATURES = USER_FEATURES + PROBLEM_FEATURES


# ── Data loading helpers ─────────────────────────────────────────────────

def _load_raw_contests(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read all JSONL contest files and flatten into interactions.

    Each row is one (user, contest, problem) triple.  Solved problems get
    ``solved=1``; problems a user did NOT attempt in a contest they entered
    get ``solved=0`` (negative samples).
    """
    contests_dir = (raw_dir or RAW_DIR) / "contests"
    jsonl_files = sorted(contests_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in {contests_dir}")

    rows: list[dict] = []

    for path in jsonl_files:
        contest_slug = path.stem
        question_ids: set[str] = set()
        earliest_ts: float | None = None
        records: list[dict] = []

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
                for qid, sub in rec.get("submissions", {}).items():
                    question_ids.add(qid)
                    ts = sub.get("date")
                    if ts is not None and (earliest_ts is None or ts < earliest_ts):
                        earliest_ts = ts

        if earliest_ts is None:
            earliest_ts = 0.0

        for rec in records:
            slug = rec.get("user_slug", "")
            rank = rec.get("rank", 0)
            subs = rec.get("submissions", {})
            solved_qids: set[str] = set()

            for qid, sub in subs.items():
                solved_qids.add(qid)
                sub_ts = sub.get("date", 0)
                rows.append({
                    "user_id": slug,
                    "contest_slug": contest_slug,
                    "problem_id": qid,
                    "solved": 1,
                    "penalty_count": sub.get("fail_count", 0),
                    "solve_time_sec": (sub_ts - earliest_ts) if sub_ts else np.nan,
                    "language": sub.get("lang", ""),
                    "rank": rank,
                })

            for qid in question_ids:
                if qid not in solved_qids:
                    rows.append({
                        "user_id": slug,
                        "contest_slug": contest_slug,
                        "problem_id": qid,
                        "solved": 0,
                        "penalty_count": 0,
                        "solve_time_sec": np.nan,
                        "language": "",
                        "rank": rank,
                    })

    df = pd.DataFrame(rows)
    df["solve_time_min"] = df["solve_time_sec"] / 60.0
    return df


def _build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-user statistics."""
    uf = (
        df.groupby("user_id")
        .agg(
            total_contests=("contest_slug", "nunique"),
            total_solved=("solved", "sum"),
            total_attempted=("solved", "count"),
            avg_rank=("rank", "mean"),
            avg_penalty_count=("penalty_count", "mean"),
            avg_solve_time=("solve_time_min", lambda s: s.dropna().mean()),
            num_languages_used=("language", lambda s: s[s != ""].nunique()),
        )
        .reset_index()
    )
    uf["overall_solve_rate"] = uf["total_solved"] / uf["total_attempted"]
    return uf


def _build_problem_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-problem statistics."""
    pf = (
        df.groupby("problem_id")
        .agg(
            prob_total_attempts=("solved", "count"),
            prob_total_solves=("solved", "sum"),
            prob_avg_penalty_count=("penalty_count", "mean"),
            prob_avg_solve_time=("solve_time_min", lambda s: s.dropna().mean()),
            prob_num_contests=("contest_slug", "nunique"),
        )
        .reset_index()
    )
    pf["prob_solve_rate"] = pf["prob_total_solves"] / pf["prob_total_attempts"]
    return pf


# ── Model ────────────────────────────────────────────────────────────────

class LightGBMRecommender(BaseRecommender):
    """LightGBM recommender that predicts solve probability.

    Unlike ALS (latent factors only) this model uses explicit user and
    problem features, giving it an advantage on cold-start users and
    interpretability through feature-importance scores.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": RANDOM_SEED,
        }
        self.model: lgb.Booster | None = None
        self._user_feats: pd.DataFrame | None = None
        self._prob_feats: pd.DataFrame | None = None

    # ── Training ─────────────────────────────────────────────────────

    def fit(self, interactions: pd.DataFrame) -> "LightGBMRecommender":
        """Train on a pre-built interactions DataFrame.

        Expected columns: user_id, problem_id, solved, penalty_count,
        solve_time_min, rank, language, contest_slug.
        """
        uf = _build_user_features(interactions)
        pf = _build_problem_features(interactions)
        self._user_feats = uf
        self._prob_feats = pf

        merged = interactions.merge(uf, on="user_id", how="left")
        merged = merged.merge(pf, on="problem_id", how="left")

        X = merged[ALL_FEATURES].fillna(0)
        y = merged["solved"]

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
        )

        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)

        self.model = lgb.train(
            self.params,
            ds_tr,
            num_boost_round=500,
            valid_sets=[ds_tr, ds_va],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50),
            ],
        )

        # Print feature importance
        imp = self.model.feature_importance(importance_type="gain")
        pairs = sorted(zip(ALL_FEATURES, imp), key=lambda p: p[1], reverse=True)
        print("\nFeature importance (gain):")
        for name, val in pairs:
            print(f"  {name:30s} {val:12.1f}")

        return self

    def fit_from_raw(self, raw_dir: Path | None = None) -> "LightGBMRecommender":
        """Convenience: load raw JSONL data, preprocess, and train."""
        print("Loading raw contest data...")
        df = _load_raw_contests(raw_dir)
        print(f"Loaded {len(df):,} interaction rows")
        return self.fit(df)

    # ── Prediction ───────────────────────────────────────────────────

    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        """Return P(solved) for each (user, problem) pair."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")

        # User feature vector (fall back to median for unknown users)
        row = self._user_feats[self._user_feats["user_id"] == user_id]
        if row.empty:
            u_vals = self._user_feats[USER_FEATURES].median()
        else:
            u_vals = row[USER_FEATURES].iloc[0]

        # Build one row per candidate problem
        feat_rows: list[list] = []
        for pid in problem_ids:
            p_row = self._prob_feats[self._prob_feats["problem_id"] == pid]
            if p_row.empty:
                p_vals = self._prob_feats[PROBLEM_FEATURES].median()
            else:
                p_vals = p_row[PROBLEM_FEATURES].iloc[0]
            feat_rows.append(list(u_vals) + list(p_vals))

        X = pd.DataFrame(feat_rows, columns=ALL_FEATURES).fillna(0)
        return self.model.predict(X)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        """Save model + feature tables."""
        path = path or (MODELS_DIR / "lightgbm")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / "model.txt"))
        self._user_feats.to_parquet(path / "user_features.parquet", index=False)
        self._prob_feats.to_parquet(path / "problem_features.parquet", index=False)
        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "LightGBMRecommender":
        """Load a previously saved model."""
        path = path or (MODELS_DIR / "lightgbm")
        inst = cls()
        inst.model = lgb.Booster(model_file=str(path / "model.txt"))
        inst._user_feats = pd.read_parquet(path / "user_features.parquet")
        inst._prob_feats = pd.read_parquet(path / "problem_features.parquet")
        return inst
