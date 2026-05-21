"""LightGBM binary classifier for problem-solve prediction.

Incorporates metadata features that standard matrix factorization ignores:
penalty counts, programming language, finish times, and user skill stats.

Predicts P(solved) for a given (user, problem) pair.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lrs.config import MODELS_DIR, RANDOM_SEED
from lrs.features.problem_features import build_problem_features
from lrs.features.user_features import build_user_features
from lrs.models.base import BaseRecommender


# Features used by the model (order matters for consistency)
USER_FEATURES = [
    "total_contests",
    "total_solved",
    "total_attempted",
    "overall_solve_rate",
    "avg_rank",
    "avg_fail_count",
    "avg_solve_time",
    "num_languages_used",
]

PROBLEM_FEATURES = [
    "total_attempts",
    "total_solves",
    "solve_rate",
    "avg_fail_count_problem",
    "avg_solve_time_problem",
    "num_contests_appeared",
]

ALL_FEATURES = USER_FEATURES + PROBLEM_FEATURES


class LightGBMRecommender(BaseRecommender):
    """LightGBM-based recommender that predicts solve probability.

    Unlike ALS which only uses the user-problem interaction matrix,
    this model leverages side features (user stats, problem stats)
    to make predictions, including for cold-start scenarios.
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
        self.user_features_df: pd.DataFrame | None = None
        self.problem_features_df: pd.DataFrame | None = None

    def _build_feature_table(
        self, interactions: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build user and problem feature tables, then join them onto
        interactions to create the training feature matrix.

        Returns (feature_matrix, user_features, problem_features).
        """
        # Build aggregated features
        user_feats = build_user_features(interactions)
        problem_feats = build_problem_features(interactions)

        # Rename columns to avoid collisions when joining
        problem_feats = problem_feats.rename(columns={
            "avg_fail_count": "avg_fail_count_problem",
            "avg_solve_time": "avg_solve_time_problem",
        })

        # Join features onto interactions
        df = interactions.merge(user_feats, on="user_slug", how="left")
        df = df.merge(problem_feats, on="question_id", how="left")

        return df, user_feats, problem_feats

    def fit(self, interactions: pd.DataFrame) -> "LightGBMRecommender":
        """Train the LightGBM model on interaction data.

        Parameters
        ----------
        interactions : pd.DataFrame
            Cleaned interactions with columns: user_slug, question_id,
            solved, fail_count, solve_time_minutes, rank, score, etc.

        Returns
        -------
        self
        """
        df, user_feats, problem_feats = self._build_feature_table(interactions)

        # Store feature tables for prediction time
        self.user_features_df = user_feats
        self.problem_features_df = problem_feats

        # Prepare X and y
        X = df[ALL_FEATURES].fillna(0)
        y = df["solved"]

        # Train/validation split (stratified to preserve class balance)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50),
            ],
        )

        # Print feature importance
        importance = self.model.feature_importance(importance_type="gain")
        feat_imp = sorted(
            zip(ALL_FEATURES, importance), key=lambda x: x[1], reverse=True
        )
        print("\nFeature importance (gain):")
        for name, imp in feat_imp:
            print(f"  {name:30s} {imp:12.1f}")

        return self

    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        """Predict solve probability for a user on each problem.

        Parameters
        ----------
        user_id : str
            user_slug to predict for.
        problem_ids : list[str]
            List of question_id strings to score.

        Returns
        -------
        np.ndarray
            Predicted P(solved) for each problem, same order as problem_ids.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Look up user features
        user_row = self.user_features_df[
            self.user_features_df["user_slug"] == user_id
        ]
        if user_row.empty:
            # Cold start: use median user features
            user_vals = self.user_features_df[USER_FEATURES].median()
        else:
            user_vals = user_row[USER_FEATURES].iloc[0]

        # Build feature rows for each problem
        rows = []
        for pid in problem_ids:
            prob_row = self.problem_features_df[
                self.problem_features_df["question_id"] == pid
            ]
            if prob_row.empty:
                prob_vals = self.problem_features_df[PROBLEM_FEATURES].median()
            else:
                prob_vals = prob_row[PROBLEM_FEATURES].iloc[0]

            row = list(user_vals) + list(prob_vals)
            rows.append(row)

        X = pd.DataFrame(rows, columns=ALL_FEATURES).fillna(0)
        return self.model.predict(X)

    def save(self, path: Path | None = None) -> Path:
        """Persist the model and feature tables to disk."""
        if path is None:
            path = MODELS_DIR / "lightgbm"
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path / "model.txt"))
        self.user_features_df.to_parquet(path / "user_features.parquet", index=False)
        self.problem_features_df.to_parquet(path / "problem_features.parquet", index=False)
        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "LightGBMRecommender":
        """Load a saved model from disk."""
        if path is None:
            path = MODELS_DIR / "lightgbm"

        instance = cls()
        instance.model = lgb.Booster(model_file=str(path / "model.txt"))
        instance.user_features_df = pd.read_parquet(path / "user_features.parquet")
        instance.problem_features_df = pd.read_parquet(path / "problem_features.parquet")
        return instance
