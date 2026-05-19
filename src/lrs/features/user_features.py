"""User-level feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lrs.utils.logging import get_logger

logger = get_logger(__name__)


def _tag_solve_rates(user_data: pd.DataFrame) -> dict[str, float]:
    """Per-tag solve rate for a user."""
    tag_stats: dict[str, list[bool]] = {}
    for _, row in user_data.iterrows():
        tags = row.get("tags")
        if not isinstance(tags, (list, np.ndarray)):
            continue
        solved = bool(row.get("solved", False))
        for tag in tags:
            tag_stats.setdefault(tag, []).append(solved)
    return {tag: float(np.mean(vals)) for tag, vals in tag_stats.items()}


def compute_user_features(
    interactions_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute per-user aggregated features."""
    logger.info("Computing user features...")

    rows: list[dict] = []
    for user_id, user_data in interactions_df.groupby("user_id"):
        n_contests = user_data["contest_id"].nunique()
        n_problems = user_data["problem_id"].nunique()
        n_solved = int(user_data["solved"].sum())
        solve_rate = n_solved / max(n_problems, 1)

        scores = user_data["normalized_score"].dropna()
        avg_score = float(scores.mean()) if len(scores) else 0.0

        if "user_rating" in user_data.columns:
            avg_rating = float(user_data["user_rating"].mean())
        elif "rank_percentile" in user_data.columns:
            avg_rating = float(user_data["rank_percentile"].mean()) * 3000
        else:
            avg_rating = 1500.0

        finish_times = user_data["finish_time_min"].dropna()
        median_finish_time = float(finish_times.median()) if len(finish_times) else None

        penalty_rate = float(user_data["penalty_count"].sum()) / max(len(user_data), 1)

        lang_counts = user_data["language"].value_counts()
        top_language = lang_counts.index[0] if len(lang_counts) else "unknown"

        tag_scores = _tag_solve_rates(user_data)

        rows.append(
            {
                "user_id": user_id,
                "total_contests": n_contests,
                "total_problems": n_problems,
                "n_solved": n_solved,
                "solve_rate": solve_rate,
                "avg_score": avg_score,
                "avg_rating": avg_rating,
                "median_finish_time": median_finish_time,
                "penalty_rate": penalty_rate,
                "top_language": top_language,
                "tag_scores": tag_scores,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(f"Computed features for {len(df):,} users")

    if output_dir:
        output_path = Path(output_dir) / "user_features.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved user features to {output_path}")

    return df


def compute_peer_tag_performance(
    interactions_df: pd.DataFrame,
    rating_band: float = 0.1,
) -> pd.DataFrame:
    """Per-contest peer tag solve rates for blind-spot detection."""
    records: list[dict] = []

    for contest_id, contest_df in interactions_df.groupby("contest_id"):
        if "rank_percentile" in contest_df.columns:
            rating = contest_df["rank_percentile"]
        else:
            rating = contest_df.get("user_rating", pd.Series(0.5, index=contest_df.index)) / 3000

        for tag in set(
            t for tags in contest_df["tags"].dropna() for t in (tags if isinstance(tags, list) else [])
        ):
            mask = contest_df["tags"].apply(
                lambda x, tag=tag: isinstance(x, list) and tag in x
            )
            tag_df = contest_df[mask]
            if tag_df.empty:
                continue

            for user_id, user_rating in rating.items():
                peer_mask = (rating - user_rating).abs() <= rating_band
                peers = tag_df[tag_df["user_id"].isin(contest_df.loc[peer_mask, "user_id"])]
                if peers.empty:
                    continue
                records.append(
                    {
                        "user_id": contest_df.loc[user_id, "user_id"]
                        if user_id in contest_df.index
                        else user_id,
                        "tag": tag,
                        "score": float(peers["solved"].mean()),
                    }
                )

    if not records:
        return pd.DataFrame(columns=["user_id", "tag", "score"])

    peer_df = pd.DataFrame(records)
    return peer_df.groupby(["user_id", "tag"], as_index=False)["score"].mean()
