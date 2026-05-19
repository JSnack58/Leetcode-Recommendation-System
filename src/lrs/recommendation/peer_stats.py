"""Peer-group tag performance for blind-spot detection."""

from __future__ import annotations

import pandas as pd


def compute_peer_tag_scores(
    interactions: pd.DataFrame,
    user_id: str,
    rating_band: float = 0.1,
) -> dict[str, float]:
    """Average tag solve rate among rating-similar users."""
    user_rows = interactions[interactions["user_id"] == user_id]
    if user_rows.empty:
        return {}

    if "user_rating" in user_rows.columns:
        user_rating = float(user_rows["user_rating"].mean()) / 3000
    elif "rank_percentile" in user_rows.columns:
        user_rating = float(user_rows["rank_percentile"].mean())
    else:
        user_rating = 0.5

    if "user_rating" in interactions.columns:
        all_ratings = interactions.groupby("user_id")["user_rating"].mean() / 3000
    else:
        all_ratings = interactions.groupby("user_id")["rank_percentile"].mean()

    peer_users = all_ratings[(all_ratings - user_rating).abs() <= rating_band].index
    peer_df = interactions[interactions["user_id"].isin(peer_users)]

    tag_scores: dict[str, list[bool]] = {}
    for _, row in peer_df.iterrows():
        tags = row.get("tags")
        if not isinstance(tags, list):
            continue
        solved = bool(row.get("solved", False))
        for tag in tags:
            tag_scores.setdefault(tag, []).append(solved)

    return {tag: float(sum(v) / len(v)) for tag, v in tag_scores.items() if v}
