"""Parse raw contest JSONL into flat user-problem interaction rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lrs.utils.logging import get_logger

logger = get_logger(__name__)


def _contest_start_time(contest_df: pd.DataFrame) -> int:
    """Infer contest start as minimum submission timestamp in that contest."""
    if "submission_date" not in contest_df.columns:
        return 0
    dates = contest_df["submission_date"].dropna()
    return int(dates.min()) if len(dates) else 0


def explode_contest_record(record: dict) -> list[dict]:
    """Expand one contest row into per-problem interaction dicts."""
    submissions = record.get("submissions") or {}
    if not submissions:
        return []

    user_slug = record.get("user_slug")
    contest_id = record.get("contest_id")
    if not user_slug or contest_id is None:
        return []

    rank = record.get("rank")
    try:
        rank_val = int(rank) if rank is not None else None
    except (TypeError, ValueError):
        rank_val = None

    rows: list[dict] = []
    for frontend_id, sub in submissions.items():
        if not isinstance(sub, dict):
            continue
        fail_count = int(sub.get("fail_count", 0) or 0)
        submission_date = sub.get("date")
        rows.append(
            {
                "user_slug": user_slug,
                "contest_id": contest_id,
                "frontend_id": str(frontend_id),
                "penalty_count": fail_count,
                "language": sub.get("lang", "unknown"),
                "submission_date": submission_date,
                "solved": fail_count == 0,
                "rank": rank_val,
            }
        )
    return rows


def _finalize_parsed_frame(flat_rows: list[dict], source_rows: int) -> pd.DataFrame:
    """Apply rank percentile, finish times, and scores to exploded rows."""
    if not flat_rows:
        raise ValueError("No interaction rows produced from contest data")

    df = pd.DataFrame(flat_rows)
    logger.info(f"Exploded {source_rows:,} contest rows -> {len(df):,} interactions")

    # Rank percentile within each contest (proxy for user_rating)
    if "rank" in df.columns:
        contest_sizes = df.groupby("contest_id")["user_slug"].transform("nunique")
        # Lower rank is better; percentile where 1.0 = top performer
        df["rank_percentile"] = 1.0 - (df["rank"].astype(float) / contest_sizes.clip(lower=1))
        df.loc[df["rank"].isna() | (df["rank"] <= 0), "rank_percentile"] = 0.5
    else:
        df["rank_percentile"] = 0.5

    # Finish time: minutes from inferred contest start to submission
    df["finish_time_min"] = None
    for contest_id, group_idx in df.groupby("contest_id").groups.items():
        group = df.loc[group_idx]
        start = _contest_start_time(group)
        if start > 0:
            mins = (group["submission_date"].astype(float) - start) / 60.0
            df.loc[group_idx, "finish_time_min"] = mins.round(2)

    # Per-problem implicit feedback score (not aggregate contest score)
    df["normalized_score"] = df["solved"].astype(float)
    df.loc[df["penalty_count"] > 0, "normalized_score"] = (
        1.0 / (1.0 + df.loc[df["penalty_count"] > 0, "penalty_count"])
    )

    return df


def parse_contest_jsonl_file(jsonl_path: Path, sample: int | None = None) -> pd.DataFrame:
    """Read a single combined JSONL file (one contest record per line)."""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Contest JSONL not found: {jsonl_path}")

    logger.info(f"Parsing combined contest file {jsonl_path}")

    flat_rows: list[dict] = []
    source_rows = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skip bad JSON at line {line_num}: {e}")
                continue

            flat_rows.extend(explode_contest_record(record))
            source_rows += 1
            if sample is not None and source_rows >= sample:
                break

    return _finalize_parsed_frame(flat_rows, source_rows)


def parse_contest_jsonl(raw_dir: Path, sample: int | None = None) -> pd.DataFrame:
    """Read all contest JSONL files in a directory and return exploded interactions.

    Args:
        raw_dir: Directory containing ``*.jsonl`` contest files.
        sample: If set, stop after this many source contest rows (dev mode).

    Returns:
        DataFrame with one row per user-problem-contest attempt.
    """
    raw_dir = Path(raw_dir)
    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {raw_dir}")

    logger.info(f"Parsing {len(jsonl_files)} contest files from {raw_dir}")

    flat_rows: list[dict] = []
    source_rows = 0

    for file_path in jsonl_files:
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skip bad JSON in {file_path.name}:{line_num}: {e}")
                    continue

                flat_rows.extend(explode_contest_record(record))
                source_rows += 1
                if sample is not None and source_rows >= sample:
                    break
        if sample is not None and source_rows >= sample:
            break

    return _finalize_parsed_frame(flat_rows, source_rows)
