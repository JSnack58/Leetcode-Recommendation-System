"""Load raw and processed data into DataFrames.

The raw-side helpers read JSONL files written by the scraper;
the processed-side helpers read parquet files produced by the preprocessor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lrs.config import RAW_DIR, PROCESSED_DIR


def load_raw_contests() -> pd.DataFrame:
    """Read all JSONL files from data/raw/contests/ and flatten into a
    user-problem interaction DataFrame.

    For each user in a contest, solved problems produce a row with solved=1.
    Problems the user did NOT attempt in that contest produce a negative row
    with solved=0.

    Returns
    -------
    pd.DataFrame
        Columns: user_slug, contest_slug, question_id, solved, fail_count,
        solve_time_seconds, lang, rank, score, finish_time, country_code.
    """
    contests_dir = RAW_DIR / "contests"
    jsonl_files = sorted(contests_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {contests_dir}")

    all_rows = []

    for jsonl_path in jsonl_files:
        contest_slug = jsonl_path.stem

        # First pass: collect all question_ids and find earliest submission
        # timestamp (proxy for contest start time).
        contest_question_ids = set()
        earliest_date = None
        records = []

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(record)
                subs = record.get("submissions", {})
                for qid, sub_info in subs.items():
                    contest_question_ids.add(qid)
                    sub_date = sub_info.get("date")
                    if sub_date is not None:
                        if earliest_date is None or sub_date < earliest_date:
                            earliest_date = sub_date

        if earliest_date is None:
            earliest_date = 0.0

        # Second pass: build rows
        for record in records:
            user_slug = record.get("user_slug", "")
            rank = record.get("rank", 0)
            score = record.get("score", 0)
            finish_time = record.get("finish_time", 0)
            country_code = record.get("country_code", "")
            subs = record.get("submissions", {})

            solved_qids = set()

            # Positive rows (solved problems)
            for qid, sub_info in subs.items():
                solved_qids.add(qid)
                sub_date = sub_info.get("date", 0)
                solve_time = sub_date - earliest_date if sub_date else np.nan
                all_rows.append({
                    "user_slug": user_slug,
                    "contest_slug": contest_slug,
                    "question_id": qid,
                    "solved": 1,
                    "fail_count": sub_info.get("fail_count", 0),
                    "solve_time_seconds": solve_time,
                    "lang": sub_info.get("lang", ""),
                    "rank": rank,
                    "score": score,
                    "finish_time": finish_time,
                    "country_code": country_code,
                })

            # Negative rows (problems the user didn't attempt)
            for qid in contest_question_ids:
                if qid not in solved_qids:
                    all_rows.append({
                        "user_slug": user_slug,
                        "contest_slug": contest_slug,
                        "question_id": qid,
                        "solved": 0,
                        "fail_count": 0,
                        "solve_time_seconds": np.nan,
                        "lang": "",
                        "rank": rank,
                        "score": score,
                        "finish_time": finish_time,
                        "country_code": country_code,
                    })

    df = pd.DataFrame(all_rows)
    df["question_id"] = df["question_id"].astype(str)
    df["solved"] = df["solved"].astype(int)
    df["fail_count"] = df["fail_count"].astype(int)
    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(int)
    return df


def load_interactions() -> pd.DataFrame:
    """Read the processed interactions parquet file."""
    path = PROCESSED_DIR / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the preprocessor first."
        )
    return pd.read_parquet(path)


def load_problems() -> pd.DataFrame:
    """Read the processed problems parquet file."""
    path = PROCESSED_DIR / "problems_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the preprocessor first."
        )
    return pd.read_parquet(path)
