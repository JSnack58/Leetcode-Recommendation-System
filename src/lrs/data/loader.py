"""Load raw and processed data into DataFrames.

The raw-side helpers read files written by ``scraper.py``; the processed-side
helpers (``load_interactions``, ``load_problems``) will be implemented once
``preprocessor.py`` lands.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from lrs import config
from lrs.data.scraper import contest_raw_dir


def iter_raw_contest_pages(
    contest_slug: str,
    *,
    raw_dir: Path | None = None,
) -> Iterator[dict]:
    """Yield the raw JSON payload for each page of a contest in page order."""
    directory = contest_raw_dir(contest_slug, raw_dir=raw_dir)
    if not directory.exists():
        return
    for path in sorted(directory.glob("page_*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def load_raw_contest(
    contest_slug: str,
    *,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Flatten one contest's raw pages into a ``total_rank`` DataFrame.

    Each row is one (user, contest) pair, with columns matching the fields
    returned by the LeetCode contest ranking API (``username``, ``rank``,
    ``score``, ``finish_time``, ...). Does not join ``submissions`` — that
    join happens in the preprocessor.
    """
    rows: list[dict] = []
    for payload in iter_raw_contest_pages(contest_slug, raw_dir=raw_dir):
        rows.extend(payload.get("total_rank", []))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["contest_slug"] = contest_slug
    return df


def load_interactions() -> pd.DataFrame:
    """Load ``data/processed/interactions.parquet``.

    Not implemented yet — depends on the preprocessor landing.
    """
    path = config.PROCESSED_DIR / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run the preprocessor first.")
    return pd.read_parquet(path)


def load_problems() -> pd.DataFrame:
    """Load ``data/processed/problems_clean.parquet``.

    Not implemented yet — depends on the preprocessor landing.
    """
    path = config.PROCESSED_DIR / "problems_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run the preprocessor first.")
    return pd.read_parquet(path)
