#!/usr/bin/env python3
"""Build processed data and features from raw contest JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from lrs.config import FEATURES_DIR, PROCESSED_DIR, RAW_DIR
from lrs.data.preprocessor import preprocess_contest_data
from lrs.features.interaction_matrix import build_interaction_matrix
from lrs.features.problem_features import compute_problem_features
from lrs.features.user_features import compute_user_features


def ensure_raw_contests(raw_dir: Path, repo_root: Path) -> Path:
    """Use data/raw/contests or symlink from contest data/."""
    contests_dir = raw_dir / "contests"
    contests_dir.mkdir(parents=True, exist_ok=True)

    if list(contests_dir.glob("*.jsonl")):
        return contests_dir

    alt = repo_root / "contest data"
    if alt.exists() and list(alt.glob("*.jsonl")):
        print(f"Using contest files from {alt}")
        return alt

    raise FileNotFoundError(
        f"No contest JSONL in {contests_dir} or {alt}. "
        "Place files under data/raw/contests/ or contest data/."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LRS dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Combined contest JSONL (default: data/raw/contests/ or contest data/)",
    )
    parser.add_argument("--sample", type=int, default=None, help="Limit source contest rows")
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    default_combined = repo_root / "combined_contest_data.jsonl"

    if args.input is not None:
        raw_file = args.input.resolve()
        raw_dir = None
        print(f"Using combined file: {raw_file}")
    elif default_combined.exists():
        raw_file = default_combined
        raw_dir = None
        print(f"Using combined file: {raw_file}")
    else:
        raw_file = None
        raw_dir = ensure_raw_contests(RAW_DIR, repo_root)

    print("=== Preprocessing contest data ===")
    interactions, _ = preprocess_contest_data(
        raw_dir=raw_dir,
        output_dir=PROCESSED_DIR,
        sample=args.sample,
        raw_file=raw_file,
    )

    if args.skip_features:
        return

    print("=== Feature engineering ===")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    problems = pd.read_parquet(PROCESSED_DIR / "problems_clean.parquet")
    compute_user_features(interactions, FEATURES_DIR)
    compute_problem_features(interactions, problems, FEATURES_DIR)
    build_interaction_matrix(interactions, FEATURES_DIR)

    print("Done. Artifacts in", PROCESSED_DIR, "and", FEATURES_DIR)


if __name__ == "__main__":
    main()
