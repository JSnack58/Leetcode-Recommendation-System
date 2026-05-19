"""Tests for contest JSONL parsing."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from lrs.data.contest_parser import explode_contest_record, parse_contest_jsonl


def test_explode_contest_record():
    record = {
        "user_slug": "alice",
        "contest_id": 100,
        "rank": 5,
        "submissions": {
            "1": {"fail_count": 0, "lang": "python", "date": 1000},
            "2": {"fail_count": 2, "lang": "cpp", "date": 2000},
        },
    }
    rows = explode_contest_record(record)
    assert len(rows) == 2
    assert rows[0]["solved"] is True
    assert rows[1]["penalty_count"] == 2
    assert rows[1]["solved"] is False


def test_parse_contest_jsonl(tmp_path: Path):
    line = json.dumps(
        {
            "user_slug": "bob",
            "contest_id": 1,
            "rank": 1,
            "submissions": {"42": {"fail_count": 0, "lang": "java", "date": 500}},
        }
    )
    f = tmp_path / "biweekly-contest-0.jsonl"
    f.write_text(line + "\n")
    df = parse_contest_jsonl(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["frontend_id"] == "42"
    assert "rank_percentile" in df.columns
