"""Tests for the contest scraper.

No live network calls — all tests inject a fake ``fetch`` callable and use a
``tmp_path`` raw directory so they run in <100ms and on CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lrs.data._rate_limiter import RateLimiter
from lrs.data.loader import load_raw_contest
from lrs.data.scraper import (
    PermanentHTTPError,
    TransientHTTPError,
    _build_url,
    _expected_page_count,
    contest_raw_dir,
    fetch_page,
    page_path,
    scrape_contest,
    scrape_contests,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _mock_payload(page: int, *, user_num: int = 50) -> dict[str, Any]:
    """Build a LeetCode-shaped page payload with ``page_size`` entries."""
    page_size = 25
    start_rank = (page - 1) * page_size + 1
    total_rank = [
        {
            "username": f"user_{r:04d}",
            "user_slug": f"user_{r:04d}",
            "rank": r,
            "score": max(0, 30 - (r // 5)),
            "finish_time": 1_700_000_000 + r,
            "country_code": "US",
        }
        for r in range(start_rank, start_rank + page_size)
    ]
    return {
        "user_num": user_num,
        "questions": [{"question_id": i, "title_slug": f"q-{i}"} for i in range(1, 5)],
        "total_rank": total_rank,
        "submissions": [{} for _ in total_rank],
    }


class RecordingFetch:
    """Deterministic ``fetch`` stub that records every URL fetched."""

    def __init__(self, user_num: int = 50) -> None:
        self.user_num = user_num
        self.calls: list[str] = []

    def __call__(self, url: str, *, timeout: float) -> dict[str, Any]:
        self.calls.append(url)
        # parse ?pagination=N out of the URL
        page = int(url.split("pagination=")[1].split("&")[0])
        return _mock_payload(page, user_num=self.user_num)


class FlakyFetch:
    """Raises ``TransientHTTPError`` a configurable number of times first."""

    def __init__(self, failures_before_success: int, user_num: int = 25) -> None:
        self.failures_left = failures_before_success
        self.user_num = user_num
        self.calls = 0

    def __call__(self, url: str, *, timeout: float) -> dict[str, Any]:
        self.calls += 1
        if self.failures_left > 0:
            self.failures_left -= 1
            raise TransientHTTPError("simulated 500")
        page = int(url.split("pagination=")[1].split("&")[0])
        return _mock_payload(page, user_num=self.user_num)


def _noop_limiter() -> RateLimiter:
    """Rate limiter that never actually sleeps."""
    calls: list[float] = []
    return RateLimiter(rps=1_000_000.0, clock=lambda: 0.0, sleep=calls.append)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_build_url_shape():
    url = _build_url("https://leetcode.com", "weekly-contest-400", 3)
    assert url == (
        "https://leetcode.com/contest/api/ranking/weekly-contest-400/?pagination=3&region=global"
    )


@pytest.mark.parametrize(
    "user_num,expected",
    [
        (0, 0),
        (1, 1),
        (25, 1),
        (26, 2),
        (50, 2),
        (30_000, 1_200),
    ],
)
def test_expected_page_count(user_num: int, expected: int):
    assert _expected_page_count(user_num) == expected


def test_page_path_zero_pads(tmp_path: Path):
    p = page_path("weekly-contest-400", 7, raw_dir=tmp_path)
    assert p == tmp_path / "contests" / "weekly-contest-400" / "page_00007.json"


# ---------------------------------------------------------------------------
# fetch_page
# ---------------------------------------------------------------------------


def test_fetch_page_returns_parsed_payload():
    fetch = RecordingFetch()
    result = fetch_page(
        "weekly-contest-400",
        1,
        fetch=fetch,
        rate_limiter=_noop_limiter(),
        base_url="https://x.test",
    )

    assert result.page == 1
    assert result.contest_slug == "weekly-contest-400"
    assert result.user_num == 50
    assert len(result.total_rank) == 25
    assert fetch.calls == [
        "https://x.test/contest/api/ranking/weekly-contest-400/?pagination=1&region=global"
    ]


def test_fetch_page_retries_transient_errors():
    fetch = FlakyFetch(failures_before_success=2)

    result = fetch_page(
        "weekly-contest-400",
        1,
        fetch=fetch,
        rate_limiter=_noop_limiter(),
        backoff_base=0.0,
        rng=lambda: 0.0,
    )

    assert result.page == 1
    assert fetch.calls == 3  # 2 failures + 1 success


def test_fetch_page_gives_up_after_retry_max():
    fetch = FlakyFetch(failures_before_success=1_000)

    with pytest.raises(TransientHTTPError):
        fetch_page(
            "weekly-contest-400",
            1,
            fetch=fetch,
            rate_limiter=_noop_limiter(),
            retry_max=2,
            backoff_base=0.0,
            rng=lambda: 0.0,
        )

    assert fetch.calls == 3  # initial + 2 retries


def test_fetch_page_does_not_retry_permanent_errors():
    def fetch(url: str, *, timeout: float) -> dict[str, Any]:
        raise PermanentHTTPError("404")

    with pytest.raises(PermanentHTTPError):
        fetch_page(
            "weekly-contest-400",
            1,
            fetch=fetch,
            rate_limiter=_noop_limiter(),
        )


# ---------------------------------------------------------------------------
# scrape_contest
# ---------------------------------------------------------------------------


def test_scrape_contest_writes_expected_pages(tmp_path: Path):
    fetch = RecordingFetch(user_num=50)  # 50 users => 2 pages

    written = scrape_contest(
        "weekly-contest-400",
        fetch=fetch,
        rate_limiter=_noop_limiter(),
        raw_dir=tmp_path,
    )

    contest_dir = contest_raw_dir("weekly-contest-400", raw_dir=tmp_path)
    assert written == 2
    assert (contest_dir / "page_00001.json").exists()
    assert (contest_dir / "page_00002.json").exists()
    assert len(fetch.calls) == 2

    # Payloads round-trip cleanly
    payload = json.loads((contest_dir / "page_00001.json").read_text())
    assert payload["user_num"] == 50
    assert len(payload["total_rank"]) == 25


def test_scrape_contest_is_resumable(tmp_path: Path):
    fetch = RecordingFetch(user_num=75)  # 3 pages

    # First run: fetch everything.
    scrape_contest(
        "weekly-contest-400",
        fetch=fetch,
        rate_limiter=_noop_limiter(),
        raw_dir=tmp_path,
    )
    assert len(fetch.calls) == 3

    # Second run: should skip all existing pages.
    fetch2 = RecordingFetch(user_num=75)
    written = scrape_contest(
        "weekly-contest-400",
        fetch=fetch2,
        rate_limiter=_noop_limiter(),
        raw_dir=tmp_path,
    )

    assert written == 0
    assert fetch2.calls == [], "no HTTP calls when all pages are on disk"


def test_scrape_contest_respects_max_pages(tmp_path: Path):
    fetch = RecordingFetch(user_num=250)  # would be 10 pages

    written = scrape_contest(
        "weekly-contest-400",
        fetch=fetch,
        rate_limiter=_noop_limiter(),
        raw_dir=tmp_path,
        max_pages=3,
    )

    assert written == 3
    assert len(fetch.calls) == 3


def test_scrape_contests_continues_after_single_failure(tmp_path: Path):
    good_fetch = RecordingFetch(user_num=25)

    def fetch(url: str, *, timeout: float) -> dict[str, Any]:
        if "weekly-contest-bad" in url:
            raise PermanentHTTPError("404")
        return good_fetch(url, timeout=timeout)

    results = scrape_contests(
        ["weekly-contest-bad", "weekly-contest-400"],
        fetch=fetch,
        rate_limiter=_noop_limiter(),
        raw_dir=tmp_path,
    )

    assert results == {"weekly-contest-bad": 0, "weekly-contest-400": 1}
    assert page_path("weekly-contest-400", 1, raw_dir=tmp_path).exists()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_load_raw_contest_flattens_pages(tmp_path: Path):
    fetch = RecordingFetch(user_num=50)
    scrape_contest(
        "weekly-contest-400",
        fetch=fetch,
        rate_limiter=_noop_limiter(),
        raw_dir=tmp_path,
    )

    df = load_raw_contest("weekly-contest-400", raw_dir=tmp_path)

    assert len(df) == 50
    assert set(df["contest_slug"]) == {"weekly-contest-400"}
    assert df["rank"].tolist() == list(range(1, 51))


def test_load_raw_contest_returns_empty_frame_for_unknown_contest(tmp_path: Path):
    df = load_raw_contest("weekly-contest-never-scraped", raw_dir=tmp_path)
    assert df.empty
