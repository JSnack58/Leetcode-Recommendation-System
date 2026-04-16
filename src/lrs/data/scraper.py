"""LeetCode contest API scraper.

Fetches raw contest participation data from the public contest ranking endpoint
and saves it to ``data/raw/contests/{contest_slug}/``.

Raw files are immutable after creation — re-running the scraper skips pages
that already exist on disk, making the job resumable after interruption.

Endpoint
--------
``GET {LEETCODE_BASE_URL}/contest/api/ranking/{contest_slug}/?pagination={page}&region=global``

Returns JSON with (among other fields):

- ``user_num`` — total number of participants for the contest
- ``questions`` — list of the 4 problems in the contest
  (``question_id``, ``credit``, ``title``, ``title_slug``, ``category_slug``)
- ``total_rank`` — 25 entries per page: ``{rank, username, score, finish_time, ...}``
- ``submissions`` — parallel to ``total_rank``; per-user dict keyed by ``question_id``
  with ``{fail_count, lang, date, submission_id}`` inside.

The scraper is transport-agnostic: it takes a ``FetchFn`` callable so tests can
inject a deterministic stub without touching the network.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lrs import config
from lrs.data._rate_limiter import RateLimiter
from lrs.utils.logging import logger

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TransientHTTPError(Exception):
    """Retryable error (HTTP 429, 5xx, network blip)."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentHTTPError(Exception):
    """Non-retryable error (HTTP 404, malformed response)."""


class _FetchProtocol(Protocol):
    def __call__(self, url: str, *, timeout: float) -> dict[str, Any]: ...


FetchFn = _FetchProtocol


@dataclass(frozen=True)
class ContestPage:
    """In-memory representation of one fetched page."""

    contest_slug: str
    page: int
    payload: dict[str, Any]

    @property
    def user_num(self) -> int:
        return int(self.payload.get("user_num", 0))

    @property
    def total_rank(self) -> list[dict[str, Any]]:
        return list(self.payload.get("total_rank", []))


# ---------------------------------------------------------------------------
# HTTP fetcher (default)
# ---------------------------------------------------------------------------


def _default_fetch(url: str, *, timeout: float) -> dict[str, Any]:
    """Default HTTP fetcher using the ``requests`` library.

    Raised errors:
    - ``TransientHTTPError`` for 429/5xx and network errors
    - ``PermanentHTTPError`` for 404/malformed JSON
    """
    # Imported lazily so tests don't need `requests` on the path.
    import requests

    headers = {"User-Agent": config.SCRAPE_USER_AGENT, "Accept": "application/json"}
    if config.LEETCODE_SESSION:
        headers["Cookie"] = f"LEETCODE_SESSION={config.LEETCODE_SESSION}"

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise TransientHTTPError(f"network error: {exc}") from exc

    if resp.status_code == 404:
        raise PermanentHTTPError(f"404 for {url}")
    if resp.status_code == 429:
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        raise TransientHTTPError(f"429 rate-limited at {url}", retry_after=retry_after)
    if 500 <= resp.status_code < 600:
        raise TransientHTTPError(f"{resp.status_code} server error at {url}")
    if resp.status_code != 200:
        raise PermanentHTTPError(f"unexpected status {resp.status_code} at {url}")

    try:
        return resp.json()
    except ValueError as exc:
        raise PermanentHTTPError(f"invalid JSON at {url}: {exc}") from exc


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core scraping
# ---------------------------------------------------------------------------


def _build_url(base_url: str, contest_slug: str, page: int) -> str:
    return f"{base_url}/contest/api/ranking/{contest_slug}/?pagination={page}&region=global"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to ``path`` atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same directory ensures the rename is atomic on all POSIX filesystems.
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        tmp_path.replace(path)
    except Exception:
        # Best-effort cleanup.
        tmp_path.unlink(missing_ok=True)
        raise


def contest_raw_dir(contest_slug: str, *, raw_dir: Path | None = None) -> Path:
    """Return the raw-files directory for a contest."""
    root = raw_dir if raw_dir is not None else config.RAW_DIR
    return root / "contests" / contest_slug


def page_path(contest_slug: str, page: int, *, raw_dir: Path | None = None) -> Path:
    """Filesystem path for one contest page's raw JSON."""
    return contest_raw_dir(contest_slug, raw_dir=raw_dir) / f"page_{page:05d}.json"


def fetch_page(
    contest_slug: str,
    page: int,
    *,
    fetch: FetchFn = _default_fetch,
    rate_limiter: RateLimiter,
    base_url: str | None = None,
    timeout: float | None = None,
    retry_max: int | None = None,
    backoff_base: float | None = None,
    rng: Callable[[], float] = random.random,
) -> ContestPage:
    """Fetch one contest ranking page with rate limiting and retry/backoff.

    Retries on ``TransientHTTPError`` using exponential backoff with jitter.
    Raises ``PermanentHTTPError`` immediately on non-retryable errors, and
    after ``retry_max`` attempts on transient errors.
    """
    base = base_url if base_url is not None else config.LEETCODE_BASE_URL
    timeout_s = timeout if timeout is not None else config.SCRAPE_TIMEOUT_SEC
    retries = retry_max if retry_max is not None else config.SCRAPE_RETRY_MAX
    backoff = backoff_base if backoff_base is not None else config.SCRAPE_BACKOFF_BASE_SEC

    url = _build_url(base, contest_slug, page)

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        rate_limiter.wait()
        try:
            payload = fetch(url, timeout=timeout_s)
            return ContestPage(contest_slug=contest_slug, page=page, payload=payload)
        except PermanentHTTPError:
            raise
        except TransientHTTPError as exc:
            last_exc = exc
            if attempt == retries:
                break
            # Honor server Retry-After if present; otherwise exponential backoff.
            if exc.retry_after is not None:
                penalty = exc.retry_after
            else:
                penalty = backoff * (2**attempt) + rng()
            logger.warning(
                "transient error (attempt {}/{}): {} — backing off {:.1f}s",
                attempt + 1,
                retries,
                exc,
                penalty,
            )
            rate_limiter.penalize(penalty)

    raise TransientHTTPError(f"exhausted retries for {url}: {last_exc}")


def _expected_page_count(user_num: int, page_size: int = config.CONTEST_PAGE_SIZE) -> int:
    if user_num <= 0:
        return 0
    return math.ceil(user_num / page_size)


def scrape_contest(
    contest_slug: str,
    *,
    fetch: FetchFn = _default_fetch,
    rate_limiter: RateLimiter | None = None,
    raw_dir: Path | None = None,
    max_pages: int | None = None,
    start_page: int = 1,
) -> int:
    """Scrape one contest's full ranking to ``data/raw/contests/{slug}/``.

    Already-downloaded pages are skipped (resumable). The first page determines
    ``user_num``, which sets the total expected page count.

    Parameters
    ----------
    contest_slug:
        e.g. ``"weekly-contest-400"`` or ``"biweekly-contest-150"``.
    fetch:
        HTTP fetcher. Defaults to a requests-based implementation.
    rate_limiter:
        If omitted, a new one is built from ``config.SCRAPE_RATE_LIMIT_RPS``.
    raw_dir:
        Override ``data/raw/`` (used by tests with tmp_path).
    max_pages:
        Cap the number of pages fetched in this run (useful for smoke tests).
    start_page:
        Page number to start from (default 1). Useful when we want to re-scrape
        from a specific offset.

    Returns
    -------
    Number of pages written (not counting pages skipped because they existed).
    """
    if rate_limiter is None:
        rate_limiter = RateLimiter(rps=config.SCRAPE_RATE_LIMIT_RPS)

    out_dir = contest_raw_dir(contest_slug, raw_dir=raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch (or reuse on-disk) page 1 to discover user_num.
    first_path = page_path(contest_slug, start_page, raw_dir=raw_dir)
    if first_path.exists():
        first_payload = json.loads(first_path.read_text(encoding="utf-8"))
        first_page = ContestPage(contest_slug, start_page, first_payload)
        logger.info("reusing existing page {} for {}", start_page, contest_slug)
        pages_written = 0
    else:
        first_page = fetch_page(contest_slug, start_page, fetch=fetch, rate_limiter=rate_limiter)
        _atomic_write_json(first_path, first_page.payload)
        pages_written = 1
        logger.info(
            "fetched page {} for {}: user_num={}",
            start_page,
            contest_slug,
            first_page.user_num,
        )

    total_pages = _expected_page_count(first_page.user_num)
    if total_pages == 0:
        logger.warning("{} reports 0 participants; nothing to scrape", contest_slug)
        return pages_written

    end_page = total_pages
    if max_pages is not None:
        end_page = min(end_page, start_page + max_pages - 1)

    logger.info(
        "{}: scraping pages {}..{} (total participants={})",
        contest_slug,
        start_page,
        end_page,
        first_page.user_num,
    )

    for page in range(start_page + 1, end_page + 1):
        out_path = page_path(contest_slug, page, raw_dir=raw_dir)
        if out_path.exists():
            continue
        contest_page = fetch_page(contest_slug, page, fetch=fetch, rate_limiter=rate_limiter)
        _atomic_write_json(out_path, contest_page.payload)
        pages_written += 1
        if pages_written % 25 == 0:
            logger.info(
                "{}: wrote {} / {} pages",
                contest_slug,
                page - start_page + 1,
                end_page - start_page + 1,
            )

    logger.info(
        "{}: done — {} new pages written, {} total pages on disk",
        contest_slug,
        pages_written,
        end_page - start_page + 1,
    )
    return pages_written


def scrape_contests(
    contest_slugs: Iterable[str],
    *,
    fetch: FetchFn = _default_fetch,
    rate_limiter: RateLimiter | None = None,
    raw_dir: Path | None = None,
    max_pages_per_contest: int | None = None,
) -> dict[str, int]:
    """Scrape multiple contests sequentially, sharing a single rate limiter.

    Returns a mapping ``{contest_slug: pages_written}``.
    """
    if rate_limiter is None:
        rate_limiter = RateLimiter(rps=config.SCRAPE_RATE_LIMIT_RPS)

    results: dict[str, int] = {}
    for slug in contest_slugs:
        try:
            written = scrape_contest(
                slug,
                fetch=fetch,
                rate_limiter=rate_limiter,
                raw_dir=raw_dir,
                max_pages=max_pages_per_contest,
            )
            results[slug] = written
        except PermanentHTTPError as exc:
            logger.error("{} — permanent error, skipping: {}", slug, exc)
            results[slug] = 0
        except TransientHTTPError as exc:
            logger.error("{} — retries exhausted, skipping: {}", slug, exc)
            results[slug] = 0
    return results
