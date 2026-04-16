# Contest Scraping Approach

- **Status**: Accepted
- **Date**: 2026-04-15
- **Deciders**: Project team

## Context and Problem Statement

Phase 0 of the roadmap requires ~1M+ user-problem interactions from roughly 50 LeetCode
contests. Each weekly contest has 20k–30k participants, paginated 25 per page on the
public ranking endpoint (`/contest/api/ranking/{slug}/?pagination={n}`). That is ~800
pages per contest and ~40,000 pages total across the target dataset.

We need a scraper that (a) does not get our IPs rate-limited, (b) is resumable after an
interruption mid-scrape, and (c) can be tested in CI without hitting the live endpoint.
This ADR documents the shape of that scraper.

## Decision Drivers

- Must produce raw data that's auditable and re-derivable (principle: raw data is immutable)
- Must be CI-testable with no network access
- Must stay well under LeetCode's rate limits for a multi-hour run
- Must be resumable — a 40,000-page job will get interrupted
- Should not require authentication; the public contest ranking endpoint doesn't need it
- Should not be tightly coupled to the `requests` library (so advanced-model CI without
  the `advanced` group can still run scraper tests)

## Considered Options

- **Option A**: Sync `requests` client + token-bucket rate limiter + per-page JSON files on disk
- **Option B**: Async `httpx` / `aiohttp` client with concurrent page fetches
- **Option C**: A dedicated crawler framework (Scrapy or Crawl4AI) as Jonathan mentioned in Discord

## Decision Outcome

**Chosen option: Option A**, because the critical constraint is *not getting rate-limited*,
and concurrency makes that constraint harder to reason about without a meaningful speedup
given our per-host limit. Per-page JSON files on disk give resumability for free (existing
files are skipped on re-run) and keep the raw layer immutable.

The scraper is structured as a pure function that takes an injected `fetch` callable, so
the HTTP client is replaceable and every test runs without any real network access.

### Positive Consequences

- Resumable by construction: interrupting the scraper and re-running picks up exactly
  where it stopped; no checkpoint file to maintain.
- Rate limit is a single number in `config.py` (`SCRAPE_RATE_LIMIT_RPS`); changing the
  throughput doesn't touch the scraper code.
- Zero live network calls in tests — the fetcher is injected, and the rate limiter
  accepts fake `clock` / `sleep` callables for deterministic microsecond tests.
- Transient failures (429, 5xx) retry with exponential backoff and honor `Retry-After`
  when present; permanent failures (404) surface immediately.

### Negative Consequences / Risks

- Sequential fetching is slow: at 0.25 rps (1 request every 4 seconds), 40,000 pages
  takes ~44 hours of wall time. *Mitigation*: multiple team members can run the scraper
  against disjoint contest ranges in parallel, sharing the raw files via git-lfs or a
  shared object store. `scrape_contests()` takes an iterable of slugs so splitting work
  is trivial.
- Writing one file per page creates a lot of small files (~800 per contest). *Mitigation*:
  downstream preprocessing concatenates these into a single parquet file, so the small-
  file penalty is paid only once per contest and only at scrape time.

## Pros and Cons of the Options

### Option A: Sync `requests` + token-bucket rate limiter + per-page files

- Pro: Simplest thing that works; ~250 LOC.
- Pro: Resumable by file existence — no separate checkpoint state.
- Pro: Trivially testable — inject a fake `fetch` function, inject a fake clock into the limiter.
- Pro: Matches project principle "raw data is immutable" — page files are never overwritten,
  writes go through a temp file + rename.
- Con: Sequential; won't benefit from multiple cores. (Not a real downside — we are
  IP-rate-limited, not CPU-limited.)

### Option B: Async `httpx` / `aiohttp` with concurrency

- Pro: Could be faster *if* LeetCode allowed concurrency.
- Con: LeetCode appears to rate-limit per-IP, so concurrency doesn't help; it just makes
  the rate limiter harder to reason about (shared state across coroutines).
- Con: Adds a new library (`httpx` or `aiohttp`) and an async programming model for one
  I/O path that the rest of the project doesn't need.
- Con: Harder to test without a mock transport.

### Option C: Scrapy or Crawl4AI

- Pro: Batteries-included — retries, rate limiting, resumption all built in.
- Con: Heavy framework for one endpoint we fully understand. Most of Scrapy's value
  is in HTML parsing and link-following, neither of which applies here.
- Con: Hides the rate limiter and retry policy behind framework config, making it
  harder to justify "we respect their rate limits" in the final report.
- Con: New dependency and mental model for the team.

## Links

- [LeetCode contest ranking endpoint reference](https://github.com/Nnadozie/leetcode-contest-api)
- [`src/lrs/data/scraper.py`](../../src/lrs/data/scraper.py)
- [`src/lrs/data/_rate_limiter.py`](../../src/lrs/data/_rate_limiter.py)
- [Data pipeline architecture doc](../architecture/data-pipeline.md)
