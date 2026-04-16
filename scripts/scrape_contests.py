"""Scrape LeetCode contest ranking pages to ``data/raw/contests/``.

Usage:
    # Scrape a single contest
    uv run python scripts/scrape_contests.py --contest weekly-contest-400

    # Scrape a range of weekly contests
    uv run python scripts/scrape_contests.py --weekly 380 400

    # Scrape a range of biweekly contests
    uv run python scripts/scrape_contests.py --biweekly 120 130

    # Smoke test: only fetch the first 2 pages of a contest
    uv run python scripts/scrape_contests.py --contest weekly-contest-400 --max-pages 2

The scraper is resumable: pages already on disk are skipped, so interrupting
and re-running the command picks up where it left off.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator

from lrs.data.scraper import scrape_contests
from lrs.utils.logging import configure_logging, logger


def _expand_contest_slugs(args: argparse.Namespace) -> Iterator[str]:
    if args.contest:
        yield from args.contest
    if args.weekly:
        start, end = args.weekly
        for n in range(start, end + 1):
            yield f"weekly-contest-{n}"
    if args.biweekly:
        start, end = args.biweekly
        for n in range(start, end + 1):
            yield f"biweekly-contest-{n}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape LeetCode contest ranking pages.",
    )
    parser.add_argument(
        "--contest",
        action="append",
        default=[],
        metavar="SLUG",
        help=("Contest slug (e.g. weekly-contest-400). Can be passed multiple times."),
    )
    parser.add_argument(
        "--weekly",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help="Inclusive range of weekly contest numbers to scrape.",
    )
    parser.add_argument(
        "--biweekly",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help="Inclusive range of biweekly contest numbers to scrape.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=("Cap the number of pages per contest (smoke-test / rate-limit friendly)."),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv)

    slugs = list(_expand_contest_slugs(args))
    if not slugs:
        logger.error("No contests specified. Use --contest, --weekly, or --biweekly.")
        return 2

    logger.info("Scraping {} contest(s): {}", len(slugs), ", ".join(slugs[:10]))
    results = scrape_contests(slugs, max_pages_per_contest=args.max_pages)
    total = sum(results.values())
    logger.info("Done. {} new pages written across {} contests.", total, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
