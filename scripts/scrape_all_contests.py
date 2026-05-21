import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scrape_contest import scrape_contest_rankings  # noqa: E402

from loguru import logger


def main(from_contest: int, to_contest: int, force: bool) -> None:
    contests = range(from_contest, to_contest - 1, -1)
    total = len(contests)

    for i, contest in enumerate(contests, start=1):
        logger.info(f"[{i}/{total}] Starting biweekly contest {contest}")
        try:
            scrape_contest_rankings(contest_number=contest, force=force)
        except Exception as e:
            logger.error(f"Contest {contest} failed, skipping: {e}")

    logger.info("All contests done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape all LeetCode biweekly contest rankings")
    parser.add_argument("--from", type=int, default=180, dest="from_contest", help="Start contest number (default: 180)")
    parser.add_argument("--to", type=int, default=1, dest="to_contest", help="End contest number inclusive (default: 0)")
    parser.add_argument("--force", action="store_true", help="Re-scrape contests even if output file already exists")
    args = parser.parse_args()

    main(from_contest=args.from_contest, to_contest=args.to_contest, force=args.force)
