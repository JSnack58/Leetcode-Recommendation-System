# TEAM SCRAPING INSTRUCTIONS
# Each machine needs its own .env.scraper with its own LeetCode credentials.
# See .env.scraper for the required variables and how to get them from Chrome DevTools.
#
# 3-way split (run one command per machine from the project root):
#   Machine 1:  python -m scripts.data_collection.run --from 180 --to 121
#   Machine 2:  python -m scripts.data_collection.run --from 120 --to 61
#   Machine 3:  python -m scripts.data_collection.run --from 60  --to 0
#
# If a run is interrupted, re-run the same command — already-scraped pages are skipped.
# To re-scrape a range from scratch: add --force

import argparse
from pathlib import Path

from loguru import logger

from .client import LeetCodeClient
from .writers import JsonlWriter
from .scrapers.contests import scrape_contest

_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "contests"


def main(from_contest: int, to_contest: int, force: bool) -> None:
    client = LeetCodeClient.from_env()

    contests = list(range(from_contest, to_contest - 1, -1))
    total = len(contests)

    for i, contest_number in enumerate(contests, start=1):
        logger.info(f"[{i}/{total}] Biweekly contest {contest_number}")
        output_file = _OUTPUT_DIR / f"biweekly-contest-{contest_number}.jsonl"

        with JsonlWriter(output_file) as writer:
            if force:
                writer.reset()
            try:
                scrape_contest(client, writer, contest_number)
            except Exception as e:
                logger.error(f"Contest {contest_number} failed, skipping: {e}")

    logger.info("All contests complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape all LeetCode biweekly contest rankings")
    parser.add_argument("--from", type=int, default=180, dest="from_contest", help="Start contest number (default: 180)")
    parser.add_argument("--to", type=int, default=0, dest="to_contest", help="End contest number inclusive (default: 0)")
    parser.add_argument("--force", action="store_true", help="Re-scrape contests even if output already exists")
    args = parser.parse_args()

    main(from_contest=args.from_contest, to_contest=args.to_contest, force=args.force)
