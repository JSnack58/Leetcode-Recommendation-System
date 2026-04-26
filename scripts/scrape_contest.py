# HOW TO GET YOUR LEETCODE COOKIES
# 1. Log into leetcode.com in Chrome
# 2. Press F12 to open DevTools
# 3. Go to the Application tab -> Cookies -> https://leetcode.com
# 4. Copy the values for "LEETCODE_SESSION", "csrftoken", and "cf_clearance"
#    (cf_clearance is set by Cloudflare after the browser passes its JS challenge)
#
# USAGE (pick one):
#   Option A - environment variables (set once, reuse):
#     export LEETCODE_SESSION=<your_value>
#     export LEETCODE_CSRF=<your_value>
#     export LEETCODE_CF_CLEARANCE=<your_value>
#     python scripts/scrape_contest.py
#
#   Option B - CLI flags:
#     python scripts/scrape_contest.py --session <your_value> --csrf <your_value> --cf-clearance <your_value>

import os
import time
import json
import argparse
from pathlib import Path
from curl_cffi import requests  # impersonates Chrome TLS fingerprint to bypass Cloudflare
from dotenv import load_dotenv
from loguru import logger

load_dotenv(Path(__file__).parent.parent / ".env.scraper")

def _load_completed_pages(output_file: Path) -> set[int]:
    """Return the set of page numbers already written to the output file."""
    completed = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["_page"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def scrape_contest_rankings(start_page=1, end_page=835, session_cookie=None, csrf_token=None, cf_clearance=None, force=False):
    session_cookie = session_cookie or os.environ.get("LEETCODE_SESSION")
    csrf_token = csrf_token or os.environ.get("LEETCODE_CSRF")
    cf_clearance = cf_clearance or os.environ.get("LEETCODE_CF_CLEARANCE")

    if not session_cookie or not csrf_token or not cf_clearance:
        raise ValueError(
            "LeetCode credentials are required.\n"
            "Set LEETCODE_SESSION, LEETCODE_CSRF, and LEETCODE_CF_CLEARANCE environment variables, "
            "or pass --session, --csrf, and --cf-clearance as CLI arguments."
        )

    base_url = "https://leetcode.com/contest/api/ranking/biweekly-contest-181/"
    # Absolute path anchored to the project root — consistent regardless of where the script is run from
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "contests"
    output_file = output_dir / "biweekly-contest-181.jsonl"
    region = "global_v2"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Cookie": f"LEETCODE_SESSION={session_cookie}; csrftoken={csrf_token}; cf_clearance={cf_clearance}",
        "Referer": "https://leetcode.com/contest/biweekly-contest-181/",
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    if force and output_file.exists():
        output_file.unlink()
        logger.info("--force: removed existing output file.")

    completed_pages = _load_completed_pages(output_file)
    logger.info(f"Starting scrape: pages {start_page}–{end_page} | already done: {len(completed_pages)}")

    with open(output_file, "a", encoding="utf-8") as out:
        for page in range(start_page, end_page + 1):
            if page in completed_pages:
                logger.debug(f"Page {page} already in output, skipping.")
                continue

            try:
                logger.info(f"Fetching page {page}/{end_page}...")
                response = requests.get(
                    base_url,
                    params={"pagination": page, "region": region},
                    headers=headers,
                    timeout=10,
                    impersonate="chrome",
                )
                response.raise_for_status()
                data = response.json()

                # Write one JSONL line per contestant entry, tagged with page number.
                # This keeps the file flat and directly loadable via pd.read_json(..., lines=True).
                for entry in data.get("total_rank", []):
                    entry["_page"] = page
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out.flush()

                logger.success(f"Page {page}: wrote {len(data.get('total_rank', []))} entries")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for page {page}: {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON for page {page}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error on page {page}: {e}")

            time.sleep(4)

    logger.info(f"Scraping complete. Output: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape LeetCode contest rankings")
    parser.add_argument("--start", type=int, default=1, help="Start page number")
    parser.add_argument("--end", type=int, default=835, help="End page number")
    parser.add_argument("--session", type=str, default=None, help="LEETCODE_SESSION cookie value (falls back to env var)")
    parser.add_argument("--csrf", type=str, default=None, help="csrftoken cookie value (falls back to env var)")
    parser.add_argument("--cf-clearance", type=str, default=None, dest="cf_clearance", help="cf_clearance cookie value (falls back to env var)")
    parser.add_argument("--force", action="store_true", help="Delete existing output file and re-scrape from scratch")
    args = parser.parse_args()

    scrape_contest_rankings(start_page=args.start, end_page=args.end, session_cookie=args.session, csrf_token=args.csrf, cf_clearance=args.cf_clearance, force=args.force)
