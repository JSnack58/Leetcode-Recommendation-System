import math
import time

from loguru import logger

from ..client import LeetCodeClient
from ..writers import JsonlWriter

_ENTRIES_PER_PAGE = 25
_RATE_LIMIT_SECONDS = 2


def scrape_contest(
    client: LeetCodeClient,
    writer: JsonlWriter,
    contest_number: int,
    start_page: int = 1,
    end_page: int | None = None,
) -> None:
    base_url = f"https://leetcode.com/contest/api/ranking/biweekly-contest-{contest_number}/"
    region = "global_v2"

    if end_page is None:
        resp = client.get(base_url, params={"pagination": 1, "region": region})
        resp.raise_for_status()
        user_num = resp.json().get("user_num", 0)
        end_page = max(1, math.ceil(user_num / _ENTRIES_PER_PAGE))
        logger.info(f"Contest {contest_number}: {user_num} users → {end_page} pages")

    already_done = sum(1 for p in range(start_page, end_page + 1) if writer.is_done(p))
    logger.info(f"Pages {start_page}–{end_page} | already done: {already_done}")

    for page in range(start_page, end_page + 1):
        if writer.is_done(page):
            logger.debug(f"Page {page} already done, skipping.")
            continue

        try:
            resp = client.get(base_url, params={"pagination": page, "region": region})
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("total_rank", [])
            writer.write_page(page, entries)
            logger.success(f"Page {page}/{end_page}: {len(entries)} entries")
        except Exception as e:
            logger.error(f"Page {page} failed: {e}")

        time.sleep(_RATE_LIMIT_SECONDS)
