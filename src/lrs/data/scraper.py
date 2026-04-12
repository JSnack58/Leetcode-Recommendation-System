"""LeetCode contest API scraper.

Fetches raw contest participation data and saves it to data/raw/contests/.
Raw files are immutable after creation — re-scraping creates new versioned files.
"""

# TODO: Implement scraping logic
# Key endpoints to investigate:
#   - /contest/api/ranking/{contest_slug}/
#   - /problems/{problem_slug}/
# Authentication: LEETCODE_SESSION cookie from config.LEETCODE_SESSION
