"""Central configuration for LRS.

All paths and constants are defined here. Environment variables override defaults.
Load .env before importing this module (done automatically via python-dotenv).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("LRS_DATA_DIR", _REPO_ROOT / "data"))
MODELS_DIR = Path(os.getenv("LRS_MODELS_DIR", _REPO_ROOT / "models"))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

# Optional session cookie for authenticated API requests. The public contest
# ranking endpoint does NOT require this — it is only used when future
# endpoints (e.g., user profile scraping) need auth.
LEETCODE_SESSION: str = os.getenv("LEETCODE_SESSION", "")

# Base URL for LeetCode API. Override to point at a local mock during tests.
LEETCODE_BASE_URL: str = os.getenv("LEETCODE_BASE_URL", "https://leetcode.com")

# User-Agent header. Identify this as a research project, not a generic bot.
SCRAPE_USER_AGENT: str = os.getenv(
    "SCRAPE_USER_AGENT",
    "LRS-Research/0.1 (SJSU CMPE 256 term project; contact kayvaun.khoshkhou@sjsu.edu)",
)

# Rate limit — requests per second for the contest ranking endpoint.
# Default 0.25 rps = 1 request every 4 seconds (conservative; see ADR-0002).
SCRAPE_RATE_LIMIT_RPS: float = float(os.getenv("SCRAPE_RATE_LIMIT_RPS", "0.25"))

# Max retries for transient HTTP failures (429, 5xx, network).
SCRAPE_RETRY_MAX: int = int(os.getenv("SCRAPE_RETRY_MAX", "5"))

# Exponential backoff base (seconds). Retry delay = base * (2 ** attempt) + jitter.
SCRAPE_BACKOFF_BASE_SEC: float = float(os.getenv("SCRAPE_BACKOFF_BASE_SEC", "2.0"))

# Per-request timeout (seconds).
SCRAPE_TIMEOUT_SEC: float = float(os.getenv("SCRAPE_TIMEOUT_SEC", "30.0"))

# Entries per contest ranking page (LeetCode's fixed page size).
CONTEST_PAGE_SIZE: int = 25

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# Recommendation tier thresholds (all overridable via environment variables)
# ---------------------------------------------------------------------------

# Edge of Competence: predicted solve probability window
EDGE_P_LOW: float = float(os.getenv("EDGE_P_LOW", "0.35"))
EDGE_P_HIGH: float = float(os.getenv("EDGE_P_HIGH", "0.65"))

# Blind Spots: minimum divergence from peer group average
BLIND_SPOT_THRESHOLD: float = float(os.getenv("BLIND_SPOT_THRESHOLD", "0.15"))

# Confidence Builders: minimum predicted solve probability
CONFIDENCE_P_MIN: float = float(os.getenv("CONFIDENCE_P_MIN", "0.75"))

# Number of problems per tier
TIER_SIZE: int = int(os.getenv("TIER_SIZE", "5"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
