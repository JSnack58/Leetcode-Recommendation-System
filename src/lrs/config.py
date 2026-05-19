"""Central configuration for LRS.

All paths and constants are defined here. Environment variables override defaults.
Load .env before importing this module (done automatically via python-dotenv).
"""

from pathlib import Path

from dotenv import load_dotenv
import os

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

LEETCODE_SESSION: str = os.getenv("LEETCODE_SESSION", "")

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

# Ensemble weights
ENSEMBLE_W_ALS: float = float(os.getenv("ENSEMBLE_W_ALS", "0.7"))
ENSEMBLE_W_CONTENT: float = float(os.getenv("ENSEMBLE_W_CONTENT", "0.3"))
GRAPH_BOOST_ALPHA: float = float(os.getenv("GRAPH_BOOST_ALPHA", "0.05"))

SIMILARITY_GRAPH_GML = DATA_DIR / "leetcode_problems_graph.gml"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
