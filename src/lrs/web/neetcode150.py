"""NeetCode 150 cold-start recommendation path."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from lrs.config import NEETCODE150_PATH

DEFAULT_PATH = NEETCODE150_PATH

DIFFICULTY_P_SOLVE = {
    "Easy": 0.72,
    "Medium": 0.52,
    "Hard": 0.38,
}


@lru_cache(maxsize=1)
def load_neetcode150(path: str | None = None) -> list[dict[str, Any]]:
    """Load ordered NeetCode 150 problem list from JSON."""
    json_path = Path(path) if path else DEFAULT_PATH
    if not json_path.exists():
        raise FileNotFoundError(f"NeetCode 150 list not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    problems = data.get("problems", data if isinstance(data, list) else [])
    return sorted(problems, key=lambda p: int(p.get("order", 0)))


def neetcode_cards(count: int, path: str | None = None) -> list[dict[str, Any]]:
    """Build UI-ready problem cards for cold-start users."""
    problems = load_neetcode150(path)[:count]
    cards: list[dict[str, Any]] = []

    for p in problems:
        order = int(p.get("order", 0))
        slug = p["slug"]
        title = p.get("title", slug.replace("-", " ").title())
        topic = p.get("topic", "General")
        difficulty = p.get("difficulty", "Medium")
        base = DIFFICULTY_P_SOLVE.get(difficulty, 0.5)
        # Slight progression: earlier problems in the roadmap feel more approachable
        progress_bonus = max(0.0, 0.08 - (order / 150) * 0.08)
        p_solve = min(0.85, base + progress_bonus)

        cards.append(
            {
                "slug": slug,
                "title": title,
                "tags": [topic, difficulty],
                "tier": "neetcode150",
                "p_solve": round(p_solve, 4),
                "signals": [
                    f"NeetCode 150 roadmap — problem #{order} in the curated interview path.",
                    f"{difficulty} · {topic}: recommended starting point without contest history.",
                ],
                "signal_count": 2,
            }
        )

    return cards
