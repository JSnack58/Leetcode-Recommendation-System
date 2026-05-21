"""Human-readable recommendation explanations for the web UI."""

from __future__ import annotations

import ast
from typing import Any


def slug_to_title(slug: str, title_map: dict[str, str]) -> str:
    return title_map.get(slug, slug.replace("-", " ").title())


def _user_tag_scores(user_feat: Any) -> dict[str, float]:
    if user_feat is None:
        return {}
    raw = user_feat.get("tag_scores") if hasattr(user_feat, "get") else None
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def build_signals(
    slug: str,
    tags: list[str],
    tier: str,
    p_solve: float,
    user_tag_scores: dict[str, float],
    peer_tag_scores: dict[str, float],
    slug_neighbors: dict[str, set[str]],
    solved_slugs: set[str],
    struggle_slugs: set[str],
    title_map: dict[str, str],
) -> list[str]:
    """Return explanation strings (one per signal)."""
    signals: list[str] = []

    # Peer blind-spot style
    for tag in tags:
        user_rate = user_tag_scores.get(tag)
        peer_rate = peer_tag_scores.get(tag)
        if user_rate is None or peer_rate is None:
            continue
        if user_rate < peer_rate - 0.15:
            user_pct = int(round(user_rate * 100))
            peer_pct = int(round(peer_rate * 100))
            signals.append(
                f"Peers at your level solve {tag} problems more often "
                f"(you: {user_pct}% vs peers: {peer_pct}%)"
            )

    # Graph similarity to solved or struggled problems
    neighbors = slug_neighbors.get(slug, set())
    for solved in solved_slugs:
        if solved in neighbors:
            title = slug_to_title(solved, title_map)
            signals.append(f"Similar to {title}, which you solved.")
            break

    if not any("Similar to" in s for s in signals):
        for struggle in struggle_slugs:
            if struggle in neighbors or slug in slug_neighbors.get(struggle, set()):
                title = slug_to_title(struggle, title_map)
                signals.append(f"Related to {title}, where you had penalties in contests.")
                break

    # Edge-of-competence / tier context
    if tier == "edge_of_competence":
        pct = int(round(p_solve * 100))
        signals.append(
            f"Edge of competence: ~{pct}% estimated solve likelihood "
            f"(relative ranking among candidates)."
        )
    elif tier == "confidence_builders":
        pct = int(round(p_solve * 100))
        signals.append(f"High confidence: ~{pct}% predicted solve rate — good for reinforcement.")
    elif tier == "blind_spots" and not signals:
        signals.append("Targets a topic where you underperform vs similar-rated peers.")

    if not signals:
        pct = int(round(p_solve * 100))
        signals.append(f"Recommended based on contest patterns (~{pct}% predicted solve rate).")

    return signals
