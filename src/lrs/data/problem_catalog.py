"""Load problem metadata from bipartite problem-tag GML graph."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from lrs.config import DATA_DIR
from lrs.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BIPARTITE_GML = DATA_DIR / "leetcode_tags_bipartite_graph.gml"


def load_problem_catalog(gml_path: Path | None = None) -> pd.DataFrame:
    """Build problem catalog from bipartite GML.

    Returns DataFrame with columns:
        frontend_id, problem_id (slug), title, tags (list[str])
    """
    gml_path = Path(gml_path or DEFAULT_BIPARTITE_GML)
    if not gml_path.exists():
        raise FileNotFoundError(f"Bipartite graph not found: {gml_path}")

    logger.info(f"Loading problem catalog from {gml_path}")
    graph = nx.read_gml(gml_path)

    problem_nodes = [
        (nid, attrs)
        for nid, attrs in graph.nodes(data=True)
        if attrs.get("type") == "problem"
    ]

    rows: list[dict] = []
    for nid, attrs in problem_nodes:
        # networkx uses GML node id as key (numeric frontend question id)
        frontend_id = str(nid)
        slug = attrs.get("titleSlug")
        if not slug:
            continue

        tags: list[str] = []
        for neighbor in graph.neighbors(nid):
            n_attrs = graph.nodes[neighbor]
            if n_attrs.get("type") == "tag":
                tags.append(n_attrs.get("title", neighbor))

        rows.append(
            {
                "frontend_id": frontend_id,
                "problem_id": slug,
                "title": attrs.get("title", slug),
                "tags": tags,
            }
        )

    catalog = pd.DataFrame(rows).drop_duplicates(subset=["frontend_id"])
    logger.info(f"Loaded {len(catalog):,} problems from catalog")
    return catalog


def join_interactions_to_catalog(
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map frontend IDs to slugs and attach tags.

    Drops rows with no catalog match.
    """
    merged = interactions.merge(
        catalog,
        on="frontend_id",
        how="inner",
        suffixes=("", "_cat"),
    )
    dropped = len(interactions) - len(merged)
    if dropped > 0:
        logger.warning(f"Dropped {dropped:,} interactions with no catalog match")

    problems_clean = catalog[["problem_id", "title", "tags"]].drop_duplicates(
        subset=["problem_id"]
    )
    return merged, problems_clean
