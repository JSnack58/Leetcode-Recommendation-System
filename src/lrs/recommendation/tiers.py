"""Three-tier recommendation logic.

This module is deliberately separate from the model layer.
Tier assignment is a business rule — it consumes raw model scores
and user features but does not depend on how scores were produced.

Tiers:
  - Edge of Competence:   solve probability in [EDGE_P_LOW, EDGE_P_HIGH]
  - Blind Spots:          tags where user underperforms vs peer group
  - Confidence Builders:  high predicted success + fast expected solve time
"""

# TODO: Implement tier assignment
# See docs/architecture/recommendation-tiers.md for full design
# Thresholds are in src/lrs/config.py and overridable via environment variables
