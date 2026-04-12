# Recommendation Tiers Design

## Overview

The system produces three distinct tiers of recommendations per user, each serving a different learning objective.

```
User Profile
(rating, tag scores, history)
        │
        ▼
  [ Model Scores ]          ← raw predicted scores for all unsolved problems
        │
        ▼
  [ Tier Assigner ]         ← src/lrs/recommendation/tiers.py
        │
        ├──► Edge of Competence
        ├──► Blind Spots
        └──► Confidence Builders
        │
        ▼
  [ Filters ]               ← remove solved, dedup, enforce tag diversity
        │
        ▼
  Final Recommendations
```

---

## Tier 1: Edge of Competence

**Goal**: Problems that challenge the user just above their current ceiling.

**Definition**: Problems where the user's predicted solve probability is in the range `[P_low, P_high]` — high enough that solving is plausible, low enough that it requires real effort.

**Signal used**:
- User's latent factor score from the CF model
- `difficulty_proxy` of unsolved problems
- User's rating percentile

**Tuning**: `P_low` and `P_high` are configurable in `src/lrs/config.py`.

---

## Tier 2: Blind Spots

**Goal**: Topics where the user underperforms compared to peers of similar rating.

**Definition**: Tags where `user_tag_score < peer_group_avg_tag_score - threshold`.

**Signal used**:
- Per-tag latent factors from the model
- Peer group defined as users within ±100 rating points
- `avg_finish_time_by_tag` and `solve_rate_by_tag` from feature engineering

**Novel aspect**: Unlike standard CF (which recommends what you're already good at), this tier deliberately targets divergence — finding gaps that standard similarity-based methods would never surface.

---

## Tier 3: Confidence Builders

**Goal**: High-success-rate problems that reinforce foundations and build speed.

**Definition**: Problems where predicted solve probability > `P_confidence` AND predicted finish time is below the user's median finish time for that difficulty level.

**Signal used**:
- High predicted solve probability (from CF or NCF)
- Temporal performance: `avg_peer_finish_time` × user's speed ratio
- Tag diversity filter to avoid over-indexing on one topic

---

## Implementation Notes

- Tier logic lives in `src/lrs/recommendation/tiers.py`, not in any model class.
- Models expose a common interface (`BaseRecommender.recommend()`) that returns raw scores.
- The tier assigner consumes those raw scores + user features to make tier assignments.
- This separation means swapping models does not require changing tier logic.

## Thresholds (configurable)

| Parameter | Default | Description |
|---|---|---|
| `EDGE_P_LOW` | 0.35 | Minimum solve probability for Edge of Competence |
| `EDGE_P_HIGH` | 0.65 | Maximum solve probability for Edge of Competence |
| `BLIND_SPOT_THRESHOLD` | 0.15 | Minimum divergence from peer average to qualify |
| `CONFIDENCE_P_MIN` | 0.75 | Minimum solve probability for Confidence Builders |
| `TIER_SIZE` | 5 | Number of problems per tier per recommendation request |

All thresholds are defined in `src/lrs/config.py` and overridable via environment variables.
