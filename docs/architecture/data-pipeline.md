# Data Pipeline

## Overview

```
LeetCode Contest API
    │
    ▼  scraper.py
data/raw/
├── contests/          ← one file per contest, immutable
└── problems/          ← problem metadata snapshot
    │
    ▼  preprocessor.py
data/processed/
├── interactions.parquet
└── problems_clean.parquet
    │
    ▼  features/*.py
data/features/
├── user_features.parquet
├── problem_features.parquet
└── interaction_matrix/
```

## Stage 1: Scraping (`src/lrs/data/scraper.py`)

**Source**: LeetCode Contest API  
**Target**: ~50 contests × 20k–30k participants = 1M+ interactions

Fields collected per contest submission:
- `user_id` — anonymized or hashed username
- `contest_id`, `contest_date`
- `problem_id`, `problem_slug`
- `solved` — boolean
- `finish_time` — seconds from contest start to submission
- `penalty_count` — number of wrong submissions
- `language` — programming language used
- `user_rating` — LeetCode contest rating at time of contest

**Rule**: Raw files are never modified after scraping. Re-scrape creates a new versioned file.

## Stage 2: Preprocessing (`src/lrs/data/preprocessor.py`)

Transformations applied:
- Drop duplicate submissions (same user + problem in same contest)
- Normalize `finish_time` to minutes; cap outliers at 90th percentile
- Join problem metadata (tags, acceptance rate, difficulty label)
- Compute a `difficulty_proxy` from `finish_time` + `penalty_count` (see ADR when written)
- Output: `data/processed/interactions.parquet` and `data/processed/problems_clean.parquet`

## Stage 3: Feature Engineering (`src/lrs/features/`)

### User Features (`user_features.py`)
- `avg_finish_time_by_tag` — per-tag average finish time
- `solve_rate_by_difficulty` — fraction solved at each difficulty proxy bucket
- `penalty_rate` — penalties / total attempts
- `rating_trajectory` — rating delta over last N contests

### Problem Features (`problem_features.py`)
- `tag_vector` — multi-hot encoding of problem tags
- `acceptance_rate` — global acceptance rate from LeetCode
- `difficulty_proxy` — contest-derived difficulty (vs static Easy/Medium/Hard)
- `avg_peer_finish_time` — average finish time across all users who attempted it

### Interaction Matrix (`interaction_matrix.py`)
- Sparse CSR matrix: rows = users, columns = problems
- Cell value = composite "rating" derived from `solved` + `finish_time` percentile
- Saved as scipy sparse matrix for CF models

## Data Schema

### `interactions.parquet`

| Column | Type | Notes |
|---|---|---|
| `user_id` | str | Hashed username |
| `contest_id` | str | e.g. `weekly-contest-400` |
| `problem_id` | str | LeetCode problem slug |
| `solved` | bool | |
| `finish_time_min` | float | Normalized finish time in minutes |
| `penalty_count` | int | |
| `language` | str | |
| `user_rating` | float | Rating at contest time |
| `difficulty_proxy` | float | Computed composite difficulty |

### `problems_clean.parquet`

| Column | Type | Notes |
|---|---|---|
| `problem_id` | str | LeetCode problem slug |
| `title` | str | |
| `tags` | list[str] | e.g. `["Dynamic Programming", "Graph"]` |
| `difficulty_label` | str | Easy / Medium / Hard |
| `acceptance_rate` | float | Global acceptance rate |
