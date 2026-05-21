# LeetCode Recommendation System

A personalized LeetCode problem recommender built on **contest performance data**, problem **tags**, and LeetCode’s **similar-problem graph**. It targets the *edge of competence* — problems hard enough to grow on, not so hard they discourage you.

**Warm start:** users found in historical contest data get ML-personalized picks.  
**Cold start:** everyone else gets the curated [**NeetCode 150**](neetcode150.json) roadmap.

---

## Features

- **Three recommendation tiers** (when contest history exists):
  - **Edge of Competence** — ~35–65% estimated solve likelihood (productive struggle)
  - **Blind Spots** — topics where you underperform vs. similar-rated peers
  - **Confidence Builders** — higher-confidence reinforcement problems
- **Hybrid scoring:** ALS collaborative filtering + tag-based content model + similarity-graph boosts
- **Web UI** — simple Flask app to enter a LeetCode username and get explanations
- **CLI** — batch JSON output for scripts and evaluation

---

## Requirements

| Requirement | Notes |
|-------------|--------|
| **Python 3.11+** | Tested on 3.11–3.13 |
| **~8 GB+ RAM** | Full contest build + model load; use `--sample` on smaller machines |
| **Disk space** | ~2 GB for processed data + model artifacts after full pipeline |

All commands below use the project **virtualenv** at `.venv/`. Do **not** use system `python` or `pip` directly — scripts will refuse to run otherwise.

---

## Quick start (full pipeline)

### 1. Clone and create the virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/Leetcode-Recommendation-System.git
cd Leetcode-Recommendation-System

python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Key settings in `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LRS_DATA_DIR` | `./data` | Data root |
| `LRS_MODELS_DIR` | `./models` | Saved models |
| `FLASK_PORT` | `5001` | Web UI port (5000 is common on macOS) |
| `NEETCODE150_PATH` | `./neetcode150.json` | Cold-start problem list |

### 3. Provide contest data

You need **historical contest participation** as JSONL (one JSON object per line per user per contest).

**Option A — single combined file (recommended)**

Place your file at the repo root:

```
combined_contest_data.jsonl
```

**Option B — per-contest files**

```
contest data/biweekly-contest-0.jsonl
contest data/biweekly-contest-1.jsonl
...
```

Or symlink/copy them into `data/raw/contests/`.

Each line should look like:

```json
{
  "contest_id": 355,
  "user_slug": "your-leetcode-slug",
  "rank": 42,
  "score": 18,
  "submissions": {
    "1888": {"fail_count": 0, "lang": "cpp", "date": 1615041137}
  }
}
```

Find your slug in your profile URL: `https://leetcode.com/u/<slug>/`

### 4. Graph data (included in repo)

These files under `data/` are required for problem slugs, tags, and similarity reranking:

| File | Purpose |
|------|---------|
| `data/leetcode_tags_bipartite_graph.gml` | Problem ↔ tag mapping |
| `data/leetcode_problems_graph.gml` | LeetCode “Similar Questions” edges |

`neetcode150.json` at the repo root powers **cold start** recommendations.

### 5. Build the dataset

```bash
# Full build (uses combined_contest_data.jsonl if present at repo root)
.venv/bin/python scripts/build_dataset.py

# Or explicitly:
.venv/bin/python scripts/build_dataset.py --input combined_contest_data.jsonl

# Quick smoke test (~1k contest rows):
.venv/bin/python scripts/build_dataset.py --sample 5000
```

**Outputs** (gitignored, created locally):

```
data/processed/interactions.parquet   # user–problem attempts
data/processed/problems_clean.parquet # slug, title, tags
data/features/user_features.parquet
data/features/problem_features.parquet
data/features/interaction_matrix.npz
```

This step can take **several minutes** on the full ~3.5M-row dataset.

### 6. Train models

```bash
.venv/bin/python scripts/train_baseline.py --model all
```

**Outputs:**

```
models/als/              # ALS collaborative filtering
models/content/            # Tag content-based model
models/ensemble.pkl        # Combined model for inference
models/calibration_als.pkl
models/calibration_content.pkl
```

Training on the **full** dataset is required for good personalized recommendations. A model trained only on a small `--sample` build will not cover most problems.

### 7. Run the web UI

```bash
make web
# or:
.venv/bin/python web/app.py
# or:
./web/run.sh
```

Open **http://127.0.0.1:5001** (or your `FLASK_PORT`).

- Enter a **LeetCode user slug** and count (e.g. `10`).
- **Warm start** — slug appears in contest data → personalized recommendations.
- **Cold start** — slug not found → first N problems from **NeetCode 150** in roadmap order.

> **First warm-start request is slow** (1–3+ min): loads ~80MB interactions parquet + ~1GB model bundle into memory. Later requests reuse the cache and are much faster. Cold start is fast (no model load).

### 8. CLI recommendations (optional)

```bash
.venv/bin/python scripts/generate_recommendations.py \
  --user-slug YOUR_SLUG \
  --output recommendations.json
```

Example output:

```json
{
  "user_slug": "foorest",
  "edge_of_competence": [{"slug": "...", "p_solve": 0.52, "tags": ["Array", "..."]}],
  "blind_spots": [],
  "confidence_builders": []
}
```

### 9. Evaluate (optional)

```bash
.venv/bin/python scripts/evaluate.py
```

Runs a temporal holdout backtest and prints Precision@10, NDCG@10, etc.

---

## Makefile shortcuts

All targets use `.venv/bin/python`:

```bash
make install     # pip install -e ".[dev]"
make build       # build_dataset.py
make train       # train_baseline.py --model all
make web         # Flask UI on :5001
# CLI: use generate_recommendations.py with --user-slug (see step 8)
make evaluate    # offline metrics
make test        # pytest
make lint        # ruff
```

---

## How recommendations work

```text
Contest JSONL  →  interactions.parquet  →  ALS + content models
       ↓                    ↓                        ↓
  Problem catalog     User features          Ensemble score
  (tag graph)              ↓                        ↓
                    Tier assigner  ←  Similarity graph boost
                           ↓
              Edge / Blind spot / Confidence tiers
```

| Component | Role |
|-----------|------|
| **ALS** | Learns user/problem patterns from contest attempts |
| **Content** | Tag cosine similarity from problems you’ve solved |
| **Similarity graph** | Small boost for problems near past struggles |
| **Tier logic** | Buckets scores into learning goals (not three full lists every time) |

**Note:** Displayed solve percentages are **relative rankings among candidates** when the model produces flat scores (common for users with little contest history).

---

## Project structure

```
Leetcode-Recommendation-System/
├── combined_contest_data.jsonl   # Your contest data (you provide)
├── neetcode150.json             # Cold-start roadmap (in repo)
├── contest data/                # Optional per-contest JSONL files
├── data/
│   ├── leetcode_tags_bipartite_graph.gml
│   ├── leetcode_problems_graph.gml
│   ├── processed/                 # Generated (gitignored)
│   └── features/                  # Generated (gitignored)
├── models/                        # Generated (gitignored)
├── src/lrs/                       # Core Python package
│   ├── data/                      # Parsing, preprocessing, catalog
│   ├── features/                  # Feature engineering
│   ├── models/                    # ALS, content, ensemble
│   ├── recommendation/            # Tiers, pipeline, graph reranker
│   └── web/                       # Flask service + NeetCode 150
├── scripts/
│   ├── build_dataset.py
│   ├── train_baseline.py
│   ├── generate_recommendations.py
│   └── evaluate.py
├── web/
│   ├── app.py                     # Flask entrypoint
│   ├── templates/
│   └── static/
├── tests/
└── docs/                          # Architecture & design docs
```

---

## Configuration

Tier thresholds (optional, in `.env`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `EDGE_P_LOW` | `0.35` | Min P(solve) for Edge of Competence |
| `EDGE_P_HIGH` | `0.65` | Max P(solve) for Edge of Competence |
| `CONFIDENCE_P_MIN` | `0.75` | Min P(solve) for Confidence Builders |
| `BLIND_SPOT_THRESHOLD` | `0.15` | Tag gap vs. peers for Blind Spots |
| `ENSEMBLE_W_ALS` | `0.7` | ALS weight in ensemble |
| `ENSEMBLE_W_CONTENT` | `0.3` | Content weight in ensemble |

---

## Troubleshooting

### `Run with .venv/bin/python` / venv error

Create and use the project venv:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python web/app.py
```

### `Missing processed data` / `Missing trained models`

Run the pipeline in order:

```bash
.venv/bin/python scripts/build_dataset.py --input combined_contest_data.jsonl
.venv/bin/python scripts/train_baseline.py --model all
```

### Web UI hangs on first recommendation

Expected for **warm start** — loading full parquet + large `ensemble.pkl`. Wait for the first response, or use cold start (unknown slug) to verify the UI instantly.

### Every problem shows the same % solve rate

Usually means models were trained on a **tiny sample** (`--sample 5000`) while inference runs against the full catalog. Rebuild and retrain on full data. Recent versions also spread scores when calibration collapses.

### User always gets cold start (NeetCode 150)

The slug must match `user_slug` in your contest JSONL **exactly** (case-sensitive). The system hashes it internally; only users present in `interactions.parquet` get warm start.

### Port already in use

Default port is **5001**. Change in `.env`:

```
FLASK_PORT=5002
```

### Out of memory during `build_dataset`

Use a smaller sample for development:

```bash
.venv/bin/python scripts/build_dataset.py --sample 10000
```

Then train and test the UI flow before running the full build.

---

## Development

```bash
# Tests
.venv/bin/pytest tests/ -v

# Lint
.venv/bin/python -m ruff check src/ tests/ scripts/ web/

# Wrapper for any script
./scripts/run.sh scripts/build_dataset.py --sample 1000
```

### Scraping new contest data (optional)

See [`scripts/data_collection/README.md`](scripts/data_collection/README.md). Requires LeetCode session cookies. Output goes to `data/raw/contests/`.

---

## What’s not in the MVP

- HTTP API / production deployment
- LightGCN, NCF, XGBoost ranker (stubs exist under `src/lrs/models/advanced/`)
- Dense shared-tags graph (too noisy; not used)

See [docs/architecture/system-overview.md](docs/architecture/system-overview.md) for the full design.

---

## Documentation

- [Architecture overview](docs/architecture/system-overview.md)
- [Data pipeline](docs/architecture/data-pipeline.md)
- [Recommendation tiers](docs/architecture/recommendation-tiers.md)
- [Roadmap](docs/planning/roadmap.md)

---

## License

Add your license here before publishing if this repo is public.
