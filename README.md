# Leetcode Recommendation System

A personalized skill-gap analyzer that recommends LeetCode problems using contest performance data. Goes beyond "similar problem" suggestions to diagnose *what you need* rather than *what you like*.

## Recommendation Tiers

| Tier | Purpose |
|---|---|
| **Edge of Competence** | Problems at your current ceiling — productive struggle zone |
| **Blind Spots** | Topics where you underperform peers at your rating level |
| **Confidence Builders** | High-success-rate problems to reinforce and speed up foundations |

## Architecture Overview

- **Data**: LeetCode contest API → ~1M+ user-problem interactions
- **Baseline models**: SVD, ALS (Collaborative Filtering), Content-Based Filtering
- **Advanced models**: Neural CF (NCF), LightGCN (Graph NN), XGBoost/LightGBM ranker
- **Novelty**: Temporal performance awareness (time-to-solve), dynamic difficulty via penalties

See [docs/architecture/system-overview.md](docs/architecture/system-overview.md) for the full design.

## Quickstart

All commands below use the project **`.venv`** — do not use system `python`/`pip` directly.

```bash
# One-time: create and install into .venv
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Copy environment config
cp .env.example .env

# Build processed data + features (auto-detects combined_contest_data.jsonl at repo root)
.venv/bin/python scripts/build_dataset.py
# Or: .venv/bin/python scripts/build_dataset.py --input combined_contest_data.jsonl
# Dev sample: .venv/bin/python scripts/build_dataset.py --sample 5000

# Train ALS + content + ensemble
.venv/bin/python scripts/train_baseline.py --model all

# Generate tiered recommendations
.venv/bin/python scripts/generate_recommendations.py --user-slug YOUR_SLUG --output recommendations.json

# Run evaluation
.venv/bin/python scripts/evaluate.py

# Web UI (Flask) — must use .venv
.venv/bin/python web/app.py
# Or: ./web/run.sh
# Or: make web
# Open http://127.0.0.1:5001
```

Activate the venv once per shell if you prefer shorter commands:

```bash
source .venv/bin/activate
python scripts/build_dataset.py   # uses venv python while activated
```

Wrapper script (always uses `.venv/bin/python`):

```bash
./scripts/run.sh scripts/build_dataset.py --input combined_contest_data.jsonl
```

## Project Structure

```
docs/           Planning, architecture docs, and ADRs
data/           Raw → processed → feature-engineered data (gitignored)
notebooks/      Exploratory analysis and model prototyping
src/lrs/        Main Python package
models/         Serialized model artifacts (gitignored)
scripts/        CLI entrypoints for training, evaluation, inference
web/            Flask UI
tests/          Unit and integration tests
```

## Development

```bash
make install    # .venv/bin/pip install -e ".[dev]"
make lint
make test
make build      # dataset pipeline
make train      # train all baseline models
make web        # Flask UI on :5001
```

## Documentation

- [Planning & Roadmap](docs/planning/roadmap.md)
- [Architecture Overview](docs/architecture/system-overview.md)
- [Architecture Decision Records](docs/adr/README.md)
- [Data Pipeline Design](docs/architecture/data-pipeline.md)
- [Recommendation Tiers Design](docs/architecture/recommendation-tiers.md)
