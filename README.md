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

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Install advanced model dependencies (PyTorch, PyG) — optional, heavy
uv sync --group advanced

# Copy environment config
cp .env.example .env
# Edit .env with your paths

# Run baseline training
uv run python scripts/train_baseline.py --model svd

# Run evaluation
uv run python scripts/evaluate.py
```

## Project Structure

```
docs/           Planning, architecture docs, and ADRs
data/           Raw → processed → feature-engineered data (gitignored)
notebooks/      Exploratory analysis and model prototyping
src/lrs/        Main Python package
models/         Serialized model artifacts (gitignored)
scripts/        CLI entrypoints for training, evaluation, inference
tests/          Unit and integration tests
```

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Run tests
uv run pytest tests/

# Or use make shortcuts
make lint
make test
```

## Documentation

- [Planning & Roadmap](docs/planning/roadmap.md)
- [Architecture Overview](docs/architecture/system-overview.md)
- [Architecture Decision Records](docs/adr/README.md)
- [Data Pipeline Design](docs/architecture/data-pipeline.md)
- [Recommendation Tiers Design](docs/architecture/recommendation-tiers.md)
