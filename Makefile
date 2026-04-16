.PHONY: install install-advanced lint typecheck test scrape train-baseline train-advanced evaluate notebook clean

# Install base dependencies
install:
	uv sync

# Install including heavy advanced model deps (PyTorch, PyG)
install-advanced:
	uv sync --group advanced

# Lint with ruff
lint:
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

# Type check
typecheck:
	uv run mypy src/

# Run all tests
test:
	uv run pytest tests/ -v

# Run tests with coverage report
coverage:
	uv run pytest tests/ --cov=lrs --cov-report=term-missing

# Scrape contest data (usage: make scrape WEEKLY="380 400")
WEEKLY ?=
BIWEEKLY ?=
CONTEST ?=
scrape:
	uv run python scripts/scrape_contests.py \
	  $(if $(WEEKLY),--weekly $(WEEKLY)) \
	  $(if $(BIWEEKLY),--biweekly $(BIWEEKLY)) \
	  $(if $(CONTEST),--contest $(CONTEST))

# Train a baseline model (usage: make train-baseline MODEL=svd)
MODEL ?= svd
train-baseline:
	uv run python scripts/train_baseline.py --model $(MODEL)

# Train an advanced model (usage: make train-advanced MODEL=ncf)
train-advanced:
	uv run python scripts/train_advanced.py --model $(MODEL)

# Run offline evaluation suite
evaluate:
	uv run python scripts/evaluate.py

# Generate batch recommendations (usage: make recommend USER=123)
USER ?= all
recommend:
	uv run python scripts/generate_recommendations.py --user $(USER)

# Strip Jupyter notebook outputs before committing
notebook-clean:
	uv run nbstripout notebooks/**/*.ipynb

# Remove generated artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
