# All targets use the project .venv — never system Python.
PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: install install-advanced venv lint typecheck test train-baseline train-advanced evaluate web notebook clean build train recommend

venv:
	@test -x $(PY) || (echo "Run: python3 -m venv .venv && $(PIP) install -e \".[dev]\"" && exit 1)

# Create .venv and install package + dev deps
install: venv
	$(PIP) install -e ".[dev]"

install-advanced: venv
	$(PIP) install -e ".[advanced,dev]"

lint: venv
	$(PY) -m ruff check src/ tests/ scripts/ web/
	$(PY) -m ruff format --check src/ tests/ scripts/ web/

typecheck: venv
	$(PY) -m mypy src/

test: venv
	$(PY) -m pytest tests/ -v

coverage: venv
	$(PY) -m pytest tests/ --cov=lrs --cov-report=term-missing

MODEL ?= svd
train-baseline: venv
	$(PY) scripts/train_baseline.py --model $(MODEL)

train-advanced: venv
	$(PY) scripts/train_advanced.py --model $(MODEL)

evaluate: venv
	$(PY) scripts/evaluate.py

web: venv
	$(PY) web/app.py

build: venv
	$(PY) scripts/build_dataset.py

train: venv
	$(PY) scripts/train_baseline.py --model all

USER ?= all
recommend: venv
	$(PY) scripts/generate_recommendations.py --user $(USER)

notebook-clean: venv
	$(PY) -m nbstripout notebooks/**/*.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
