# Convenience targets. Everything runs inside the project's .venv.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup run ask test test-all lint format typecheck check scenarios reindex clean

setup: ## Create the venv and install runtime + dev dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt -r requirements-tts.txt -r requirements-dev.txt
	$(PIP) install -e .

run: ## Start the interactive assistant
	$(PY) -m voice_rag

ask: ## One-shot question, e.g. make ask Q="Does the L1 lamp have a battery?"
	$(PY) -m voice_rag --ask "$(Q)"

reindex: ## Force a rebuild of the local index (no TTS, no question)
	$(PY) -m voice_rag --reindex --ask "" --no-tts

test: ## Unit tests only (no models needed)
	$(PY) -m pytest

test-all: ## Unit + integration tests (needs Ollama running with the models pulled)
	$(PY) -m pytest -m "" -o addopts=""

lint: ## Ruff lint + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## Auto-format with ruff
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck: ## mypy in strict mode
	$(PY) -m mypy

check: lint typecheck test ## Everything CI runs

scenarios: ## Run the six scenarios on each model and write results/RESULTS.md
	$(PY) scripts/run_scenarios.py

clean:
	rm -rf .index .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
