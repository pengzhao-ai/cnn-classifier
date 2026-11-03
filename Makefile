.DEFAULT_GOAL := help

UV ?= uv

.PHONY: help setup sync test coverage lint format typecheck docs build gate clean

help:  ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: sync  ## Install all dependencies, including every extra
sync:  ## Sync the uv environment with all extras (non-editable; see README)
	$(UV) sync --all-extras --no-editable

test:  ## Run the test suite
	$(UV) run --no-sync pytest

coverage:  ## Run tests with the core coverage gate
	$(UV) run --no-sync pytest --cov=cnn_classifier --cov-report=term-missing --cov-fail-under=90

lint:  ## Check lint and formatting
	$(UV) run --no-sync ruff check .
	$(UV) run --no-sync ruff format --check .

format:  ## Apply lint fixes and formatting
	$(UV) run --no-sync ruff check --fix .
	$(UV) run --no-sync ruff format .

typecheck:  ## Type-check src/ with mypy
	$(UV) run --no-sync mypy src

docs:  ## Build the documentation with warnings as errors
	$(UV) run --no-sync mkdocs build --strict

build:  ## Build the wheel and source distribution
	$(UV) build

gate: lint typecheck test coverage docs build  ## Run the local quality gate

clean:  ## Remove local caches and build outputs
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov site dist build
	rm -f .coverage .coverage.* coverage.xml
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
