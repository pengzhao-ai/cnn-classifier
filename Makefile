.DEFAULT_GOAL := help

UV ?= uv

.PHONY: help setup sync test coverage lint format typecheck docs build gate clean

help:  ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: sync  ## Install all dependencies, including every extra
sync:  ## Sync the uv environment with all extras
	$(UV) sync --all-extras
	$(UV) pip install -e . --config-settings editable_mode=compat

test:  ## Run the test suite
	$(UV) run pytest

coverage:  ## Run tests with the core coverage gate
	$(UV) run pytest --cov=cnn_classifier --cov-report=term-missing --cov-fail-under=90

lint:  ## Check lint and formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Apply lint fixes and formatting
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:  ## Type-check src/ with mypy
	$(UV) run mypy src

docs:  ## Build the documentation with warnings as errors
	$(UV) run mkdocs build --strict

build:  ## Build the wheel and source distribution
	$(UV) build

gate: lint typecheck test coverage docs build  ## Run the local quality gate

clean:  ## Remove local caches and build outputs
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov site dist build
	rm -f .coverage .coverage.* coverage.xml
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
