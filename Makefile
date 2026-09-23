# Developer entry points. `make` with no arguments prints the target list.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON  ?= python3
PIP     ?= $(PYTHON) -m pip
COMPOSE ?= docker compose

# Where the local (non-container) commands expect HAPI to be listening.
FHIR_BASE_URL ?= http://localhost:8080/fhir
# Model served by `make serve-model`, and the port the app expects it on.
LLM_MODEL     ?= mlx-community/Qwen3.5-9B-MLX-4bit
VLLM_PORT     ?= 8001
PATIENTS      ?= 120
SEED          ?= 42

.PHONY: help install lint format typecheck check test test-integration test-all cov \
        run demo serve-model seed bench up down down-volumes logs clean

help: ## Show this help
	@echo "fhir-healthcare-ai - available targets:"
	@echo
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in editable mode with dev dependencies
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || \
		{ echo ">>> Python 3.11+ is required; $(PYTHON) is $$($(PYTHON) --version 2>&1)"; exit 1; }
	@# Virtualenvs made by `uv venv` (or `python -m venv --without-pip`) have no pip.
	@if $(PYTHON) -m pip --version >/dev/null 2>&1; then \
		$(PIP) install -e ".[dev]"; \
	elif command -v uv >/dev/null 2>&1; then \
		uv pip install --python "$$($(PYTHON) -c 'import sys; print(sys.executable)')" -e ".[dev]"; \
	else \
		$(PYTHON) -m ensurepip --upgrade && $(PIP) install -e ".[dev]"; \
	fi

lint: ## Run ruff lint checks
	ruff check .

format: ## Auto-format the codebase with ruff
	ruff format .

typecheck: ## Run mypy over src/
	mypy src

check: lint typecheck test bench ## Everything CI runs

test: ## Run unit tests (no live FHIR server required)
	pytest -m "not integration"

test-integration: ## Run integration tests (requires a live FHIR server; see `make up`)
	FHIR_BASE_URL=$(FHIR_BASE_URL) pytest -m integration

test-all: ## Run the full suite, unit and integration
	FHIR_BASE_URL=$(FHIR_BASE_URL) pytest

cov: ## Run unit tests with a coverage report
	pytest -m "not integration" \
		--cov=fhir_healthcare_ai --cov-report=term-missing --cov-report=xml

demo: ## Run the API on an in-memory synthetic population: no Docker, no model, no key
	FHIR_IN_MEMORY=true LLM_PROVIDER=mock LOG_JSON=false \
		uvicorn fhir_healthcare_ai.api.main:app --host 127.0.0.1 --port 8000

run: ## Run the API locally with auto-reload on http://127.0.0.1:8000
	uvicorn fhir_healthcare_ai.api.main:app --reload --host 127.0.0.1 --port 8000

serve-model: ## Serve the local model with vLLM (vllm-metal on a Mac) on port 8001
	vllm serve $(LLM_MODEL) --port $(VLLM_PORT) --max-model-len 16384 \
		--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'

seed: ## Load synthetic patients into the FHIR server (idempotent, PUT-based)
	fhir-ai-seed --push --wait-for-server \
		--patients $(PATIENTS) --seed $(SEED) --fhir-base-url $(FHIR_BASE_URL)

bench: ## Run the benchmark suite against the current planner
	fhir-ai-bench

up: ## Build and start the full stack (HAPI + seeder + API) in the background
	$(COMPOSE) up -d --build

down: ## Stop the stack, keeping the seeded FHIR data
	$(COMPOSE) down

down-volumes: ## DESTRUCTIVE: stop the stack and delete all volumes (seeded data is lost)
	@echo ">>> This deletes the HAPI database and all generated data. Ctrl-C within 5s to abort."
	@sleep 5
	$(COMPOSE) down -v

logs: ## Tail logs from every running service
	$(COMPOSE) logs -f

clean: ## Remove caches, build artifacts and coverage output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage .coverage.* \
		coverage.xml htmlcov build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
