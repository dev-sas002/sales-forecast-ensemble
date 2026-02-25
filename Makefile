# Everything runs in the container, so these work on a clean checkout with
# nothing installed but Docker. The *-local targets use the active Python env.
COMPOSE = docker compose
RUN     = $(COMPOSE) run --rm --no-deps api

.PHONY: help build up down logs pipeline backtest benchmark test test-local \
        lint format clean

help:
	@echo "make up          Build, seed and start the API (:8360) and dashboard (:8361)"
	@echo "make pipeline    Generate data and run ingest -> features -> train -> ensemble"
	@echo "make backtest    Rolling-origin backtest against a seasonal-naive baseline"
	@echo "make benchmark   Measure the scalability numbers quoted in the README"
	@echo "make test        Run the test suite in the container"
	@echo "make test-local  Run the test suite against the active Python env"
	@echo "make lint        ruff check"
	@echo "make down        Stop everything and remove the data volume"

build:
	$(COMPOSE) build

# `up` waits for the seed job, so the dashboard is populated on first load
# rather than showing an empty state.
up: build
	$(COMPOSE) up -d --wait
	@echo "API       http://localhost:8360/docs"
	@echo "Dashboard http://localhost:8361"

logs:
	$(COMPOSE) logs -f api dashboard

pipeline:
	$(RUN) python -m scripts.run_pipeline --weeks 130 --horizon 12

backtest:
	$(RUN) python -m src.evaluation.backtest --folds 4 --horizon 8

benchmark:
	$(RUN) python -m scripts.benchmark

test:
	$(RUN) python -m pytest tests/ -q

# For a checkout with requirements-core.txt and requirements-dev.txt installed
# locally; no Docker, no image build. macOS needs `brew install libomp` first,
# or LightGBM fails to load its own shared library.
test-local:
	python -m pytest tests/ -q

lint:
	$(RUN) ruff check src tests scripts dags streamlit_demo.py

format:
	$(RUN) ruff format src tests scripts dags streamlit_demo.py

down:
	$(COMPOSE) down -v

clean: down
	rm -rf .pytest_cache .ruff_cache artifacts/models data
