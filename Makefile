.PHONY: help install install-dev lint format test test-fast coverage \
        seed ingest features train backtest segment baskets drift pipeline \
        api dashboard mlflow up down clean

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PY) -m pip install -e ".[api,ml,dash]"

install-dev:  ## Install everything, including test and lint tooling
	$(PY) -m pip install -e ".[api,ml,dash,dev,orchestration]"

lint:  ## Lint
	ruff check .

format:  ## Auto-format and fix
	ruff check --fix .
	ruff format .

test:  ## Run the full test suite
	pytest

test-fast:  ## Skip slow model fits
	pytest -m "not slow" -q

coverage:  ## Test with a coverage report
	pytest --cov=src/retail_intel --cov=app --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

# --- data and models ------------------------------------------------------
seed:  ## Populate the warehouse with generated data (no raw file needed)
	$(PY) -m pipelines.flow --synthetic --jobs 0

ingest:  ## Ingest the raw workbook (set SOURCE=path/to/file.xlsx)
	$(PY) -m retail_intel.data.ingest --path $(or $(SOURCE),data/online_retail_II.xlsx)

features:  ## Build the weekly modelling table
	$(PY) -m retail_intel.data.features

backtest:  ## Rolling-origin backtest across every SKU
	$(PY) -m retail_intel.forecasting.backtest --jobs 0

train:  ## Backtest, select champions, persist and register
	$(PY) -m retail_intel.forecasting.train --jobs 0

segment:  ## RFM segments, CLV and uplift targeting
	$(PY) -m retail_intel.segmentation.pipeline

baskets:  ## Mine product association rules
	$(PY) -m retail_intel.recommend.market_basket

drift:  ## Check the serving window for drift
	$(PY) -m retail_intel.monitoring.drift

pipeline:  ## Run everything end to end (set SOURCE for real data)
	$(PY) -m pipelines.flow $(if $(SOURCE),--source $(SOURCE),--synthetic) --jobs 0

# --- services -------------------------------------------------------------
api:  ## Serve the API on :8000
	uvicorn app.main:app --reload --port 8000

dashboard:  ## Serve the dashboard on :8501
	streamlit run app/dashboard.py

mlflow:  ## Serve the MLflow UI on :5000
	mlflow ui --backend-store-uri $(or $(MLFLOW_TRACKING_URI),file:./mlruns) --port 5000

up:  ## Start the full stack in Docker
	docker compose up --build -d

down:  ## Stop the stack
	docker compose down

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
