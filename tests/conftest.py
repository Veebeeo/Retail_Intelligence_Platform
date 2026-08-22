"""Shared fixtures.

Every test runs against an isolated SQLite database seeded from the synthetic
generator, so the suite needs no Postgres, no credentials and no network — CI
can run it on a clean checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# The package lives under src/; make it importable without an editable install
# so `pytest` works straight after a clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session", autouse=True)
def _isolated_env(tmp_path_factory):
    """Point every module at a throwaway database and model directory."""
    root = tmp_path_factory.mktemp("retail")
    os.environ["DATABASE_URL"] = f"sqlite:///{root / 'test.db'}"
    os.environ["MODEL_DIR"] = str(root / "models")
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{root / 'mlruns'}"
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ.pop("POSTGRES_USER", None)

    from retail_intel.config import get_settings

    get_settings.cache_clear()
    yield root


@pytest.fixture(scope="session")
def raw_transactions() -> pd.DataFrame:
    """Small, deterministic, and deliberately messy."""
    from retail_intel.data.synthetic import make_transactions

    return make_transactions(n_customers=120, n_skus=12, weeks=80, seed=42)


@pytest.fixture(scope="session")
def clean_transactions(raw_transactions) -> pd.DataFrame:
    from retail_intel.data.ingest import clean

    return clean(raw_transactions)


@pytest.fixture(scope="session")
def weekly_panel(clean_transactions) -> pd.DataFrame:
    from retail_intel.data.features import add_features, select_top_skus, to_weekly_panel

    return add_features(to_weekly_panel(select_top_skus(clean_transactions, 12)))


@pytest.fixture(scope="session")
def seeded_db(clean_transactions, weekly_panel):
    """A database containing transactions and features."""
    from retail_intel.data.models import create_all
    from retail_intel.db import write_table

    create_all()
    write_table(clean_transactions, "transactions")
    write_table(weekly_panel, "ml_weekly_features")
    return True


@pytest.fixture(scope="session")
def sample_series(weekly_panel) -> pd.Series:
    """One SKU's weekly demand, long enough to fit and backtest."""
    sku = weekly_panel.groupby("stock_code").size().idxmax()
    return weekly_panel.loc[weekly_panel["stock_code"] == sku, "weekly_sales"].reset_index(
        drop=True
    )
