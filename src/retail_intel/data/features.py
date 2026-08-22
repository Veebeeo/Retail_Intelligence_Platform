"""``transactions`` -> ``ml_weekly_features``.

This is the pipeline the old README documented as ``scripts/ingest_features.py``
but which only ever existed as notebook cells, so a clone of the repo could not
reproduce the modelling table.

Two things it does that the notebook did not:

* **Reindex onto a dense weekly grid.** Grouping by week only yields rows for
  weeks that had a sale, so ``shift(1)`` silently meant "previous week *with a
  sale*", which is not the same lag. Zero-demand weeks are real signal in
  retail and are now represented.
* **Compute rolling windows shifted by one period.** ``rolling(4).mean()``
  includes the current week, so a model trained on it sees part of its own
  target. Every window here ends at ``t-1``.
"""

from __future__ import annotations

import argparse

import pandas as pd

from retail_intel.config import get_settings
from retail_intel.data.contracts import WeeklyFeatureSchema, validate
from retail_intel.db import read_sql, write_table
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)

LAGS = (1, 2, 4, 52)
ROLLING_WINDOWS = (4, 12)


def load_transactions() -> pd.DataFrame:
    df = read_sql(
        "SELECT stock_code, invoice_date, quantity, total_price FROM transactions"
    )
    if df.empty:
        raise RuntimeError("`transactions` is empty. Run the ingest pipeline first.")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def select_top_skus(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep the ``top_n`` SKUs by total units sold.

    Long-tail SKUs have too few observations to fit a seasonal model and would
    dominate any averaged error metric with noise.
    """
    top = df.groupby("stock_code")["quantity"].sum().nlargest(top_n).index
    logger.info("Retained %d of %d SKUs by volume", len(top), df["stock_code"].nunique())
    return df[df["stock_code"].isin(top)].copy()


def to_weekly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to weekly buckets and reindex onto a dense per-SKU grid."""
    weekly = (
        df.groupby([pd.Grouper(key="invoice_date", freq="W-MON"), "stock_code"])
        .agg(weekly_sales=("quantity", "sum"), weekly_revenue=("total_price", "sum"))
        .reset_index()
        .rename(columns={"invoice_date": "week"})
    )

    # A SKU's series runs from its first sale to the global end of data. Weeks
    # before its first appearance are "not stocked yet", not "zero demand".
    full_range = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="W-MON")
    panels = []
    for sku, grp in weekly.groupby("stock_code", sort=False):
        idx = full_range[full_range >= grp["week"].min()]
        panel = (
            grp.set_index("week")
            .reindex(idx)
            .assign(stock_code=sku)
            .fillna({"weekly_sales": 0.0, "weekly_revenue": 0.0})
            .rename_axis("week")
            .reset_index()
        )
        panels.append(panel)

    out = pd.concat(panels, ignore_index=True).sort_values(["stock_code", "week"])
    filled = len(out) - len(weekly)
    logger.info("Weekly panel: %d rows across %d SKUs (%d zero-demand weeks made explicit)",
                len(out), out["stock_code"].nunique(), filled)
    return out.reset_index(drop=True)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling and calendar features.

    Everything is grouped by SKU so no feature ever reaches across SKUs, and
    every window is shifted so no feature contains the current week's target.
    """
    df = panel.sort_values(["stock_code", "week"]).copy()
    grp = df.groupby("stock_code", sort=False)["weekly_sales"]

    for lag in LAGS:
        df[f"lag_{lag}_week"] = grp.shift(lag)

    prior = grp.shift(1)  # everything below is computed on t-1 and earlier
    for window in ROLLING_WINDOWS:
        rolled = prior.groupby(df["stock_code"], sort=False).rolling(window, min_periods=1)
        df[f"rolling_{window}_wk_avg"] = rolled.mean().reset_index(level=0, drop=True)
        if window == 4:
            df["rolling_4_wk_std"] = (
                rolled.std().reset_index(level=0, drop=True).fillna(0.0)
            )

    df["month"] = df["week"].dt.month.astype(int)
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["weeks_since_start"] = df.groupby("stock_code", sort=False).cumcount().astype(int)

    return df[list(WeeklyFeatureSchema.columns)].reset_index(drop=True)


def run(top_n: int | None = None, min_weeks: int | None = None) -> pd.DataFrame:
    settings = get_settings()
    top_n = top_n or settings.top_n_skus
    min_weeks = min_weeks or settings.min_train_weeks

    df = select_top_skus(load_transactions(), top_n)
    panel = to_weekly_panel(df)
    features = add_features(panel)

    # Drop SKUs too short to backtest meaningfully.
    lengths = features.groupby("stock_code").size()
    keep = lengths[lengths >= min_weeks].index
    dropped = len(lengths) - len(keep)
    if dropped:
        logger.warning("Dropped %d SKUs with fewer than %d weeks of history", dropped, min_weeks)
    features = features[features["stock_code"].isin(keep)].reset_index(drop=True)

    features = validate(features, WeeklyFeatureSchema)
    write_table(features, "ml_weekly_features")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the weekly modelling table.")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--min-weeks", type=int, default=None)
    args = parser.parse_args()
    out = run(args.top_n, args.min_weeks)
    logger.info("Feature build complete: %d rows, %d SKUs", len(out), out["stock_code"].nunique())


if __name__ == "__main__":
    main()
