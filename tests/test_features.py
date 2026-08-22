"""Feature engineering — specifically, the absence of target leakage.

A leaking feature makes backtest scores look excellent and production
forecasts fail, and nothing else in the pipeline will catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retail_intel.data.features import add_features, select_top_skus, to_weekly_panel


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """One SKU selling every week, one SKU with gaps."""
    weeks = pd.date_range("2010-01-04", periods=20, freq="W-MON")
    rows = []
    for i, week in enumerate(weeks):
        rows.append(
            {
                "invoice_date": week,
                "stock_code": "A",
                "quantity": 10 + i,
                "total_price": (10 + i) * 2.0,
            }
        )
        if i % 3 == 0:
            rows.append(
                {"invoice_date": week, "stock_code": "B", "quantity": 5, "total_price": 10.0}
            )
    return pd.DataFrame(rows)


def test_weeks_without_sales_become_explicit_zeros(toy_panel):
    """SKU B only sells every third week; the gaps must appear as zero demand,
    otherwise `shift(1)` means 'previous week that had a sale'."""
    panel = to_weekly_panel(toy_panel)
    b = panel[panel["stock_code"] == "B"]
    assert len(b) == 20
    assert (b["weekly_sales"] == 0).sum() > 0


def test_lag_features_match_the_actual_previous_week(toy_panel):
    feat = add_features(to_weekly_panel(toy_panel))
    a = feat[feat["stock_code"] == "A"].reset_index(drop=True)
    np.testing.assert_allclose(a["lag_1_week"].iloc[1:], a["weekly_sales"].iloc[:-1], rtol=1e-9)
    np.testing.assert_allclose(a["lag_2_week"].iloc[2:], a["weekly_sales"].iloc[:-2], rtol=1e-9)


def test_rolling_window_excludes_the_current_week(toy_panel):
    """The leakage bug: `rolling(4).mean()` includes row t, so a model sees
    part of its own target."""
    feat = add_features(to_weekly_panel(toy_panel))
    a = feat[feat["stock_code"] == "A"].reset_index(drop=True)
    for t in range(5, len(a)):
        expected = a["weekly_sales"].iloc[t - 4 : t].mean()
        assert a["rolling_4_wk_avg"].iloc[t] == pytest.approx(expected)


def test_first_row_of_each_sku_has_no_lag(toy_panel):
    feat = add_features(to_weekly_panel(toy_panel))
    for _, grp in feat.groupby("stock_code"):
        assert pd.isna(grp.sort_values("week")["lag_1_week"].iloc[0])


def test_features_never_cross_sku_boundaries(toy_panel):
    """Lag 1 of a SKU's first week must be NaN, not the last week of the
    previous SKU in the frame."""
    feat = add_features(to_weekly_panel(toy_panel)).sort_values(["stock_code", "week"])
    firsts = feat.groupby("stock_code").head(1)
    assert firsts["lag_1_week"].isna().all()


def test_select_top_skus_keeps_the_highest_volume(clean_transactions):
    top = select_top_skus(clean_transactions, 3)
    assert top["stock_code"].nunique() == 3
    volumes = clean_transactions.groupby("stock_code")["quantity"].sum()
    assert set(top["stock_code"].unique()) == set(volumes.nlargest(3).index)


def test_calendar_features_are_in_range(weekly_panel):
    assert weekly_panel["month"].between(1, 12).all()
    assert weekly_panel["week_of_year"].between(1, 53).all()
    assert (weekly_panel["weeks_since_start"] >= 0).all()


def test_panel_has_no_duplicate_sku_weeks(weekly_panel):
    assert not weekly_panel.duplicated(subset=["stock_code", "week"]).any()
