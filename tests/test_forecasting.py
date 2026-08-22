"""Forecaster contract, baselines and backtest mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retail_intel.forecasting.backtest import (
    backtest_series,
    pick_champions,
    rolling_origin_splits,
    summarise,
)
from retail_intel.forecasting.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from retail_intel.forecasting.registry import BASELINE_MODELS, available_models, build

BASELINE_NAMES = tuple(BASELINE_MODELS)


@pytest.fixture
def seasonal_series() -> pd.Series:
    t = np.arange(120)
    rng = np.random.default_rng(0)
    return pd.Series(
        np.clip(50 + 15 * np.sin(2 * np.pi * t / 52) + 0.1 * t + rng.normal(0, 4, 120), 0, None)
    )


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_every_baseline_honours_the_forecaster_contract(name, seasonal_series):
    pred = build(name, seasonal_period=52).fit(seasonal_series).predict(4)
    assert list(pred.columns) == ["yhat", "yhat_lower", "yhat_upper"]
    assert len(pred) == 4
    assert pred.notna().all().all()


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_forecasts_are_never_negative(name):
    """Demand cannot be negative; the base class clips so no implementation
    has to remember to."""
    y = pd.Series([5.0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0])
    pred = build(name, seasonal_period=4).fit(y).predict(6)
    assert (pred.to_numpy() >= 0).all()


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_interval_bounds_bracket_the_point_forecast(name, seasonal_series):
    pred = build(name, seasonal_period=52).fit(seasonal_series).predict(6)
    assert (pred["yhat_lower"] <= pred["yhat"]).all()
    assert (pred["yhat_upper"] >= pred["yhat"]).all()


def test_naive_repeats_the_last_value(seasonal_series):
    pred = NaiveForecaster().fit(seasonal_series).predict(3)
    assert pred["yhat"].nunique() == 1
    assert pred["yhat"].iloc[0] == pytest.approx(seasonal_series.iloc[-1])


def test_seasonal_naive_repeats_one_period_back():
    y = pd.Series(list(range(1, 105)), dtype=float)
    pred = SeasonalNaiveForecaster(seasonal_period=52).fit(y).predict(3)
    np.testing.assert_allclose(pred["yhat"], y.iloc[-52:-49].to_numpy())


def test_seasonal_naive_degrades_gracefully_on_short_history():
    """Most SKUs will not have two years of history; failing is not an option."""
    y = pd.Series([5.0, 7, 6, 8, 9])
    model = SeasonalNaiveForecaster(seasonal_period=52).fit(y)
    assert model._degraded
    assert model.predict(3)["yhat"].iloc[0] == pytest.approx(y.mean())


def test_moving_average_uses_the_configured_window():
    y = pd.Series([1.0, 2, 3, 4, 100, 100, 100, 100])
    assert MovingAverageForecaster(window=4).fit(y).predict(1)["yhat"].iloc[0] == pytest.approx(100)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        NaiveForecaster().predict(1)


def test_fitting_an_empty_series_raises():
    with pytest.raises(ValueError, match="empty"):
        NaiveForecaster().fit(pd.Series([], dtype=float))


def test_zero_horizon_raises(seasonal_series):
    with pytest.raises(ValueError, match="horizon"):
        NaiveForecaster().fit(seasonal_series).predict(0)


# --- backtest mechanics ---------------------------------------------------
def test_splits_train_only_on_the_past():
    for train_end, test_end in rolling_origin_splits(104, 4, 5, 26):
        assert train_end < test_end
        assert test_end <= 104


def test_split_windows_do_not_overlap():
    splits = rolling_origin_splits(104, 4, 5, 26)
    for (_, prev_end), (next_start, _) in zip(splits, splits[1:], strict=False):
        assert next_start >= prev_end


def test_splits_respect_the_minimum_training_size():
    assert rolling_origin_splits(20, 4, 5, 26) == []
    assert all(t >= 26 for t, _ in rolling_origin_splits(60, 4, 5, 26))


def test_backtest_produces_a_result_per_model_and_fold(seasonal_series):
    results = backtest_series(seasonal_series, BASELINE_NAMES, 4, 2, 26, 52, "TEST")
    assert len(results) == len(BASELINE_NAMES) * 2
    assert {r.model for r in results} == set(BASELINE_NAMES)
    assert all(r.error is None for r in results)


def test_summarise_ranks_by_mase_and_reports_win_rate(seasonal_series):
    rows = []
    for sku in ("A", "B", "C"):
        for res in backtest_series(seasonal_series, BASELINE_NAMES, 4, 2, 26, 52, sku):
            rows.append(
                {
                    "stock_code": res.stock_code,
                    "model": res.model,
                    "fold": res.fold,
                    "train_end": res.train_end,
                    "fit_seconds": res.fit_seconds,
                    "error": res.error,
                    **res.metrics,
                }
            )
    summary = summarise(pd.DataFrame(rows))
    assert summary["mase_mean"].is_monotonic_increasing
    assert summary.loc[summary["model"] == "seasonal_naive", "win_rate_vs_baseline"].iloc[0] == 0.0
    assert summary["win_rate_vs_baseline"].between(0, 100).all()


def test_champion_selection_picks_the_lowest_mase_per_sku(seasonal_series):
    rows = []
    for sku in ("A", "B"):
        for res in backtest_series(seasonal_series, BASELINE_NAMES, 4, 2, 26, 52, sku):
            rows.append(
                {
                    "stock_code": res.stock_code,
                    "model": res.model,
                    "fold": res.fold,
                    "train_end": res.train_end,
                    "fit_seconds": res.fit_seconds,
                    "error": res.error,
                    **res.metrics,
                }
            )
    folds = pd.DataFrame(rows)
    champs = pick_champions(folds)
    assert set(champs["stock_code"]) == {"A", "B"}
    for row in champs.itertuples():
        best = folds[folds["stock_code"] == row.stock_code].groupby("model")["mase"].mean().idxmin()
        assert row.champion == best


def test_a_failing_model_is_recorded_not_raised():
    """One pathological SKU must not abandon a 100-SKU run."""
    results = backtest_series(pd.Series([1.0] * 40), ("xgboost",), 4, 1, 26, 52, "FLAT")
    assert all(r.model == "xgboost" for r in results)


def test_registry_reports_available_models():
    names = available_models()
    assert set(BASELINE_NAMES) <= set(names)


def test_unknown_model_name_raises():
    with pytest.raises(KeyError, match="Unknown forecaster"):
        build("does_not_exist")
