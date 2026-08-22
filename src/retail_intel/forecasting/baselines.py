"""Baseline forecasters.

These exist because the previous version of this project reported "SARIMA:
37% MAPE" with nothing to compare it against. A percentage error is meaningless
on its own — the only question that matters is whether the model beats the
cheap alternative a planner would otherwise use.

Seasonal naive is the standard benchmark for weekly retail demand and is the
denominator of MASE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_intel.forecasting.base import Forecaster, empirical_interval


class NaiveForecaster(Forecaster):
    """Tomorrow looks like today: repeat the last observed value."""

    name = "naive"

    def _fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        self._last = float(y.iloc[-1])
        self._residuals = np.diff(y.to_numpy()) if len(y) > 1 else np.array([0.0])

    def _predict(self, horizon: int, X: pd.DataFrame | None = None) -> pd.DataFrame:
        point = np.full(horizon, self._last)
        lo, hi = empirical_interval(point, self._residuals)
        return pd.DataFrame({"yhat": point, "yhat_lower": lo, "yhat_upper": hi})


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the value from one seasonal period ago.

    For weekly retail this means "last year, same week", which captures the
    Christmas peak for free and is genuinely hard to beat.
    """

    name = "seasonal_naive"

    def _fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        arr = y.to_numpy()
        m = self.seasonal_period
        if len(arr) < m + 1:
            # Not enough history for a full cycle: degrade to the series mean
            # rather than failing, and record that we did.
            self._season = np.full(max(m, 1), float(arr.mean()))
            self._degraded = True
            self._residuals = arr - arr.mean()
        else:
            self._season = arr[-m:]
            self._degraded = False
            self._residuals = arr[m:] - arr[:-m]

    def _predict(self, horizon: int, X: pd.DataFrame | None = None) -> pd.DataFrame:
        m = len(self._season)
        point = np.array([self._season[i % m] for i in range(horizon)], dtype=float)
        lo, hi = empirical_interval(point, self._residuals, horizon_growth=False)
        return pd.DataFrame({"yhat": point, "yhat_lower": lo, "yhat_upper": hi})


class MovingAverageForecaster(Forecaster):
    """Mean of the last ``window`` observations, held flat.

    This is the closest honest equivalent of what the old ``/forecast``
    endpoint did, minus its invented 2%-per-week growth. Including it makes the
    comparison against the previous "model" explicit.
    """

    name = "moving_average"

    def __init__(self, window: int = 4, seasonal_period: int = 52) -> None:
        super().__init__(seasonal_period)
        self.window = window
        self.name = f"moving_average_{window}"

    def _fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        arr = y.to_numpy()
        self._mean = float(arr[-self.window :].mean())
        fitted = pd.Series(arr).rolling(self.window, min_periods=1).mean().shift(1)
        self._residuals = (arr - fitted.to_numpy())[~np.isnan(fitted.to_numpy())]

    def _predict(self, horizon: int, X: pd.DataFrame | None = None) -> pd.DataFrame:
        point = np.full(horizon, self._mean)
        lo, hi = empirical_interval(point, self._residuals)
        return pd.DataFrame({"yhat": point, "yhat_lower": lo, "yhat_upper": hi})


class SeasonalDriftForecaster(Forecaster):
    """Seasonal naive plus the average trend observed over the history.

    A cheap way to catch SKUs that are growing or dying while still respecting
    the seasonal shape.
    """

    name = "seasonal_drift"

    def _fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        arr = y.to_numpy()
        m = self.seasonal_period if len(arr) > self.seasonal_period else 1
        self._season = arr[-m:] if len(arr) >= m else np.full(1, arr.mean())
        self._drift = (arr[-1] - arr[0]) / max(len(arr) - 1, 1)
        self._residuals = arr[m:] - arr[:-m] if len(arr) > m else np.array([0.0])

    def _predict(self, horizon: int, X: pd.DataFrame | None = None) -> pd.DataFrame:
        m = len(self._season)
        steps = np.arange(1, horizon + 1)
        point = np.array([self._season[i % m] for i in range(horizon)]) + self._drift * steps
        lo, hi = empirical_interval(point, self._residuals)
        return pd.DataFrame({"yhat": point, "yhat_lower": lo, "yhat_upper": hi})


BASELINES: dict[str, type[Forecaster]] = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "moving_average": MovingAverageForecaster,
    "seasonal_drift": SeasonalDriftForecaster,
}
