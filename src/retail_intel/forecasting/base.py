"""A single interface every forecaster implements.

The original ``train_models.py`` handled SARIMA, Prophet and XGBoost with three
different bespoke code paths, which is why only one of them could ever be
served. Behind one interface, the backtester and the training pipeline treat
them identically and any of them can become the champion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Fit on a univariate weekly series, predict ``h`` steps with an interval."""

    name: str = "forecaster"
    #: Whether ``fit`` needs the exogenous feature frame as well as the target.
    requires_features: bool = False

    def __init__(self, seasonal_period: int = 52) -> None:
        self.seasonal_period = seasonal_period
        self._fitted = False
        self._y_train: np.ndarray | None = None

    @abstractmethod
    def _fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int, X: pd.DataFrame | None = None) -> pd.DataFrame: ...

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> Forecaster:
        y = pd.Series(y).astype(float).reset_index(drop=True)
        if y.empty:
            raise ValueError(f"{self.name}: cannot fit on an empty series")
        self._y_train = y.to_numpy()
        self._fit(y, X)
        self._fitted = True
        return self

    def predict(self, horizon: int, X: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return columns ``yhat``, ``yhat_lower``, ``yhat_upper``.

        Demand cannot be negative, so every forecaster's output is clipped at
        zero here rather than in each implementation.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        out = self._predict(horizon, X)
        missing = {"yhat", "yhat_lower", "yhat_upper"} - set(out.columns)
        if missing:
            raise RuntimeError(f"{self.name}: _predict omitted {sorted(missing)}")

        out = out.clip(lower=0.0)
        # Clipping can invert a wide interval whose lower bound went negative.
        out["yhat_lower"] = np.minimum(out["yhat_lower"], out["yhat"])
        out["yhat_upper"] = np.maximum(out["yhat_upper"], out["yhat"])
        return out.reset_index(drop=True)

    @property
    def y_train(self) -> np.ndarray:
        if self._y_train is None:
            raise RuntimeError(f"{self.name}: not fitted")
        return self._y_train

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, fitted={self._fitted})"


def empirical_interval(
    point: np.ndarray, residuals: np.ndarray, level: float = 0.95, horizon_growth: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Build a prediction interval from residual quantiles.

    Used by models that do not produce their own interval. Residual quantiles
    beat a normal assumption here because weekly demand errors are right-skewed
    (promotions produce large positive surprises, but demand cannot undershoot
    past zero).

    ``horizon_growth`` widens the interval by sqrt(h), reflecting that a
    4-week-ahead forecast is less certain than a 1-week-ahead one.
    """
    if residuals.size == 0:
        return point.copy(), point.copy()

    alpha = (1 - level) / 2
    lo_q, hi_q = np.quantile(residuals, [alpha, 1 - alpha])
    steps = np.arange(1, len(point) + 1)
    scale = np.sqrt(steps) if horizon_growth else np.ones_like(steps, dtype=float)
    return point + lo_q * scale, point + hi_q * scale
