"""Statistical and machine-learning forecasters.

Heavy dependencies (Prophet, XGBoost) are imported inside ``_fit`` so that the
API container can load this module — and serve a SARIMA or baseline champion —
without carrying them. That is what lets the runtime image stay slim while the
training image is free to be large.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from retail_intel.forecasting.base import Forecaster, empirical_interval
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)


class SarimaForecaster(Forecaster):
    """Seasonal ARIMA via ``statsmodels`` SARIMAX.

    Note on the seasonal period: the old README documented ``(1,1,1,52)`` while
    the code actually ran ``(1,1,0,4)``. A period of 4 is not a seasonal cycle
    in weekly data — it is a monthly rhythm — which is one reason the reported
    figure could not be reproduced. This uses the configured weekly period and
    falls back automatically when a SKU has too little history to support it.
    """

    name = "sarima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] | None = None,
        seasonal_period: int = 52,
    ) -> None:
        super().__init__(seasonal_period)
        self.order = order
        self.seasonal_order = seasonal_order or (1, 1, 0, seasonal_period)

    def _fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        seasonal_order = self.seasonal_order
        # SARIMAX needs at least two full cycles to estimate a seasonal term.
        if len(y) < 2 * seasonal_order[3]:
            logger.debug(
                "Series of %d weeks too short for period %d; fitting non-seasonal ARIMA",
                len(y),
                seasonal_order[3],
            )
            seasonal_order = (0, 0, 0, 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = model.fit(disp=False, maxiter=200)
        self._effective_seasonal_order = seasonal_order

    def _predict(self, horizon: int, X: pd.DataFrame | None = None) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._result.get_forecast(steps=horizon)
            ci = fc.conf_int(alpha=0.05)
        return pd.DataFrame(
            {
                "yhat": np.asarray(fc.predicted_mean, dtype=float),
                "yhat_lower": np.asarray(ci.iloc[:, 0], dtype=float),
                "yhat_upper": np.asarray(ci.iloc[:, 1], dtype=float),
            }
        )

    @property
    def params(self) -> dict:
        return {
            "order": str(self.order),
            "seasonal_order": str(getattr(self, "_effective_seasonal_order", self.seasonal_order)),
        }


class ProphetForecaster(Forecaster):
    """Meta's Prophet on a weekly index.

    Weekly and daily seasonality are switched off because the data is already
    aggregated to weeks — leaving them on fits noise to a frequency the series
    cannot contain.
    """

    name = "prophet"

    def __init__(self, seasonal_period: int = 52, start_date: pd.Timestamp | None = None) -> None:
        super().__init__(seasonal_period)
        self.start_date = start_date or pd.Timestamp("2010-01-04")

    def _fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        from prophet import Prophet

        self._dates = pd.date_range(self.start_date, periods=len(y), freq="W-MON")
        frame = pd.DataFrame({"ds": self._dates, "y": y.to_numpy()})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = Prophet(
                yearly_seasonality=len(y) >= 2 * self.seasonal_period,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                seasonality_mode="multiplicative",
            )
            self._model.fit(frame)

    def _predict(self, horizon: int, X: pd.DataFrame | None = None) -> pd.DataFrame:
        future = pd.DataFrame(
            {
                "ds": pd.date_range(
                    self._dates[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON"
                )
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._model.predict(future)
        return pd.DataFrame(
            {
                "yhat": fc["yhat"].to_numpy(),
                "yhat_lower": fc["yhat_lower"].to_numpy(),
                "yhat_upper": fc["yhat_upper"].to_numpy(),
            }
        )

    @property
    def params(self) -> dict:
        return {"seasonality_mode": "multiplicative", "interval_width": 0.95}


class XGBoostForecaster(Forecaster):
    """Gradient-boosted trees over lag and calendar features, forecast recursively.

    Two corrections versus the original implementation:

    * It builds its own lag matrix from the series rather than reading
      pre-computed columns, so the features at prediction time are constructed
      exactly the way they were at training time.
    * Multi-step forecasting is genuinely recursive: each predicted week is fed
      back in as the lag input for the next. The old code scored a 4-week
      horizon using *actual* lag values from the test set — information that
      does not exist when forecasting forward, which flattered its numbers.

    Trees cannot extrapolate beyond the range of the training target, so a
    linear trend term is fitted first and the residual is modelled.
    """

    name = "xgboost"

    def __init__(
        self,
        seasonal_period: int = 52,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        super().__init__(seasonal_period)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.lags = (1, 2, 3, 4, 8, 12)

    def _design_row(self, history: np.ndarray, t_index: int) -> np.ndarray:
        """Feature vector for one step, built only from data strictly before it."""
        feats = [history[-lag] if len(history) >= lag else history.mean() for lag in self.lags]
        for window in (4, 12):
            tail = history[-window:] if len(history) >= 1 else np.array([0.0])
            feats.extend([tail.mean(), tail.std() if len(tail) > 1 else 0.0])
        if len(history) >= self.seasonal_period:
            feats.append(history[-self.seasonal_period])
        else:
            feats.append(history.mean())
        week_of_year = (t_index % 52) + 1
        feats.extend(
            [
                week_of_year,
                np.sin(2 * np.pi * week_of_year / 52),
                np.cos(2 * np.pi * week_of_year / 52),
                t_index,
            ]
        )
        return np.asarray(feats, dtype=float)

    def _fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        arr = y.to_numpy(dtype=float)
        start = max(self.lags)
        if len(arr) <= start + 1:
            raise ValueError(f"xgboost needs more than {start + 1} observations, got {len(arr)}")

        # Detrend so the trees model a stationary residual.
        t = np.arange(len(arr), dtype=float)
        self._trend_coef = np.polyfit(t, arr, 1)
        detrended = arr - np.polyval(self._trend_coef, t)

        rows, targets = [], []
        for i in range(start, len(arr)):
            rows.append(self._design_row(detrended[:i], i))
            targets.append(detrended[i])

        design = np.vstack(rows)
        target_vec = np.asarray(targets)

        # Calibrate the prediction interval on held-out data. In-sample
        # residuals from a boosted model are near zero -- it has memorised the
        # training rows -- which would produce an interval far too narrow to
        # size safety stock from. Fit on the first 80%, measure error on the
        # last 20%, then refit on everything and keep those honest residuals.
        self._residuals = self._calibration_residuals(design, target_vec)

        self._model = self._new_model()
        self._model.fit(design, target_vec)
        self._detrended_history = detrended
        self._n_train = len(arr)

    def _new_model(self):
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=2,
            verbosity=0,
        )

    def _calibration_residuals(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        split = int(len(target) * 0.8)
        if split < 5 or len(target) - split < 3:
            # Too little data to hold anything out; fall back to in-sample
            # error inflated by a factor that reflects its known optimism.
            model = self._new_model().fit(design, target)
            return (target - model.predict(design)) * 3.0
        model = self._new_model().fit(design[:split], target[:split])
        return target[split:] - model.predict(design[split:])

    def _predict(self, horizon: int, X: pd.DataFrame | None = None) -> pd.DataFrame:
        history = self._detrended_history.copy()
        preds = []
        for step in range(horizon):
            t_index = self._n_train + step
            yhat = float(self._model.predict(self._design_row(history, t_index).reshape(1, -1))[0])
            preds.append(yhat)
            history = np.append(history, yhat)  # recursive: feed the prediction back

        t_future = np.arange(self._n_train, self._n_train + horizon, dtype=float)
        point = np.asarray(preds) + np.polyval(self._trend_coef, t_future)
        lo, hi = empirical_interval(point, self._residuals)
        return pd.DataFrame({"yhat": point, "yhat_lower": lo, "yhat_upper": hi})

    @property
    def params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "lags": str(self.lags),
        }
