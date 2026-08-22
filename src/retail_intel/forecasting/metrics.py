"""Forecast accuracy metrics.

The original project reported MAPE alone. That is a poor choice for this data
and it is worth being explicit about why, because it changes which model looks
best:

* **MAPE is undefined at zero.** Weekly SKU demand hits zero regularly. The old
  code only avoided a division by zero because it dropped zero-demand weeks by
  accident (see ``data.features``). With those weeks restored, MAPE is
  infinite.
* **MAPE is asymmetric.** It penalises over-forecasting more heavily than
  under-forecasting, so it systematically prefers models that under-predict —
  exactly the wrong bias for inventory, where a stockout usually costs more
  than a week of holding.

So MASE is the headline metric here: it divides absolute error by the in-sample
error of a seasonal-naive forecast, which makes it scale-free, defined at zero,
and directly interpretable — **MASE < 1 means the model beats seasonal naive**.
WAPE is reported alongside because it is what a planner intuitively reads as
"percentage error", and MAPE is kept purely for continuity with the previous
numbers.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent.

    Zero actuals are excluded rather than allowed to produce infinity; the
    count of excluded points is what makes this metric untrustworthy here.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    mask = np.abs(y_true) > EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent, bounded at 200 and defined when actual is 0."""
    y_true, y_pred = _arrays(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > EPS
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: total error / total actual, in percent.

    This is the metric a demand planner actually cares about, because it
    weights each SKU-week by its volume instead of treating a 1-unit miss on a
    2-unit item the same as a 1-unit miss on a 500-unit item.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    total = np.sum(np.abs(y_true))
    if total < EPS:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


def mase(y_true, y_pred, y_train, seasonal_period: int = 1) -> float:
    """Mean absolute scaled error.

    The scale is the mean absolute error of a seasonal-naive forecast computed
    **in sample** on ``y_train``, as defined by Hyndman & Koehler (2006). Values
    below 1 beat seasonal naive on the training history.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float).ravel()

    m = seasonal_period if len(y_train) > seasonal_period else 1
    if len(y_train) <= m:
        return float("nan")

    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale < EPS:
        # A perfectly flat history has no scale to divide by.
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def bias(y_true, y_pred) -> float:
    """Mean forecast error as a percentage of mean actual.

    Positive means the model over-forecasts. Reported because a model can have
    good absolute error while being consistently biased, which quietly builds
    up or drains inventory.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    mean_actual = np.mean(y_true)
    if abs(mean_actual) < EPS:
        return float("nan")
    return float(np.mean(y_pred - y_true) / mean_actual * 100)


def coverage(y_true, lower, upper) -> float:
    """Share of actuals falling inside the prediction interval, in percent.

    A nominal 95% interval that covers 60% of actuals is not a 95% interval,
    and safety stock derived from it will be wrong.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    return float(np.mean((y_true >= lower) & (y_true <= upper)) * 100)


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Quantile (pinball) loss — how the interval bounds themselves are scored."""
    y_true, y_pred = _arrays(y_true, y_pred)
    delta = y_true - y_pred
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1) * delta)))


def evaluate(
    y_true,
    y_pred,
    y_train=None,
    lower=None,
    upper=None,
    seasonal_period: int = 52,
) -> dict[str, float]:
    """Compute the full metric set in one call."""
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "bias_pct": bias(y_true, y_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, seasonal_period)
    if lower is not None and upper is not None:
        out["coverage_pct"] = coverage(y_true, lower, upper)
        out["pinball_10"] = pinball_loss(y_true, lower, 0.10)
        out["pinball_90"] = pinball_loss(y_true, upper, 0.90)
    return out
