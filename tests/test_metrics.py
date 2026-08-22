"""Forecast metrics, including the edge cases that make MAPE unusable here."""

from __future__ import annotations

import numpy as np
import pytest

from retail_intel.forecasting import metrics as M


def test_perfect_forecast_scores_zero_error():
    y = np.array([10.0, 20, 30, 40])
    assert M.mae(y, y) == 0
    assert M.rmse(y, y) == 0
    assert M.smape(y, y) == 0
    assert M.wape(y, y) == 0


def test_mape_skips_zero_actuals_instead_of_returning_infinity():
    """Weekly SKU demand hits zero regularly, which is exactly why MAPE is not
    the headline metric in this project."""
    y = np.array([0.0, 10.0])
    result = M.mape(y, np.array([5.0, 11.0]))
    assert np.isfinite(result)
    assert result == pytest.approx(10.0)


def test_mape_is_nan_when_every_actual_is_zero():
    assert np.isnan(M.mape(np.zeros(4), np.ones(4)))


def test_smape_is_defined_at_zero_actuals():
    assert np.isfinite(M.smape(np.array([0.0, 10]), np.array([5.0, 11])))


def test_mase_below_one_means_better_than_seasonal_naive():
    # Not perfectly seasonal: an exactly repeating history gives a zero scale
    # and MASE is then undefined (covered by the next test).
    train = np.array([10.0, 20, 12, 22, 9, 19, 11, 21])
    y_true = np.array([10.0, 20])
    assert M.mase(y_true, y_true, train, seasonal_period=2) == 0.0
    assert M.mase(y_true, np.array([30.0, 40]), train, seasonal_period=2) > 1.0


def test_mase_is_nan_for_a_flat_history():
    """A constant series gives a zero scale, so MASE cannot be formed."""
    assert np.isnan(M.mase(np.array([1.0, 2]), np.array([1.0, 2]), np.ones(20), 1))


def test_wape_weights_by_volume():
    """A 1-unit miss on a 500-unit item matters less than on a 2-unit item.
    MAPE treats them identically; WAPE does not."""
    y = np.array([500.0, 2.0])
    high_volume_miss = M.wape(y, np.array([501.0, 2.0]))
    low_volume_miss = M.wape(y, np.array([500.0, 3.0]))
    assert high_volume_miss == pytest.approx(low_volume_miss)
    assert M.mape(y, np.array([501.0, 2.0])) < M.mape(y, np.array([500.0, 3.0]))


def test_bias_sign_indicates_direction():
    y = np.array([10.0, 10, 10])
    assert M.bias(y, np.array([12.0, 12, 12])) > 0
    assert M.bias(y, np.array([8.0, 8, 8])) < 0


def test_coverage_counts_actuals_inside_the_interval():
    y = np.array([1.0, 2, 3, 4])
    assert M.coverage(y, y - 1, y + 1) == 100.0
    assert M.coverage(y, y + 5, y + 6) == 0.0
    assert M.coverage(y, np.array([0.0, 0, 10, 10]), np.array([2.0, 3, 20, 20])) == 50.0


def test_pinball_loss_penalises_asymmetrically():
    y = np.array([10.0])
    over = M.pinball_loss(y, np.array([12.0]), 0.9)
    under = M.pinball_loss(y, np.array([8.0]), 0.9)
    assert under > over  # a 90th-percentile forecast should rarely fall short


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        M.mae(np.ones(3), np.ones(4))


def test_evaluate_returns_the_full_metric_set():
    y, p, train = np.array([10.0, 12, 14]), np.array([11.0, 11, 15]), np.arange(1, 30.0)
    out = M.evaluate(y, p, train, p - 2, p + 2, seasonal_period=4)
    assert {"mae", "rmse", "mape", "smape", "wape", "bias_pct", "mase", "coverage_pct"} <= set(out)
