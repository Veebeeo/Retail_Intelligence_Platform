"""Inventory economics: the translation from forecast error into money."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from retail_intel.business import inventory as INV


def test_optimal_service_level_equals_the_critical_ratio():
    assert INV.optimal_service_level(9.0, 1.0) == pytest.approx(0.9)
    assert INV.optimal_service_level(1.0, 1.0) == pytest.approx(0.5)


def test_expensive_stockouts_push_the_service_level_up():
    assert INV.optimal_service_level(100.0, 0.1) > INV.optimal_service_level(2.0, 1.0)


def test_service_level_is_clamped_to_a_sane_range():
    assert INV.optimal_service_level(1e9, 1e-9) <= 0.999
    assert INV.optimal_service_level(0.0, 5.0) >= 0.5


def test_safety_stock_matches_the_textbook_formula():
    expected = stats.norm.ppf(0.95) * 10 * np.sqrt(2)
    assert INV.safety_stock(10, 2, 0.95) == pytest.approx(expected)


def test_safety_stock_grows_with_the_square_root_of_lead_time():
    """Variance of a sum of independent weekly demands grows linearly, so its
    standard deviation grows with the square root."""
    one, four = INV.safety_stock(10, 1, 0.95), INV.safety_stock(10, 4, 0.95)
    assert four == pytest.approx(one * 2, rel=1e-6)


def test_higher_service_level_requires_more_safety_stock():
    assert INV.safety_stock(10, 2, 0.99) > INV.safety_stock(10, 2, 0.90)


def test_a_perfect_forecast_needs_no_buffer():
    assert INV.safety_stock(0.0, 4, 0.95) == 0.0


def test_reorder_point_is_lead_time_demand_plus_buffer():
    policy = INV.build_policy(
        "X", np.array([50.0, 50, 50, 50]), 6.0, lead_time_weeks=2, service_level=0.95
    )
    assert policy.expected_lead_time_demand == pytest.approx(100.0)
    assert policy.reorder_point == pytest.approx(100.0 + policy.safety_stock)


def test_policy_values_are_plain_floats():
    """numpy scalars break JSON serialisation in the API response."""
    policy = INV.build_policy("X", np.array([50.0, 50]), 6.0)
    assert isinstance(policy.safety_stock, float)
    assert isinstance(policy.reorder_point, float)


def test_a_better_forecast_costs_less_at_the_same_service_level():
    """The claim the whole forecasting effort has to earn."""
    rng = np.random.default_rng(0)
    actual = rng.poisson(50, 40).astype(float)
    accurate = actual + rng.normal(0, 3, 40)
    poor = actual + rng.normal(0, 25, 40)

    results = INV.compare_models(
        actual,
        {"accurate": accurate, "seasonal_naive": poor},
        {"accurate": 3.0, "seasonal_naive": 25.0},
    )
    assert results[0].model == "accurate"
    savings = INV.savings_vs_baseline(results)
    assert savings["champion"] == "accurate"
    assert savings["pct_saving"] > 0


def test_understocking_incurs_stockout_cost():
    cost = INV.simulate_cost(
        np.full(10, 100.0),
        np.full(10, 10.0),
        forecast_std=0.0,
        holding_cost_per_unit_week=1.0,
        stockout_cost_per_unit=5.0,
    )
    assert cost.units_short > 0
    assert cost.holding_cost == 0
    assert cost.fill_rate < 100


def test_overstocking_incurs_holding_cost():
    cost = INV.simulate_cost(
        np.full(10, 10.0),
        np.full(10, 100.0),
        forecast_std=0.0,
        holding_cost_per_unit_week=1.0,
        stockout_cost_per_unit=5.0,
    )
    assert cost.units_held > 0
    assert cost.stockout_cost == 0
    assert cost.fill_rate == 100


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="must match"):
        INV.simulate_cost(np.ones(5), np.ones(3), 1.0)


def test_savings_requires_the_named_baseline():
    results = INV.compare_models(np.ones(5), {"a": np.ones(5)}, {"a": 1.0})
    with pytest.raises(KeyError, match="baseline"):
        INV.savings_vs_baseline(results, baseline="not_present")
