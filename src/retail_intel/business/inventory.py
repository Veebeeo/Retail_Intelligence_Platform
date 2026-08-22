"""Turning forecast error into money.

A MASE of 0.8 means nothing to anyone who buys stock for a living. This module
converts a forecast and its uncertainty into the two numbers a planner acts on
— how much to hold, and when to reorder — and then prices the consequences of
getting it wrong.

The model is the classic newsvendor / continuous-review (s, Q) formulation:

    safety stock   = z(SL) * sigma_demand * sqrt(lead_time)
    reorder point  = expected demand over lead time + safety stock

where ``z(SL)`` is the normal quantile at the target service level. The square
root of lead time is there because the variance of a sum of independent weekly
demands grows linearly with the number of weeks, so its standard deviation
grows with the square root.

The critical-ratio version is also provided, because when stockout and holding
costs are known the *optimal* service level is not a policy choice at all — it
falls out of the cost ratio:

    SL* = Cu / (Cu + Co)

with ``Cu`` the cost of being one unit short and ``Co`` the cost of holding one
unit too many. That is the number this project uses to argue that a better
forecast is worth money rather than just a better metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from retail_intel.config import get_settings


@dataclass(frozen=True)
class InventoryPolicy:
    """A reorder policy for one SKU."""

    stock_code: str
    expected_lead_time_demand: float
    demand_std: float
    safety_stock: float
    reorder_point: float
    service_level: float
    z_score: float
    lead_time_weeks: int

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class CostBreakdown:
    """What a forecast cost over an evaluation window."""

    model: str
    holding_cost: float
    stockout_cost: float
    total_cost: float
    units_held: float
    units_short: float
    fill_rate: float
    periods: int

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def optimal_service_level(unit_stockout_cost: float, unit_holding_cost: float) -> float:
    """Newsvendor critical ratio: the service level that minimises expected cost.

    Clamped to [0.50, 0.999] because the formula runs to 1.0 as holding cost
    approaches zero, and no real operation holds infinite stock.
    """
    if unit_stockout_cost <= 0:
        return 0.5
    ratio = unit_stockout_cost / (unit_stockout_cost + max(unit_holding_cost, 1e-9))
    return float(np.clip(ratio, 0.50, 0.999))


def safety_stock(demand_std: float, lead_time_weeks: int, service_level: float) -> float:
    """z(SL) * sigma * sqrt(L)."""
    z = float(stats.norm.ppf(service_level))
    return float(max(0.0, z * demand_std * np.sqrt(max(lead_time_weeks, 1))))


def build_policy(
    stock_code: str,
    forecast: np.ndarray,
    forecast_std: float,
    lead_time_weeks: int | None = None,
    service_level: float | None = None,
) -> InventoryPolicy:
    """Turn a demand forecast and its uncertainty into a reorder policy.

    ``forecast_std`` should come from the *forecast error* distribution, not
    from the variability of demand itself. Using demand variance is a common
    mistake that over-sizes safety stock: a model that predicts a seasonal peak
    correctly does not need buffer against that peak.
    """
    settings = get_settings()
    lead_time_weeks = lead_time_weeks or settings.lead_time_weeks
    service_level = service_level if service_level is not None else settings.service_level

    forecast = np.asarray(forecast, dtype=float)
    horizon = min(lead_time_weeks, len(forecast))
    expected = float(forecast[:horizon].sum())
    ss = safety_stock(forecast_std, lead_time_weeks, service_level)

    return InventoryPolicy(
        stock_code=stock_code,
        expected_lead_time_demand=expected,
        demand_std=float(forecast_std),
        safety_stock=float(ss),
        reorder_point=float(expected + ss),
        service_level=service_level,
        z_score=float(stats.norm.ppf(service_level)),
        lead_time_weeks=lead_time_weeks,
    )


def simulate_cost(
    actual: np.ndarray,
    forecast: np.ndarray,
    forecast_std: float,
    model: str = "model",
    holding_cost_per_unit_week: float | None = None,
    stockout_cost_per_unit: float | None = None,
    service_level: float | None = None,
    lead_time_weeks: int | None = None,
) -> CostBreakdown:
    """Price a forecast by simulating the inventory it would have driven.

    Each period we order up to ``forecast + safety stock``, then compare against
    what actually sold. Leftover units accrue holding cost; unmet demand accrues
    stockout cost. This is deliberately a simple order-up-to policy rather than
    a full supply-chain simulation — the point is a like-for-like comparison
    between forecasts, not an operations research exercise.
    """
    settings = get_settings()
    holding = (
        holding_cost_per_unit_week
        if holding_cost_per_unit_week is not None
        else settings.holding_cost_per_unit_week
    )
    stockout = (
        stockout_cost_per_unit
        if stockout_cost_per_unit is not None
        else settings.stockout_cost_per_unit
    )
    service_level = service_level if service_level is not None else settings.service_level
    lead_time_weeks = lead_time_weeks or settings.lead_time_weeks

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError(f"actual {actual.shape} and forecast {forecast.shape} must match")

    buffer = safety_stock(forecast_std, lead_time_weeks, service_level)
    order_up_to = np.maximum(forecast + buffer, 0.0)

    leftover = np.maximum(order_up_to - actual, 0.0)
    shortfall = np.maximum(actual - order_up_to, 0.0)

    units_held = float(leftover.sum())
    units_short = float(shortfall.sum())
    total_demand = float(actual.sum())

    return CostBreakdown(
        model=model,
        holding_cost=units_held * holding,
        stockout_cost=units_short * stockout,
        total_cost=units_held * holding + units_short * stockout,
        units_held=units_held,
        units_short=units_short,
        fill_rate=float(1 - units_short / total_demand) * 100 if total_demand > 0 else 100.0,
        periods=len(actual),
    )


def compare_models(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    forecast_stds: dict[str, float],
    **kwargs,
) -> list[CostBreakdown]:
    """Cost every candidate forecast on the same actuals, cheapest first."""
    results = [
        simulate_cost(
            actual, fc, forecast_stds.get(name, float(np.std(actual))), model=name, **kwargs
        )
        for name, fc in forecasts.items()
    ]
    return sorted(results, key=lambda r: r.total_cost)


def savings_vs_baseline(results: list[CostBreakdown], baseline: str = "seasonal_naive") -> dict:
    """Express the champion's advantage over the baseline in money and percent.

    This is the sentence the whole forecasting effort has to earn: not "MASE
    improved by 0.2" but "the same service level costs X% less to deliver".
    """
    by_name = {r.model: r for r in results}
    if baseline not in by_name:
        raise KeyError(f"No baseline '{baseline}' in results: {sorted(by_name)}")

    base = by_name[baseline]
    best = min(results, key=lambda r: r.total_cost)
    delta = base.total_cost - best.total_cost

    return {
        "baseline": baseline,
        "baseline_cost": round(base.total_cost, 2),
        "champion": best.model,
        "champion_cost": round(best.total_cost, 2),
        "absolute_saving": round(delta, 2),
        "pct_saving": round(delta / base.total_cost * 100, 2) if base.total_cost > 0 else 0.0,
        "fill_rate_baseline": round(base.fill_rate, 2),
        "fill_rate_champion": round(best.fill_rate, 2),
    }
