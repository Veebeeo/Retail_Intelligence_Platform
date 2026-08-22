"""Demand forecasting and inventory endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Path, status

from app.dependencies import get_bundle
from app.schemas import (
    ForecastRequest,
    ForecastResponse,
    InventoryRequest,
    InventoryResponse,
)
from retail_intel.business import inventory as INV
from retail_intel.config import get_settings
from retail_intel.forecasting.serving import bundle_summary, forecast
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["forecasting"])


@router.post("/forecast", response_model=ForecastResponse)
def post_forecast(payload: ForecastRequest):
    """Forecast weekly demand for a SKU using its champion model.

    The response carries its own provenance — which model, how it scored
    against the baseline in backtesting, and when it was trained — so a
    consumer can tell a well-validated forecast from a weak one.
    """
    bundle = get_bundle()
    try:
        return forecast(payload.stock_code, payload.horizon_weeks, bundle)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No trained model for stock code '{payload.stock_code}'. "
                f"{len(bundle.skus)} SKUs are available; see GET /models/skus."
            ),
        ) from None


@router.post("/inventory/policy", response_model=InventoryResponse)
def post_inventory_policy(payload: InventoryRequest):
    """Turn a forecast into a reorder point and safety stock.

    When both cost parameters are supplied the service level is *derived* from
    the newsvendor critical ratio rather than taken as an input — if you know
    what a stockout and a unit of excess stock each cost, the optimal service
    level is determined, not a policy choice.
    """
    bundle = get_bundle()
    settings = get_settings()
    sku = payload.stock_code
    if not bundle.has(sku):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No trained model for '{sku}'.")

    if payload.unit_holding_cost is not None and payload.unit_stockout_cost is not None:
        service_level = INV.optimal_service_level(
            payload.unit_stockout_cost, payload.unit_holding_cost
        )
        source = "derived from the newsvendor critical ratio Cu/(Cu+Co)"
    else:
        service_level = payload.service_level or settings.service_level
        source = "supplied" if payload.service_level else "configured default"

    horizon = max(payload.lead_time_weeks, 1)
    preds = bundle.models[sku].predict(horizon)["yhat"].to_numpy()
    residual_std = float(bundle.meta.get(sku, {}).get("residual_std") or np.std(preds))

    policy = INV.build_policy(sku, preds, residual_std, payload.lead_time_weeks, service_level)
    model_name = bundle.meta.get(sku, {}).get("model", "unknown")

    return InventoryResponse(
        stock_code=sku,
        model=model_name,
        expected_lead_time_demand=round(policy.expected_lead_time_demand, 2),
        safety_stock=round(policy.safety_stock, 2),
        reorder_point=round(policy.reorder_point, 2),
        service_level=round(policy.service_level, 4),
        service_level_source=source,
        z_score=round(policy.z_score, 4),
        lead_time_weeks=policy.lead_time_weeks,
        forecast_error_std=round(residual_std, 3),
        explanation=(
            f"Reorder when stock reaches {policy.reorder_point:.0f} units. That covers "
            f"{policy.expected_lead_time_demand:.0f} units of expected demand over the "
            f"{policy.lead_time_weeks}-week lead time, plus {policy.safety_stock:.0f} units of "
            f"safety stock to hold a {policy.service_level:.1%} service level given a forecast "
            f"error standard deviation of {residual_std:.1f} units."
        ),
    )


@router.get("/models")
def get_models():
    """Registry summary: which models are serving, and how well they scored."""
    return bundle_summary(get_bundle())


@router.get("/models/skus")
def get_skus():
    """Every SKU with a trained champion."""
    bundle = get_bundle()
    return {"n_skus": len(bundle.skus), "stock_codes": bundle.skus}


@router.get("/models/{stock_code}")
def get_model_detail(stock_code: str = Path(..., min_length=1, max_length=20)):
    """Backtest provenance for one SKU's champion."""
    bundle = get_bundle()
    sku = stock_code.strip().upper()
    if sku not in bundle.meta:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No model for '{sku}'.")
    return {"stock_code": sku, **bundle.meta[sku], "model_version": bundle.version}
