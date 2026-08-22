"""Health, drift and analytics endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import text

from retail_intel import __version__
from retail_intel.config import get_settings
from retail_intel.db import get_engine, read_sql, table_exists
from retail_intel.forecasting.serving import try_load_bundle
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["monitoring"])


@router.get("/health")
def health():
    """Liveness and readiness.

    Reports *degraded* rather than *healthy* when the database or the models
    are missing, so a container orchestrator and a human both get the truth.
    The old health check returned a hard-coded "healthy" regardless of state.
    """
    db_status = "connected"
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        db_status = f"unavailable: {type(exc).__name__}"

    bundle = try_load_bundle()
    healthy = db_status == "connected" and bundle is not None

    return {
        "status": "healthy" if healthy else "degraded",
        "version": __version__,
        "database": db_status,
        "models_loaded": bundle is not None,
        "model_version": bundle.version if bundle else None,
        "n_skus": len(bundle.forecasts) if bundle else None,
        "detail": None if healthy else "Run the training pipeline and check DATABASE_URL.",
    }


@router.get("/drift")
def get_drift():
    """The most recent drift report produced by the monitoring pipeline."""
    path = get_settings().report_dir / "drift_report.json"
    if not path.exists():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No drift report yet. Run `python -m retail_intel.monitoring.drift`.",
        )
    return json.loads(path.read_text())


@router.get("/analytics/revenue")
def revenue_trend(weeks: int = Query(52, ge=4, le=260)):
    """Weekly revenue and order counts, for the dashboard's trend view."""
    if not table_exists("ml_weekly_features"):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Features have not been built yet."
        )

    df = read_sql(
        """
        SELECT week,
               SUM(weekly_sales)   AS units,
               SUM(weekly_revenue) AS revenue
        FROM ml_weekly_features
        GROUP BY week
        ORDER BY week DESC
        LIMIT :weeks
        """,
        {"weeks": weeks},
    )
    df = df.sort_values("week")
    df["week"] = df["week"].astype(str)
    return {"weeks": len(df), "series": df.round(2).to_dict(orient="records")}


@router.get("/analytics/top-skus")
def top_skus(limit: int = Query(10, ge=1, le=100)):
    """Highest-revenue SKUs over the whole history."""
    if not table_exists("ml_weekly_features"):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Features have not been built yet."
        )

    df = read_sql(
        """
        SELECT stock_code,
               SUM(weekly_sales)   AS total_units,
               SUM(weekly_revenue) AS total_revenue,
               AVG(weekly_sales)   AS avg_weekly_units
        FROM ml_weekly_features
        GROUP BY stock_code
        ORDER BY total_revenue DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )
    return {"skus": df.round(2).to_dict(orient="records")}
