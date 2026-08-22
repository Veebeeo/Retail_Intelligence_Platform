"""Customer segmentation, lifetime value and cross-sell endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.schemas import (
    RecommendationItem,
    RecommendResponse,
    SegmentMetrics,
    SegmentRequest,
    SegmentResponse,
    SegmentValue,
)
from retail_intel.db import read_sql, table_exists
from retail_intel.logging_conf import get_logger
from retail_intel.segmentation.strategies import strategy_for

logger = get_logger(__name__)
router = APIRouter(tags=["customers"])


@router.post("/segment", response_model=SegmentResponse)
def post_segment(payload: SegmentRequest):
    """Look up a customer's segment, predicted value and next best action.

    The query is parameterised. The original built it with an f-string, so
    ``customer_id`` was injectable straight into the database.
    """
    if not table_exists("customer_segments"):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Customer segments have not been built yet. Run the segmentation pipeline.",
        )

    df = read_sql(
        """
        SELECT customer_id, recency, frequency, monetary, tenure, avg_order_value,
               segment_label, predicted_purchases_90d, predicted_clv_90d,
               churn_probability, recommended_action
        FROM customer_segments
        WHERE customer_id = :customer_id
        """,
        {"customer_id": payload.customer_id},
    )
    if df.empty:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Customer {payload.customer_id} not found.")

    row = df.iloc[0]

    def opt(col: str) -> float | None:
        value = row.get(col)
        return None if value is None or value != value else float(value)  # NaN check

    label = str(row["segment_label"])
    return SegmentResponse(
        customer_id=int(row["customer_id"]),
        metrics=SegmentMetrics(
            days_since_last_purchase=int(row["recency"]),
            total_lifetime_orders=int(row["frequency"]),
            total_monetary_spend=round(float(row["monetary"]), 2),
            average_order_value=opt("avg_order_value"),
            tenure_days=int(row["tenure"]) if row.get("tenure") == row.get("tenure") else None,
        ),
        segment=label,
        recommended_action=str(row.get("recommended_action") or strategy_for(label)),
        value=SegmentValue(
            predicted_purchases_90d=opt("predicted_purchases_90d"),
            predicted_clv_90d=opt("predicted_clv_90d"),
            churn_probability=opt("churn_probability"),
        ),
        uplift_segment=None,
    )


@router.get("/segments/summary")
def get_segment_summary():
    """Segment sizes, revenue share and predicted value."""
    if not table_exists("customer_segments"):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Segments have not been built yet."
        )

    df = read_sql(
        """
        SELECT segment_label,
               COUNT(*)                    AS customers,
               AVG(recency)                AS avg_recency_days,
               AVG(frequency)              AS avg_frequency,
               AVG(monetary)               AS avg_monetary,
               SUM(monetary)               AS total_revenue,
               AVG(predicted_clv_90d)      AS avg_predicted_clv_90d,
               AVG(churn_probability)      AS avg_churn_probability
        FROM customer_segments
        GROUP BY segment_label
        ORDER BY total_revenue DESC
        """
    )
    total = float(df["total_revenue"].sum()) or 1.0
    df["revenue_share_pct"] = (df["total_revenue"] / total * 100).round(2)
    df["recommended_action"] = df["segment_label"].map(strategy_for)
    return {"segments": df.round(3).to_dict(orient="records")}


@router.get("/customers/at-risk")
def get_at_risk(
    limit: int = Query(20, ge=1, le=500),
    min_clv: float = Query(0.0, ge=0),
):
    """Customers with high predicted value *and* high churn probability.

    This is the list the win-back budget should be spent on — the intersection
    of "worth keeping" and "about to leave". Ranking by value alone puts loyal
    customers at the top, who need nothing.
    """
    if not table_exists("customer_segments"):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Segments have not been built yet."
        )

    df = read_sql(
        """
        SELECT customer_id, segment_label, recency, frequency, monetary,
               predicted_clv_90d, churn_probability
        FROM customer_segments
        WHERE churn_probability > 0.5 AND predicted_clv_90d >= :min_clv
        ORDER BY predicted_clv_90d DESC
        LIMIT :limit
        """,
        {"limit": limit, "min_clv": min_clv},
    )
    return {
        "n_customers": len(df),
        "total_revenue_at_risk": round(float(df["predicted_clv_90d"].sum()), 2) if len(df) else 0.0,
        "customers": df.round(3).to_dict(orient="records"),
    }


@router.get("/recommend/{stock_code}", response_model=RecommendResponse)
def get_recommendations(
    stock_code: str = Path(..., min_length=1, max_length=20),
    top_n: int = Query(5, ge=1, le=20),
):
    """Cross-sell suggestions for a SKU, ranked by lift."""
    if not table_exists("product_associations"):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Association rules have not been mined yet. Run the market-basket pipeline.",
        )

    sku = stock_code.strip().upper()
    df = read_sql(
        """
        SELECT consequent, consequent_desc, lift, confidence, support
        FROM product_associations
        WHERE antecedent = :sku
        ORDER BY lift DESC
        LIMIT :top_n
        """,
        {"sku": sku, "top_n": top_n},
    )
    if df.empty:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No association rules for '{sku}'. It may be below the support threshold.",
        )

    return RecommendResponse(
        stock_code=sku,
        recommendations=[
            RecommendationItem(
                stock_code=row.consequent,
                description=row.consequent_desc,
                lift=round(float(row.lift), 3),
                confidence=round(float(row.confidence), 4),
                support=round(float(row.support), 5),
                interpretation=(
                    f"Buyers of {sku} are {row.lift:.1f}x more likely than average to "
                    f"also buy {row.consequent}."
                ),
            )
            for row in df.itertuples()
        ],
    )
