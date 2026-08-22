"""Request and response models.

The originals accepted a bare string stock code and a float customer id with
no constraints, then interpolated both into SQL. Validation happens here now,
before anything reaches the database layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    stock_code: str = Field(..., min_length=1, max_length=20, examples=["85123A"])
    horizon_weeks: int = Field(4, ge=1, le=26)

    @field_validator("stock_code")
    @classmethod
    def normalise(cls, v: str) -> str:
        # Stock codes are alphanumeric. Rejecting anything else is defence in
        # depth: queries are parameterised, but a value that cannot be a stock
        # code should not reach the database at all.
        v = v.strip().upper()
        if not v.isalnum():
            raise ValueError("stock_code must be alphanumeric")
        return v


class PredictionPoint(BaseModel):
    week_horizon: int
    week_starting: str
    predicted_quantity: float
    lower_95: float
    upper_95: float


class ForecastResponse(BaseModel):
    stock_code: str
    horizon_weeks: int
    model: str = Field(..., description="The champion model selected for this SKU by backtest.")
    model_version: int
    trained_at: str
    backtest_mase: float | None = Field(
        None, description="Mean absolute scaled error. Below 1 beats seasonal naive."
    )
    baseline_mase: float | None = None
    improvement_vs_baseline_pct: float | None = None
    residual_std: float | None = None
    predictions: list[PredictionPoint]


class InventoryRequest(BaseModel):
    stock_code: str = Field(..., min_length=1, max_length=20)
    lead_time_weeks: int = Field(2, ge=1, le=26)
    service_level: float | None = Field(None, ge=0.5, lt=1.0)
    unit_holding_cost: float | None = Field(None, ge=0)
    unit_stockout_cost: float | None = Field(None, ge=0)

    @field_validator("stock_code")
    @classmethod
    def normalise(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.isalnum():
            raise ValueError("stock_code must be alphanumeric")
        return v


class InventoryResponse(BaseModel):
    stock_code: str
    model: str
    expected_lead_time_demand: float
    safety_stock: float
    reorder_point: float
    service_level: float
    service_level_source: str
    z_score: float
    lead_time_weeks: int
    forecast_error_std: float
    explanation: str


class SegmentRequest(BaseModel):
    customer_id: int = Field(..., ge=0, examples=[17841])


class SegmentMetrics(BaseModel):
    days_since_last_purchase: int
    total_lifetime_orders: int
    total_monetary_spend: float
    average_order_value: float | None = None
    tenure_days: int | None = None


class SegmentValue(BaseModel):
    predicted_purchases_90d: float | None = None
    predicted_clv_90d: float | None = None
    churn_probability: float | None = None


class SegmentResponse(BaseModel):
    customer_id: int
    metrics: SegmentMetrics
    segment: str
    recommended_action: str
    value: SegmentValue
    uplift_segment: str | None = None


class RecommendationItem(BaseModel):
    stock_code: str
    description: str | None = None
    lift: float
    confidence: float
    support: float
    interpretation: str


class RecommendResponse(BaseModel):
    stock_code: str
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    models_loaded: bool
    model_version: int | None = None
    n_skus: int | None = None
    detail: str | None = None
