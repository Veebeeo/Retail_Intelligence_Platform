"""Data contracts.

Every table crossing a pipeline boundary is validated against a schema before
it is written. A pipeline that fails loudly on bad input is worth more than one
that silently trains on it: the original code would happily have fitted a model
to negative demand.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)

# Raw retail line items after cleaning, before they hit `transactions`.
TransactionSchema = DataFrameSchema(
    {
        "invoice_no": Column(str, nullable=False),
        "stock_code": Column(str, nullable=False),
        "description": Column(str, nullable=True),
        "quantity": Column(int, Check.gt(0), nullable=False),
        "invoice_date": Column("datetime64[ns]", nullable=False),
        "unit_price": Column(float, Check.gt(0), nullable=False),
        "customer_id": Column("int64", nullable=False),
        "country": Column(str, nullable=True),
        "total_price": Column(float, Check.gt(0), nullable=False),
    },
    strict=True,
    coerce=True,
    name="transactions",
)

# Weekly per-SKU demand with engineered features.
WeeklyFeatureSchema = DataFrameSchema(
    {
        "stock_code": Column(str, nullable=False),
        "week": Column("datetime64[ns]", nullable=False),
        "weekly_sales": Column(float, Check.ge(0), nullable=False),
        "weekly_revenue": Column(float, Check.ge(0), nullable=True),
        "lag_1_week": Column(float, nullable=True),
        "lag_2_week": Column(float, nullable=True),
        "lag_4_week": Column(float, nullable=True),
        "lag_52_week": Column(float, nullable=True),
        "rolling_4_wk_avg": Column(float, nullable=True),
        "rolling_4_wk_std": Column(float, nullable=True),
        "rolling_12_wk_avg": Column(float, nullable=True),
        "month": Column(int, Check.in_range(1, 12), nullable=False),
        "week_of_year": Column(int, Check.in_range(1, 53), nullable=False),
        "weeks_since_start": Column(int, Check.ge(0), nullable=False),
    },
    strict=True,
    coerce=True,
    name="ml_weekly_features",
)

# RFM aggregates before clustering.
RFMSchema = DataFrameSchema(
    {
        "customer_id": Column("int64", nullable=False, unique=True),
        "recency": Column(int, Check.ge(0), nullable=False),
        "frequency": Column(int, Check.ge(1), nullable=False),
        "monetary": Column(float, Check.gt(0), nullable=False),
        "tenure": Column(int, Check.ge(0), nullable=False),
        "avg_order_value": Column(float, Check.gt(0), nullable=False),
    },
    strict=True,
    coerce=True,
    name="rfm",
)


def validate(df: pd.DataFrame, schema: DataFrameSchema, *, lazy: bool = True) -> pd.DataFrame:
    """Validate ``df``, logging a readable summary before re-raising on failure.

    Pandera's default error is a wall of text; collecting failure cases first
    makes the actual problem obvious in CI logs.
    """
    try:
        validated = schema.validate(df, lazy=lazy)
    except pa.errors.SchemaErrors as exc:
        summary = (
            exc.failure_cases.groupby(["column", "check"], dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        logger.error(
            "Contract '%s' failed on %d rows:\n%s", schema.name, len(exc.failure_cases), summary
        )
        raise
    logger.info("Contract '%s' passed: %d rows, %d columns", schema.name, *validated.shape)
    return validated
