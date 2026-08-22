"""SQLAlchemy schema.

Two changes from the original: ``create_all`` is an explicit call rather than an
import-time side effect, and ``customer_id`` is an integer. It was a float only
because pandas widens a column containing NaN, which is a property of the load
step, not of the data — the ingest pipeline drops those rows anyway.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    """One line item on one invoice."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_no = Column(String(20), nullable=False, index=True)
    stock_code = Column(String(20), nullable=False, index=True)
    description = Column(String(255))
    quantity = Column(Integer, nullable=False)
    invoice_date = Column(DateTime, nullable=False, index=True)
    unit_price = Column(Float, nullable=False)
    customer_id = Column(BigInteger, nullable=False, index=True)
    country = Column(String(64))
    total_price = Column(Float, nullable=False)

    __table_args__ = (
        # The two access patterns that matter: per-SKU time slices for
        # forecasting, per-customer rollups for RFM/CLV.
        Index("ix_transactions_sku_date", "stock_code", "invoice_date"),
        Index("ix_transactions_customer_date", "customer_id", "invoice_date"),
    )


class WeeklyFeature(Base):
    """Weekly demand per SKU with lag/rolling/calendar features."""

    __tablename__ = "ml_weekly_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), nullable=False, index=True)
    week = Column(Date, nullable=False, index=True)
    weekly_sales = Column(Float, nullable=False)
    weekly_revenue = Column(Float)
    lag_1_week = Column(Float)
    lag_2_week = Column(Float)
    lag_4_week = Column(Float)
    lag_52_week = Column(Float)
    rolling_4_wk_avg = Column(Float)
    rolling_4_wk_std = Column(Float)
    rolling_12_wk_avg = Column(Float)
    month = Column(Integer)
    week_of_year = Column(Integer)
    weeks_since_start = Column(Integer)

    __table_args__ = (Index("ix_weekly_sku_week", "stock_code", "week", unique=True),)


class CustomerSegment(Base):
    """RFM segment plus predicted lifetime value for one customer."""

    __tablename__ = "customer_segments"

    customer_id = Column(BigInteger, primary_key=True)
    recency = Column(Integer, nullable=False)
    frequency = Column(Integer, nullable=False)
    monetary = Column(Float, nullable=False)
    tenure = Column(Integer)
    avg_order_value = Column(Float)
    cluster = Column(Integer, nullable=False)
    segment_label = Column(String(64), nullable=False, index=True)
    predicted_purchases_90d = Column(Float)
    predicted_avg_order_value = Column(Float)
    predicted_clv_90d = Column(Float)
    probability_alive = Column(Float)
    churn_probability = Column(Float)
    recommended_action = Column(String(255))
    uplift_segment = Column(String(32))
    updated_at = Column(DateTime, default=datetime.utcnow)


def create_all() -> None:
    """Create every table. Explicit, so importing the schema is side-effect free."""
    from retail_intel.db import get_engine

    Base.metadata.create_all(bind=get_engine())
