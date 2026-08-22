"""Cleaning rules, which is where silent data corruption starts."""

from __future__ import annotations

import pandas as pd
import pytest

from retail_intel.data.ingest import clean


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_cancellation_removes_both_the_credit_note_and_the_original_sale():
    """The bug this replaced: only the credit note was dropped, so a refunded
    order still counted as revenue."""
    raw = _frame(
        [
            {
                "Invoice": "A1",
                "StockCode": "85123A",
                "Description": "x",
                "Quantity": 6,
                "InvoiceDate": "2011-01-01",
                "Price": 2.5,
                "Customer ID": 17850.0,
                "Country": "UK",
            },
            {
                "Invoice": "C9",
                "StockCode": "85123A",
                "Description": "x",
                "Quantity": -6,
                "InvoiceDate": "2011-01-04",
                "Price": 2.5,
                "Customer ID": 17850.0,
                "Country": "UK",
            },
            {
                "Invoice": "A2",
                "StockCode": "22423",
                "Description": "y",
                "Quantity": 3,
                "InvoiceDate": "2011-01-01",
                "Price": 4.0,
                "Customer ID": 13047.0,
                "Country": "UK",
            },
        ]
    )
    out = clean(raw)
    assert list(out["invoice_no"]) == ["A2"]
    assert out["total_price"].sum() == pytest.approx(12.0)


def test_cancellation_only_consumes_one_matching_sale():
    """Two identical purchases and one cancellation should leave one purchase."""
    rows = [
        {
            "Invoice": f"A{i}",
            "StockCode": "85123A",
            "Description": "x",
            "Quantity": 6,
            "InvoiceDate": "2011-01-01",
            "Price": 2.5,
            "Customer ID": 17850.0,
            "Country": "UK",
        }
        for i in range(2)
    ]
    rows.append(
        {
            "Invoice": "C1",
            "StockCode": "85123A",
            "Description": "x",
            "Quantity": -6,
            "InvoiceDate": "2011-01-05",
            "Price": 2.5,
            "Customer ID": 17850.0,
            "Country": "UK",
        }
    )
    assert len(clean(_frame(rows))) == 1


def test_rows_without_a_customer_id_are_dropped(raw_transactions):
    assert clean(raw_transactions)["customer_id"].notna().all()


def test_service_codes_are_excluded(clean_transactions):
    """POST, M, DOT and friends are not inventory and must not reach the SKU list."""
    codes = set(clean_transactions["stock_code"])
    assert not codes & {"POST", "M", "DOT", "BANK CHARGES"}


def test_no_non_positive_quantities_or_prices(clean_transactions):
    assert (clean_transactions["quantity"] > 0).all()
    assert (clean_transactions["unit_price"] > 0).all()


def test_total_price_is_consistent(clean_transactions):
    expected = clean_transactions["quantity"] * clean_transactions["unit_price"]
    pd.testing.assert_series_equal(clean_transactions["total_price"], expected, check_names=False)


def test_customer_id_is_an_integer(clean_transactions):
    """It was a float purely because pandas widens a column containing NaN."""
    assert clean_transactions["customer_id"].dtype.kind == "i"


def test_duplicate_lines_are_removed():
    row = {
        "Invoice": "A1",
        "StockCode": "85123A",
        "Description": "x",
        "Quantity": 6,
        "InvoiceDate": "2011-01-01",
        "Price": 2.5,
        "Customer ID": 17850.0,
        "Country": "UK",
    }
    assert len(clean(_frame([row, dict(row)]))) == 1


def test_missing_required_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        clean(_frame([{"Invoice": "A1", "StockCode": "X"}]))
