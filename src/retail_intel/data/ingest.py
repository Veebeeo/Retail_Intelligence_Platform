"""Raw retail workbook -> validated ``transactions`` table.

Cleaning decisions and why they are made:

* Rows with no customer id are dropped. RFM and CLV are per-customer, so an
  anonymous line item cannot contribute to either.
* Cancellations (invoice prefixed ``C``) are dropped *together with the
  original line they reverse*. The previous version dropped only the credit
  note, which left the original sale counted as revenue that was in fact
  refunded — inflating both monetary value and demand.
* Non-product stock codes (POSTAGE, DOT, M, BANK CHARGES...) are dropped. They
  are not inventory and would pollute the top-SKU selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retail_intel.data.contracts import TransactionSchema, validate
from retail_intel.data.models import create_all
from retail_intel.db import write_table
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)

COLUMN_MAP = {
    "Invoice": "invoice_no",
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "UnitPrice": "unit_price",
    "Customer ID": "customer_id",
    "CustomerID": "customer_id",
    "Country": "country",
}

# Service lines and adjustments, not sellable inventory.
NON_PRODUCT_CODES = {
    "POST",
    "DOT",
    "M",
    "m",
    "C2",
    "CRUK",
    "BANK CHARGES",
    "B",
    "AMAZONFEE",
    "S",
    "D",
    "PADS",
    "GIFT",
}


def read_raw(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Read the source workbook or CSV.

    ``online_retail_II.xlsx`` ships two sheets (2009-2010 and 2010-2011).
    Passing ``sheet_name=None`` reads and concatenates both, which roughly
    doubles the history available for seasonal models.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    frames = pd.read_excel(path, sheet_name=sheet_name)
    if isinstance(frames, dict):
        logger.info("Reading %d sheets: %s", len(frames), list(frames))
        return pd.concat(frames.values(), ignore_index=True)
    return frames


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules described in the module docstring."""
    start = len(df)
    df = df.rename(columns=COLUMN_MAP).copy()

    required = {"invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"}
    if missing := required - set(df.columns):
        raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")

    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

    df = df.dropna(subset=["customer_id", "invoice_date"])
    df["customer_id"] = df["customer_id"].astype("int64")

    df = _remove_cancellations(df)

    df = df[~df["stock_code"].isin({c.upper() for c in NON_PRODUCT_CODES})]
    # Codes that are pure letters are adjustments; real SKUs are numeric with an
    # optional letter suffix (e.g. 85123A).
    df = df[df["stock_code"].str.contains(r"\d", regex=True)]

    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    df = df.drop_duplicates(
        subset=["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price"]
    )

    df["description"] = df.get("description", pd.Series(dtype=str)).astype(str).str.strip()
    df["country"] = df.get("country", pd.Series(dtype=str)).astype(str).str.strip()
    df["total_price"] = df["quantity"] * df["unit_price"]

    df = df[list(TransactionSchema.columns)].reset_index(drop=True)
    logger.info(
        "Cleaned %d -> %d rows (%.1f%% retained)", start, len(df), 100 * len(df) / max(start, 1)
    )
    return df


def _remove_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Drop credit notes *and* the sales they reverse.

    A cancellation carries a negative quantity for some (customer, sku). We
    match it back to the most recent prior positive line for the same customer
    and SKU with the same absolute quantity, and remove both.
    """
    is_cancel = df["invoice_no"].str.upper().str.startswith("C")
    cancels = df[is_cancel]
    if cancels.empty:
        return df

    sales = df[~is_cancel]
    # Build a key of (customer, sku, |qty|); each cancellation consumes one
    # matching sale.
    cancel_keys = list(
        zip(cancels["customer_id"], cancels["stock_code"], cancels["quantity"].abs(), strict=True)
    )
    sale_keys = pd.Series(
        list(zip(sales["customer_id"], sales["stock_code"], sales["quantity"], strict=True)),
        index=sales.index,
    )

    to_drop: list[int] = []
    remaining = sale_keys.copy()
    for key in cancel_keys:
        matches = remaining.index[remaining == key]
        if len(matches):
            idx = matches[-1]  # most recent matching sale
            to_drop.append(idx)
            remaining = remaining.drop(idx)

    logger.info("Removed %d credit notes and %d reversed sale lines", len(cancels), len(to_drop))
    return df.drop(index=list(cancels.index) + to_drop)


def run(path: str | Path, sheet_name: str | int | None = None, if_exists: str = "replace") -> int:
    create_all()
    raw = read_raw(path, sheet_name)
    cleaned = validate(clean(raw), TransactionSchema)
    return write_table(cleaned, "transactions", if_exists=if_exists)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest raw retail transactions into the warehouse."
    )
    parser.add_argument("--path", default="data/online_retail_II.xlsx")
    parser.add_argument(
        "--sheet", default=None, help="Sheet name; omit to read and concatenate every sheet."
    )
    parser.add_argument("--if-exists", default="replace", choices=["replace", "append"])
    args = parser.parse_args()
    rows = run(args.path, args.sheet, args.if_exists)
    logger.info("Ingest complete: %d transactions", rows)


if __name__ == "__main__":
    main()
