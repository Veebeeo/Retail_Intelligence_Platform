"""Market-basket analysis: which products sell together.

The invoice data already in the warehouse supports this with no extra
collection — each invoice is a basket. Association rules turn that into a
cross-sell recommendation for any SKU.

The metric that matters is **lift**, not confidence. Confidence answers "of
people who bought A, what share also bought B", which ranks best-sellers at the
top no matter what A is — recommending the most popular item to everyone is not
a recommendation. Lift divides that by B's baseline rate, so it answers the
useful question: does buying A actually make B *more likely* than chance?
A lift of 1 means the two are independent.
"""

from __future__ import annotations

import argparse

import pandas as pd

from retail_intel.db import read_sql, write_table
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)


def load_baskets(min_basket_size: int = 2, max_basket_size: int = 60) -> pd.DataFrame:
    """One row per (invoice, SKU).

    Single-item invoices carry no co-occurrence information and only inflate
    the denominator of every support calculation. Very large baskets are
    usually wholesale orders rather than a shopper's basket, and they create
    spurious associations between everything they contain.
    """
    df = read_sql("SELECT invoice_no, stock_code, description FROM transactions")
    if df.empty:
        raise RuntimeError("`transactions` is empty. Run the ingest pipeline first.")

    df = df.drop_duplicates(subset=["invoice_no", "stock_code"])
    sizes = df.groupby("invoice_no")["stock_code"].transform("size")
    kept = df[(sizes >= min_basket_size) & (sizes <= max_basket_size)]

    logger.info(
        "Baskets: %d invoices usable of %d (sizes %d-%d)",
        kept["invoice_no"].nunique(),
        df["invoice_no"].nunique(),
        min_basket_size,
        max_basket_size,
    )
    return kept


def build_matrix(baskets: pd.DataFrame, top_n_items: int = 300) -> pd.DataFrame:
    """One-hot basket x item matrix.

    Restricted to the ``top_n_items`` most frequent SKUs: the matrix is dense
    and its memory cost grows with the item count, while rare items cannot
    clear any sensible support threshold anyway.
    """
    top_items = baskets["stock_code"].value_counts().nlargest(top_n_items).index
    subset = baskets[baskets["stock_code"].isin(top_items)]

    matrix = (
        subset.assign(present=1)
        .pivot_table(index="invoice_no", columns="stock_code", values="present", fill_value=0)
        .astype(bool)
    )
    logger.info("Basket matrix: %d invoices x %d items", *matrix.shape)
    return matrix


def mine_rules(
    matrix: pd.DataFrame,
    min_support: float = 0.02,
    min_lift: float = 1.5,
    max_len: int = 2,
) -> pd.DataFrame:
    """Mine association rules with FP-Growth.

    FP-Growth rather than Apriori: it makes a single pass to build a prefix
    tree instead of re-scanning the data once per itemset size, which matters
    at 20k+ baskets. ``max_len=2`` keeps rules to pairs, which is what a
    "customers also bought" widget can actually use.
    """
    from mlxtend.frequent_patterns import association_rules, fpgrowth

    itemsets = fpgrowth(matrix, min_support=min_support, use_colnames=True, max_len=max_len)
    if itemsets.empty:
        logger.warning("No itemsets at min_support=%.3f. Try lowering it.", min_support)
        return pd.DataFrame()

    # mlxtend 0.24 added a required `num_itemsets` argument. Support both
    # signatures rather than pinning the library to one minor version.
    try:
        rules = association_rules(
            itemsets, num_itemsets=len(matrix), metric="lift", min_threshold=min_lift
        )
    except TypeError:
        rules = association_rules(itemsets, metric="lift", min_threshold=min_lift)

    if rules.empty:
        logger.warning("No rules at min_lift=%.2f.", min_lift)
        return pd.DataFrame()

    rules = rules[
        (rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)
    ].copy()
    rules["antecedent"] = rules["antecedents"].apply(lambda s: next(iter(s)))
    rules["consequent"] = rules["consequents"].apply(lambda s: next(iter(s)))

    columns = ["antecedent", "consequent", "support", "confidence", "lift"]
    columns += [c for c in ("leverage", "conviction") if c in rules.columns]
    out = rules[columns].sort_values("lift", ascending=False).reset_index(drop=True)

    logger.info(
        "Mined %d pairwise rules (lift %.2f-%.2f)", len(out), out["lift"].min(), out["lift"].max()
    )
    return out


def attach_descriptions(rules: pd.DataFrame, baskets: pd.DataFrame) -> pd.DataFrame:
    """Add product names so rules are readable without a second lookup."""
    names = (
        baskets.dropna(subset=["description"])
        .groupby("stock_code")["description"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
    )
    out = rules.copy()
    out["antecedent_desc"] = out["antecedent"].map(names).fillna("")
    out["consequent_desc"] = out["consequent"].map(names).fillna("")
    return out


def recommend(rules: pd.DataFrame, stock_code: str, top_n: int = 5) -> list[dict]:
    """Top cross-sell candidates for one SKU, ranked by lift."""
    matches = rules[rules["antecedent"] == stock_code].nlargest(top_n, "lift")
    return [
        {
            "stock_code": row.consequent,
            "description": getattr(row, "consequent_desc", ""),
            "lift": round(float(row.lift), 3),
            "confidence": round(float(row.confidence), 4),
            "support": round(float(row.support), 5),
            "interpretation": (
                f"Buyers of this SKU are {row.lift:.1f}x more likely than average "
                f"to also buy {row.consequent}."
            ),
        }
        for row in matches.itertuples()
    ]


def run(min_support: float = 0.02, min_lift: float = 1.5, top_n_items: int = 300) -> pd.DataFrame:
    baskets = load_baskets()
    matrix = build_matrix(baskets, top_n_items)
    rules = mine_rules(matrix, min_support, min_lift)
    if rules.empty:
        # Retry once at a lower bar rather than writing an empty table: support
        # thresholds are data-dependent and 2% is only a reasonable default.
        logger.info("Retrying at min_support=%.3f", min_support / 4)
        rules = mine_rules(matrix, min_support / 4, min_lift)
    if rules.empty:
        return rules

    rules = attach_descriptions(rules, baskets)
    write_table(rules, "product_associations")
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine product association rules.")
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--min-lift", type=float, default=1.5)
    parser.add_argument("--top-items", type=int, default=300)
    args = parser.parse_args()

    rules = run(args.min_support, args.min_lift, args.top_items)
    if rules.empty:
        logger.warning("No association rules found.")
        return
    logger.info("Top rules by lift:\n%s", rules.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
