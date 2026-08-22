"""Synthetic transaction generator.

Serves two purposes: deterministic fixtures for the test suite, and a way for
anyone cloning the repo to run the whole pipeline end to end without obtaining
the source workbook first.

It is built around **shopping trips** rather than per-SKU demand draws, because
that is what the real data is: one invoice is one customer's basket containing
several products. Generating SKU demand directly would produce single-item
invoices, and market-basket analysis would have nothing to find.

The generative process:

* Each SKU has a base popularity, a linear trend, an annual seasonal shape and
  occasional promotional spikes.
* Customers are acquired over time and silently churn — nobody cancels in
  non-contractual retail, they just stop coming back. Acquisition roughly
  balances churn, so the active population is stationary and drift monitoring
  is not just detecting the generator winding down.
* Some SKU pairs are deliberately **complementary**: buying one raises the
  probability of the other appearing in the same basket, which is the signal
  the association-rule miner should recover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNTRIES = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"]


def make_transactions(
    n_customers: int = 400,
    n_skus: int = 30,
    start: str = "2010-01-04",
    weeks: int = 104,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a transaction-level frame matching the real schema."""
    rng = np.random.default_rng(seed)
    week_index = pd.date_range(start, periods=weeks, freq="W-MON")

    skus = [f"{85000 + i}{'A' if i % 4 == 0 else ''}" for i in range(n_skus)]
    popularity = rng.gamma(shape=2.0, scale=1.0, size=n_skus) + 0.3
    popularity = popularity / popularity.sum()
    trend = rng.normal(0.002, 0.004, size=n_skus)
    seasonal_amp = rng.uniform(0.2, 0.7, size=n_skus)
    seasonal_phase = rng.uniform(0, 2 * np.pi, size=n_skus)
    unit_price = np.round(rng.gamma(shape=2.0, scale=1.8, size=n_skus) + 0.5, 2)

    # Complementary pairs: buying `a` makes `b` far more likely in the same
    # basket. These are the rules market-basket analysis should recover.
    n_pairs = max(1, n_skus // 5)
    pair_a = rng.choice(n_skus, n_pairs, replace=False)
    pair_b = np.array([rng.choice([j for j in range(n_skus) if j != a]) for a in pair_a])
    complements = dict(zip(pair_a, pair_b, strict=True))

    # Customer population: staggered acquisition, geometric lifespan.
    cust_ids = rng.choice(np.arange(12000, 12000 + n_customers * 3), n_customers, replace=False)
    lifespan = rng.geometric(p=0.015, size=n_customers)

    # A stationary active population. If every customer were acquired uniformly
    # across the window the active count would ramp up all the way through, and
    # drift monitoring would just be detecting the generator warming up. So
    # roughly half the customers are already active at week 0 with a random
    # amount of their lifespan already elapsed -- the steady state a real
    # business is in -- and the rest are acquired as replacements over time.
    incumbent = rng.random(n_customers) < 0.55
    elapsed = np.where(incumbent, rng.integers(0, np.maximum(lifespan, 1)), 0)
    cust_start = np.where(incumbent, 0, rng.integers(0, weeks, size=n_customers))
    cust_end = np.minimum(cust_start + np.maximum(lifespan - elapsed, 1), weeks)
    # Heavy-tailed shopping frequency: a few customers shop most weeks.
    cust_trip_rate = np.clip(rng.beta(1.2, 8.0, size=n_customers), 0.01, 0.9)
    cust_basket_size = rng.gamma(shape=2.5, scale=1.4, size=n_customers) + 1
    cust_country = rng.choice(COUNTRIES, n_customers, p=[0.75, 0.06, 0.05, 0.05, 0.05, 0.04])

    records: list[dict] = []
    invoice_counter = 500_000

    for w, week_start in enumerate(week_index):
        # Per-SKU appeal this week: trend, annual seasonality, promotions.
        phase = 2 * np.pi * (week_start.dayofyear / 365.25)
        season = 1.0 + seasonal_amp * np.sin(phase + seasonal_phase)
        promo = np.where(rng.random(n_skus) < 0.03, rng.uniform(1.8, 3.5, n_skus), 1.0)
        appeal = np.clip(popularity * (1 + trend * w) * season * promo, 1e-6, None)
        appeal = appeal / appeal.sum()

        active = np.flatnonzero((cust_start <= w) & (w < cust_end))
        if active.size == 0:
            continue

        shopping = active[rng.random(active.size) < cust_trip_rate[active]]
        for cust in shopping:
            n_items = max(1, int(rng.poisson(cust_basket_size[cust])))
            n_items = min(n_items, n_skus)
            chosen = list(rng.choice(n_skus, size=n_items, replace=False, p=appeal))

            # Pull in complements, which is what creates learnable rules.
            for item in list(chosen):
                partner = complements.get(item)
                if partner is not None and partner not in chosen and rng.random() < 0.55:
                    chosen.append(partner)

            invoice_counter += 1
            invoice = str(invoice_counter)
            offset = int(rng.integers(0, 7))
            for item in chosen:
                qty = int(rng.geometric(p=0.35))
                records.append(
                    {
                        "Invoice": invoice,
                        "StockCode": skus[item],
                        "Description": f"PRODUCT {skus[item]}",
                        "Quantity": qty,
                        "InvoiceDate": week_start + pd.Timedelta(days=offset),
                        "Price": float(unit_price[item]),
                        "Customer ID": float(cust_ids[cust]),
                        "Country": cust_country[cust],
                    }
                )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("Generated no transactions; increase n_customers or weeks.")

    return _inject_noise(df, rng).sort_values("InvoiceDate").reset_index(drop=True)


def _inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the messiness the cleaning step exists to handle."""
    n = len(df)

    # ~1.5% of lines are later cancelled: emit a matching credit note.
    cancel_idx = rng.choice(n, size=max(1, int(n * 0.015)), replace=False)
    cancels = df.iloc[cancel_idx].copy()
    cancels["Invoice"] = "C" + cancels["Invoice"].astype(str)
    cancels["Quantity"] = -cancels["Quantity"]
    cancels["InvoiceDate"] = cancels["InvoiceDate"] + pd.Timedelta(days=3)

    # ~2% of rows have no customer id (till sales).
    anon = df.sample(frac=0.02, random_state=1).copy()
    anon["Customer ID"] = np.nan

    # Postage and manual-adjustment lines.
    service = df.sample(n=min(200, n), random_state=2).copy()
    service["StockCode"] = rng.choice(["POST", "M", "DOT", "BANK CHARGES"], len(service))
    service["Description"] = "SERVICE LINE"
    service["Quantity"] = 1

    dupes = df.sample(frac=0.005, random_state=3).copy()

    return pd.concat([df, cancels, anon, service, dupes], ignore_index=True)


def write_fixture(path: str, **kwargs) -> pd.DataFrame:
    """Generate and persist a CSV fixture."""
    df = make_transactions(**kwargs)
    df.to_csv(path, index=False)
    return df
