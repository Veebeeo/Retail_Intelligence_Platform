"""RFM feature construction and K-means segmentation.

Corrections to the original ``cluster_customers.py``:

* ``K=4`` was hard-coded with a comment claiming the elbow and silhouette
  methods had chosen it; neither was computed. ``choose_k`` now actually runs
  the search and records the scores.
* Labels were assigned by sorting clusters on **monetary value alone**, so a
  cluster of recent, frequent, modest spenders could be labelled
  "Hibernating / At-Risk". Labelling now scores clusters on all three
  dimensions.
* "Leal Customers" — a typo for "Loyal" — was written into the database and
  returned by the live API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from retail_intel.data.contracts import RFMSchema, validate
from retail_intel.db import read_sql
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)

RFM_COLUMNS = ["recency", "frequency", "monetary"]


def load_transactions() -> pd.DataFrame:
    df = read_sql(
        "SELECT customer_id, invoice_no, invoice_date, quantity, unit_price, total_price "
        "FROM transactions"
    )
    if df.empty:
        raise RuntimeError("`transactions` is empty. Run the ingest pipeline first.")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate transactions into per-customer RFM features.

    The snapshot date is one day after the last invoice in the data, so a
    customer who bought on the final day has recency 1 rather than 0 — which
    keeps the log transform below well defined.
    """
    snapshot_date = snapshot_date or df["invoice_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("invoice_date", lambda s: (snapshot_date - s.max()).days),
        frequency=("invoice_no", "nunique"),
        monetary=("total_price", "sum"),
        first_purchase=("invoice_date", "min"),
    )
    rfm["tenure"] = (snapshot_date - rfm["first_purchase"]).dt.days
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    rfm = rfm.drop(columns=["first_purchase"]).reset_index()

    rfm = rfm[rfm["monetary"] > 0]
    logger.info("Built RFM for %d customers (snapshot %s)", len(rfm), snapshot_date.date())
    return validate(rfm, RFMSchema)


def preprocess(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform then standardise.

    All three RFM measures are heavily right-skewed; without the log, K-means
    (which minimises squared Euclidean distance) puts every big spender in a
    singleton cluster.
    """
    logged = np.log1p(rfm[RFM_COLUMNS].to_numpy(dtype=float))
    scaler = StandardScaler()
    return scaler.fit_transform(logged), scaler


def choose_k(
    X: np.ndarray, k_range: range = range(2, 9), random_state: int = 42
) -> tuple[int, pd.DataFrame]:
    """Search K by silhouette, recording inertia and Calinski-Harabasz too.

    Silhouette decides because it measures separation directly. Inertia is
    reported so the elbow is visible, but it falls monotonically with K and so
    cannot pick a value on its own — which is the flaw in citing "the elbow
    method" without a second criterion.
    """
    rows = []
    # Silhouette on every customer is O(n^2); a sample is enough to rank K.
    sample = X if len(X) <= 5000 else X[np.random.default_rng(random_state).choice(len(X), 5000, replace=False)]

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sample_labels = labels if len(X) <= 5000 else km.predict(sample)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(sample, sample_labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            }
        )

    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    logger.info("Selected K=%d by silhouette:\n%s", best_k, scores.round(4).to_string(index=False))
    return best_k, scores


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    return km.fit_predict(X), km


def assign_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    """Name each cluster from its full RFM profile, not monetary value alone.

    Each cluster is scored on percentile-ranked recency (inverted, so higher is
    better), frequency and monetary value. The combination decides the name, so
    a cluster that spends heavily but has not bought in a year is correctly
    called at-risk rather than "Champions".
    """
    profile = rfm.groupby("cluster")[RFM_COLUMNS].mean()

    # Percentile rank each dimension across clusters; invert recency because
    # a *low* recency (bought recently) is the good outcome.
    ranked = profile.rank(pct=True)
    ranked["recency"] = 1 - ranked["recency"]
    ranked["value_score"] = ranked[["frequency", "monetary"]].mean(axis=1)

    labels: dict[int, str] = {}
    for cluster, row in ranked.iterrows():
        active = row["recency"] >= 0.5
        valuable = row["value_score"] >= 0.5
        if active and valuable:
            labels[cluster] = "Champions"
        elif active and not valuable:
            labels[cluster] = "New / Promising"
        elif not active and valuable:
            labels[cluster] = "At-Risk High Value"
        else:
            labels[cluster] = "Hibernating"

    # With more clusters than archetypes, disambiguate duplicates by value.
    seen: dict[str, int] = {}
    for cluster in sorted(labels, key=lambda c: -ranked.loc[c, "value_score"]):
        name = labels[cluster]
        if name in seen:
            seen[name] += 1
            labels[cluster] = f"{name} (tier {seen[name]})"
        else:
            seen[name] = 1

    out = rfm.copy()
    out["segment_label"] = out["cluster"].map(labels)
    logger.info("Segment sizes:\n%s", out["segment_label"].value_counts().to_string())
    return out


def profile_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Human-readable summary of each segment."""
    return (
        rfm.groupby("segment_label")
        .agg(
            customers=("customer_id", "count"),
            avg_recency_days=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
            avg_order_value=("avg_order_value", "mean"),
        )
        .assign(revenue_share_pct=lambda d: d["total_revenue"] / d["total_revenue"].sum() * 100)
        .sort_values("total_revenue", ascending=False)
        .round(2)
        .reset_index()
    )
