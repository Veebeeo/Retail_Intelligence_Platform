"""Customer analytics pipeline: RFM segments, CLV, and uplift targeting.

Runs the three customer models in one pass and writes a single wide
``customer_segments`` table, so a lookup by customer id returns the behavioural
segment, the forward-looking value, the churn probability and the campaign
recommendation together.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from retail_intel.config import get_settings
from retail_intel.db import write_table
from retail_intel.logging_conf import get_logger
from retail_intel.segmentation import clv as CLV
from retail_intel.segmentation import rfm as RFM
from retail_intel.segmentation import uplift as UPLIFT
from retail_intel.segmentation.strategies import STRATEGIES, strategy_for

logger = get_logger(__name__)

__all__ = ["STRATEGIES", "main", "run", "strategy_for"]


def run(
    clv_horizon_days: int = 90,
    k: int | None = None,
    run_uplift: bool = True,
) -> dict:
    settings = get_settings()

    transactions = RFM.load_transactions()
    rfm = RFM.build_rfm(transactions)

    # --- behavioural segmentation --------------------------------------
    X, _ = RFM.preprocess(rfm)
    if k is None:
        k, k_scores = RFM.choose_k(X, random_state=settings.random_seed)
    else:
        k_scores = pd.DataFrame()
    rfm["cluster"], _ = RFM.fit_kmeans(X, k, random_state=settings.random_seed)
    rfm = RFM.assign_labels(rfm)
    profile = RFM.profile_segments(rfm)

    # --- forward-looking value -----------------------------------------
    logger.info("Fitting CLV models...")
    clv_table, clv_params = CLV.fit_predict(transactions, days=clv_horizon_days)

    merged = rfm.merge(clv_table, on="customer_id", how="left")
    merged = merged.rename(
        columns={
            f"predicted_purchases_{clv_horizon_days}d": "predicted_purchases_90d",
            f"predicted_clv_{clv_horizon_days}d": "predicted_clv_90d",
        }
    )
    merged["recommended_action"] = merged["segment_label"].map(strategy_for)
    merged["updated_at"] = pd.Timestamp.utcnow().tz_localize(None)

    # --- campaign targeting --------------------------------------------
    uplift_report: dict = {}
    if run_uplift:
        try:
            logger.info("Validating uplift model on a simulated campaign...")
            result = UPLIFT.run_validation(rfm, seed=settings.random_seed)
            merged["uplift_segment"] = result["segments"].reindex(merged.index).to_numpy()
            uplift_report = {
                key: result[key]
                for key in (
                    "metrics",
                    "rank_correlation_with_truth",
                    "optimal_targeting_fraction",
                    "net_profit_at_optimum",
                    "net_profit_targeting_all",
                    "segment_counts",
                    "note",
                )
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Uplift validation skipped: %s", exc)

    write_table(merged, "customer_segments")

    report = {
        "n_customers": len(merged),
        "k_selected": int(k),
        "k_search": k_scores.to_dict(orient="records") if len(k_scores) else [],
        "segment_profile": profile.to_dict(orient="records"),
        "clv": clv_params,
        "clv_summary": {
            "total_predicted_clv": round(float(merged["predicted_clv_90d"].sum()), 2),
            "mean_predicted_clv": round(float(merged["predicted_clv_90d"].mean()), 2),
            "mean_churn_probability": round(float(merged["churn_probability"].mean()), 4),
            "high_risk_high_value_customers": int(
                (
                    (merged["churn_probability"] > 0.5)
                    & (merged["predicted_clv_90d"] > merged["predicted_clv_90d"].quantile(0.75))
                ).sum()
            ),
        },
        "uplift": uplift_report,
    }

    settings.report_dir.mkdir(parents=True, exist_ok=True)
    (settings.report_dir / "segmentation_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    profile.to_csv(settings.report_dir / "segment_profile.csv", index=False)

    logger.info("Segment profile:\n%s", profile.to_string(index=False))
    logger.info("CLV: %s", report["clv_summary"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build customer segments, CLV and uplift targeting."
    )
    parser.add_argument("--clv-days", type=int, default=90)
    parser.add_argument("--k", type=int, default=None, help="Force K instead of searching.")
    parser.add_argument("--no-uplift", action="store_true")
    args = parser.parse_args()

    report = run(clv_horizon_days=args.clv_days, k=args.k, run_uplift=not args.no_uplift)
    logger.info(
        "Segmentation complete for %d customers (K=%d)", report["n_customers"], report["k_selected"]
    )


if __name__ == "__main__":
    main()
