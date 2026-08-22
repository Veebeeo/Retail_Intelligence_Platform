"""Data and prediction drift detection.

The old README claimed the platform "tracks data-drift". Nothing in the
repository did. This is that component.

Three complementary tests, because no single one catches everything:

* **PSI (Population Stability Index)** — the industry default for tabular
  monitoring. Bins the reference distribution and measures how much probability
  mass moved. Conventional reading: < 0.1 stable, 0.1-0.25 moderate, > 0.25
  significant.
* **Kolmogorov-Smirnov** — the largest gap between the two empirical CDFs.
  Sensitive to shape changes PSI's coarse bins can miss, and comes with a
  p-value.
* **Chi-square** — for categorical columns, where distance metrics do not apply.

A note on p-values in monitoring: with enough rows, *every* comparison becomes
significant, because no two weeks of real data are ever drawn from an identical
distribution. So the effect sizes (PSI, KS statistic) decide the verdict here
and the p-values are reported as supporting detail, not as the trigger.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy import stats

from retail_intel.config import get_settings
from retail_intel.db import read_sql
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


@dataclass
class DriftResult:
    feature: str
    test: str
    statistic: float
    p_value: float | None
    severity: str
    reference_mean: float | None = None
    current_mean: float | None = None
    pct_change: float | None = None

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between two numeric samples.

    Bin edges come from the reference quantiles, so bins are equally populated
    at training time and any imbalance in the current window is real movement
    rather than an artefact of arbitrary cut-points.
    """
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = np.asarray(current, dtype=float)
    current = current[np.isfinite(current)]
    if len(reference) < n_bins or len(current) == 0:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)

    # An empty bin makes the log term infinite; the conventional fix is a small
    # floor, which bounds the contribution of a bin that emptied out.
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def severity_from_psi(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def numeric_drift(
    reference: pd.Series, current: pd.Series, feature: str, n_bins: int = 10
) -> list[DriftResult]:
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return []

    psi = population_stability_index(ref, cur, n_bins)
    ks_stat, ks_p = stats.ks_2samp(ref, cur)
    ref_mean, cur_mean = float(ref.mean()), float(cur.mean())

    return [
        DriftResult(
            feature,
            "psi",
            psi,
            None,
            severity_from_psi(psi),
            ref_mean,
            cur_mean,
            (cur_mean - ref_mean) / ref_mean * 100 if abs(ref_mean) > 1e-9 else None,
        ),
        DriftResult(
            feature,
            "ks",
            float(ks_stat),
            float(ks_p),
            # KS statistic is a max CDF gap in [0,1]; 0.1/0.2 are the usual
            # informal thresholds for "worth looking at" and "clearly moved".
            "significant" if ks_stat > 0.2 else "moderate" if ks_stat > 0.1 else "stable",
            ref_mean,
            cur_mean,
            None,
        ),
    ]


def categorical_drift(
    reference: pd.Series, current: pd.Series, feature: str, top_n: int = 20
) -> list[DriftResult]:
    categories = reference.value_counts().nlargest(top_n).index
    ref_counts = (
        reference[reference.isin(categories)].value_counts().reindex(categories, fill_value=0)
    )
    cur_counts = current[current.isin(categories)].value_counts().reindex(categories, fill_value=0)

    if cur_counts.sum() == 0 or ref_counts.sum() == 0:
        return []

    # Scale the reference to the current sample size so chi-square compares
    # shape rather than volume.
    expected = ref_counts / ref_counts.sum() * cur_counts.sum()
    expected = expected.clip(lower=1e-6)
    chi2 = float(((cur_counts - expected) ** 2 / expected).sum())
    dof = max(len(categories) - 1, 1)
    p = float(1 - stats.chi2.cdf(chi2, dof))

    ref_pct = (ref_counts / ref_counts.sum()).to_numpy()
    cur_pct = (cur_counts / cur_counts.sum()).to_numpy()
    psi = float(
        np.sum(
            (cur_pct - ref_pct)
            * np.log(np.clip(cur_pct, 1e-6, None) / np.clip(ref_pct, 1e-6, None))
        )
    )

    return [
        DriftResult(feature, "chi2", chi2, p, severity_from_psi(psi)),
        DriftResult(feature, "psi_categorical", psi, None, severity_from_psi(psi)),
    ]


def compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run every applicable test over two frames."""
    numeric_cols = numeric_cols or [
        c for c in reference.select_dtypes(include=[np.number]).columns if c in current.columns
    ]
    categorical_cols = categorical_cols or []

    results: list[DriftResult] = []
    for col in numeric_cols:
        results.extend(numeric_drift(reference[col], current[col], col))
    for col in categorical_cols:
        if col in reference.columns and col in current.columns:
            results.extend(categorical_drift(reference[col], current[col], col))

    return pd.DataFrame([r.to_dict() for r in results])


def split_reference_current(
    df: pd.DataFrame, date_col: str = "week", current_weeks: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed frame into a training reference and a recent window."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = df[date_col].max() - pd.Timedelta(weeks=current_weeks)
    return df[df[date_col] <= cutoff], df[df[date_col] > cutoff]


def run(current_weeks: int = 8) -> dict:
    """Compare the most recent weeks of features against the earlier history."""
    settings = get_settings()

    features = read_sql(
        "SELECT stock_code, week, weekly_sales, weekly_revenue, lag_1_week, "
        "rolling_4_wk_avg, month FROM ml_weekly_features ORDER BY week"
    )
    if features.empty:
        raise RuntimeError("`ml_weekly_features` is empty. Run the feature pipeline first.")

    reference, current = split_reference_current(features, "week", current_weeks)
    logger.info(
        "Drift check: %d reference rows vs %d rows from the last %d weeks",
        len(reference),
        len(current),
        current_weeks,
    )

    results = compare(
        reference,
        current,
        numeric_cols=["weekly_sales", "weekly_revenue", "lag_1_week", "rolling_4_wk_avg"],
        categorical_cols=["stock_code"],
    )

    flagged = results[results["severity"] != "stable"]
    verdict = (
        "significant"
        if (results["severity"] == "significant").any()
        else "moderate"
        if (results["severity"] == "moderate").any()
        else "stable"
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "current_window_weeks": current_weeks,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "verdict": verdict,
        "n_flagged": len(flagged),
        "results": results.to_dict(orient="records"),
        "recommendation": {
            "stable": "No action. Continue on the current champion models.",
            "moderate": "Watch. Re-run the backtest and confirm champion selection still holds.",
            "significant": "Retrain. The serving distribution has moved away from the training data.",
        }[verdict],
    }

    settings.report_dir.mkdir(parents=True, exist_ok=True)
    (settings.report_dir / "drift_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )

    logger.info("Drift verdict: %s (%d flagged tests)", verdict, len(flagged))
    if len(flagged):
        logger.info("Flagged:\n%s", flagged.to_string(index=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for data drift in the serving window.")
    parser.add_argument(
        "--weeks", type=int, default=8, help="Size of the recent comparison window."
    )
    args = parser.parse_args()
    report = run(args.weeks)
    logger.info("%s", report["recommendation"])


if __name__ == "__main__":
    main()
