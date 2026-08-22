"""Drift detection and data contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retail_intel.data.contracts import RFMSchema, WeeklyFeatureSchema, validate
from retail_intel.monitoring.drift import (
    categorical_drift,
    compare,
    numeric_drift,
    population_stability_index,
    severity_from_psi,
    split_reference_current,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_psi_is_near_zero_for_identical_distributions(rng):
    a, b = rng.normal(50, 10, 5000), rng.normal(50, 10, 5000)
    assert population_stability_index(a, b) < 0.1


def test_psi_detects_a_mean_shift(rng):
    a, b = rng.normal(50, 10, 5000), rng.normal(65, 10, 5000)
    assert population_stability_index(a, b) > 0.25


def test_psi_detects_a_variance_change(rng):
    a, b = rng.normal(50, 5, 5000), rng.normal(50, 25, 5000)
    assert population_stability_index(a, b) > 0.25


def test_psi_grows_with_the_size_of_the_shift(rng):
    base = rng.normal(50, 10, 5000)
    small = population_stability_index(base, rng.normal(53, 10, 5000))
    large = population_stability_index(base, rng.normal(70, 10, 5000))
    assert large > small


def test_psi_is_finite_when_a_bin_empties(rng):
    """An empty bin makes the log term infinite without a floor."""
    assert np.isfinite(population_stability_index(rng.normal(50, 10, 1000), np.full(500, 50.0)))


def test_psi_handles_too_little_data():
    assert population_stability_index(np.array([1.0, 2]), np.array([1.0])) == 0.0


def test_severity_thresholds_follow_convention():
    assert severity_from_psi(0.05) == "stable"
    assert severity_from_psi(0.15) == "moderate"
    assert severity_from_psi(0.40) == "significant"


def test_numeric_drift_reports_both_tests(rng):
    results = numeric_drift(
        pd.Series(rng.normal(50, 10, 2000)), pd.Series(rng.normal(70, 10, 2000)), "units"
    )
    assert {r.test for r in results} == {"psi", "ks"}
    assert all(r.severity == "significant" for r in results)
    assert results[0].pct_change > 0


def test_categorical_drift_detects_a_mix_change(rng):
    reference = pd.Series(rng.choice(list("ABCD"), 2000))
    current = pd.Series(rng.choice(list("ABCD"), 2000, p=[0.85, 0.05, 0.05, 0.05]))
    results = categorical_drift(reference, current, "country")
    assert any(r.severity != "stable" for r in results)


def test_compare_covers_every_requested_column(rng):
    reference = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(5, 2, 500)})
    current = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(9, 2, 500)})
    results = compare(reference, current, numeric_cols=["a", "b"])
    assert set(results["feature"]) == {"a", "b"}
    assert (results[results["feature"] == "b"]["severity"] != "stable").any()


def test_split_puts_the_recent_window_last():
    frame = pd.DataFrame(
        {"week": pd.date_range("2010-01-04", periods=52, freq="W-MON"), "x": range(52)}
    )
    reference, current = split_reference_current(frame, "week", current_weeks=8)
    assert len(current) == 8
    assert current["week"].min() > reference["week"].max()


# --- contracts ------------------------------------------------------------
def test_valid_features_pass_the_contract(weekly_panel):
    assert len(validate(weekly_panel, WeeklyFeatureSchema)) == len(weekly_panel)


def test_negative_demand_fails_the_contract(weekly_panel):
    """A pipeline that silently trains on negative demand is worse than one
    that stops."""
    import pandera as pa

    broken = weekly_panel.copy()
    broken.loc[0, "weekly_sales"] = -5.0
    with pytest.raises(pa.errors.SchemaErrors):
        validate(broken, WeeklyFeatureSchema)


def test_an_out_of_range_month_fails_the_contract(weekly_panel):
    import pandera as pa

    broken = weekly_panel.copy()
    broken.loc[0, "month"] = 13
    with pytest.raises(pa.errors.SchemaErrors):
        validate(broken, WeeklyFeatureSchema)


def test_duplicate_customers_fail_the_rfm_contract():
    import pandera as pa

    duplicated = pd.DataFrame(
        {
            "customer_id": [1, 1],
            "recency": [5, 6],
            "frequency": [2, 3],
            "monetary": [10.0, 20.0],
            "tenure": [50, 60],
            "avg_order_value": [5.0, 6.67],
        }
    )
    with pytest.raises(pa.errors.SchemaErrors):
        validate(duplicated, RFMSchema)


def test_unexpected_columns_fail_the_contract(weekly_panel):
    """Strict schemas catch a renamed or added column before it reaches a model."""
    import pandera as pa

    with pytest.raises(pa.errors.SchemaErrors):
        validate(weekly_panel.assign(surprise=1), WeeklyFeatureSchema)


def test_drift_report_is_json_serialisable(seeded_db):
    """PSI has no p-value, so the results frame carries NaN. `json.dumps`
    emits a bare `NaN`, which is not valid JSON and made /drift return 500."""
    import json

    from retail_intel.monitoring.drift import run

    report = run(current_weeks=6)
    encoded = json.dumps(report, allow_nan=False)
    assert "NaN" not in encoded
    assert json.loads(encoded)["verdict"] in {"stable", "moderate", "significant"}


def test_drift_endpoint_returns_a_report(seeded_db):
    from fastapi.testclient import TestClient

    from app.main import app
    from retail_intel.monitoring.drift import run

    run(current_weeks=6)
    with TestClient(app) as client:
        response = client.get("/drift")
    assert response.status_code == 200
    assert response.json()["verdict"] in {"stable", "moderate", "significant"}
