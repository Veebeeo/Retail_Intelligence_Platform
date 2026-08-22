"""API contract, error handling and injection resistance."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(seeded_db):
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def trained_client(seeded_db, weekly_panel):
    """A client with champion models actually loaded."""
    from retail_intel.forecasting import serving
    from retail_intel.forecasting.train import run

    run(panel=weekly_panel, horizon=4, n_folds=1, log_to_mlflow=False)
    serving.reset_cache()

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_root_lists_the_endpoints(client):
    body = client.get("/").json()
    assert "endpoints" in body
    assert "/forecast" in body["endpoints"]["forecasting"]


def test_health_reports_real_state_not_a_constant(client):
    """The old health check returned a hard-coded 'healthy' regardless."""
    body = client.get("/health").json()
    assert body["status"] in {"healthy", "degraded"}
    assert body["database"] == "connected"
    assert isinstance(body["models_loaded"], bool)


# --- injection ------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        "85123A'; DROP TABLE transactions; --",
        "' OR '1'='1",
        "85123A UNION SELECT * FROM customer_segments",
        "'; DELETE FROM ml_weekly_features WHERE '1'='1",
    ],
)
def test_forecast_rejects_sql_injection_attempts(client, payload):
    """The original interpolated stock_code straight into the query string."""
    response = client.post("/forecast", json={"stock_code": payload, "horizon_weeks": 4})
    assert response.status_code == 422  # rejected by validation, never reaches SQL


def test_the_transactions_table_survives_injection_attempts(client):
    client.post("/forecast", json={"stock_code": "X'; DROP TABLE transactions; --"})
    from retail_intel.db import read_sql

    assert read_sql("SELECT COUNT(*) AS n FROM transactions")["n"].iloc[0] > 0


def test_segment_rejects_a_non_numeric_customer_id(client):
    response = client.post("/segment", json={"customer_id": "1 OR 1=1"})
    assert response.status_code == 422


def test_segment_query_is_parameterised(client, seeded_db):
    """A crafted numeric-looking id must not widen the result set."""
    response = client.post("/segment", json={"customer_id": 999999999})
    assert response.status_code in {404, 503}
    assert response.status_code != 200


# --- validation -----------------------------------------------------------
@pytest.mark.parametrize("horizon", [0, -1, 27, 1000])
def test_forecast_horizon_is_bounded(client, horizon):
    response = client.post("/forecast", json={"stock_code": "85123A", "horizon_weeks": horizon})
    assert response.status_code == 422


def test_stock_code_is_normalised_to_upper_case(trained_client):
    skus = trained_client.get("/models/skus").json()["stock_codes"]
    response = trained_client.post(
        "/forecast", json={"stock_code": skus[0].lower(), "horizon_weeks": 2}
    )
    assert response.status_code == 200
    assert response.json()["stock_code"] == skus[0]


def test_empty_stock_code_is_rejected(client):
    assert client.post("/forecast", json={"stock_code": ""}).status_code == 422


# --- forecasting ----------------------------------------------------------
def test_forecast_returns_a_real_model_not_a_growth_heuristic(trained_client):
    """The old endpoint returned recent_avg * (1 + 0.02 * week) for every SKU."""
    skus = trained_client.get("/models/skus").json()["stock_codes"]
    body = trained_client.post("/forecast", json={"stock_code": skus[0], "horizon_weeks": 4}).json()

    assert body["model"] not in {"", None}
    assert body["model_version"] >= 1
    assert body["backtest_mase"] is not None

    values = [p["predicted_quantity"] for p in body["predictions"]]
    ratios = [values[i + 1] / values[i] for i in range(len(values) - 1) if values[i] > 0]
    assert not all(abs(r - 1.02) < 1e-6 for r in ratios), "still a fixed 2% growth curve"


def test_forecast_includes_prediction_intervals(trained_client):
    skus = trained_client.get("/models/skus").json()["stock_codes"]
    body = trained_client.post("/forecast", json={"stock_code": skus[0], "horizon_weeks": 4}).json()
    for point in body["predictions"]:
        assert point["lower_95"] <= point["predicted_quantity"] <= point["upper_95"]
        assert point["predicted_quantity"] >= 0
        assert point["week_starting"]


def test_unknown_sku_returns_404(trained_client):
    response = trained_client.post("/forecast", json={"stock_code": "ZZZZZZ99", "horizon_weeks": 4})
    assert response.status_code == 404
    assert "No trained model" in response.json()["detail"]


def test_models_endpoint_reports_registry_state(trained_client):
    body = trained_client.get("/models").json()
    assert body["n_skus"] > 0
    assert isinstance(body["model_mix"], dict)
    assert 0 <= body["pct_skus_beating_baseline"] <= 100


def test_model_detail_carries_backtest_provenance(trained_client):
    skus = trained_client.get("/models/skus").json()["stock_codes"]
    body = trained_client.get(f"/models/{skus[0]}").json()
    assert "backtest_mase" in body
    assert "baseline_mase" in body


# --- inventory ------------------------------------------------------------
def test_inventory_policy_returns_an_actionable_reorder_point(trained_client):
    skus = trained_client.get("/models/skus").json()["stock_codes"]
    body = trained_client.post(
        "/inventory/policy", json={"stock_code": skus[0], "lead_time_weeks": 2}
    ).json()

    assert body["reorder_point"] >= body["expected_lead_time_demand"]
    assert body["safety_stock"] >= 0
    assert "Reorder when stock reaches" in body["explanation"]


def test_supplying_costs_derives_the_service_level(trained_client):
    """With both costs known the optimal service level is determined, not chosen."""
    skus = trained_client.get("/models/skus").json()["stock_codes"]
    body = trained_client.post(
        "/inventory/policy",
        json={
            "stock_code": skus[0],
            "lead_time_weeks": 2,
            "unit_holding_cost": 0.1,
            "unit_stockout_cost": 9.9,
        },
    ).json()
    assert "critical ratio" in body["service_level_source"]
    assert body["service_level"] == pytest.approx(0.99, abs=0.01)


# --- customers ------------------------------------------------------------
def test_segment_lookup_returns_metrics_and_an_action(client, seeded_db):
    from retail_intel.segmentation.pipeline import run

    run(clv_horizon_days=90, k=3, run_uplift=False)

    from retail_intel.db import read_sql

    customer = int(
        read_sql("SELECT customer_id FROM customer_segments LIMIT 1")["customer_id"].iloc[0]
    )
    body = client.post("/segment", json={"customer_id": customer}).json()

    assert body["customer_id"] == customer
    assert body["recommended_action"]
    assert "Leal" not in body["segment"]
    assert body["metrics"]["total_lifetime_orders"] >= 1


def test_segment_summary_reports_revenue_share(client, seeded_db):
    segments = client.get("/segments/summary").json()["segments"]
    assert segments
    assert sum(s["revenue_share_pct"] for s in segments) == pytest.approx(100, abs=1)


def test_missing_customer_returns_404(client):
    assert client.post("/segment", json={"customer_id": 987654321}).status_code == 404


def test_at_risk_list_is_ranked_by_predicted_value(client, seeded_db):
    body = client.get("/customers/at-risk", params={"limit": 10}).json()
    values = [c["predicted_clv_90d"] for c in body["customers"]]
    assert values == sorted(values, reverse=True)


# --- degraded operation ---------------------------------------------------
def test_forecast_returns_503_when_no_models_are_loaded(seeded_db, monkeypatch):
    """It says so, rather than inventing a number."""
    from retail_intel.forecasting import serving

    serving.reset_cache()
    # Patch where it is *used*: app.dependencies imported the name directly.
    monkeypatch.setattr("app.dependencies.load_bundle", _raise_not_available)

    from app.main import app

    with TestClient(app) as no_model_client:
        response = no_model_client.post("/forecast", json={"stock_code": "85123A"})
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"]
    serving.reset_cache()


def _raise_not_available(*args, **kwargs):
    from retail_intel.forecasting.serving import ModelNotAvailable

    raise ModelNotAvailable("No trained models at models/champions.pkl.")
