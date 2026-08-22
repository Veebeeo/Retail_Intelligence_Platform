"""RFM, CLV and uplift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retail_intel.segmentation import clv as CLV
from retail_intel.segmentation import rfm as RFM
from retail_intel.segmentation import uplift as UP
from retail_intel.segmentation.pipeline import STRATEGIES, strategy_for


@pytest.fixture(scope="module")
def rfm_table(clean_transactions) -> pd.DataFrame:
    return RFM.build_rfm(clean_transactions)


@pytest.fixture(scope="module")
def large_rfm_table() -> pd.DataFrame:
    """A bigger population, for tests that need statistical power."""
    from retail_intel.data.ingest import clean
    from retail_intel.data.synthetic import make_transactions

    return RFM.build_rfm(clean(make_transactions(n_customers=900, n_skus=15, weeks=90, seed=11)))


def test_rfm_metrics_are_well_formed(rfm_table):
    assert (rfm_table["recency"] >= 0).all()
    assert (rfm_table["frequency"] >= 1).all()
    assert (rfm_table["monetary"] > 0).all()
    assert rfm_table["customer_id"].is_unique


def test_recency_is_measured_from_the_last_purchase(clean_transactions):
    snapshot = clean_transactions["invoice_date"].max() + pd.Timedelta(days=1)
    table = RFM.build_rfm(clean_transactions, snapshot)
    customer = table.iloc[0]["customer_id"]
    last = clean_transactions.loc[
        clean_transactions["customer_id"] == customer, "invoice_date"
    ].max()
    assert table.iloc[0]["recency"] == (snapshot - last).days


def test_average_order_value_is_consistent(rfm_table):
    expected = rfm_table["monetary"] / rfm_table["frequency"]
    np.testing.assert_allclose(rfm_table["avg_order_value"], expected, rtol=1e-9)


def test_choose_k_searches_rather_than_hard_coding(rfm_table):
    """The original hard-coded K=4 while claiming to have run the elbow and
    silhouette methods."""
    X, _ = RFM.preprocess(rfm_table)
    k, scores = RFM.choose_k(X, k_range=range(2, 6))
    assert 2 <= k <= 5
    assert set(scores.columns) == {"k", "inertia", "silhouette", "calinski_harabasz"}
    assert k == int(scores.loc[scores["silhouette"].idxmax(), "k"])
    # Inertia always falls with K, which is why it cannot pick K alone.
    assert scores["inertia"].is_monotonic_decreasing


def test_labels_use_all_three_rfm_dimensions():
    """Sorting on monetary value alone let a recent, frequent, modest-spending
    cluster be labelled 'Hibernating'."""
    rfm = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "recency": [5, 400, 10, 500],
            "frequency": [50, 40, 30, 1],
            "monetary": [5000.0, 4000.0, 900.0, 20.0],
            "cluster": [0, 1, 2, 3],
        }
    )
    labelled = RFM.assign_labels(rfm)
    labels = dict(zip(labelled["cluster"], labelled["segment_label"], strict=True))
    # Cluster 1 spends heavily but has not bought in over a year.
    assert "At-Risk" in labels[1]
    assert labels[0] == "Champions"
    assert labels[3] == "Hibernating"


def test_no_leal_customers_typo():
    """This typo was written into the database and returned by the live API."""
    assert "Leal Customers" not in STRATEGIES
    assert all("Leal" not in name for name in STRATEGIES)


def test_every_segment_has_a_strategy():
    for label in STRATEGIES:
        assert strategy_for(label) == STRATEGIES[label]
    assert strategy_for("Champions (tier 2)") == STRATEGIES["Champions"]


# --- CLV ------------------------------------------------------------------
@pytest.fixture(scope="module")
def cbs(clean_transactions) -> pd.DataFrame:
    return CLV.build_cbs(clean_transactions)


def test_cbs_frequency_counts_repeat_purchases_not_total(cbs):
    """BG/NBD 'frequency' is repeat purchases, so a one-off buyer scores 0.
    Conflating it with RFM frequency is the classic way to get this wrong."""
    assert (cbs["frequency"] == cbs["n_orders"] - 1).all()
    assert (cbs["frequency"] >= 0).all()


def test_cbs_recency_never_exceeds_observation_time(cbs):
    assert (cbs["recency"] <= cbs["T"]).all()


def test_bgnbd_fits_positive_parameters(cbs):
    params = CLV.fit_bgnbd(cbs)
    assert all(v > 0 for v in (params.r, params.alpha, params.a, params.b))
    assert np.isfinite(params.log_likelihood)


def test_predicted_purchases_are_finite_and_non_negative(cbs):
    """Regression test: an `a` below 1 flipped the sign of the closed form and
    made every prediction clip to zero."""
    params = CLV.fit_bgnbd(cbs)
    predicted = CLV.predict_purchases(cbs, params, days=90)
    assert np.isfinite(predicted).all()
    assert (predicted >= 0).all()
    assert predicted.max() > 0


def test_probability_alive_is_a_probability(cbs):
    alive = CLV.probability_alive(cbs, CLV.fit_bgnbd(cbs))
    assert ((alive >= 0) & (alive <= 1)).all()


def test_a_recently_active_customer_is_likelier_alive_than_a_lapsed_one():
    params = CLV.BGNBDParams(r=0.6, alpha=20.0, a=1.5, b=13.0, log_likelihood=-1.0, converged=True)
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "frequency": [10.0, 10.0],
            "recency": [340.0, 30.0],
            "T": [350.0, 350.0],
        }
    )
    alive = CLV.probability_alive(frame, params)
    assert alive[0] > alive[1]


def test_gamma_gamma_shrinks_sparse_customers_towards_the_population_mean():
    params = CLV.GammaGammaParams(p=2.3, q=3.9, v=29.6, converged=True)
    frame = pd.DataFrame({"frequency": [0.0, 50.0], "monetary_value": [500.0, 500.0]})
    values = CLV.predict_average_value(frame, params)
    population_mean = params.p * params.v / (params.q - 1)
    assert values[0] == pytest.approx(population_mean)
    assert values[1] > values[0]  # 50 observations carry real individual signal


def test_gamma_gamma_needs_enough_repeat_customers():
    tiny = pd.DataFrame({"frequency": [1.0], "monetary_value": [10.0], "customer_id": [1]})
    with pytest.raises(ValueError, match="at least 10"):
        CLV.fit_gamma_gamma(tiny)


def test_clv_falls_with_a_longer_discounting_horizon(cbs):
    bg, gg = CLV.fit_bgnbd(cbs), CLV.fit_gamma_gamma(cbs)
    undiscounted = CLV.predict_clv(cbs, bg, gg, 90, annual_discount_rate=0.0)
    discounted = CLV.predict_clv(cbs, bg, gg, 90, annual_discount_rate=0.5)
    assert discounted["predicted_clv_90d"].sum() < undiscounted["predicted_clv_90d"].sum()


# --- uplift ---------------------------------------------------------------
def test_uplift_recovers_a_known_simulated_effect(large_rfm_table):
    """The estimator is validated against ground truth, because the source data
    has no control group to estimate a real effect from.

    Uses a larger population than the other tests on purpose: a T-learner takes
    the difference of two independently fitted models, so its variance is high
    and a few hundred customers per arm is the realistic floor.
    """
    rfm_table = large_rfm_table
    treatment, outcome, true_uplift = UP.simulate_campaign(rfm_table, seed=7)
    features = rfm_table[["recency", "frequency", "monetary", "tenure", "avg_order_value"]]

    model = UP.TLearner(random_state=7).fit(features, treatment, outcome)
    predicted = model.predict_uplift(features)

    correlation = pd.Series(predicted).corr(pd.Series(true_uplift), method="spearman")
    assert correlation > 0.2, f"uplift ranking uncorrelated with truth ({correlation:.3f})"


def test_uplift_requires_both_arms(rfm_table):
    """Without a randomised holdout there is no causal effect to estimate."""
    features = rfm_table[["recency", "frequency", "monetary"]]
    all_treated = np.ones(len(features), dtype=int)
    outcome = np.tile([0, 1], len(features))[: len(features)]
    with pytest.raises(ValueError, match="control arm"):
        UP.TLearner().fit(features, all_treated, outcome)


def test_qini_curve_starts_at_zero_and_covers_the_population():
    rng = np.random.default_rng(0)
    n = 400
    scores, treatment = rng.random(n), rng.integers(0, 2, n)
    outcome = rng.integers(0, 2, n)
    curve = UP.qini_curve(scores, treatment, outcome, n_bins=10)
    assert curve["qini"].iloc[0] == 0.0
    assert curve["fraction"].iloc[-1] == pytest.approx(1.0)
    assert curve["fraction"].is_monotonic_increasing


def test_classify_assigns_known_uplift_archetypes(rfm_table):
    treatment, outcome, _ = UP.simulate_campaign(rfm_table, seed=7)
    features = rfm_table[["recency", "frequency", "monetary", "tenure", "avg_order_value"]]
    segments = UP.TLearner(random_state=7).fit(features, treatment, outcome).classify(features)
    assert set(segments.unique()) <= set(UP.SEGMENT_NAMES)


def test_campaign_economics_prices_contact_cost(rfm_table):
    treatment, outcome, _ = UP.simulate_campaign(rfm_table, seed=7)
    features = rfm_table[["recency", "frequency", "monetary", "tenure", "avg_order_value"]]
    predicted = (
        UP.TLearner(random_state=7).fit(features, treatment, outcome).predict_uplift(features)
    )

    economics = UP.campaign_economics(predicted, treatment, outcome, cost_per_contact=0.5)
    np.testing.assert_allclose(
        economics["net_profit"], economics["gross_margin"] - economics["contact_cost"], atol=1e-6
    )
    assert economics["contact_cost"].is_monotonic_increasing
