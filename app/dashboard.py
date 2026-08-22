"""Retail Intelligence dashboard.

Rebuilt from a two-box API demo into a decision tool. The questions it answers
are the ones a planner or a marketer actually has: how much of this do I order
and when, which customers should I spend retention budget on, what should I
cross-sell, and can I trust the models today.

(The previous version opened with ``import streamlit as str``, which shadows
the ``str`` builtin for the whole module.)
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 30

st.set_page_config(
    page_title="Retail Intelligence Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_get(path: str, **params) -> tuple[dict | None, str | None]:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, f"Could not reach the API at {API_BASE_URL}: {exc}"
    return _unpack(response)


def api_post(path: str, payload: dict) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, f"Could not reach the API at {API_BASE_URL}: {exc}"
    return _unpack(response)


def _unpack(response: requests.Response) -> tuple[dict | None, str | None]:
    if response.status_code == 200:
        return response.json(), None
    try:
        return None, response.json().get("detail", response.text)
    except ValueError:
        return None, f"HTTP {response.status_code}"


@st.cache_data(ttl=300)
def cached_get(path: str, **params):
    return api_get(path, **params)


# ---------------------------------------------------------------------------
# Sidebar: service state
# ---------------------------------------------------------------------------
health, health_error = api_get("/health")

with st.sidebar:
    st.title("📦 Retail Intelligence")
    st.caption(f"API: `{API_BASE_URL}`")

    if health_error:
        st.error("API unreachable")
        st.caption(health_error)
    elif health["status"] == "healthy":
        st.success(f"Healthy · v{health['version']}")
        st.caption(f"Models v{health['model_version']} · {health['n_skus']} SKUs")
    else:
        st.warning("Degraded")
        st.caption(f"Database: {health['database']}")
        st.caption(f"Models loaded: {health['models_loaded']}")
        if health.get("detail"):
            st.caption(health["detail"])

    page = st.radio(
        "View",
        ["Demand & Reordering", "Customer Value", "Cross-Sell", "Model Performance"],
        label_visibility="collapsed",
    )

if health_error:
    st.error(
        f"The dashboard cannot reach the API at `{API_BASE_URL}`.\n\n"
        "Start it with `make api` (or `uvicorn app.main:app --reload`), or set "
        "`API_BASE_URL` to point at a running instance."
    )
    st.stop()


@st.cache_data(ttl=300)
def sku_options() -> list[str]:
    data, _ = api_get("/models/skus")
    return data["stock_codes"] if data else []


# ---------------------------------------------------------------------------
# Demand & reordering
# ---------------------------------------------------------------------------
if page == "Demand & Reordering":
    st.header("Demand forecast and reorder policy")
    st.caption(
        "Each SKU is served by the model that won its rolling-origin backtest. "
        "Where no model beat the seasonal-naive baseline, the baseline is served — "
        "the panel below always says which."
    )

    skus = sku_options()
    if not skus:
        st.warning("No trained models available. Run the training pipeline first.")
        st.stop()

    left, mid, right = st.columns([2, 1, 1])
    sku = left.selectbox("SKU", skus, index=0)
    horizon = mid.slider("Horizon (weeks)", 1, 12, 4)
    lead_time = right.slider("Lead time (weeks)", 1, 8, 2)

    with st.expander("Inventory cost assumptions"):
        c1, c2 = st.columns(2)
        holding = c1.number_input("Holding cost per unit per week (£)", 0.0, 100.0, 0.15, 0.05)
        stockout = c2.number_input("Stockout cost per unit (£)", 0.0, 1000.0, 2.50, 0.25)
        st.caption(
            "With both costs supplied the service level is derived from the newsvendor "
            "critical ratio Cu/(Cu+Co) rather than picked by hand."
        )

    forecast, err = api_post("/forecast", {"stock_code": sku, "horizon_weeks": horizon})
    if err:
        st.error(err)
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Champion model", forecast["model"])
    mase = forecast.get("backtest_mase")
    baseline_mase = forecast.get("baseline_mase")
    m2.metric(
        "Backtest MASE",
        f"{mase:.3f}" if mase else "n/a",
        delta=f"{forecast['improvement_vs_baseline_pct']:.1f}% vs baseline"
        if forecast.get("improvement_vs_baseline_pct")
        else None,
    )
    m3.metric("Baseline MASE", f"{baseline_mase:.3f}" if baseline_mase else "n/a")
    total = sum(p["predicted_quantity"] for p in forecast["predictions"])
    m4.metric(f"Demand next {horizon}w", f"{total:,.0f} units")

    if mase and mase >= 1:
        st.warning(
            f"MASE of {mase:.2f} means this SKU's forecast is no better than repeating "
            "last year's value. Treat the numbers below as indicative only."
        )

    fdf = pd.DataFrame(forecast["predictions"])
    fdf["week_starting"] = pd.to_datetime(fdf["week_starting"])

    band = (
        alt.Chart(fdf)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("week_starting:T", title="Week"),
            y=alt.Y("lower_95:Q", title="Units"),
            y2="upper_95:Q",
        )
    )
    line = (
        alt.Chart(fdf)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x="week_starting:T",
            y="predicted_quantity:Q",
            tooltip=["week_starting:T", "predicted_quantity:Q", "lower_95:Q", "upper_95:Q"],
        )
    )
    st.altair_chart((band + line).properties(height=320), use_container_width=True)
    st.caption(
        "Shaded band is the 95% prediction interval — the range safety stock is sized against."
    )

    st.subheader("Reorder policy")
    policy, perr = api_post(
        "/inventory/policy",
        {
            "stock_code": sku,
            "lead_time_weeks": lead_time,
            "unit_holding_cost": holding,
            "unit_stockout_cost": stockout,
        },
    )
    if perr:
        st.error(perr)
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Reorder point", f"{policy['reorder_point']:,.0f} units")
        p2.metric("Safety stock", f"{policy['safety_stock']:,.0f} units")
        p3.metric("Service level", f"{policy['service_level']:.1%}")
        p4.metric("Lead-time demand", f"{policy['expected_lead_time_demand']:,.0f} units")
        st.info(policy["explanation"])
        st.caption(f"Service level {policy['service_level_source']}.")

    with st.expander("Forecast detail"):
        st.dataframe(fdf, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Customer value
# ---------------------------------------------------------------------------
elif page == "Customer Value":
    st.header("Customer value and retention")
    st.caption(
        "Segments describe past behaviour; BG/NBD and Gamma-Gamma predict forward value "
        "and churn risk. Retention budget belongs where the two overlap."
    )

    summary, serr = cached_get("/segments/summary")
    if serr:
        st.warning(serr)
    else:
        seg = pd.DataFrame(summary["segments"])
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Segments")
            st.dataframe(
                seg[
                    [
                        "segment_label",
                        "customers",
                        "avg_recency_days",
                        "avg_frequency",
                        "avg_monetary",
                        "revenue_share_pct",
                        "avg_churn_probability",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.subheader("Revenue share")
            st.altair_chart(
                alt.Chart(seg)
                .mark_arc(innerRadius=55)
                .encode(
                    theta="total_revenue:Q",
                    color=alt.Color("segment_label:N", title="Segment"),
                    tooltip=["segment_label:N", "revenue_share_pct:Q"],
                )
                .properties(height=280),
                use_container_width=True,
            )

        st.subheader("Recommended actions")
        for row in seg.itertuples():
            st.markdown(
                f"**{row.segment_label}** ({int(row.customers)} customers) — {row.recommended_action}"
            )

    st.divider()
    st.subheader("Retention priority list")
    st.caption(
        "High predicted value *and* high churn probability. Ranking by past spend alone "
        "puts loyal customers at the top, who need nothing."
    )
    limit = st.slider("Customers to show", 5, 100, 20)
    at_risk, aerr = cached_get("/customers/at-risk", limit=limit)
    if aerr:
        st.warning(aerr)
    elif at_risk["n_customers"] == 0:
        st.info("No customers currently meet the at-risk threshold.")
    else:
        st.metric("Predicted 90-day value at risk", f"£{at_risk['total_revenue_at_risk']:,.2f}")
        st.dataframe(pd.DataFrame(at_risk["customers"]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Look up a customer")
    customer_id = st.number_input("Customer ID", min_value=0, value=0, step=1)
    if st.button("Fetch profile") and customer_id:
        profile, cerr = api_post("/segment", {"customer_id": int(customer_id)})
        if cerr:
            st.error(cerr)
        else:
            metrics, value = profile["metrics"], profile["value"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Segment", profile["segment"])
            k2.metric("Lifetime spend", f"£{metrics['total_monetary_spend']:,.2f}")
            k3.metric(
                "Predicted 90-day value",
                f"£{value['predicted_clv_90d']:,.2f}" if value.get("predicted_clv_90d") else "n/a",
            )
            k4.metric(
                "Churn probability",
                f"{value['churn_probability']:.0%}"
                if value.get("churn_probability") is not None
                else "n/a",
            )
            st.info(f"**Recommended action:** {profile['recommended_action']}")
            st.json(metrics)


# ---------------------------------------------------------------------------
# Cross-sell
# ---------------------------------------------------------------------------
elif page == "Cross-Sell":
    st.header("Cross-sell recommendations")
    st.caption(
        "Association rules mined from invoice baskets, ranked by **lift** rather than "
        "confidence. Confidence alone just surfaces best-sellers; lift measures whether "
        "buying A genuinely makes B more likely than chance."
    )

    skus = sku_options()
    sku = st.selectbox("SKU", skus) if skus else st.text_input("SKU", "85123A")
    top_n = st.slider("Recommendations", 1, 20, 5)

    if sku:
        recs, rerr = cached_get(f"/recommend/{sku}", top_n=top_n)
        if rerr:
            st.warning(rerr)
        else:
            rdf = pd.DataFrame(recs["recommendations"])
            st.altair_chart(
                alt.Chart(rdf)
                .mark_bar()
                .encode(
                    x=alt.X("lift:Q", title="Lift (1.0 = independent)"),
                    y=alt.Y("stock_code:N", sort="-x", title="Product"),
                    tooltip=["stock_code:N", "description:N", "lift:Q", "confidence:Q"],
                )
                .properties(height=max(160, 45 * len(rdf))),
                use_container_width=True,
            )
            for row in rdf.itertuples():
                st.markdown(f"- **{row.stock_code}** — {row.interpretation}")
            st.dataframe(rdf, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
else:
    st.header("Model performance and monitoring")

    models, merr = cached_get("/models")
    if merr:
        st.warning(merr)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bundle version", models["version"])
        m2.metric("SKUs served", models["n_skus"])
        m3.metric(
            "Median MASE",
            f"{models['median_backtest_mase']:.3f}"
            if models.get("median_backtest_mase")
            else "n/a",
        )
        m4.metric("Beating baseline", f"{models['pct_skus_beating_baseline']:.0f}%")
        st.caption(f"Trained {models['trained_at']}")

        st.subheader("Which model won where")
        st.caption(
            "No single model wins everywhere. A steady high-volume SKU and an "
            "intermittent long-tail one are different forecasting problems, so the "
            "champion is chosen per SKU."
        )
        mix = pd.DataFrame(
            [{"model": k, "skus": v} for k, v in models["model_mix"].items()]
        ).sort_values("skus", ascending=False)
        st.altair_chart(
            alt.Chart(mix)
            .mark_bar()
            .encode(
                x=alt.X("skus:Q", title="SKUs"),
                y=alt.Y("model:N", sort="-x", title=None),
                tooltip=["model:N", "skus:Q"],
            )
            .properties(height=max(150, 45 * len(mix))),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Data drift")
    drift, derr = cached_get("/drift")
    if derr:
        st.info(derr)
    else:
        verdict = drift["verdict"]
        {"stable": st.success, "moderate": st.warning, "significant": st.error}[verdict](
            f"**{verdict.title()}** — {drift['recommendation']}"
        )
        ddf = pd.DataFrame(drift["results"])
        if not ddf.empty:
            st.dataframe(
                ddf[ddf["test"] == "psi"][
                    [
                        "feature",
                        "statistic",
                        "severity",
                        "reference_mean",
                        "current_mean",
                        "pct_change",
                    ]
                ].rename(columns={"statistic": "PSI"}),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "PSI reading: below 0.10 stable, 0.10–0.25 moderate, above 0.25 significant."
            )

    st.divider()
    st.subheader("Revenue trend")
    revenue, verr = cached_get("/analytics/revenue", weeks=104)
    if verr:
        st.info(verr)
    else:
        rev = pd.DataFrame(revenue["series"])
        rev["week"] = pd.to_datetime(rev["week"])
        st.altair_chart(
            alt.Chart(rev)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("week:T", title="Week"),
                y=alt.Y("revenue:Q", title="Revenue (£)"),
                tooltip=["week:T", "revenue:Q", "units:Q"],
            )
            .properties(height=280),
            use_container_width=True,
        )
