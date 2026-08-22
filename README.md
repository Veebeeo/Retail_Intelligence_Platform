# Retail Intelligence Platform

Demand forecasting, inventory optimisation and customer value modelling over
two years of UK online retail transactions.

The point of the project is not that it forecasts. It is that every forecast is
**measured against the cheap alternative** a planner would otherwise use, only
served when it wins, and translated into a reorder quantity and a cost — so the
question "is this model worth anything" has an answer in currency.

```
Raw workbook ──► transactions ──► weekly features ──┬──► backtest ──► champions ──► API ──► dashboard
   (~1M rows)      (Postgres)      (lag/rolling)     │      (MLflow)      (per SKU)
                        │                            └──► drift monitor
                        ├──► RFM segments + BG/NBD CLV + uplift targeting
                        └──► market-basket association rules
```

---

## Contents

- [What it does](#what-it-does)
- [Results](#results)
- [Design decisions](#design-decisions)
- [Quick start](#quick-start)
- [API](#api)
- [Project layout](#project-layout)
- [Testing](#testing)
- [What this repository used to be](#what-this-repository-used-to-be)
- [Limitations](#limitations)

---

## What it does

### 1. Demand forecasting with honest evaluation

Seven candidate models — four baselines, SARIMA, Prophet, XGBoost — behind one
interface, evaluated by **rolling-origin cross-validation across every SKU**
rather than a single split on the single easiest series.

The champion is selected **per SKU** on MASE, and **only promoted if it beats
seasonal naive on that SKU**. Where nothing beats the baseline, the baseline is
what gets served. A model that loses to a one-line heuristic is not worth
shipping.

Headline metric is MASE, not MAPE. MAPE is undefined when weekly demand hits
zero (which it does, constantly) and it is asymmetric in the wrong direction
for inventory — it prefers models that under-forecast, and under-forecasting is
what causes stockouts. See [MODEL_CARD.md](MODEL_CARD.md).

### 2. Inventory decisions, not just predictions

A forecast is worth nothing until it becomes an order quantity:

```
SL*           = Cu / (Cu + Co)              # newsvendor critical ratio
safety stock  = z(SL) · σ_error · √(lead time)
reorder point = expected lead-time demand + safety stock
```

Safety stock is sized from the **forecast error** distribution, not from demand
variance — a common mistake that over-buys by charging the model for
seasonality it predicted correctly. Supply both cost parameters and the service
level is *derived* rather than assumed.

`POST /inventory/policy` returns a reorder point with an explanation in plain
English.

### 3. Customer value, forward-looking

RFM K-means describes what a customer *has* done. It cannot say whether a quiet
customer has churned or is simply between purchases — the central problem in
non-contractual retail, where nobody cancels, they just stop coming back.

**BG/NBD** (purchase frequency + dropout) and **Gamma-Gamma** (spend per
transaction) are implemented directly on scipy — `lifetimes` is unmaintained and
breaks on modern pandas — giving expected purchases, predicted 90-day value and
a churn probability per customer.

Validated on a time holdout: **0.86 correlation** between predicted and actual
purchases over a 180-day future window.

`GET /customers/at-risk` returns the intersection of high predicted value and
high churn probability — which is where retention budget belongs. Ranking by
past spend alone puts loyal customers at the top, and they need nothing.

### 4. Uplift modelling for campaign targeting

CLV ranks customers by worth. Targeting the top of that ranking still wastes
most of the budget, because the highest-value customers include a large group
who would have bought anyway — the discount sent to them is margin given away.

A **T-learner** estimates the *causal* effect of contact per individual,
splitting customers into persuadables, sure things, lost causes and sleeping
dogs, evaluated with Qini and priced with a campaign profit curve.

> **On the data.** Estimating uplift honestly requires a randomised holdout, and
> Online Retail II has no campaign log and no control group. Rather than imply a
> causal result the data cannot support, the estimator is validated against a
> **simulated campaign with known ground truth** — it recovers the planted
> effect with a Spearman correlation of ~0.5 and identifies a profitable
> targeting depth of ~20% against a *loss* from contacting everyone. This is a
> validated implementation waiting for real campaign data, not a claimed result.

### 5. Cross-sell recommendations

FP-Growth association rules over invoice baskets, ranked by **lift** rather than
confidence. Confidence alone just resurfaces best-sellers regardless of the
input SKU; lift measures whether buying A genuinely makes B more likely than
chance.

### 6. Drift monitoring

PSI, Kolmogorov–Smirnov and chi-square against the training reference window,
producing a verdict and a retrain recommendation.

On p-values in monitoring: with enough rows *every* comparison becomes
significant, because no two weeks of real data come from an identical
distribution. Effect sizes decide the verdict here; p-values are supporting
detail.

---

## Results

Reproduce with `make backtest`. Reports land in `reports/`.

> **These numbers come from generated data.** Online Retail II is not
> redistributable and CI runs on a clean checkout, so the repository ships a
> generator producing realistic series — trend, annual seasonality, promotional
> spikes, customer acquisition and churn, and deliberately planted complementary
> product pairs. The reports in `reports/` are committed so the figures below
> have a backing artifact, but **re-run against the real workbook before quoting
> any of them.**

The committed run: 20 SKUs, 3 rolling origins, 4-week horizon, 60 folds per
model. Full output in [`reports/backtest_summary.csv`](reports/backtest_summary.csv).

| Model | MASE (mean) | MASE (median) | WAPE | Bias | Coverage | Wins vs baseline | Fit time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `moving_average` | **0.878** | 0.827 | 108.7% | +42.1% | 96.7% | 50% | 0.004s |
| `xgboost` | 0.882 | 0.812 | 102.9% | +34.5% | 87.5% | 50% | 0.512s |
| `seasonal_naive` | 0.920 | 0.830 | 115.6% | +20.0% | 93.8% | — *(baseline)* | 0.005s |
| `seasonal_drift` | 0.922 | 0.820 | 115.9% | +19.1% | 97.1% | 60% | 0.006s |
| `naive` | 0.988 | 0.887 | 119.2% | +50.8% | 97.9% | 45% | 0.005s |
| `sarima` | 1.017 | 0.948 | 111.8% | +33.9% | 87.9% | 45% | 8.911s |

Three things worth reading off that table.

**SARIMA — the model this project previously declared its production champion —
is the worst of the six, and the only one scoring above 1.0.** A MASE above 1
means it loses to repeating last year's value, while costing ~1800x more compute
per fit. That is not a bug: it is the expected outcome for short, noisy,
promotion-driven SKU series with barely two seasonal cycles of history. It is
also precisely what reporting a single MAPE figure with no baseline conceals.

**No model dominates.** The best mean MASE belongs to a 4-week moving average,
but it wins on only half the SKUs. Champion mix across the catalogue:

| Champion | SKUs |
| --- | --- |
| `naive` | 5 |
| `moving_average` | 4 |
| `seasonal_naive` *(nothing beat the baseline)* | 4 |
| `xgboost` | 3 |
| `sarima` | 2 |
| `seasonal_drift` | 2 |

That spread is the argument for per-SKU selection. Picking one global winner
would have been wrong for 16 of 20 SKUs.

**Per-SKU selection beats every individual model.** Mean champion MASE is
**0.729** against the best single model's 0.878, with a median improvement of
**16.9%** over baseline. 80% of SKUs get a model that beats seasonal naive; the
remaining 20% are served by the baseline itself, which is the honest outcome
rather than a failure.

Every model over-forecasts on this data (positive bias throughout) — worth
knowing, since sustained over-forecasting quietly accumulates stock.

---

## Design decisions

Decisions worth defending, and why:

| Decision | Reasoning |
| --- | --- |
| MASE over MAPE | MAPE is undefined at zero demand and asymmetric against over-forecasting — the wrong bias for inventory |
| Per-SKU champions | Steady high-volume and intermittent long-tail SKUs are different problems; no model wins both |
| Baseline veto | A candidate that loses to seasonal naive is not promoted, ever |
| Rolling-origin CV | One split gives one number from one arbitrary cut-off, on whichever series happened to be chosen |
| Safety stock from error, not demand variance | Predicted seasonality is not uncertainty and should not be buffered against |
| BG/NBD over an RFM recency cut-off | Distinguishes "churned" from "slow but alive"; a recency threshold cannot |
| Uplift over propensity | Targeting who *responds* rather than who *buys*; the difference is the whole marketing budget |
| Lift over confidence | Confidence ranks best-sellers regardless of input |
| Optional heavy dependencies | The registry probes each candidate; a broken Prophet install skips that model instead of aborting a 100-SKU run |
| Serving image carries statsmodels and xgboost | A champion bundle is a pickle of fitted models, so serving needs whatever library produced the winner — the training-only stack (mlflow, mlxtend, pandera) is what gets left out |
| Spawn-based process pool | Forking a process holding threaded-BLAS locks deadlocks the children |

---

## Quick start

### Run it without any data

Everything works from a clean clone — the generator produces a realistic
warehouse:

```bash
make install-dev
cp .env.example .env          # defaults to SQLite; no Postgres required
make seed                     # generate → ingest → features → train → segment → baskets → drift
make api                      # http://localhost:8000/docs
make dashboard                # http://localhost:8501
```

### Run it on the real dataset

Download [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
to `data/online_retail_II.xlsx`, then:

```bash
make pipeline SOURCE=data/online_retail_II.xlsx
```

Or step by step:

```bash
make ingest SOURCE=data/online_retail_II.xlsx   # → transactions
make features                                   # → ml_weekly_features
make backtest                                   # → reports/backtest_summary.csv
make train                                      # → models/champions.pkl + MLflow
make segment                                    # → customer_segments (RFM + CLV + uplift)
make baskets                                    # → product_associations
make drift                                      # → reports/drift_report.json
```

### Docker

```bash
make up     # Postgres + MLflow + API + dashboard
make down
```

Services: API on `:8001`, dashboard on `:8501`, MLflow on `:5000`, Postgres on
`:5430`. Every service has a healthcheck and dependencies wait on health rather
than on the container merely existing.

### Orchestration

The pipeline runs as plain Python by default, with Prefect as an optional
extra — the API image does not need an orchestrator to serve a forecast:

```bash
python -m pipelines.flow --synthetic          # plain
python -m pipelines.flow --synthetic --prefect  # with retries and task-level observability
```

---

## API

Interactive docs at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `POST /forecast` | Demand forecast with 95% interval, model name and backtest score |
| `POST /inventory/policy` | Reorder point and safety stock, with a plain-English explanation |
| `GET /models` | Registry summary: version, model mix, median MASE, % beating baseline |
| `GET /models/{sku}` | Backtest provenance for one SKU |
| `POST /segment` | Customer segment, predicted CLV, churn probability, next action |
| `GET /customers/at-risk` | High value ∩ high churn risk, ranked |
| `GET /segments/summary` | Segment sizes, revenue share, recommended actions |
| `GET /recommend/{sku}` | Cross-sell candidates ranked by lift |
| `GET /drift` | Latest drift report and retrain recommendation |
| `GET /health` | Real readiness — reports *degraded*, not a hard-coded "healthy" |

Every forecast response carries its own provenance:

```json
{
  "stock_code": "85123A",
  "model": "seasonal_drift",
  "model_version": 3,
  "backtest_mase": 0.71,
  "baseline_mase": 0.94,
  "improvement_vs_baseline_pct": 24.5,
  "predictions": [
    {"week_horizon": 1, "week_starting": "2011-12-12",
     "predicted_quantity": 412.3, "lower_95": 288.1, "upper_95": 561.7}
  ]
}
```

You can tell a well-validated forecast from a weak one without leaving the
response.

---

## Project layout

```
src/retail_intel/
├── config.py              # pydantic-settings; every tunable in one place
├── db.py                  # lazy engine, parameterised queries only
├── data/
│   ├── contracts.py       # Pandera schemas at every pipeline boundary
│   ├── ingest.py          # raw → transactions
│   ├── features.py        # transactions → ml_weekly_features
│   ├── models.py          # SQLAlchemy schema
│   └── synthetic.py       # generator: trend, seasonality, churn, complements
├── forecasting/
│   ├── base.py            # the Forecaster interface
│   ├── baselines.py       # naive, seasonal naive, moving average, drift
│   ├── models.py          # SARIMA, Prophet, XGBoost
│   ├── metrics.py         # MASE, WAPE, sMAPE, bias, coverage, pinball
│   ├── backtest.py        # rolling-origin CV, parallel
│   ├── registry.py        # availability probing
│   ├── train.py           # champion selection, persistence, MLflow
│   └── serving.py         # load once, serve fast
├── business/inventory.py  # newsvendor, safety stock, cost simulation
├── segmentation/
│   ├── rfm.py             # RFM + K search + labelling
│   ├── clv.py             # BG/NBD + Gamma-Gamma
│   ├── uplift.py          # T-learner, Qini, campaign economics
│   └── pipeline.py        # combined customer table
├── recommend/market_basket.py
└── monitoring/drift.py    # PSI, KS, chi-square

app/                       # FastAPI (routers, schemas) + Streamlit dashboard
pipelines/flow.py          # end-to-end orchestration
tests/                     # 146 tests
```

---

## Testing

```bash
make test        # 146 tests
make coverage    # 78%
make lint
```

The suite runs against SQLite seeded from the generator — no Postgres, no
credentials, no network. CI runs lint, tests on Python 3.11 and 3.12, a full
end-to-end pipeline on generated data, and a Docker build that must answer
`/health` before it passes.

Tests worth pointing at:

- `test_features.py` — asserts rolling windows exclude the current week, so a
  leaking feature cannot silently reappear
- `test_api.py` — SQL injection payloads against both endpoints, plus an
  assertion that the forecast is not a fixed 2%-per-week curve
- `test_customers.py` — the T-learner must recover a *known* simulated uplift
- `test_pipeline.py` — the miner must recover deliberately planted product
  associations

---

## What this repository used to be

Worth stating plainly, since the history is public.

| Was | Now |
| --- | --- |
| `/forecast` returned `recent_avg * (1 + 0.02 * week)` while the docs described a SARIMA production model | Serves the per-SKU champion selected by backtest |
| Both endpoints interpolated user input into SQL with f-strings | Parameterised throughout, with validation before the query and tests that prove it |
| `import streamlit as str` | Fixed, and the dashboard rebuilt as a decision tool |
| `ingest_features.py` documented but absent — features lived only in notebook cells | Exists; the pipeline is reproducible from a clone |
| `rolling(4).mean()` included the current week | Every window shifted; a test enforces it |
| Cancellations dropped the credit note but kept the original sale | Both removed, so refunds stop counting as revenue |
| Weekly grouping silently skipped zero-demand weeks | Dense weekly grid |
| `K=4` hard-coded with a comment claiming elbow and silhouette had chosen it | K actually searched, scores recorded |
| Clusters labelled by monetary value alone | Labelled on all three RFM dimensions |
| `"Leal Customers"` typo in the database and live API responses | Fixed |
| MAPE only, no baseline, one SKU, one split | MASE/WAPE/bias/coverage, four baselines, every SKU, rolling origin |
| Model card compared `0.3712` against `40.6340` as if both were the same unit | Consistent units, baselines, honest caveats |
| `mlflow.db` and `mlruns/` committed | Untracked |
| Unpinned dependencies; `xgboost`/`prophet` imported but not declared | Pinned and split by role |
| No tests, no CI | 146 tests, 78% coverage, four CI jobs |
| Container ran as root, no healthcheck | Unprivileged user, healthchecks throughout |

---

## Limitations

Stated because they are real, not because they are small:

- **Top 100 SKUs.** The long tail is intermittent demand and wants Croston's
  method, which is not implemented.
- **No exogenous features.** Price, promotions, holidays and stockouts are not
  modelled. A week that sold nothing *because the item was out of stock* is
  recorded as zero demand — the single largest source of bias in retail
  forecasting, and untreated here.
- **Two years of history** means at most two observations of any annual peak, so
  seasonal terms are weakly identified.
- **UK-dominated** (~90% of rows).
- **Uplift is validated, not measured.** No control group exists in this data.
- **Committed metrics are from generated data.** Re-run on the real workbook.
- **Forecasts are not causal.** Nothing here says what demand would be at a
  different price.

---

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii),
UCI Machine Learning Repository. Not redistributed here.

## References

- Hyndman & Koehler (2006), *Another look at measures of forecast accuracy* — MASE
- Fader, Hardie & Lee (2005), *"Counting Your Customers" the Easy Way* — BG/NBD
- Fader & Hardie (2013), *The Gamma-Gamma Model of Monetary Value*
- Radcliffe (2007), *Using Control Groups to Target on Predicted Lift* — Qini

## Licence

MIT
