# Model Card — Retail Demand Forecasting

## What this replaces

The previous version of this card reported:

```
MAPE of SARIMA  = 0.3712
MAPE of Prophet = 40.6340
MAPE of XGBoost = 0.5360
```

Three problems with that table:

1. **Mixed units.** SARIMA and XGBoost are fractions (37%, 54%); Prophet is a
   percentage (40.6%). Read consistently, Prophet is either the best model or
   4063% wrong. It cannot be compared to the others as written.
2. **No baseline.** There was nothing to say whether 37% error is good. For
   weekly retail demand it usually is not — repeating last year's value is a
   strong benchmark and it is not in the table.
3. **One SKU, one split.** All three numbers come from a single 80/20 split on
   the single highest-volume SKU, then generalised to a claim about the model.

The model was also never served. The production `/forecast` endpoint returned
`recent_average * (1 + 0.02 * week)`, a hard-coded growth curve, for every SKU.

## Task

Forecast weekly unit demand per SKU, 1–12 weeks ahead, for the top 100 SKUs by
volume. The output feeds reorder-point and safety-stock calculations, so both
the point forecast and a calibrated prediction interval matter.

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) —
UK online gift retailer, Dec 2009 to Dec 2011, ~1M line items.

| Step | Effect |
| --- | --- |
| Drop rows without a customer id | Needed for RFM and CLV |
| Remove cancellations **and the sales they reverse** | Refunded orders no longer count as revenue |
| Drop service codes (POST, M, DOT, BANK CHARGES) | Not inventory |
| Deduplicate identical lines | |
| Aggregate to weekly buckets per SKU | W-MON frequency |
| Reindex onto a dense weekly grid | Zero-demand weeks made explicit |

That last step matters more than it looks. Grouping by week produces rows only
for weeks with a sale, so `shift(1)` silently means "the previous week that had
a sale". Restoring zero-demand weeks changes both the lag features and the
error metrics — and it is what makes MAPE undefined on this data.

## Metrics, and why not MAPE

**MASE** is the headline metric. It divides absolute error by the in-sample
error of a seasonal-naive forecast, so it is scale-free, defined at zero
demand, and directly interpretable: **MASE < 1 beats seasonal naive.**

MAPE is reported for continuity only. It is a poor choice here:

- **Undefined at zero.** Weekly SKU demand hits zero regularly.
- **Asymmetric.** It penalises over-forecasting harder than under-forecasting,
  so it systematically favours models that under-predict — the wrong bias for
  inventory, where a stockout usually costs more than a week of holding.

Also reported: WAPE (volume-weighted, what a planner reads as "percent error"),
bias (direction, since a low-error model can still quietly drain stock), and
interval coverage (a nominal 95% interval covering 60% of actuals will size
safety stock wrongly).

## Evaluation protocol

Rolling-origin cross-validation. Five forecast origins step forward through
time; each trains only on data before its origin and tests on the following
4 weeks. Test windows do not overlap. Every SKU is evaluated, not one.

Selection is **per SKU** on mean MASE. A steady high-volume SKU and an
intermittent long-tail one are different forecasting problems and no single
model wins both.

**A candidate is only promoted if it beats seasonal naive on that SKU.**
Otherwise the baseline is the champion. Shipping a complex model that loses to
a one-line heuristic is worse than shipping the heuristic.

## Candidate models

| Model | Notes |
| --- | --- |
| `naive` | Last observed value |
| `seasonal_naive` | Value from 52 weeks ago. **The benchmark.** |
| `moving_average` | 4-week mean, held flat |
| `seasonal_drift` | Seasonal naive plus average trend |
| `sarima` | `(1,1,1)(1,1,0,52)`, auto-degrading when history is short |
| `prophet` | Multiplicative seasonality, weekly index |
| `xgboost` | Lag/calendar features, recursive multi-step |

Two corrections to the earlier implementations:

- The old README documented SARIMA as `(1,1,1)(1,1,1,52)` while the code ran
  `(1,1,0,4)`. A period of 4 is a monthly rhythm, not an annual cycle in weekly
  data. That is one reason the published figure could not be reproduced.
- XGBoost was scored on a 4-week horizon using **actual** lag values from the
  test set — information unavailable when forecasting forward. It now forecasts
  recursively, feeding each prediction back as the next step's lag. Its
  prediction interval is calibrated on held-out residuals, because a boosted
  model's in-sample residuals are near zero and produce intervals far too
  narrow to size safety stock from.

Prophet is optional. It ships a compiled Stan backend that fails on some
platforms; the registry probes each candidate at start-up and skips any whose
dependency is unavailable rather than aborting the run.

## Results

Run `make backtest` to regenerate `reports/backtest_summary.csv`.

**The numbers in this repository's committed reports were produced on
synthetic data**, because Online Retail II is not redistributable and CI must
run on a clean checkout. Re-run against the real workbook before quoting any
figure:

```bash
make ingest SOURCE=data/online_retail_II.xlsx
make features
make train
```

What the synthetic backtest shows, and what is worth checking on real data:

- Classical and naive methods are competitive. In one representative run SARIMA
  scored a mean MASE of **1.06** — *worse than seasonal naive* — with a **+34%**
  forecast bias, while a 4-week moving average scored **0.83**.
- Champion mix is mixed across SKUs. No single model wins everywhere, which is
  the argument for per-SKU selection.
- If SARIMA turns out to lose to seasonal naive on the real data too, that is a
  finding worth reporting, not a failure worth hiding. It is the expected
  outcome for short, noisy, promotion-driven SKU series.

## Business translation

Accuracy is converted to money rather than left as a metric. Given holding cost
`Co` and stockout cost `Cu`, the newsvendor critical ratio gives the optimal
service level directly:

```
SL* = Cu / (Cu + Co)
```

and safety stock follows from the forecast **error** distribution — not demand
variance, which over-sizes buffers by charging the model for seasonality it
predicted correctly:

```
safety stock  = z(SL) · σ_error · √(lead time)
reorder point = expected lead-time demand + safety stock
```

`retail_intel.business.inventory.compare_models` prices competing forecasts as
holding plus stockout cost over the same actuals, so model choice can be argued
in currency.

## Limitations

- **Top 100 SKUs only.** The long tail is intermittent demand, which wants
  Croston's method or a zero-inflated model, not what is here.
- **Two years of history** gives at most two observations of any annual
  seasonal peak. Seasonal terms are weakly identified.
- **No exogenous features.** Price, promotions, stockouts and holidays are not
  modelled. A demand series that fell because the item was out of stock is
  recorded as low demand — the single largest source of bias in retail
  forecasting, and not addressed here.
- **UK-dominated** (~90% of rows), so this generalises to one market.
- **Recursive multi-step forecasting compounds error** with horizon. The
  intervals widen with √h to reflect it, but a 12-week forecast is much weaker
  than a 1-week one.
- **Point forecasts are not causal.** Nothing here says what demand *would* be
  under a different price.

## Reproducing

```bash
make install-dev
make pipeline SOURCE=data/online_retail_II.xlsx   # or `make seed` for generated data
make test
```

Every run writes `reports/backtest_summary.csv`, `reports/champions.csv` and
`reports/backtest_folds.csv`, and logs parameters and metrics to MLflow.
