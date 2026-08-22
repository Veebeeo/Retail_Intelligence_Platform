"""Per-SKU training, champion selection and model persistence.

What changed from the original ``scripts/train_models.py``:

* It trained on **one** SKU and logged three runs. Nothing was ever registered
  or written anywhere the API could reach, which is why the served endpoint
  fell back to an invented growth rate.
* Champion selection was a manual reading of a metrics table. Here it is the
  output of the backtest — chosen per SKU, on MASE, against the baseline.
* A model is only promoted if it actually beats seasonal naive on that SKU.
  Otherwise the baseline *is* the champion. Shipping a complicated model that
  loses to a one-line heuristic is worse than shipping the heuristic.

Artifacts are written twice: to MLflow for lineage and comparison, and to a
plain joblib bundle under ``MODEL_DIR`` that the API loads at start-up. The
second path means serving does not require a reachable tracking server.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from retail_intel.config import get_settings
from retail_intel.db import read_sql, write_table
from retail_intel.forecasting import metrics as M
from retail_intel.forecasting.backtest import pick_champions, summarise
from retail_intel.forecasting.backtest import run as run_backtest
from retail_intel.forecasting.registry import available_models, build
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)

BASELINE = "seasonal_naive"


def load_panel() -> pd.DataFrame:
    df = read_sql(
        "SELECT stock_code, week, weekly_sales FROM ml_weekly_features ORDER BY stock_code, week"
    )
    if df.empty:
        raise RuntimeError("`ml_weekly_features` is empty. Run the feature pipeline first.")
    df["week"] = pd.to_datetime(df["week"])
    return df


def fit_champion(y: pd.Series, model_name: str, seasonal_period: int):
    """Refit the chosen model on the full history, ready to serve."""
    model = build(model_name, seasonal_period=seasonal_period)
    model.fit(y)
    return model


def residual_std_from_folds(
    folds: pd.DataFrame, stock_code: str, model: str, fallback: float
) -> float:
    """Out-of-sample forecast-error sigma, for sizing safety stock.

    Taken from the backtest folds rather than re-fitting the model on a fresh
    holdout. Two reasons: the backtest already measured exactly this quantity
    across several forecast origins, so the average is more stable than a
    single tail split; and refitting doubled the cost of the training job for
    no new information -- a SARIMA fit on two years of weekly history with a
    52-period seasonal term takes minutes, and it was being paid twice per SKU.

    RMSE is the right column: it is the root mean squared *forecast error*,
    which is the standard deviation of that error for an unbiased forecast.
    """
    rows = folds[
        (folds["stock_code"] == stock_code) & (folds["model"] == model) & folds["error"].isna()
    ]
    if rows.empty or not np.isfinite(rows["rmse"]).any():
        return float(fallback)
    return float(rows["rmse"].mean())


def run(
    panel: pd.DataFrame | None = None,
    horizon: int | None = None,
    n_folds: int | None = None,
    max_skus: int | None = None,
    n_jobs: int = 1,
    log_to_mlflow: bool = True,
) -> dict:
    settings = get_settings()
    horizon = horizon or settings.forecast_horizon
    n_folds = n_folds or settings.backtest_folds
    panel = load_panel() if panel is None else panel

    logger.info("Backtesting to select champions...")
    folds = run_backtest(
        panel=panel, horizon=horizon, n_folds=n_folds, max_skus=max_skus, n_jobs=n_jobs
    )
    summary = summarise(folds, baseline=BASELINE)
    champions = pick_champions(folds, baseline=BASELINE)

    logger.info("Backtest leaderboard:\n%s", summary.round(3).to_string(index=False))

    # Demote any champion that does not actually beat the baseline.
    demoted = champions["champion_mase"] >= champions["baseline_mase"]
    n_demoted = int(demoted.sum())
    champions.loc[demoted, "champion"] = BASELINE
    if n_demoted:
        logger.info(
            "%d/%d SKUs kept the %s baseline: no candidate beat it",
            n_demoted,
            len(champions),
            BASELINE,
        )

    bundle = _fit_and_persist(panel, champions, horizon, settings, folds)
    if log_to_mlflow:
        _log_mlflow(summary, champions, bundle, settings, horizon, n_folds)

    _persist_reports(summary, champions, folds, settings)
    return {
        "summary": summary,
        "champions": champions,
        "folds": folds,
        "n_models": len(bundle["forecasts"]),
    }


def _fit_and_persist(
    panel: pd.DataFrame, champions: pd.DataFrame, horizon: int, settings, folds: pd.DataFrame
) -> dict:
    """Refit each SKU's champion on full history and write one servable bundle.

    The bundle stores the **forecast path**, not the fitted model object.

    These are univariate models with no exogenous inputs, and no new data
    arrives between retrains, so the forecast for any horizon is fully
    determined the moment the model is fitted. Persisting the path is exactly
    equivalent to persisting the model and calling predict later.

    It is also the difference between a bundle of kilobytes and one of
    gigabytes. A fitted SARIMAX result with a 52-period seasonal term pickles
    to roughly 220 MB because it carries the full Kalman smoother state for
    every observation; twenty SARIMA champions alone produced a 445 MB file.
    Storing paths also means the serving image needs neither statsmodels nor
    xgboost, since there is nothing left to unpickle but numbers.
    """
    max_horizon = max(settings.max_forecast_horizon, horizon)
    forecasts: dict[str, dict[str, list[float]]] = {}
    meta: dict[str, dict] = {}

    for row in champions.itertuples():
        y = panel.loc[panel["stock_code"] == row.stock_code, "weekly_sales"].reset_index(drop=True)
        try:
            model = fit_champion(y, row.champion, settings.seasonal_period)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s: %s failed to fit (%s); using %s", row.stock_code, row.champion, exc, BASELINE
            )
            model = fit_champion(y, BASELINE, settings.seasonal_period)

        path = model.predict(max_horizon)
        forecasts[row.stock_code] = {
            "yhat": [round(float(v), 4) for v in path["yhat"]],
            "yhat_lower": [round(float(v), 4) for v in path["yhat_lower"]],
            "yhat_upper": [round(float(v), 4) for v in path["yhat_upper"]],
        }
        meta[row.stock_code] = {
            "model": getattr(model, "name", row.champion),
            "backtest_mase": float(row.champion_mase),
            "baseline_mase": float(row.baseline_mase),
            "improvement_pct": float(row.improvement_pct)
            if np.isfinite(row.improvement_pct)
            else None,
            "residual_std": residual_std_from_folds(
                folds, row.stock_code, row.champion, fallback=float(y.std())
            ),
            "n_weeks": int(len(y)),
            "last_week": str(panel.loc[panel["stock_code"] == row.stock_code, "week"].max().date()),
            "mean_weekly_sales": float(y.mean()),
        }

    bundle = {
        "forecasts": forecasts,
        "meta": meta,
        "seasonal_period": settings.seasonal_period,
        "horizon": horizon,
        "max_horizon": max_horizon,
        "trained_at": datetime.now(UTC).isoformat(),
        "version": _next_version(settings.model_path),
    }

    path = settings.model_path
    path.mkdir(parents=True, exist_ok=True)
    target = path / "champions.pkl"
    with target.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (path / "champions_meta.json").write_text(
        json.dumps(
            {"version": bundle["version"], "trained_at": bundle["trained_at"], "meta": meta},
            indent=2,
            default=str,
        )
    )
    logger.info(
        "Persisted %d champion forecast paths (%d weeks each) to %s -- %.1f KB",
        len(forecasts),
        max_horizon,
        target,
        target.stat().st_size / 1024,
    )
    return bundle


def _next_version(path: Path) -> int:
    meta_file = path / "champions_meta.json"
    if not meta_file.exists():
        return 1
    try:
        return int(json.loads(meta_file.read_text()).get("version", 0)) + 1
    except Exception:  # noqa: BLE001
        return 1


def _log_mlflow(summary, champions, bundle, settings, horizon: int, n_folds: int) -> None:
    """Log the run to MLflow and register the bundle.

    Wrapped so a missing or unreachable tracking server degrades to a warning:
    training that produced a servable model should not be thrown away because
    the experiment tracker is down.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; skipping experiment logging")
        return

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)

        with mlflow.start_run(run_name=f"champions_v{bundle['version']}"):
            mlflow.log_params(
                {
                    "horizon_weeks": horizon,
                    "backtest_folds": n_folds,
                    "seasonal_period": settings.seasonal_period,
                    "n_skus": len(champions),
                    "candidate_models": ",".join(available_models()),
                    "selection_metric": "MASE",
                    "baseline": BASELINE,
                }
            )
            for row in summary.itertuples():
                for metric in (
                    "mase_mean",
                    "mase_median",
                    "wape_mean",
                    "rmse_mean",
                    "coverage_pct",
                ):
                    value = getattr(row, metric, None)
                    if value is not None and np.isfinite(value):
                        mlflow.log_metric(f"{row.model}__{metric}", float(value))

            beat = float((champions["champion"] != BASELINE).mean() * 100)
            mlflow.log_metric("pct_skus_beating_baseline", beat)
            mlflow.log_metric("mean_champion_mase", float(champions["champion_mase"].mean()))

            model_dir = settings.model_path
            mlflow.log_artifacts(str(model_dir), artifact_path="champions")
            mlflow.set_tag(
                "champion_mix", json.dumps(champions["champion"].value_counts().to_dict())
            )
        logger.info("Logged run to MLflow at %s", settings.mlflow_tracking_uri)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging failed (%s); model artifacts were still written", exc)


def _persist_reports(summary, champions, folds, settings) -> None:
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(settings.report_dir / "backtest_summary.csv", index=False)
    champions.to_csv(settings.report_dir / "champions.csv", index=False)
    folds.to_csv(settings.report_dir / "backtest_folds.csv", index=False)

    try:
        write_table(champions, "model_champions")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write champions to the database: %s", exc)


def evaluate_holdout(panel: pd.DataFrame, horizon: int, seasonal_period: int) -> pd.DataFrame:
    """Score every model on one final untouched holdout for the model card."""
    rows = []
    for sku, grp in panel.groupby("stock_code"):
        y = grp["weekly_sales"].reset_index(drop=True)
        if len(y) < horizon + 30:
            continue
        train, test = y.iloc[:-horizon], y.iloc[-horizon:]
        for name in available_models():
            try:
                pred = build(name, seasonal_period=seasonal_period).fit(train).predict(horizon)
                rows.append(
                    {
                        "stock_code": sku,
                        "model": name,
                        **M.evaluate(
                            test.to_numpy(),
                            pred["yhat"].to_numpy(),
                            train.to_numpy(),
                            pred["yhat_lower"].to_numpy(),
                            pred["yhat_upper"].to_numpy(),
                            seasonal_period,
                        ),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register per-SKU champion forecasters.")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--max-skus", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers; 0 uses every core.")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    out = run(
        horizon=args.horizon,
        n_folds=args.folds,
        max_skus=args.max_skus,
        n_jobs=args.jobs,
        log_to_mlflow=not args.no_mlflow,
    )
    champs = out["champions"]
    logger.info("Trained %d champion models", out["n_models"])
    logger.info("Champion mix:\n%s", champs["champion"].value_counts().to_string())
    beat = (champs["champion"] != BASELINE).mean() * 100
    logger.info("%.1f%% of SKUs use a model that beats the %s baseline", beat, BASELINE)


if __name__ == "__main__":
    main()
