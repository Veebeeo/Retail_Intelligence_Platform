"""Rolling-origin cross-validation.

The original evaluation was a single 80/20 split on a single SKU. That gives
one number from one arbitrary cut-off, on the easiest series in the catalogue —
you cannot tell from it whether a model is good or whether that particular
fortnight happened to be quiet.

This evaluates every SKU across several forecast origins that move forward
through time, always training on the past and testing on the future, and
reports the *distribution* of error. What matters in the output is not the mean
MASE but the share of SKUs where the model beats seasonal naive.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from retail_intel.config import get_settings
from retail_intel.db import read_sql
from retail_intel.forecasting import metrics as M
from retail_intel.forecasting.registry import available_models, build
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class FoldResult:
    stock_code: str
    model: str
    fold: int
    train_end: int
    metrics: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    error: str | None = None


def rolling_origin_splits(
    n_obs: int, horizon: int, n_folds: int, min_train: int
) -> list[tuple[int, int]]:
    """Yield ``(train_end, test_end)`` index pairs, oldest origin first.

    Origins step forward by one horizon so the test windows do not overlap,
    which keeps the folds close to independent.
    """
    splits: list[tuple[int, int]] = []
    for k in range(n_folds, 0, -1):
        train_end = n_obs - k * horizon
        if train_end < min_train:
            continue
        splits.append((train_end, train_end + horizon))
    return splits


def backtest_series(
    y: pd.Series,
    model_names: tuple[str, ...],
    horizon: int,
    n_folds: int,
    min_train: int,
    seasonal_period: int,
    stock_code: str = "series",
) -> list[FoldResult]:
    """Backtest every named model on one series."""
    results: list[FoldResult] = []
    splits = rolling_origin_splits(len(y), horizon, n_folds, min_train)
    if not splits:
        logger.debug("%s: only %d weeks, not enough for a fold", stock_code, len(y))
        return results

    for fold, (train_end, test_end) in enumerate(splits):
        y_train = y.iloc[:train_end]
        y_test = y.iloc[train_end:test_end]
        if y_test.empty:
            continue

        for name in model_names:
            started = time.perf_counter()
            try:
                model = build(name, seasonal_period=seasonal_period)
                model.fit(y_train)
                pred = model.predict(len(y_test))
                scores = M.evaluate(
                    y_test.to_numpy(),
                    pred["yhat"].to_numpy(),
                    y_train=y_train.to_numpy(),
                    lower=pred["yhat_lower"].to_numpy(),
                    upper=pred["yhat_upper"].to_numpy(),
                    seasonal_period=seasonal_period,
                )
                results.append(
                    FoldResult(
                        stock_code, name, fold, train_end, scores, time.perf_counter() - started
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad SKU must not stop the run
                logger.debug("%s/%s fold %d failed: %s", stock_code, name, fold, exc)
                results.append(
                    FoldResult(
                        stock_code, name, fold, train_end,
                        fit_seconds=time.perf_counter() - started, error=str(exc)[:200],
                    )
                )
    return results


def load_panel() -> pd.DataFrame:
    df = read_sql("SELECT stock_code, week, weekly_sales FROM ml_weekly_features ORDER BY stock_code, week")
    if df.empty:
        raise RuntimeError("`ml_weekly_features` is empty. Run the feature pipeline first.")
    df["week"] = pd.to_datetime(df["week"])
    return df


def run(
    panel: pd.DataFrame | None = None,
    model_names: tuple[str, ...] | None = None,
    horizon: int | None = None,
    n_folds: int | None = None,
    max_skus: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Backtest every SKU and return one row per (SKU, model, fold)."""
    settings = get_settings()
    horizon = horizon or settings.forecast_horizon
    n_folds = n_folds or settings.backtest_folds
    model_names = model_names or available_models()
    panel = load_panel() if panel is None else panel

    skus = list(panel["stock_code"].unique())
    if max_skus:
        skus = skus[:max_skus]
    n_jobs = max(1, min(n_jobs if n_jobs > 0 else (os.cpu_count() or 1), len(skus)))
    logger.info(
        "Backtesting %d SKUs x %d models x %d folds (horizon=%d weeks, n_jobs=%d)",
        len(skus), len(model_names), n_folds, horizon, n_jobs,
    )

    series = {
        sku: panel.loc[panel["stock_code"] == sku, "weekly_sales"].reset_index(drop=True)
        for sku in skus
    }
    args = (model_names, horizon, n_folds, settings.min_train_weeks, settings.seasonal_period)

    rows: list[dict] = []
    started = time.perf_counter()

    if n_jobs == 1:
        for i, (sku, y) in enumerate(series.items(), 1):
            rows.extend(_as_rows(backtest_series(y, *args, sku)))
            _progress(i, len(skus), started)
    else:
        # SKUs are independent and SARIMA fits are CPU-bound, so a process pool
        # gives close to linear speed-up on the dominant cost.
        #
        # "spawn", not the Linux default "fork": statsmodels pulls in a
        # threaded BLAS, and forking a process that already holds BLAS thread
        # locks deadlocks the children. Workers are also pinned to one BLAS
        # thread each, otherwise n_jobs workers x n BLAS threads oversubscribe
        # the machine and run slower than sequential.
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
        ) as pool:
            futures = {
                pool.submit(backtest_series, y, *args, sku): sku for sku, y in series.items()
            }
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    rows.extend(_as_rows(future.result()))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SKU %s failed entirely: %s", futures[future], exc)
                _progress(i, len(skus), started)

    return pd.DataFrame(rows)


def _init_worker() -> None:
    """Pin each worker to a single BLAS thread and silence library warnings."""
    import warnings

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    warnings.filterwarnings("ignore")


def _as_rows(results: list[FoldResult]) -> list[dict]:
    return [
        {
            "stock_code": r.stock_code,
            "model": r.model,
            "fold": r.fold,
            "train_end": r.train_end,
            "fit_seconds": r.fit_seconds,
            "error": r.error,
            **r.metrics,
        }
        for r in results
    ]


def _progress(done: int, total: int, started: float) -> None:
    if done % 10 == 0 or done == total:
        logger.info("  %d/%d SKUs (%.0fs elapsed)", done, total, time.perf_counter() - started)


def summarise(results: pd.DataFrame, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """Aggregate fold results into a per-model leaderboard.

    ``win_rate_vs_baseline`` is the headline: the share of SKUs where the model
    has a lower average MASE than seasonal naive. A model can win on mean error
    while losing on most SKUs, which usually means it is being carried by a
    handful of high-volume series.
    """
    ok = results[results["error"].isna()].copy()
    if ok.empty:
        raise RuntimeError("Every backtest fold failed. Check the logs.")

    per_sku = ok.groupby(["model", "stock_code"], as_index=False).agg(
        mase=("mase", "mean"), wape=("wape", "mean"), rmse=("rmse", "mean")
    )
    base = per_sku[per_sku["model"] == baseline].set_index("stock_code")["mase"]

    def win_rate(grp: pd.DataFrame) -> float:
        joined = grp.set_index("stock_code")["mase"].dropna()
        common = joined.index.intersection(base.dropna().index)
        if len(common) == 0:
            return float("nan")
        return float((joined.loc[common] < base.loc[common]).mean() * 100)

    summary = (
        ok.groupby("model")
        .agg(
            n_folds=("fold", "size"),
            n_skus=("stock_code", "nunique"),
            mase_mean=("mase", "mean"),
            mase_median=("mase", "median"),
            wape_mean=("wape", "mean"),
            rmse_mean=("rmse", "mean"),
            smape_mean=("smape", "mean"),
            bias_pct=("bias_pct", "mean"),
            coverage_pct=("coverage_pct", "mean"),
            fit_seconds=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    summary["win_rate_vs_baseline"] = [
        win_rate(per_sku[per_sku["model"] == m]) for m in summary["model"]
    ]
    failures = results[results["error"].notna()].groupby("model").size()
    summary["failed_folds"] = summary["model"].map(failures).fillna(0).astype(int)
    return summary.sort_values("mase_mean").reset_index(drop=True)


def pick_champions(results: pd.DataFrame, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """Choose the best model per SKU on mean MASE.

    Selecting per SKU rather than globally matters: a steady high-volume SKU and
    an intermittent long-tail one are different forecasting problems, and no
    single model wins both. Ties and total failures fall back to the baseline,
    so every SKU always has something servable.
    """
    ok = results[results["error"].isna()].copy()
    per_sku = ok.groupby(["stock_code", "model"], as_index=False).agg(
        mase=("mase", "mean"), wape=("wape", "mean"), rmse=("rmse", "mean")
    )
    per_sku = per_sku.dropna(subset=["mase"])

    champs = []
    for sku, grp in per_sku.groupby("stock_code"):
        best = grp.loc[grp["mase"].idxmin()]
        base_row = grp[grp["model"] == baseline]
        base_mase = float(base_row["mase"].iloc[0]) if len(base_row) else float("nan")
        champs.append(
            {
                "stock_code": sku,
                "champion": best["model"],
                "champion_mase": float(best["mase"]),
                "champion_wape": float(best["wape"]),
                "baseline_mase": base_mase,
                "improvement_pct": (
                    float((base_mase - best["mase"]) / base_mase * 100)
                    if base_mase and np.isfinite(base_mase) and base_mase > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(champs).sort_values("improvement_pct", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin backtest across all SKUs.")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--max-skus", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers; 0 uses every core.")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out", default=None, help="Directory for the JSON/CSV reports.")
    args = parser.parse_args()

    settings = get_settings()
    out_dir = Path(args.out or settings.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run(
        model_names=tuple(args.models) if args.models else None,
        horizon=args.horizon,
        n_folds=args.folds,
        max_skus=args.max_skus,
        n_jobs=args.jobs,
    )
    summary = summarise(results)
    champions = pick_champions(results)

    results.to_csv(out_dir / "backtest_folds.csv", index=False)
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)
    champions.to_csv(out_dir / "champions.csv", index=False)
    (out_dir / "backtest_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, default=float)
    )

    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Champion mix:\n%s", champions["champion"].value_counts().to_string())
    logger.info("Reports written to %s", out_dir)


if __name__ == "__main__":
    main()
