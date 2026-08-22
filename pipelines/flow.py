"""End-to-end orchestration.

Runs the full chain in dependency order:

    ingest -> features -> train/select champions -> segment -> baskets -> drift

Two execution modes, deliberately:

* **Plain Python** (``python -m pipelines.flow``) — no orchestration dependency
  at all. This is what CI runs and what a reviewer can execute after cloning.
* **Prefect** (``python -m pipelines.flow --prefect``) — the same steps wrapped
  as a flow with retries, task-level observability and a deployable schedule.

Keeping Prefect optional is the point: the API image does not need an
orchestrator on its dependency list to serve a forecast, and the pipeline does
not need one to be reproducible.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from retail_intel.config import get_settings
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class StepResult:
    name: str
    status: str
    seconds: float
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def step_ingest(path: str, sheet: str | None = None) -> dict:
    from retail_intel.data.ingest import run

    return {"rows": run(path, sheet)}


def step_seed_synthetic(n_customers: int = 500, n_skus: int = 40, weeks: int = 130) -> dict:
    """Populate the warehouse with generated data.

    Lets the whole pipeline be demonstrated without the source workbook, which
    is not redistributable.
    """
    from retail_intel.data.contracts import TransactionSchema, validate
    from retail_intel.data.ingest import clean
    from retail_intel.data.models import create_all
    from retail_intel.data.synthetic import make_transactions
    from retail_intel.db import write_table

    create_all()
    raw = make_transactions(n_customers=n_customers, n_skus=n_skus, weeks=weeks)
    rows = write_table(validate(clean(raw), TransactionSchema), "transactions")
    return {"rows": rows, "source": "synthetic"}


def step_features(top_n: int | None = None) -> dict:
    from retail_intel.data.features import run

    df = run(top_n=top_n)
    return {"rows": len(df), "skus": int(df["stock_code"].nunique())}


def step_train(horizon: int | None = None, folds: int | None = None, jobs: int = 0) -> dict:
    from retail_intel.forecasting.train import run

    out = run(horizon=horizon, n_folds=folds, n_jobs=jobs)
    champions = out["champions"]
    return {
        "n_models": out["n_models"],
        "champion_mix": champions["champion"].value_counts().to_dict(),
        "mean_champion_mase": round(float(champions["champion_mase"].mean()), 4),
        "pct_beating_baseline": round(
            float((champions["champion"] != "seasonal_naive").mean() * 100), 1
        ),
    }


def step_segment(clv_days: int = 90) -> dict:
    from retail_intel.segmentation.pipeline import run

    report = run(clv_horizon_days=clv_days)
    return {
        "n_customers": report["n_customers"],
        "k": report["k_selected"],
        "clv": report["clv_summary"],
    }


def step_baskets(min_support: float = 0.01, min_lift: float = 1.2) -> dict:
    from retail_intel.recommend.market_basket import run

    rules = run(min_support=min_support, min_lift=min_lift)
    return {
        "n_rules": len(rules),
        "max_lift": round(float(rules["lift"].max()), 3) if len(rules) else None,
    }


def step_drift(weeks: int = 8) -> dict:
    from retail_intel.monitoring.drift import run

    report = run(weeks)
    return {"verdict": report["verdict"], "n_flagged": report["n_flagged"]}


STEPS = {
    "features": step_features,
    "train": step_train,
    "segment": step_segment,
    "baskets": step_baskets,
    "drift": step_drift,
}


def run_pipeline(
    source: str | None = None,
    sheet: str | None = None,
    synthetic: bool = False,
    skip: tuple[str, ...] = (),
    fail_fast: bool = False,
    **step_kwargs,
) -> list[StepResult]:
    """Run every step, recording timing and outcome.

    Later steps are independent of each other (segments do not depend on
    forecasts), so by default one failure does not abandon the run — the
    summary reports exactly what succeeded.
    """
    settings = get_settings()
    results: list[StepResult] = []

    ordered: list[tuple[str, Any]] = []
    if synthetic:
        ordered.append(("seed", lambda: step_seed_synthetic(**step_kwargs.get("seed", {}))))
    elif source:
        ordered.append(("ingest", lambda: step_ingest(source, sheet)))

    for name, fn in STEPS.items():
        if name not in skip:
            ordered.append((name, lambda fn=fn, name=name: fn(**step_kwargs.get(name, {}))))

    for name, fn in ordered:
        logger.info("=" * 70)
        logger.info("STEP: %s", name)
        logger.info("=" * 70)
        started = time.perf_counter()
        try:
            detail = fn()
            results.append(StepResult(name, "success", time.perf_counter() - started, detail))
            logger.info("%s completed in %.1fs: %s", name, results[-1].seconds, detail)
        except Exception as exc:  # noqa: BLE001
            results.append(
                StepResult(name, "failed", time.perf_counter() - started, error=str(exc)[:500])
            )
            logger.exception("%s failed", name)
            if fail_fast:
                break

    _write_manifest(results, settings)
    return results


def _write_manifest(results: list[StepResult], settings) -> None:
    """Record what ran, in what order, with what outcome.

    A pipeline that leaves no trace of its own execution cannot be debugged
    after the fact.
    """
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_at": datetime.now(UTC).isoformat(),
        "total_seconds": round(sum(r.seconds for r in results), 2),
        "steps": [
            {
                "name": r.name,
                "status": r.status,
                "seconds": round(r.seconds, 2),
                "detail": r.detail,
                "error": r.error,
            }
            for r in results
        ],
    }
    (settings.report_dir / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )


def summarise(results: list[StepResult]) -> str:
    lines = ["", "PIPELINE SUMMARY", "-" * 70]
    for r in results:
        mark = "OK  " if r.status == "success" else "FAIL"
        lines.append(f"{mark}  {r.name:<12} {r.seconds:>7.1f}s  {r.error or r.detail}")
    lines.append("-" * 70)
    failed = [r.name for r in results if r.status != "success"]
    lines.append(f"{len(results) - len(failed)}/{len(results)} steps succeeded")
    if failed:
        lines.append(f"Failed: {', '.join(failed)}")
    return "\n".join(lines)


def run_with_prefect(**kwargs):
    """Same pipeline as a Prefect flow, with per-task retries."""
    try:
        from prefect import flow, task
    except ImportError as exc:
        raise SystemExit(
            "Prefect is not installed. Install it with `pip install -e '.[orchestration]'`, "
            "or drop --prefect to run the pipeline directly."
        ) from exc

    features_t = task(name="features", retries=1)(step_features)
    train_t = task(name="train", retries=1, retry_delay_seconds=30)(step_train)
    segment_t = task(name="segment", retries=1)(step_segment)
    baskets_t = task(name="baskets", retries=1)(step_baskets)
    drift_t = task(name="drift")(step_drift)

    @flow(name="retail-intelligence", log_prints=True)
    def retail_flow(synthetic: bool = False, source: str | None = None):
        if synthetic:
            task(name="seed")(step_seed_synthetic)()
        elif source:
            task(name="ingest", retries=2)(step_ingest)(source)
        features_t()
        # These four are independent of each other and could be run
        # concurrently; kept sequential so a laptop is not oversubscribed by
        # four parallel model-fitting tasks.
        train_t()
        segment_t()
        baskets_t()
        drift_t()

    return retail_flow(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full retail intelligence pipeline.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", help="Path to the raw workbook or CSV to ingest.")
    source.add_argument(
        "--synthetic", action="store_true", help="Seed generated data instead of ingesting."
    )
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--skip", nargs="*", default=[], choices=list(STEPS))
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--prefect", action="store_true", help="Run as a Prefect flow.")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel workers for training.")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    args = parser.parse_args()

    if args.prefect:
        run_with_prefect(synthetic=args.synthetic, source=args.source)
        return

    results = run_pipeline(
        source=args.source,
        sheet=args.sheet,
        synthetic=args.synthetic,
        skip=tuple(args.skip),
        fail_fast=args.fail_fast,
        features={"top_n": args.top_n},
        train={"jobs": args.jobs, "folds": args.folds},
    )
    logger.info("%s", summarise(results))
    if any(r.status != "success" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
