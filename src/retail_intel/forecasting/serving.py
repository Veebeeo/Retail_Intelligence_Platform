"""Loading trained champions at serving time.

The bundle holds **precomputed forecast paths**, not fitted model objects.
Because every champion is univariate with no exogenous inputs, and no new data
arrives between retrains, the forecast for any horizon is fixed the moment the
model is fitted — so storing the path is equivalent to storing the model, and
serving becomes an array slice rather than a model call.

That choice is what keeps the bundle small enough to ship. A fitted SARIMAX
result with a 52-period seasonal term pickles to roughly 220 MB because it
carries the Kalman smoother state for every observation; an earlier version of
this bundle was 445 MB for twenty SKUs. It also means the serving image needs
neither statsmodels nor xgboost.

If no bundle exists the loader says so plainly — the API then reports itself
degraded rather than quietly inventing numbers, which is what the version
before this one did.
"""

from __future__ import annotations

import json
import pickle
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from retail_intel.config import get_settings
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)

_LOCK = threading.Lock()
_BUNDLE: ChampionBundle | None = None


@dataclass
class ChampionBundle:
    #: stock_code -> {"yhat": [...], "yhat_lower": [...], "yhat_upper": [...]}
    forecasts: dict[str, dict[str, list[float]]]
    meta: dict[str, dict]
    seasonal_period: int
    horizon: int
    max_horizon: int
    trained_at: str
    version: int
    source: Path

    def has(self, stock_code: str) -> bool:
        return stock_code in self.forecasts

    @property
    def skus(self) -> list[str]:
        return sorted(self.forecasts)

    def model_mix(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.meta.values():
            counts[m["model"]] = counts.get(m["model"], 0) + 1
        return counts


class ModelNotAvailable(RuntimeError):
    """Raised when no trained bundle is on disk."""


def load_bundle(path: Path | None = None, force: bool = False) -> ChampionBundle:
    """Load (and cache) the champion bundle."""
    global _BUNDLE
    with _LOCK:
        if _BUNDLE is not None and not force:
            return _BUNDLE

        settings = get_settings()
        target = Path(path) if path else settings.model_path / "champions.pkl"
        if not target.exists():
            raise ModelNotAvailable(
                f"No trained models at {target}. Run `python -m retail_intel.forecasting.train` "
                "(or `make train`) first."
            )

        try:
            with target.open("rb") as fh:
                raw = pickle.load(fh)  # noqa: S301 - our own artifact, from the training job
        except ModuleNotFoundError as exc:
            # A bundle is a pickle of fitted model objects, so loading it needs
            # whatever library produced the champion. Say which one is missing
            # rather than surfacing a traceback from inside pickle.
            raise ModelNotAvailable(
                f"The champion bundle at {target} contains a model from '{exc.name}', "
                f"which is not installed in this environment. Either install {exc.name}, "
                "or retrain with that model excluded from the candidate set."
            ) from exc

        if "forecasts" not in raw:
            raise ModelNotAvailable(
                f"The bundle at {target} was written by an older version that stored "
                "model objects. Re-run the training pipeline to regenerate it."
            )

        _BUNDLE = ChampionBundle(
            forecasts=raw["forecasts"],
            meta=raw["meta"],
            seasonal_period=raw.get("seasonal_period", 52),
            horizon=raw.get("horizon", 4),
            max_horizon=raw.get("max_horizon", 26),
            trained_at=raw.get("trained_at", "unknown"),
            version=int(raw.get("version", 1)),
            source=target,
        )
        logger.info(
            "Loaded champion bundle v%d (%d SKUs, %d-week paths, trained %s): %s",
            _BUNDLE.version,
            len(_BUNDLE.forecasts),
            _BUNDLE.max_horizon,
            _BUNDLE.trained_at,
            _BUNDLE.model_mix(),
        )
        return _BUNDLE


def try_load_bundle() -> ChampionBundle | None:
    """Load the bundle, returning None instead of raising. For start-up probes."""
    try:
        return load_bundle()
    except ModelNotAvailable as exc:
        logger.warning("%s", exc)
        return None


def reset_cache() -> None:
    global _BUNDLE
    with _LOCK:
        _BUNDLE = None


def forecast(stock_code: str, horizon: int = 4, bundle: ChampionBundle | None = None) -> dict:
    """Forecast one SKU with its champion model.

    Returns the point forecast, a 95% interval and the provenance a consumer
    needs to judge it: which model produced it, how it scored in backtesting,
    and when it was trained.
    """
    bundle = bundle or load_bundle()
    if not bundle.has(stock_code):
        raise KeyError(stock_code)
    if horizon > bundle.max_horizon:
        raise ValueError(
            f"Horizon {horizon} exceeds the {bundle.max_horizon} weeks precomputed at "
            "training time. Raise MAX_FORECAST_HORIZON and retrain."
        )

    path = bundle.forecasts[stock_code]
    meta = bundle.meta.get(stock_code, {})

    last_week = pd.Timestamp(meta.get("last_week", datetime.now().date()))
    weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")

    return {
        "stock_code": stock_code,
        "horizon_weeks": horizon,
        "model": meta.get("model", "unknown"),
        "model_version": bundle.version,
        "trained_at": bundle.trained_at,
        "backtest_mase": meta.get("backtest_mase"),
        "baseline_mase": meta.get("baseline_mase"),
        "improvement_vs_baseline_pct": meta.get("improvement_pct"),
        "residual_std": meta.get("residual_std"),
        "predictions": [
            {
                "week_horizon": i + 1,
                "week_starting": weeks[i].date().isoformat(),
                "predicted_quantity": round(float(path["yhat"][i]), 2),
                "lower_95": round(float(path["yhat_lower"][i]), 2),
                "upper_95": round(float(path["yhat_upper"][i]), 2),
            }
            for i in range(horizon)
        ],
    }


def forecast_array(
    stock_code: str, horizon: int, bundle: ChampionBundle | None = None
) -> np.ndarray:
    """Just the point forecast, for the inventory calculators."""
    bundle = bundle or load_bundle()
    if not bundle.has(stock_code):
        raise KeyError(stock_code)
    return np.asarray(bundle.forecasts[stock_code]["yhat"][:horizon], dtype=float)


def bundle_summary(bundle: ChampionBundle | None = None) -> dict:
    """Registry-level view for the /models endpoint and the dashboard."""
    bundle = bundle or load_bundle()
    mases = [m["backtest_mase"] for m in bundle.meta.values() if m.get("backtest_mase") is not None]
    beating = [m for m in bundle.meta.values() if m.get("model") != "seasonal_naive"]
    return {
        "version": bundle.version,
        "trained_at": bundle.trained_at,
        "n_skus": len(bundle.forecasts),
        "seasonal_period": bundle.seasonal_period,
        "model_mix": bundle.model_mix(),
        "mean_backtest_mase": round(float(np.mean(mases)), 4) if mases else None,
        "median_backtest_mase": round(float(np.median(mases)), 4) if mases else None,
        "pct_skus_beating_baseline": round(len(beating) / max(len(bundle.meta), 1) * 100, 1),
        "source": str(bundle.source),
    }


def load_meta_json(path: Path | None = None) -> dict:
    """Read the sidecar JSON without unpickling the models."""
    settings = get_settings()
    target = Path(path) if path else settings.model_path / "champions_meta.json"
    if not target.exists():
        raise ModelNotAvailable(f"No metadata at {target}")
    return json.loads(target.read_text())
