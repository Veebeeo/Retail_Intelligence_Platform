"""Model catalogue with availability probing.

Prophet and XGBoost are optional extras. A training run should skip a model
whose dependency is missing or broken, log why, and carry on with the rest —
not abort. Prophet in particular is fragile: it ships a compiled Stan backend
that fails on some platforms even after a clean ``pip install``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from retail_intel.forecasting.base import Forecaster
from retail_intel.forecasting.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalDriftForecaster,
    SeasonalNaiveForecaster,
)
from retail_intel.forecasting.models import (
    ProphetForecaster,
    SarimaForecaster,
    XGBoostForecaster,
)
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)

#: Baselines are always available -- pure numpy/pandas.
BASELINE_MODELS: dict[str, Callable[..., Forecaster]] = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "moving_average": MovingAverageForecaster,
    "seasonal_drift": SeasonalDriftForecaster,
}

#: Candidates, mapped to the import that must succeed for them to be usable.
CANDIDATE_MODELS: dict[str, tuple[Callable[..., Forecaster], str]] = {
    "sarima": (SarimaForecaster, "statsmodels"),
    "prophet": (ProphetForecaster, "prophet"),
    "xgboost": (XGBoostForecaster, "xgboost"),
}


def _probe(name: str) -> bool:
    """Return True if ``name`` can actually fit, not merely import.

    Prophet imports fine and then fails when it tries to load its Stan backend,
    so a plain ``import`` check is not enough.
    """
    import importlib

    _, module = CANDIDATE_MODELS[name]
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - any import failure disqualifies it
        logger.warning("Model '%s' unavailable: %s", name, exc)
        return False

    if name == "prophet":
        try:
            from prophet import Prophet

            Prophet()  # constructing it is what loads the Stan backend
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Model 'prophet' imported but its Stan backend failed to load (%s). "
                "Skipping it. Fix with: python -c 'import cmdstanpy; "
                "cmdstanpy.install_cmdstan(overwrite=True)'",
                exc,
            )
            return False
    return True


@lru_cache(maxsize=1)
def available_models() -> tuple[str, ...]:
    """Names of every model that can be fitted in this environment."""
    usable = [n for n in CANDIDATE_MODELS if _probe(n)]
    names = (*BASELINE_MODELS, *usable)
    logger.info("Available forecasters: %s", ", ".join(names))
    return names


def build(name: str, seasonal_period: int = 52, **kwargs) -> Forecaster:
    """Instantiate a forecaster by name."""
    if name in BASELINE_MODELS:
        return BASELINE_MODELS[name](seasonal_period=seasonal_period, **kwargs)
    if name in CANDIDATE_MODELS:
        factory, _ = CANDIDATE_MODELS[name]
        return factory(seasonal_period=seasonal_period, **kwargs)
    raise KeyError(f"Unknown forecaster '{name}'. Known: {sorted({*BASELINE_MODELS, *CANDIDATE_MODELS})}")


def build_all(seasonal_period: int = 52, include: tuple[str, ...] | None = None) -> list[Forecaster]:
    """Instantiate every available forecaster, or the subset in ``include``."""
    names = include or available_models()
    return [build(n, seasonal_period) for n in names if n in {*BASELINE_MODELS, *CANDIDATE_MODELS}]
