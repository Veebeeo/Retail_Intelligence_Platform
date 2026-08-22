"""Shared API dependencies."""

from __future__ import annotations

from fastapi import HTTPException, status

from retail_intel.forecasting.serving import ChampionBundle, ModelNotAvailable, load_bundle


def get_bundle() -> ChampionBundle:
    """Return the loaded champion bundle, or fail with a clear 503.

    The previous API had no models at all and papered over it by returning a
    made-up growth curve. If models are missing now, callers are told.
    """
    try:
        return load_bundle()
    except ModelNotAvailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Forecasting models are not loaded on this instance. "
                f"{exc} Until then, forecast endpoints are unavailable."
            ),
        ) from exc
