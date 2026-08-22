"""Retail Intelligence API.

What changed from the previous version, in order of importance:

1. ``/forecast`` serves an actual trained model. It previously returned
   ``recent_average * (1 + 0.02 * week)`` — a hard-coded 2%-per-week growth
   curve — while the documentation described a SARIMA production model. No
   trained model existed anywhere in the repository.
2. Both endpoints interpolated user input into SQL with f-strings. Every query
   is now parameterised, and inputs are validated before they reach the
   database.
3. The service reports itself degraded when the database or models are
   unavailable, instead of returning a fixed "healthy".
4. Models load once at start-up rather than per request.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import customers, forecast, monitoring
from retail_intel import __version__
from retail_intel.forecasting.serving import try_load_bundle
from retail_intel.logging_conf import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

DESCRIPTION = """
Demand forecasting, inventory policy and customer value modelling over retail
transaction data.

**Forecasts** come from a per-SKU champion model chosen by rolling-origin
backtesting against a seasonal-naive baseline. Every forecast response reports
which model produced it and how it scored, so you can tell a well-validated
forecast from a weak one. A SKU whose candidates all lost to the baseline is
served *by* the baseline.

**Inventory** endpoints convert a forecast and its error distribution into a
reorder point and safety stock. Supply both cost parameters and the service
level is derived from the newsvendor critical ratio rather than assumed.

**Customer** endpoints combine RFM segments with BG/NBD and Gamma-Gamma
lifetime value, so you can rank by predicted future value and churn risk rather
than by past spend alone.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at start-up.

    Unpickling per request would add tens of milliseconds and defeat the point
    of persisting them. A missing bundle is logged, not fatal: the analytics
    and customer endpoints still work without it.
    """
    bundle = try_load_bundle()
    if bundle:
        logger.info(
            "API ready with champion bundle v%d covering %d SKUs",
            bundle.version,
            len(bundle.models),
        )
    else:
        logger.warning("API starting WITHOUT forecasting models; /forecast will return 503")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Retail Intelligence API",
    description=DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# The Streamlit dashboard is served from a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(monitoring.router)
app.include_router(forecast.router)
app.include_router(customers.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the traceback, return a generic message.

    Database errors can carry table names, column names and fragments of the
    connection string. Those belong in the logs, not in an HTTP response.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. The failure has been logged."},
    )


@app.get("/", tags=["monitoring"])
def read_root():
    return {
        "service": "Retail Intelligence API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": {
            "forecasting": ["/forecast", "/models", "/models/skus", "/models/{stock_code}"],
            "inventory": ["/inventory/policy"],
            "customers": [
                "/segment",
                "/segments/summary",
                "/customers/at-risk",
                "/recommend/{stock_code}",
            ],
            "monitoring": ["/health", "/drift", "/analytics/revenue", "/analytics/top-skus"],
        },
    }
