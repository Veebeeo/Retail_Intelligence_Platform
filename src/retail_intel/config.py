"""Central configuration.

Every tunable lives here so pipelines, the API and the dashboard agree on
things like the forecast horizon or the seasonal period. Values come from the
environment (or a local ``.env``); the defaults are the ones used to produce
the numbers reported in the README.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- storage -----------------------------------------------------------
    database_url: str | None = None
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None
    postgres_host: str = "postgres_db"
    postgres_port: int = 5432

    # --- tracking ----------------------------------------------------------
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "retail_demand_forecasting"
    registered_model_name: str = "retail_demand_forecaster"
    model_dir: Path = Path("./models")
    report_dir: Path = Path("./reports")

    # --- modelling ---------------------------------------------------------
    top_n_skus: int = Field(100, description="SKUs retained by sales volume for modelling.")
    seasonal_period: int = Field(
        52, description="Weeks in a seasonal cycle. Weekly retail data is annual."
    )
    forecast_horizon: int = Field(4, description="Weeks ahead the champion is selected on.")
    min_train_weeks: int = Field(
        26, description="Minimum history before a SKU is eligible for modelling."
    )
    backtest_folds: int = Field(5, description="Rolling-origin evaluation folds.")
    random_seed: int = 42

    # --- inventory economics ----------------------------------------------
    lead_time_weeks: int = 2
    service_level: float = Field(0.95, ge=0.5, lt=1.0)
    holding_cost_per_unit_week: float = 0.15
    stockout_cost_per_unit: float = 2.50

    # --- runtime -----------------------------------------------------------
    api_base_url: str = "http://localhost:8000"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _resolve_database_url(self) -> Settings:
        """Assemble a URL from the POSTGRES_* parts when DATABASE_URL is unset.

        Raising here rather than at import time in each module means a missing
        credential surfaces once, with a message that says what to do.
        """
        if self.database_url:
            # SQLAlchemy 2.x removed the bare `postgres://` alias that some
            # managed providers still hand out.
            if self.database_url.startswith("postgres://"):
                self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
            return self

        if self.postgres_user and self.postgres_password and self.postgres_db:
            self.database_url = (
                f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
            return self

        raise ValueError(
            "No database configuration found. Set DATABASE_URL, or all of "
            "POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB. "
            "Copy .env.example to .env to get started."
        )

    @property
    def model_path(self) -> Path:
        return self.model_dir / "champions"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
