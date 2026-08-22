"""Structured logging shared by pipelines and the API."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Idempotently install a single stdout handler.

    Container platforms collect stdout, so everything goes there rather than to
    a file.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Read the level straight from the environment rather than via Settings:
    # logging must work even when the database is unconfigured, otherwise a
    # config error can never be logged.
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These libraries are chatty at INFO and drown out pipeline output.
    for noisy in ("cmdstanpy", "prophet", "matplotlib", "sqlalchemy.engine", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
