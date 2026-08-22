"""Database access.

Two rules this module exists to enforce:

1. One engine per process, created lazily. The previous version built an engine
   at import time in several modules, which meant importing anything required a
   reachable database.
2. Callers never build SQL by hand. :func:`read_sql` takes bound parameters
   only, so a stock code arriving from an HTTP request cannot alter the query.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from retail_intel.config import get_settings
from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    url = settings.database_url
    assert url is not None  # guaranteed by Settings validation

    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # SQLite has no pooling story worth configuring, and the API's
        # threadpool needs cross-thread access.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs |= {"pool_size": 5, "max_overflow": 10, "pool_recycle": 1800}

    logger.info("Creating engine for %s", _redact(url))
    return create_engine(url, **kwargs)


def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def read_sql(query: str, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Run a parameterised SELECT.

    ``query`` must use ``:name`` placeholders. Interpolating values into the
    string instead is what made the original ``/forecast`` and ``/segment``
    endpoints injectable.
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=dict(params or {}))


def write_table(df: pd.DataFrame, table: str, if_exists: str = "replace") -> int:
    """Write a DataFrame to ``table`` and return the row count."""
    with get_engine().begin() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False, chunksize=10_000)
    logger.info("Wrote %d rows to %s", len(df), table)
    return len(df)


def table_exists(table: str) -> bool:
    from sqlalchemy import inspect

    return inspect(get_engine()).has_table(table)


def _redact(url: str) -> str:
    """Strip credentials so connection strings are safe to log."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    return f"{scheme}://***@{rest.rpartition('@')[2]}"
