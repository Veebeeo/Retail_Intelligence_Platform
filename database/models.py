"""Deprecated location. See ``retail_intel.data.models``.

Kept because the README and older notebooks reference it. Unlike the original,
importing this does *not* connect to a database or run DDL as a side effect.
"""

from retail_intel.data.models import (  # noqa: F401
    Base,
    CustomerSegment,
    Transaction,
    WeeklyFeature,
    create_all,
)
from retail_intel.db import get_engine

__all__ = ["Base", "Transaction", "CustomerSegment", "WeeklyFeature", "create_all", "get_engine"]


def __getattr__(name: str):
    # `from database.models import engine` used to work because the module built
    # an engine at import time. Resolve it lazily instead.
    if name == "engine":
        return get_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
