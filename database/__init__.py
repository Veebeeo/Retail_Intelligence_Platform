"""Backwards-compatible shim.

The schema moved to ``retail_intel.data.models``. This module re-exports it so
older imports (``from database.models import engine``) keep working.
"""

from retail_intel.data.models import Base, Transaction  # noqa: F401
