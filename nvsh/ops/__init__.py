"""Operation table: the 16 registered operations, their argument specs, and validation.

Re-exports the frozen dataclasses and the public API so callers can use
``import nvsh.ops`` as a single entry-point.
"""

from __future__ import annotations

from nvsh.ops._model import ArgSpec, Operation, ValidationError
from nvsh.ops.table import OPERATIONS, get, names, validate

__all__ = [
    "ArgSpec",
    "Operation",
    "OPERATIONS",
    "ValidationError",
    "get",
    "names",
    "validate",
]
