"""Frozen dataclasses for the operation table model.

``ArgSpec`` describes a single argument (string or choice).
``Operation`` describes a registered operation (name, description, read-only
flag, tuple of arguments).
``ValidationError`` is returned by ``validate()`` when the caller passes
bad input; it never raises.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArgSpec:
    """One argument declaration on an operation.

    ``kind`` is ``"str"`` for a free-form string value or ``"choice"``
    for a fixed set of allowed values stored in ``choices``.
    """

    name: str
    kind: str
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class Operation:
    """One registered operation and its argument specification.

    ``read_only`` is ``True`` when the operation is safe to call without
    side-effects; ``False`` means it may mutate state.
    """

    name: str
    description: str
    read_only: bool
    args: tuple[ArgSpec, ...] = ()


@dataclass(frozen=True)
class ValidationError:
    """A validation failure returned by ``validate()`` (never raises).

    ``code`` is one of the recognised strings; ``message`` is a one-line
    human-readable description that names the operation and argument.
    """

    code: str
    message: str
