"""Platform detection: what machine nvsh is running on, file-first.

Public surface: :func:`detect`, :class:`Platform`, :class:`Value`.

``detect()`` never raises for a missing file, missing binary or unparseable
content — every fact nvsh looks for comes back as a :class:`Value`, present
or absent, always carrying the source that was checked. Nothing is silently
omitted. See ``docs/platforms.md`` for the full source-to-value mapping and
the commands used to verify each platform.
"""

from __future__ import annotations

from ._detect import detect
from ._model import Platform, Value

__all__ = ["detect", "Platform", "Value"]
