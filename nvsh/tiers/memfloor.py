"""Memory floor check: read MemAvailable from /proc/meminfo.

No psutil — pure stdlib.  Used by the tier loader to skip loading a tier
when the local machine has less free memory than a configured floor.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

MEMINFO_PATH = "/proc/meminfo"


Reader = Callable[[], str]  # returns the text of /proc/meminfo; may raise OSError


@dataclasses.dataclass(frozen=True)
class FloorResult:
    """Result of a memory floor check.

    ``ok`` is True when the tier may load.  ``status`` is a one-line note
    for the operator — empty string when there is nothing to say.
    """

    ok: bool
    available_mb: int | None
    status: str


def default_reader() -> str:
    """Read ``MEMINFO_PATH`` and return its UTF-8 text."""
    return Path(MEMINFO_PATH).read_text(encoding="utf-8")


def available_mb(reader: Reader = default_reader) -> int | None:
    """Return available memory in MiB (floor division), or None on error.

    Finds the line that starts with ``MemAvailable:`` and parses the kB
    value.  Returns ``None`` if the reader raises, the line is missing, or
    the number cannot be parsed.  Never raises.
    """
    try:
        text = reader()
    except (OSError, ValueError):  # ValueError covers a UnicodeDecodeError
        return None

    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    kb = int(parts[1])
                    return kb // 1024
                except (ValueError, IndexError):
                    return None
    return None


def check_floor(floor_mb: int, reader: Reader = default_reader) -> FloorResult:
    """Check whether available memory meets *floor_mb*.

    Rules
    -----
    * ``floor_mb <= 0``  -> always ok (check disabled).
    * ``available_mb`` is ``None``  -> ok, note that the check could not run.
    * ``available < floor_mb``  -> decline with a status line.
    * otherwise  -> ok, empty status.

    Never raises.
    """
    if floor_mb <= 0:
        # Check disabled — let the tier load regardless.
        try:
            avail = available_mb(reader)
        except Exception:
            avail = None
        return FloorResult(ok=True, available_mb=avail, status="")

    avail = available_mb(reader)

    if avail is None:
        return FloorResult(
            ok=True,
            available_mb=None,
            status="memory floor not checked: could not read /proc/meminfo",
        )

    if avail < floor_mb:
        return FloorResult(
            ok=False,
            available_mb=avail,
            status=f"local tier skipped: {avail} MB available, floor is {floor_mb} MB",
        )

    return FloorResult(ok=True, available_mb=avail, status="")
