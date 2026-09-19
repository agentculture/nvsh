"""Tests for nvsh.tiers.memfloor: memory floor check on /proc/meminfo.

Covers spec targets c33, h28: read MemAvailable from /proc/meminfo via an
injected reader, never require psutil, and return a simple FloorResult
telling a tier whether it may load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from nvsh.tiers.memfloor import (
    MEMINFO_PATH,
    available_mb,
    check_floor,
)


def _fake_reader(text: str) -> Callable[[], str]:
    """Return a reader that echoes *text* (or raises OSError when *text* is None)."""
    if text is None:

        def failing_reader() -> str:
            raise OSError("no /proc/meminfo")

        return failing_reader  # type: ignore[return-value]

    def good_reader() -> str:
        return text

    return good_reader


# -- parsing /proc/meminfo --


def test_available_mb_parses_kb_to_mb():
    """MemAvailable: 2048000 kB -> 2000."""
    text = "MemAvailable:    2048000 kB\n"
    reader = _fake_reader(text)
    assert available_mb(reader) == 2000


def test_missing_line_passes_with_note():
    """No MemAvailable line -> None, check_floor treats it as a note."""
    text = "MemTotal:       16384000 kB\n"
    reader = _fake_reader(text)
    assert available_mb(reader) is None
    result = check_floor(512, reader)
    assert result.ok is True
    assert result.available_mb is None
    assert "could not read" in result.status.lower()


def test_garbage_number_passes_with_note():
    """MemAvailable: lots kB -> None (unparseable), check_floor treats it as a note."""
    text = "MemAvailable: lots kB\n"
    reader = _fake_reader(text)
    assert available_mb(reader) is None
    result = check_floor(512, reader)
    assert result.ok is True
    assert result.available_mb is None
    assert "could not read" in result.status.lower()


# -- floor decision logic --


def test_below_floor_declines_with_one_status_line():
    """When available < floor: ok=False, status has no newline, contains both numbers."""
    text = "MemAvailable:    256000 kB\n"  # 250 MB
    reader = _fake_reader(text)
    floor_mb = 512
    result = check_floor(floor_mb, reader)
    assert result.ok is False
    assert result.available_mb == 250
    assert "\n" not in result.status
    assert "250" in result.status
    assert str(floor_mb) in result.status


def test_at_or_above_floor_passes():
    """available == floor passes (ok=True)."""
    text = "MemAvailable:    524288 kB\n"  # 512 MB
    reader = _fake_reader(text)
    floor_mb = 512
    result = check_floor(floor_mb, reader)
    assert result.ok is True
    assert result.available_mb == 512
    assert result.status == ""


def test_floor_zero_disables_check():
    """floor_mb <= 0 always returns ok=True regardless of available memory."""
    text = "MemAvailable:        0 kB\n"
    reader = _fake_reader(text)
    result = check_floor(0, reader)
    assert result.ok is True
    assert result.available_mb == 0

    result_neg = check_floor(-1, reader)
    assert result_neg.ok is True


def test_unreadable_meminfo_passes_with_note():
    """Reader raises OSError -> ok=True, status mentions not checking."""
    reader = _fake_reader(None)
    result = check_floor(512, reader)
    assert result.ok is True
    assert result.available_mb is None
    assert "not checked" in result.status.lower()


# -- default reader on the host machine --


def test_default_reader_on_this_machine():
    """On a Linux box with /proc/meminfo, available_mb() returns a positive int."""
    if not Path(MEMINFO_PATH).exists():
        pytest.skip(f"{MEMINFO_PATH} does not exist on this machine")
    result = available_mb()
    assert isinstance(result, int)
    assert result > 0


def test_undecodable_meminfo_passes_with_note():
    def reader() -> str:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    result = check_floor(1024, reader)
    assert result.ok is True
    assert result.available_mb is None
