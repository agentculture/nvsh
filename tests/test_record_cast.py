"""Tests for scripts/record-cast.py.

Acceptance criteria covered (task t4):

* ``record`` sets the pty window size with ``TIOCSWINSZ`` to
  ``--cols``/``--rows`` before exec (read back here with ``stty size``
  inside the child).
* ``--timestamp N`` pins the header timestamp; ``--clean-env`` starts the
  child from an allowlist (``PATH``, ``TERM``, ``HOME``, ``LANG``,
  ``XDG_*``, ``NVSH_*``, ``COLUMNS``, ``LINES``) plus ``--env`` overrides.
* Two ``--feed`` runs of ``'printf hi\\r|1'`` with ``--timestamp 0`` produce
  casts that are identical once each event's per-line stamp is stripped.

``scripts/`` is not an installed package, so the script under test is
invoked as a subprocess with ``sys.executable`` rather than imported.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "record-cast.py"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" and sys.platform != "darwin",
    reason="record-cast.py needs a real pty (pty.fork)",
)


def _shell() -> list[str]:
    if shutil.which("bash"):
        return ["bash", "--norc", "--noprofile", "-i"]
    return ["sh", "-i"]


def _record(out: Path, *extra: str) -> subprocess.CompletedProcess:
    args = [
        sys.executable,
        str(SCRIPT),
        "record",
        str(out),
        "--feed",
        r"printf hi\r|1",
        "--timestamp",
        "0",
        "--cols",
        "80",
        "--rows",
        "24",
        "--clean-env",
        "--env",
        "PS1=RC$ ",
        "--settle",
        "2",
        *extra,
        "--",
        *_shell(),
    ]
    return subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)


def _load_events(path: Path) -> tuple[dict, list[list]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = json.loads(lines[0])
    events = [json.loads(line) for line in lines[1:]]
    return header, events


def test_record_sets_window_size_via_tiocswinsz(tmp_path):
    out = tmp_path / "size.cast"
    _record(out, "--feed", r"printf hi\r|1", "--feed", r"stty size\r|1")
    _, events = _load_events(out)
    combined = "".join(event[2] for event in events)
    assert "24 80" in combined


def test_record_timestamp_and_clean_env(tmp_path, monkeypatch):
    # Qodo 5/8 (PR #14): the clean env must actually drop what is not on
    # the allowlist -- an NVSH_* behaviour knob and an XDG_* directory both
    # vanish, while an explicit --env override survives.
    monkeypatch.setenv("NVSH_NO_DAEMON", "leak")
    monkeypatch.setenv("XDG_CACHE_HOME", "/leaked/cache")
    out = tmp_path / "env.cast"
    _record(
        out,
        "--env",
        "DEMO_KEEP=kept",
        "--feed",
        r"echo A${NVSH_NO_DAEMON}B${XDG_CACHE_HOME}C $DEMO_KEEP\r|1",
    )
    header, events = _load_events(out)
    assert header["timestamp"] == 0
    assert header["width"] == 80
    assert header["height"] == 24
    combined = "".join(data for _stamp, _kind, data in events)
    assert "ABC kept" in combined
    assert "leak" not in combined


def test_scrub_survives_a_token_split_across_pty_reads(tmp_path):
    # Qodo 3 (PR #14): a token straddling two pty reads must still be
    # replaced. The child prints the halves with a pause between them.
    rules = tmp_path / "rules"
    rules.write_text("SECRETHOST=[host]\n", encoding="utf-8")
    out = tmp_path / "split.cast"
    _record(
        out,
        "--replace-from",
        str(rules),
        "--feed",
        r"printf SECRE; sleep 0.4; printf THOST; echo\r|2",
    )
    _, events = _load_events(out)
    combined = "".join(data for _stamp, _kind, data in events)
    assert "[host]" in combined
    assert "SECRETHOST" not in combined.replace("printf SECRE; sleep 0.4; printf THOST", "")


def test_two_feed_runs_are_identical_except_stamps(tmp_path):
    out1 = tmp_path / "run1.cast"
    out2 = tmp_path / "run2.cast"
    _record(out1)
    _record(out2)

    header1, events1 = _load_events(out1)
    header2, events2 = _load_events(out2)

    # The header carries the pinned timestamp, so it should match exactly.
    assert header1 == header2

    # Each event is [stamp, kind, data]. The *content* (kind + data) must
    # match once the per-line stamp is stripped, but two runs of the same
    # pty session are not guaranteed to chunk their reads at identical
    # byte boundaries -- a single burst of child output (e.g. the shell's
    # "exit" echo followed by the bracketed-paste-off sequence) can land
    # in one os.read() on one run and split across two on another, purely
    # from scheduler timing. That is not a determinism bug in what was
    # recorded, only in how it was batched, so compare the concatenated
    # "kind + data" stream rather than requiring the same event count.
    def _stream(events: list[list]) -> str:
        kinds = [event[1] for event in events]
        assert set(kinds) <= {"o"}, "unexpected event kind in a --feed recording"
        return "".join(event[2] for event in events)

    assert _stream(events1) == _stream(events2)
