"""Capture layer: session log, OSC 133 slicing, tmux pipe-pane, bounding.

The bash hook cannot see a failed command's output directly (see
``docs/architecture.md``'s "Output capture reuses Ghostty's own markers"
section): each interactive session runs under a per-session typescript
(``script -qfc "$BASH" "$log"``) or, inside tmux, under ``tmux pipe-pane -o``
writing to the same log path. Ghostty's shell integration emits OSC 133
``C`` (command start) / ``D`` (command end) markers around every command's
real output; :func:`last_slice` recovers exactly the last failed command's
output by slicing the log between the last ``C`` and the following ``D``,
never by re-running anything.

This module owns only the Python side. The bash side (the ``exec`` into
``script``, the ``EXIT`` trap that calls :func:`cleanup`, and the
``tmux pipe-pane`` invocation) lives in ``nvsh/shell/hook.bash`` (task t20)
and calls :func:`wrapper_command` / :func:`tmux_pipe_command` to render the
exact shell text to run.

Pipeline for :func:`last_slice`, in order (see the "bound, then strip, then
redact" rule in ``CLAUDE.md``'s device-context section):

1. Locate the last OSC 133 ``C``..``D`` region in the raw bytes.
2. Bound it to ``limit`` bytes (head + tail, with a truncation marker).
3. Strip terminal escape sequences (OSC, CSI, single-char ESC) and
   normalise CR.
4. Decode UTF-8 with ``errors="replace"`` (never raises on invalid bytes).
5. Redact via :func:`nvsh.redact.redact_report` — the single choke point
   for anything leaving the process.

The session log itself is never read into the returned :class:`Slice`: only
the bounded, stripped, redacted slice is ever handed to a caller (an agent
backend, ``--show-context``, or ``nvsh capture --show``), and the log's own
path never appears in that text.

Stdlib only, no third-party dependency (``dependencies = []`` in
``pyproject.toml`` stays empty).
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from nvsh.redact import redact_report

#: Default cap on the bytes handed back by :func:`last_slice`: half becomes
#: the head, half the tail, with a truncation marker in between.
DEFAULT_LIMIT = 64 * 1024

#: OSC 133 "C" (command start) marker: ``ESC ] 133 ; C`` then optional
#: parameters up to the terminator, which is either BEL (``\x07``) or the
#: two-byte String Terminator ``ESC \``.
_OSC_133_C_RE = re.compile(rb"\x1b\]133;C[^\x07\x1b]*(?:\x07|\x1b\\)")

#: OSC 133 "D" (command end) marker: ``ESC ] 133 ; D`` then ``;<exit code>``
#: and optional further parameters, terminated the same way as "C".
_OSC_133_D_RE = re.compile(rb"\x1b\]133;D[^\x07\x1b]*(?:\x07|\x1b\\)")

#: Any OSC (Operating System Command) sequence: ``ESC ]`` up to BEL or ST.
_OSC_RE = rb"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"

#: A CSI (Control Sequence Introducer) sequence: ``ESC [`` parameter/
#: intermediate bytes, then one final byte in ``@``-``~``.
_CSI_RE = rb"\x1b\[[0-9;?]*[ -/]*[@-~]"

#: A single-character ESC sequence that is neither OSC nor CSI (e.g. ``ESC
#: c`` full reset, ``ESC =``/``ESC >`` keypad mode).
_SIMPLE_ESC_RE = rb"\x1b[@-Z\\-_]"

_ESCAPE_RE = re.compile(b"|".join((_OSC_RE, _CSI_RE, _SIMPLE_ESC_RE)))

_TRUNCATION_TEMPLATE = "\n[... {n} bytes truncated ...]\n"


@dataclass(frozen=True)
class LogHandle:
    """A session log opened by :func:`open_session_log`."""

    path: Path
    pid: int


@dataclass(frozen=True)
class WrapperCommand:
    """The ``script(1)`` invocation the bash hook execs into.

    ``argv`` is the argv vector; ``bash_line`` is the ready-to-paste bash
    snippet (exporting ``NVSH_WRAPPED=1`` first, so a re-sourced hook never
    nests a second ``script`` inside the first).
    """

    argv: list[str]
    bash_line: str


@dataclass(frozen=True)
class Slice:
    """The bounded, stripped, redacted output of one failed command."""

    text: str
    status: str  # "ok" | "partial" | "truncated" | "no capture"
    source: str  # "script" | "tmux" | "none"
    redaction_rules: list[str] = field(default_factory=list)
    bytes_total: int = 0


def _no_capture(source: str) -> Slice:
    return Slice(text="", status="no capture", source=source, redaction_rules=[], bytes_total=0)


# ---------------------------------------------------------------------------
# Log location
# ---------------------------------------------------------------------------


def _runtime_dir(env: dict) -> Path:
    """``$XDG_RUNTIME_DIR/nvsh``, falling back to ``/tmp/nvsh-<uid>`` (0700)."""
    xdg = env.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "nvsh"
    # Per-uid, not a shared/predictable path an attacker could pre-create:
    # open_session_log() still creates this directory mode 0700 and the log
    # file mode 0600 before anything is written to it.
    return Path(f"/tmp/nvsh-{os.getuid()}")  # nosec B108


def session_log_path(env: dict, pid: int) -> Path:
    """Return the session log path for shell ``pid`` under ``env``.

    ``$XDG_RUNTIME_DIR/nvsh/<pid>.log`` when ``XDG_RUNTIME_DIR`` is set,
    else ``/tmp/nvsh-<uid>/<pid>.log``.
    """
    return _runtime_dir(env) / f"{pid}.log"


def capture_source(env: dict) -> str:
    """Return which capture source is active for the current shell, from ``env``.

    ``"tmux"`` when ``$TMUX`` is set (pipe-pane is the source, regardless of
    whether the shell is also ``script``-wrapped — tmux takes precedence
    since it is the outer multiplexer); ``"script"`` when ``NVSH_WRAPPED``
    is set (the hook already execed into ``script``); ``"none"`` otherwise.
    """
    if env.get("TMUX"):
        return "tmux"
    if env.get("NVSH_WRAPPED"):
        return "script"
    return "none"


# ---------------------------------------------------------------------------
# Session log lifecycle
# ---------------------------------------------------------------------------


def open_session_log(env: dict, pid: int) -> LogHandle | None:
    """Create and return a fresh session log, or ``None`` when capture must not start.

    Refuses (returns ``None``) when:

    * ``NVSH_WRAPPED`` is already set — a second ``script`` would nest.
    * ``$TMUX`` or ``$STY`` is set — the shell is already inside a
      multiplexer, whose own ``pipe-pane`` (or, for screen, a different
      capture path) is the correct source instead of a nested ``script``.
    * ``script`` is not on ``PATH`` — nothing to exec into.

    On success, creates the log directory (mode 0700) and an empty log file
    (mode 0600) and returns a :class:`LogHandle`. Never raises: a directory
    or file creation failure is treated the same as "refuse to start" by
    letting the underlying ``OSError`` propagate is NOT done here — callers
    (the bash hook) already guard capture as best-effort, but to keep this
    a pure decision function any unexpected ``OSError`` from creating the
    directory/file is left to the caller, since a permissions problem here
    is a genuine environment fault worth surfacing via ``nvsh doctor``.
    """
    if env.get("NVSH_WRAPPED"):
        return None
    if env.get("TMUX") or env.get("STY"):
        return None
    if shutil.which("script") is None:
        return None

    path = session_log_path(env, pid)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.close(fd)
    os.chmod(path, 0o600)
    return LogHandle(path=path, pid=pid)


def cleanup(log: LogHandle) -> None:
    """Remove the session log on shell exit (the ``EXIT`` trap). Never raises."""
    try:
        log.path.unlink(missing_ok=True)
    except OSError:
        pass


def wrapper_command(env: dict, log: Path) -> WrapperCommand:
    """Render the ``script(1)`` wrapper invocation for the bash hook.

    ``env`` is accepted for symmetry with the rest of this module's API
    (and so a future revision can vary the rendered line by environment)
    but is not currently consulted.
    """
    argv = ["script", "-qfc", "$BASH", str(log)]
    bash_line = f'export NVSH_WRAPPED=1\nexec script -qfc "$BASH" "{log}"'
    return WrapperCommand(argv=argv, bash_line=bash_line)


def tmux_pipe_command(log: Path) -> str:
    """Render the ``tmux pipe-pane -o`` invocation that appends to ``log``."""
    return f"tmux pipe-pane -o \"cat >> '{log}'\""


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def _locate_last_region(data: bytes) -> tuple[bytes, str] | None:
    """Return ``(raw_slice, "ok"|"partial")`` for the last C..D region, or ``None``.

    A session log can end with a trailing, incomplete ``C`` marker that has
    no matching ``D`` (e.g. the shell's own ``exit`` command: ``script(1)``
    terminates before it can write ``D`` for it). That trailing marker is
    not the failed command we are after, so the search starts from the
    *last complete pair*: find the last ``D``, then the last ``C`` that
    precedes it. Only when no ``D`` exists anywhere does a lone trailing
    ``C`` (command still running, or ``script(1)`` killed mid-command)
    count, and it is reported as ``"partial"``.
    """
    d_matches = list(_OSC_133_D_RE.finditer(data))
    if d_matches:
        last_d = d_matches[-1]
        c_before = None
        for match in _OSC_133_C_RE.finditer(data[: last_d.start()]):
            c_before = match
        if c_before is not None:
            return data[c_before.end() : last_d.start()], "ok"

    c_matches = list(_OSC_133_C_RE.finditer(data))
    if not c_matches:
        return None
    last_c = c_matches[-1]
    return data[last_c.end() :], "partial"


def _bound(raw: bytes, limit: int) -> tuple[bytes, bool]:
    """Cap ``raw`` to ``limit`` bytes (head + tail + marker). Returns ``(bytes, truncated)``."""
    if len(raw) <= limit:
        return raw, False
    half = limit // 2
    head = raw[:half]
    tail = raw[-half:] if half else b""
    marker = _TRUNCATION_TEMPLATE.format(n=len(raw) - len(head) - len(tail)).encode("ascii")
    return head + marker + tail, True


def _strip_escapes(raw: bytes) -> bytes:
    stripped = _ESCAPE_RE.sub(b"", raw)
    stripped = stripped.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return stripped


def last_slice(log: Path, limit: int = DEFAULT_LIMIT, source: str = "script") -> Slice:
    """Return the bounded, stripped, redacted output of the last failed command.

    Never raises. A missing or unreadable log, or a log with no OSC 133 "C"
    marker at all, yields ``status="no capture"`` and empty text. A "C" with
    no following "D" (``script(1)`` killed mid-command) yields
    ``status="partial"`` with whatever followed "C". Bounding to ``limit``
    bytes (head + tail with a truncation marker) overrides the status to
    ``"truncated"``. The log's own path never appears in the returned text.
    """
    try:
        data = log.read_bytes()
    except OSError:
        return _no_capture(source)

    region = _locate_last_region(data)
    if region is None:
        return _no_capture(source)
    raw_slice, region_status = region
    bytes_total = len(raw_slice)

    bounded, truncated = _bound(raw_slice, limit)
    status = "truncated" if truncated else region_status

    stripped = _strip_escapes(bounded)
    text = stripped.decode("utf-8", errors="replace")

    redacted_bytes, rules = redact_report(text.encode("utf-8", errors="replace"))
    redacted_text = redacted_bytes.decode("utf-8", errors="replace")

    return Slice(
        text=redacted_text,
        status=status,
        source=source,
        redaction_rules=rules,
        bytes_total=bytes_total,
    )
