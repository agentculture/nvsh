"""Tier 1: the Needle3 tool-selecting model, in a child process (task t9).

The engine is a native library loaded through ``ctypes`` by the
``cactus-needle`` package. nvsh never loads it in-process: a native crash,
an ``abort()`` or a runaway allocation inside that library would take the
per-user daemon -- and with it the warm session of every hooked shell --
down with it. So the engine lives in a child process nvsh starts, speaks to
over a private pipe, and kills without ceremony whenever it misbehaves
(spec target c32).

The protocol is deliberately tiny: 4-byte big-endian length, then UTF-8
JSON, bounded at :data:`MAX_FRAME_BYTES`, one request frame in and one
response frame out. Reads are deadline-bounded with ``select`` on the raw
fd, never a blocking ``read()``, so a hung child costs the operator a
timeout and nothing more. The child's stderr goes to ``DEVNULL`` so it can
never fill a pipe nobody drains and deadlock the worker.

:meth:`NeedleTier.select` **never raises**: every failure -- a missing
engine, a dead child, a garbled frame, a timeout -- comes back as a
:class:`~nvsh.tiers.base.Decline` with a reason code. And it never
executes anything: the worker's raw calls go through
:func:`~nvsh.tiers.base.decide` and no further (grounding, rendering and
approval are the router's and the operator's job).

This module is **not** importable on the hot success path: import it lazily
from any CLI/doctor/daemon code.
"""

from __future__ import annotations

import json
import os
import select
import struct
import subprocess  # nosec B404 - fixed argv list below, no shell=True
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

from ..agent._env import child_env
from ..agent._subprocess import kill_tree
from ..agent.base import AgentContext, AgentRequest
from .base import Decline, DeclineReason, Tier, TierDecision, decide
from .memfloor import FloorResult

#: Hard bound on one protocol frame. The biggest thing that legitimately
#: crosses the pipe is a handful of selected calls; anything past this is a
#: garbled or hostile length header and the child is killed rather than
#: trusted to say how much memory to allocate.
MAX_FRAME_BYTES = 1024 * 1024

_HEADER = struct.Struct(">I")
HEADER_BYTES = _HEADER.size

#: How long the parent waits for one reply by default. Generous because the
#: first request pays the engine's cold load (measured 5.7 s on a DGX Spark,
#: spike s13); the router gives it a tighter budget when it has one.
DEFAULT_TIMEOUT_SECONDS = 30.0

AvailabilityFn = Callable[[], str | None]
FloorFn = Callable[[], FloorResult]


def pack_frame(message: object) -> bytes:
    """Encode *message* as one length-prefixed JSON frame.

    Raises ``ValueError`` when the encoded payload would exceed
    :data:`MAX_FRAME_BYTES`, so an oversized frame is never written to the
    pipe in the first place.
    """
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"frame of {len(payload)} bytes is too large (max {MAX_FRAME_BYTES})")
    return _HEADER.pack(len(payload)) + payload


def default_worker_argv() -> list[str]:
    """The stock worker command: this interpreter running the worker module."""
    return [sys.executable, "-m", "nvsh.tiers.needle_worker"]


def _one_line(text: str) -> str:
    """Collapse *text* to a single line so ``status()`` stays one line."""
    return " ".join(str(text).split())


class NeedleTier(Tier):
    """Needle3 selection, isolated in a child process. Selects, never runs."""

    name = "needle"

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        worker_argv: list[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        min_confidence: float = 0.0,
        availability: AvailabilityFn | None = None,
        floor_check: FloorFn | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Cheap: records configuration and spawns nothing.

        The child starts on the first :meth:`select`. ``worker_argv`` and
        ``availability`` are injectable so tests drive a fake worker and
        never need ``cactus-needle`` installed.
        """
        self._weights_path = str(weights_path) if weights_path is not None else None
        self._worker_argv = list(worker_argv) if worker_argv else default_worker_argv()
        self._timeout = float(timeout)
        self._min_confidence = float(min_confidence)
        self._availability = availability or self._default_availability
        self._floor_check = floor_check
        self._env = dict(env) if env is not None else None
        self._proc: subprocess.Popen | None = None
        self._next_id = 0

    # -- public surface --

    def select(self, request: AgentRequest, context: AgentContext) -> TierDecision | Decline:
        """Propose one operation for *request*, or decline. Never raises."""
        try:
            return self._select(request)
        except Exception as exc:  # the child is untrusted; so is its timing
            self._shutdown()
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"needle tier failed: {type(exc).__name__}: {exc}",
            )

    def status(self) -> str:
        """One human-readable line: why the tier is unavailable, or that it is ready."""
        reason = self._availability()
        if reason:
            return f"needle tier unavailable: {_one_line(reason)}"
        if self._proc is None or self._proc.poll() is not None:
            return "needle tier ready (worker not running)"
        return f"needle tier ready (worker pid {self._proc.pid})"

    def close(self) -> None:
        """Kill the worker and everything in its process group. Never raises."""
        self._shutdown()

    # -- selection --

    def _select(self, request: AgentRequest) -> TierDecision | Decline:
        floor = self._floor_check() if self._floor_check is not None else None
        if floor is not None and not floor.ok:
            return Decline(reason=DeclineReason.MEMORY_FLOOR, detail=_one_line(floor.status))

        reason = self._availability()
        if reason:
            return Decline(reason=DeclineReason.TIER_UNAVAILABLE, detail=_one_line(reason))

        started = self._ensure_child()
        if isinstance(started, Decline):
            return started

        self._next_id += 1
        message = {
            "id": self._next_id,
            "op": "select",
            "text": _prompt_text(request),
            "weights": self._weights_path,
        }
        reply = self._exchange(started, message)
        if isinstance(reply, Decline):
            return reply
        if not reply.get("ok"):
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"needle worker: {_one_line(str(reply.get('error', 'unknown error')))}",
            )
        return decide(
            reply.get("calls"),
            reply.get("confidence"),
            min_confidence=self._min_confidence,
        )

    def _exchange(self, proc: subprocess.Popen, message: dict) -> dict | Decline:
        """Send one request frame and read one response frame, or decline."""
        deadline = time.monotonic() + self._timeout
        if proc.stdin is None or proc.stdout is None:
            self._shutdown()
            return Decline(
                reason=DeclineReason.TIER_ERROR, detail="needle worker has no pipes to speak on"
            )
        try:
            _write_all(proc.stdin.fileno(), pack_frame(message))
        except (OSError, ValueError) as exc:
            self._shutdown()
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"needle worker could not be reached: {exc}",
            )
        return self._receive(proc.stdout.fileno(), deadline)

    def _receive(self, fd: int, deadline: float) -> dict | Decline:

        header = _read_exact(fd, HEADER_BYTES, deadline)
        if not isinstance(header, bytes):
            return self._dead_child(header)

        size = _HEADER.unpack(header)[0]
        if size > MAX_FRAME_BYTES:
            self._shutdown()
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"needle worker announced a frame of {size} bytes, too large",
            )

        payload = _read_exact(fd, size, deadline)
        if not isinstance(payload, bytes):
            return self._dead_child(payload)

        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            self._shutdown()
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"needle worker sent a garbled frame: {exc}",
            )
        if not isinstance(decoded, dict):
            self._shutdown()
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"needle worker sent a {type(decoded).__name__}, not a response object",
            )
        return decoded

    def _dead_child(self, outcome: object) -> Decline:
        """Kill the child and explain why the read did not finish."""
        self._shutdown()
        if outcome == _TIMEOUT:
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"needle worker did not answer within {self._timeout:g}s",
            )
        return Decline(
            reason=DeclineReason.TIER_ERROR,
            detail="needle worker exited without answering",
        )

    # -- child lifecycle --

    def _ensure_child(self) -> subprocess.Popen | Decline:
        """The running worker, restarted first when the previous one died."""
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._shutdown()
        try:
            self._proc = subprocess.Popen(  # nosec B603 - fixed argv list, no shell
                self._worker_argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # nothing to drain, nothing to deadlock
                bufsize=0,  # raw pipes: framing is done by hand, not by a buffer
                env=child_env(self._env),
                # Its own process group, so a hung or crashed engine can be
                # killed whole -- including anything cactus-needle spawned.
                start_new_session=True,
            )
        except OSError as exc:
            self._proc = None
            return Decline(
                reason=DeclineReason.TIER_UNAVAILABLE,
                detail=f"could not start the needle worker: {exc}",
            )
        return self._proc

    def _shutdown(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        kill_tree(proc)
        for stream in (proc.stdin, proc.stdout):
            _close_quietly(stream)

    # -- availability --

    def _default_availability(self) -> str | None:
        """Why Tier 1 cannot run right now, or ``None`` when it can.

        Reports *every* problem found in one line, so an operator missing
        both the package and the weights is told both at once.
        """
        problems = [problem for problem in (_engine_problem(), self._weights_problem()) if problem]
        return "; ".join(problems) or None

    def _weights_problem(self) -> str | None:
        if self._weights_path is not None:
            if Path(self._weights_path).is_file():
                return None
            return f"needle weights not found: {self._weights_path}"
        from . import fetch  # lazy: fetch pulls urllib, the hot path must not

        resolved = fetch.resolve("weights")
        if isinstance(resolved, fetch.FetchProblem):
            return f"needle weights unusable: {resolved.message}"
        self._weights_path = str(resolved)
        return None


def _engine_problem() -> str | None:
    """``None`` when ``cactus-needle`` is importable, else why it is not."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("needle")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return "cactus-needle is not installed"
    return None


def _prompt_text(request: AgentRequest) -> str:
    """What the selector is shown: the operator's own words, nothing else."""
    return request.prompt or request.ask or ""


# -- raw pipe I/O --

#: Sentinel distinguishing "the deadline passed" from "the pipe hit EOF".
_TIMEOUT = "timeout"
_EOF = "eof"


def _write_all(fd: int, payload: bytes) -> None:
    """Write every byte of *payload* to *fd* (raw pipes do short writes)."""
    view = memoryview(payload)
    while view:
        view = view[os.write(fd, view) :]


def _read_exact(fd: int, size: int, deadline: float) -> bytes | str:
    """Read exactly *size* bytes before *deadline*, or a sentinel.

    Waits with ``select`` on the raw fd rather than blocking in ``read()``,
    so a child that stops talking costs a bounded wait. Returns
    :data:`_TIMEOUT` when the deadline passes and :data:`_EOF` when the pipe
    closes early.
    """
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        budget = deadline - time.monotonic()
        if budget <= 0:
            return _TIMEOUT
        ready, _, _ = select.select([fd], [], [], budget)
        if not ready:
            return _TIMEOUT
        chunk = os.read(fd, remaining)
        if not chunk:
            return _EOF
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _close_quietly(stream: object) -> None:
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        close()
    except OSError:
        pass  # the pipe is already gone; nothing to release
