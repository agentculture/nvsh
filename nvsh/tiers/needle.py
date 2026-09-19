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
import threading
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

#: How long the parent waits for one reply by default. This tier exists to
#: be fast: a warm selection is tens of milliseconds and even the cold load
#: measured 5.7 s on a DGX Spark (spike s13), so ten seconds is generous for
#: a working child and short enough that a broken one does not hold the
#: operator's prompt. The router may give it a tighter budget still.
DEFAULT_TIMEOUT_SECONDS = 10.0

#: Upper bound on the prompt handed to the selector. Tier 1 is for short
#: operator asks; anything longer belongs to a tier that can read it.
MAX_PROMPT_CHARS = 4000

#: Grace period for killing a child that already missed the tier's own
#: deadline (a timed-out write/read, or an early EOF). ``kill_tree``'s own
#: default (2.0s) is meant for an operator-requested ``close()``, where
#: nothing else is waiting; here the operator's request already sat out the
#: full ``self._timeout`` before ``_dead_child`` runs, so a resistant child
#: must not add up to two more full grace periods (SIGTERM wait, then
#: SIGKILL wait) on top of that (Qodo #4053821262).
_DEAD_CHILD_KILL_GRACE = 0.5

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
        tuned: bool = False,
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
        never need ``cactus-needle`` installed. ``tuned`` says the weights
        are a fine-tune rather than the stock model: the library then
        reports no confidence at all, so it stays off by default.
        """
        self._weights_path = str(weights_path) if weights_path is not None else None
        self._tuned = bool(tuned)
        self._worker_argv = list(worker_argv) if worker_argv else default_worker_argv()
        self._timeout = float(timeout)
        self._min_confidence = float(min_confidence)
        self._availability = availability or self._default_availability
        self._floor_check = floor_check
        self._env = dict(env) if env is not None else None
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self._home: object = None  # a needle_home.NeedleHome once staged
        # One turn at a time: the daemon shares a tier across handler threads
        # and two interleaved writes would splice two frames into nonsense.
        self._lock = threading.Lock()

    # -- public surface --

    def select(self, request: AgentRequest, context: AgentContext) -> TierDecision | Decline:
        """Propose one operation for *request*, or decline. Never raises."""
        try:
            with self._lock:
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
        message = {"id": self._next_id, "op": "select", "text": _prompt_text(request)}
        message.update(self._engine_fields())
        reply = self._exchange(started, message)
        if isinstance(reply, Decline):
            return reply
        if reply.get("id") != self._next_id:
            # A reply that does not answer this request means the stream is
            # out of step; nothing on it can be trusted from here on.
            self._shutdown()
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"needle worker answered request {reply.get('id')!r}, expected"
                f" {self._next_id}",
            )
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
            sent = _write_all(proc.stdin.fileno(), pack_frame(message), deadline)
        except (OSError, ValueError) as exc:
            self._shutdown()
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"needle worker could not be reached: {exc}",
            )
        if not sent:
            return self._dead_child(_TIMEOUT)
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
        except ValueError as exc:
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
        """Kill the child and explain why the read did not finish.

        Uses :data:`_DEAD_CHILD_KILL_GRACE` rather than ``_shutdown``'s
        default: the tier's own deadline already elapsed by the time this
        runs, so the decline must not sit behind two more multi-second
        waits for a child that resists termination.
        """
        self._shutdown(grace=_DEAD_CHILD_KILL_GRACE)
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

    def _shutdown(self, *, grace: float | None = None) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if grace is None:
            kill_tree(proc)
        else:
            kill_tree(proc, grace=grace)
        for stream in (proc.stdin, proc.stdout):
            _close_quietly(stream)

    # -- the staged engine --

    def _engine_fields(self) -> dict[str, object]:
        """What the request frame tells the child about its engine."""
        home = self._home
        return {
            "lib": str(getattr(home, "lib", "")) or None,
            "weights": str(getattr(home, "weights", "") or self._weights_path or "") or None,
            "home": str(getattr(home, "home", "")) or None,
            "tuned": self._tuned,
        }

    # -- availability --

    def _default_availability(self) -> str | None:
        """Why Tier 1 cannot run right now, or ``None`` when it can.

        Reports *every* problem found in one line, so an operator missing
        both the package and the staged files is told both at once.
        """
        problems = [problem for problem in (_engine_problem(), self._staging_problem()) if problem]
        return "; ".join(problems) or None

    def _staging_problem(self) -> str | None:
        """Stage the pinned engine and weights, or say what is in the way.

        An explicitly configured ``weights_path`` is taken at its word (an
        operator pointing at their own file, and what the tests use); the
        default path goes through :func:`nvsh.tiers.needle_home.stage`,
        which extracts the pinned engine and never opens a socket.
        """
        if self._home is not None:
            return None
        if self._weights_path is not None:
            if Path(self._weights_path).is_file():
                return None
            return f"needle weights not found: {self._weights_path}"
        from . import needle_home  # lazy: zipfile/urllib must stay off the hot path

        staged = needle_home.stage()
        if isinstance(staged, needle_home.FetchProblem):
            return f"needle engine files unusable: {staged.message}"
        self._home = staged
        self._weights_path = str(staged.weights)
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
    """What the selector is shown: the operator's own words, clamped.

    Tier 1 answers short operator asks; a selector prompt is never pages
    long. Clamping here also keeps the request frame small enough that
    writing it cannot sit in the pipe behind a child that has stopped
    reading (the write has a deadline, but a small frame never reaches it).
    """
    return (request.prompt or request.ask or "")[:MAX_PROMPT_CHARS]


# -- raw pipe I/O --

#: Sentinel distinguishing "the deadline passed" from "the pipe hit EOF".
_TIMEOUT = "timeout"
_EOF = "eof"


def _write_all(fd: int, payload: bytes, deadline: float) -> bool:
    """Write every byte of *payload* to *fd* before *deadline*.

    Raw pipes do short writes, and a child that has stopped reading fills
    the pipe buffer -- at which point an unguarded ``os.write`` blocks
    forever. So writability is waited for with ``select`` too. ``False``
    means the deadline passed with bytes still unwritten.
    """
    view = memoryview(payload)
    while view:
        budget = deadline - time.monotonic()
        if budget <= 0:
            return False
        _, ready, _ = select.select([], [fd], [], budget)
        if not ready:
            return False
        view = view[os.write(fd, view) :]
    return True


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
