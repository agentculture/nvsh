"""Shared subprocess mapping base for thin headless-CLI adapters.

``claude``, ``codex`` and ``qwen`` are all "run a subprocess, read its
stdout line by line, translate each line into an :class:`AgentEvent`"
adapters that differ only in argv and line format. This module factors the
shared plumbing (spawn, cancel-between-yields, exit-code-to-ERROR mapping,
idempotent close) so each concrete adapter only supplies ``_argv`` and
``_parse_line``.
"""

from __future__ import annotations

import subprocess  # nosec B404 - fixed argv lists below, no shell=True
import threading
from collections import deque
from typing import IO, Iterator

from ..redact import redact
from ._env import child_env
from .base import AgentContext, AgentEvent, AgentRequest, EventKind, NvshAgent
from .prompt import build_full_prompt as _build_full_prompt
from .prompt import build_prompt as _build_prompt
from .prompt import build_system_prompt as _build_system_prompt

#: Every backend composes its prompt with the *same* function, so the bytes
#: ``nvsh context --show`` prints are the bytes each adapter sends. Composing
#: it twice (once here, once in ``pi``) let the device context be dropped
#: whenever a caller also supplied a prompt -- which the failure client
#: always does -- so the platform block never reached these backends.
build_prompt = _build_prompt

#: The same goes for the system brief (deviation d19): one text, delivered
#: through whichever channel the backend has -- a flag where one exists,
#: the head of the prompt where none does.
build_system_prompt = _build_system_prompt
build_full_prompt = _build_full_prompt


#: How many stderr lines are kept for a non-zero exit. Enough to explain a
#: failure, bounded so a chatty backend cannot grow the daemon's memory.
_STDERR_TAIL_LINES = 200

#: How long the drain thread is given to finish once the child is gone.
_STDERR_JOIN_TIMEOUT = 2.0

#: Default seconds :func:`escalate_close` spends on each rung of its
#: wait -> terminate -> kill escalation.
CLOSE_WAIT_SECONDS = 2.0


def escalate_close(
    proc: subprocess.Popen | None,
    *,
    wait: float | None = None,
    grace: float | None = None,
) -> int | None:
    """Close *proc*'s stdin and see it out: wait, then terminate, then kill.

    One helper for every adapter (deviation d5). Each wave-2 adapter grew
    its own copy of this loop -- ``PiAgent.close``, ``AcpAgent._wait_out``,
    ``CodexAgent._wait_out``, ``AgyAgent._terminate``,
    :meth:`SubprocessAgent.close` -- which is exactly the kind of drift that
    leaves one harness's child running after ``nvsh uninstall`` while the
    others exit cleanly. The rungs, in order:

    1. **Close stdin.** Every long-lived harness nvsh drives (pi's rpc loop,
       ACP, ``codex app-server``, ``claude`` in stream-json input mode)
       exits on EOF, so that is the only rung most closes ever reach.
    2. **Wait** up to ``wait`` seconds for that clean exit.
    3. **``terminate()``**, then wait up to ``grace`` seconds.
    4. **``kill()``**, then wait up to ``grace`` seconds.

    Returns the child's exit status, or ``None`` when it survived even
    ``kill()`` (a process stuck in uninterruptible sleep, or one already
    reaped by somebody else) -- waiting on such a child forever is how
    teardown hangs, so this never does. Never raises: a close on the failure
    path must not itself become the failure.
    """
    if proc is None:
        return None
    # Resolved here, not in the signature's defaults, so the module constant
    # is read at call time (a default argument would freeze it at import).
    wait = CLOSE_WAIT_SECONDS if wait is None else wait
    grace = wait if grace is None else grace
    stdin = getattr(proc, "stdin", None)
    if stdin is not None:
        try:
            stdin.close()
        except (OSError, ValueError):  # OSError covers BrokenPipeError
            pass
    for escalate, timeout in ((None, wait), (proc.terminate, grace), (proc.kill, grace)):
        if escalate is not None:
            try:
                escalate()
            except (OSError, ValueError):  # already gone, or already reaped
                return proc.poll()
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
    return proc.poll()


def _drain(stream: IO[str], sink: deque[str]) -> None:
    """Read ``stream`` to EOF into ``sink`` (bounded). Never raises."""
    try:
        for line in stream:
            sink.append(line)
    except (OSError, ValueError):  # closed underneath us by cancel/terminate
        pass


#: Argv fragments that would let a harness approve its own tool calls.
#: "Propose, don't run" is not a policy an operator may configure away
#: through ``extra_args``; every adapter refuses these at construction.
BYPASS_ARGS = frozenset(
    {
        "--dangerously-skip-permissions",
        "--dangerously-bypass-approvals-and-sandbox",
        "--full-auto",
        "--yolo",
        "--trust-all-tools",
        "danger-full-access",
        "bypassPermissions",
    }
)


def reject_bypass_args(extra_args: list[str], harness: str) -> None:
    """Refuse ``extra_args`` that would bypass nvsh's approval gate."""
    for arg in extra_args:
        if arg in BYPASS_ARGS or any(
            token in arg for token in BYPASS_ARGS if token.startswith("--")
        ):
            raise ValueError(f"{harness}: extra_args may not bypass approval ({arg!r})")


def redacted_tail(tail: deque[str] | list[str]) -> str:
    """Join a stderr tail into one string, redacted before anyone sees it.

    Shared by every subprocess-backed adapter (and reusable by any other
    adapter that keeps its own stderr tail, e.g. ``pi``) so "redact the
    stderr before it reaches an ERROR event or a log line" is one choke
    point instead of one per adapter. ``redact`` operates on bytes, so the
    joined text round-trips through UTF-8 with ``surrogateescape`` the same
    way ``nvsh.redact`` itself does.
    """
    text = "".join(tail)
    redacted_bytes = redact(text.encode("utf-8", errors="surrogateescape"))
    return redacted_bytes.decode("utf-8", errors="surrogateescape").strip()


class SubprocessAgent(NvshAgent):
    """Spawns ``binary`` and maps its stdout lines to :class:`AgentEvent`.

    Subclasses set :attr:`binary` and implement :meth:`_argv` and
    :meth:`_parse_line`. ``env`` (test-only override) lets the conformance
    suite point PATH at ``tests/fakes/`` without touching the real PATH.
    """

    binary: str = ""

    def __init__(self, config: dict | None = None, *, env: dict[str, str] | None = None) -> None:
        self._config = dict(config or {})
        self._env = env
        self._proc: subprocess.Popen | None = None
        self._cancelled = False
        self._closed = False

    def start(self) -> None:
        self._cancelled = False
        self._closed = False

    def _argv(self, request: AgentRequest, context: AgentContext) -> list[str]:
        raise NotImplementedError

    def _parse_line(self, line: str) -> AgentEvent | None:
        """Return an ``AgentEvent`` for one stdout line, or ``None`` to skip it."""
        raise NotImplementedError

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        argv = self._argv(request, context)
        try:
            self._proc = subprocess.Popen(  # nosec B603 - argv is a fixed list, no shell
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_env(self._env),
            )
        except OSError as exc:
            yield AgentEvent(kind=EventKind.ERROR, error=f"failed to start {argv[0]}: {exc}")
            return

        # Drain stderr concurrently. Reading it only once stdout hits EOF
        # deadlocks any backend that writes more than a pipe buffer's worth
        # of warnings while it is still running: the child blocks on stderr,
        # so it never closes stdout, so the hook waits out the daemon's turn
        # timeout instead of showing the diagnosis. Only a bounded tail is
        # kept -- that is all a non-zero exit needs to be explainable.
        tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        drain: threading.Thread | None = None
        if self._proc.stderr is not None:
            drain = threading.Thread(target=_drain, args=(self._proc.stderr, tail), daemon=True)
            drain.start()

        try:
            assert self._proc.stdout is not None
            for raw_line in self._proc.stdout:
                if self._cancelled:
                    return
                event = self._parse_line(raw_line.rstrip("\n"))
                if event is None:
                    continue
                yield event
                if event.kind in (EventKind.ERROR, EventKind.DONE):
                    return
            if self._cancelled:
                return
            yield self._exit_event(argv, tail, drain)
        finally:
            # Terminating closes the child's end of the pipe, so the drain
            # thread sees EOF and returns; join it so no reader outlives the
            # turn (cancellation included).
            self._terminate_if_running()
            if drain is not None:
                drain.join(timeout=_STDERR_JOIN_TIMEOUT)

    def _exit_event(
        self, argv: list[str], tail: deque[str], drain: threading.Thread | None
    ) -> AgentEvent:
        """How the child finished: DONE on 0, else ERROR quoting its stderr.

        The drain thread is joined only on the failure path, and only then,
        because that is the one case whose message needs the whole tail.
        """
        assert self._proc is not None
        rc = self._proc.wait()
        if rc == 0:
            return AgentEvent(kind=EventKind.DONE)
        if drain is not None:
            drain.join(timeout=_STDERR_JOIN_TIMEOUT)
        stderr = redacted_tail(tail)
        return AgentEvent(kind=EventKind.ERROR, error=stderr or f"{argv[0]} exited {rc}")

    def cancel(self) -> None:
        self._cancelled = True
        self._terminate_if_running()

    def _terminate_if_running(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def close(self) -> None:
        # Close, not cancel: the stdin rung matters here and must not run
        # mid-turn (``claude`` keeps stdin a live pipe for the whole turn),
        # which is why ``cancel``/``run``'s teardown still go through
        # ``_terminate_if_running`` instead.
        if self._closed:
            return
        self._closed = True
        escalate_close(self._proc)
