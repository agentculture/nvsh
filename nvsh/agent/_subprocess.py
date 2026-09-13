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


def _drain(stream: IO[str], sink: deque[str]) -> None:
    """Read ``stream`` to EOF into ``sink`` (bounded). Never raises."""
    try:
        for line in stream:
            sink.append(line)
    except (OSError, ValueError):  # closed underneath us by cancel/terminate
        pass


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
                env=self._env,
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
        stderr = "".join(tail)
        return AgentEvent(kind=EventKind.ERROR, error=stderr.strip() or f"{argv[0]} exited {rc}")

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
        if self._closed:
            return
        self._terminate_if_running()
        self._closed = True
