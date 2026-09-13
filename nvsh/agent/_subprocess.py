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
from typing import Iterator

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
            rc = self._proc.wait()
            if rc != 0:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                yield AgentEvent(
                    kind=EventKind.ERROR, error=stderr.strip() or f"{argv[0]} exited {rc}"
                )
            else:
                yield AgentEvent(kind=EventKind.DONE)
        finally:
            self._terminate_if_running()

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
