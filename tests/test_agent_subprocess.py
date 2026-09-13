"""Tests for the shared subprocess adapter base (``nvsh/agent/_subprocess.py``).

Added for the PR #8 review finding "verbose adapter errors can stall
diagnosis": a backend that writes more stderr than a pipe buffer holds while
it is still running must not wedge the turn until the daemon's timeout.
Every test here drives a *real* child process, because the failure is a pipe
deadlock and a mocked stream cannot reproduce it.
"""

from __future__ import annotations

import sys
import threading

import pytest

from nvsh.agent._subprocess import SubprocessAgent
from nvsh.agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Capabilities,
    EventKind,
    RequestKind,
)

#: More than any plausible pipe buffer (Linux defaults to 64 KiB).
NOISE_BYTES = 1_000_000


class _ScriptedAgent(SubprocessAgent):
    """Runs a python snippet as its "backend" and echoes stdout lines."""

    binary = sys.executable

    def __init__(self, script: str) -> None:
        super().__init__({})
        self._script = script

    def _argv(self, request: AgentRequest, context: AgentContext) -> list[str]:
        return [sys.executable, "-c", self._script]

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def _parse_line(self, line: str) -> AgentEvent | None:
        if not line:
            return None
        if line == "__DONE__":
            return AgentEvent(kind=EventKind.DONE)
        return AgentEvent(kind=EventKind.TEXT_DELTA, text=line)


def _request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, prompt="why?", command="ls /nope", exit_code=2)


def _collect(agent: SubprocessAgent, timeout: float = 30.0) -> list[AgentEvent]:
    """Run the agent on a worker thread so a deadlock fails instead of hanging."""
    events: list[AgentEvent] = []
    error: list[BaseException] = []

    def work() -> None:
        try:
            events.extend(agent.run(_request(), AgentContext()))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            error.append(exc)

    thread = threading.Thread(target=work, daemon=True)
    agent.start()
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        agent.cancel()
        thread.join(5)
        pytest.fail(f"adapter deadlocked: no result within {timeout}s")
    if error:
        raise error[0]
    return events


def test_noisy_stderr_before_stdout_does_not_deadlock():
    script = (
        "import sys\n"
        f"sys.stderr.write('warning: deprecated\\n' * {NOISE_BYTES // 24})\n"
        "sys.stderr.flush()\n"
        "print('hello')\n"
        "print('__DONE__')\n"
    )
    events = _collect(_ScriptedAgent(script))
    kinds = [event.kind for event in events]
    assert EventKind.TEXT_DELTA in kinds
    assert kinds[-1] is EventKind.DONE


def test_noisy_stderr_and_a_nonzero_exit_reports_the_stderr_tail():
    script = (
        "import sys\n"
        f"sys.stderr.write('noise\\n' * {NOISE_BYTES // 6})\n"
        "sys.stderr.write('the real reason\\n')\n"
        "sys.stderr.flush()\n"
        "sys.exit(7)\n"
    )
    events = _collect(_ScriptedAgent(script))
    assert events[-1].kind is EventKind.ERROR
    assert "the real reason" in (events[-1].error or "")


def test_quiet_backend_still_reports_its_stderr_on_failure():
    script = "import sys\nsys.stderr.write('boom\\n')\nsys.exit(2)\n"
    events = _collect(_ScriptedAgent(script))
    assert events[-1].kind is EventKind.ERROR
    assert "boom" in (events[-1].error or "")


def test_exit_code_is_reported_when_stderr_is_empty():
    events = _collect(_ScriptedAgent("import sys\nsys.exit(3)\n"))
    assert events[-1].kind is EventKind.ERROR
    assert "3" in (events[-1].error or "")


def test_clean_run_ends_in_done():
    events = _collect(_ScriptedAgent("print('hi')\n"))
    assert events[-1].kind is EventKind.DONE
