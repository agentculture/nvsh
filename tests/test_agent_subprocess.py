"""Tests for the shared subprocess adapter base (``nvsh/agent/_subprocess.py``).

Added for the PR #8 review finding "verbose adapter errors can stall
diagnosis": a backend that writes more stderr than a pipe buffer holds while
it is still running must not wedge the turn until the daemon's timeout.
Every test here drives a *real* child process, because the failure is a pipe
deadlock and a mocked stream cannot reproduce it.

Also covers subprocess hygiene shared by every adapter (spec targets c42,
h38, c44, h39): the child environment is scrubbed of Claude-Code-in-the-
parent markers (``nvsh/agent/_env.py::child_env``) before any adapter
spawns a backend, and a stderr tail is redacted
(``nvsh/redact.py::redact``) before it reaches an ``ERROR`` event or a log
line (``nvsh/agent/_subprocess.py::redacted_tail``).
"""

from __future__ import annotations

import sys
import threading
from collections import deque

import pytest

from nvsh.agent._env import child_env
from nvsh.agent._subprocess import SubprocessAgent, redacted_tail
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


# ---------------------------------------------------------------------------
# nvsh/agent/_env.py::child_env
# ---------------------------------------------------------------------------


def test_child_env_drops_claudecode_and_claude_code_star(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "12345")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("SOME_OTHER_VAR", "kept")

    env = child_env()

    assert "CLAUDECODE" not in env
    assert "CLAUDE_CODE_SSE_PORT" not in env
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert not any(key.startswith("CLAUDE_CODE_") for key in env)
    assert env.get("SOME_OTHER_VAR") == "kept"


def test_child_env_scrubs_a_provided_base_mapping():
    base = {
        "CLAUDECODE": "1",
        "CLAUDE_CODE_FOO": "bar",
        "PATH": "/usr/bin",
    }

    env = child_env(base)

    assert env == {"PATH": "/usr/bin"}
    # The input mapping itself is untouched (no aliasing surprises).
    assert base["CLAUDECODE"] == "1"


def test_subprocess_agent_child_does_not_see_claudecode_markers(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    script = (
        "import os\n"
        "print('CLAUDECODE=' + os.environ.get('CLAUDECODE', '<absent>'))\n"
        "print('CLAUDE_CODE_ENTRYPOINT=' + os.environ.get('CLAUDE_CODE_ENTRYPOINT', '<absent>'))\n"
        "print('__DONE__')\n"
    )
    events = _collect(_ScriptedAgent(script))
    texts = [event.text for event in events if event.kind is EventKind.TEXT_DELTA]
    assert "CLAUDECODE=<absent>" in texts
    assert "CLAUDE_CODE_ENTRYPOINT=<absent>" in texts


# ---------------------------------------------------------------------------
# nvsh/agent/_subprocess.py::redacted_tail and the ERROR path's redaction
# ---------------------------------------------------------------------------


def test_redacted_tail_helper_redacts_a_deque_of_lines():
    tail: deque[str] = deque(["plain line\n", "HF_TOKEN=abc123def4567890secret\n"])
    text = redacted_tail(tail)
    assert "HF_TOKEN=abc123def4567890secret" not in text
    assert "<REDACTED:env_assignment>" in text
    assert "plain line" in text


def test_stderr_tail_is_redacted_before_the_error_event():
    """Acceptance criterion 2: a fake that prints ``HF_TOKEN=abc`` on stderr
    and exits 1 yields an ERROR whose text contains the redacted form and
    not the token."""
    script = "import sys\nsys.stderr.write('HF_TOKEN=abc\\n')\nsys.exit(1)\n"
    events = _collect(_ScriptedAgent(script))
    assert events[-1].kind is EventKind.ERROR
    error_text = events[-1].error or ""
    assert "HF_TOKEN=abc" not in error_text
    assert "<REDACTED:env_assignment>" in error_text
