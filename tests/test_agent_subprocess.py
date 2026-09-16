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

import os
import signal
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import threading
from collections import deque

import pytest

from nvsh.agent._env import child_env
from nvsh.agent._subprocess import SubprocessAgent, escalate_close, kill_tree, redacted_tail
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
    # The fake token is assembled at runtime so secret scanners (GitGuardian,
    # scripts/scan-secrets.py) never see a literal high-entropy value.
    fake_token = "abc123" + "def456" + "7890" + "secret"
    tail: deque[str] = deque(["plain line\n", f"HF_TOKEN={fake_token}\n"])
    text = redacted_tail(tail)
    assert f"HF_TOKEN={fake_token}" not in text
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


# ---------------------------------------------------------------------------
# nvsh/agent/_subprocess.py::escalate_close -- the one wait/terminate/kill
# escalation every adapter closes through (task t16, deviation d5)
# ---------------------------------------------------------------------------


def _spawn(script: str) -> subprocess.Popen:
    proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "up"  # it is really running
    return proc


#: Reads stdin to EOF and exits 0 -- what every long-lived harness does.
_EOF_EXITS = "import sys\nprint('up')\nsys.stdin.read()\nsys.exit(0)\n"

#: Ignores its stdin entirely; only a signal ends it.
_IGNORES_EOF = "import sys, time\nprint('up')\ntime.sleep(600)\n"

#: Ignores stdin *and* SIGTERM; only SIGKILL ends it.
_IGNORES_SIGTERM = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('up')\n"
    "time.sleep(600)\n"
)


def test_escalate_close_reaps_a_child_that_exits_on_stdin_eof():
    """Rung 1: closing stdin is enough, and no signal is ever sent."""
    proc = _spawn(_EOF_EXITS)
    assert escalate_close(proc, wait=5.0, grace=0.2) == 0


def test_escalate_close_terminates_a_child_that_ignores_stdin():
    """Rung 2: EOF did nothing, so ``terminate()`` does."""
    proc = _spawn(_IGNORES_EOF)
    assert escalate_close(proc, wait=0.2, grace=5.0) == -signal.SIGTERM


def test_escalate_close_kills_a_child_that_ignores_sigterm():
    """Rung 3: a harness that traps SIGTERM is still gone when close returns."""
    proc = _spawn(_IGNORES_SIGTERM)
    assert escalate_close(proc, wait=0.2, grace=0.2) == -signal.SIGKILL
    assert proc.poll() is not None


def test_escalate_close_is_a_no_op_without_a_process():
    assert escalate_close(None) is None


def test_escalate_close_survives_a_child_someone_else_already_reaped():
    """Teardown on the failure path must never raise, whatever it finds."""
    proc = _spawn(_EOF_EXITS)
    proc.kill()
    proc.wait(timeout=5)
    assert escalate_close(proc, wait=0.2, grace=0.2) is not None


def test_subprocess_agent_close_escalates_and_leaves_no_child():
    """``SubprocessAgent.close`` routes through the shared helper, so a
    stream-json harness that ignores SIGTERM is gone once close returns."""
    agent = _ScriptedAgent(_IGNORES_SIGTERM)
    agent.start()
    agent._proc = _spawn(_IGNORES_SIGTERM)  # stand in for a live turn's child
    pid = agent._proc.pid
    agent.close()
    assert agent._proc.poll() is not None
    with pytest.raises(OSError):
        os.kill(pid, 0)


# ---------------------------------------------------------------------------
# nvsh/agent/_subprocess.py::kill_tree and NvshAgent.force_stop -- the
# process-group stop (plan reliable-agent-stop, task t2; covers c27/h22/c14/h13)
# ---------------------------------------------------------------------------

#: Spawns a sleeping grandchild, reports its pid, then sleeps itself.
_SPAWNS_GRANDCHILD = (
    "import subprocess, sys, time\n"
    "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
    "print(g.pid, flush=True)\n"
    "time.sleep(600)\n"
)


def _pid_alive(pid: int) -> bool:
    """True while *pid* is a live (non-zombie) process."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            stat = handle.read()
    except OSError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    # The state field follows the parenthesised comm, which may contain spaces.
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def _wait_gone(pids: list[int], within: float = 3.0) -> list[int]:
    import time

    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if _pid_alive(pid)]
        if not alive:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if _pid_alive(pid)]


def _spawn_tree() -> tuple[subprocess.Popen, int]:
    proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-u", "-c", _SPAWNS_GRANDCHILD],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    grandchild = int(proc.stdout.readline().strip())
    assert _pid_alive(grandchild)
    return proc, grandchild


def test_kill_tree_kills_the_child_and_its_grandchild():
    proc, grandchild = _spawn_tree()
    kill_tree(proc, grace=2.0)
    assert _wait_gone([proc.pid, grandchild]) == []


def test_kill_tree_kills_a_tree_whose_leader_ignores_sigterm():
    script = "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n" + _SPAWNS_GRANDCHILD
    proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    grandchild = int(proc.stdout.readline().strip())
    assert kill_tree(proc, grace=0.2) == -signal.SIGKILL
    assert _wait_gone([proc.pid, grandchild]) == []


def test_kill_tree_on_an_already_dead_process_does_not_raise():
    proc = _spawn(_EOF_EXITS)
    proc.kill()
    proc.wait(timeout=5)
    assert kill_tree(proc, grace=0.2) == -signal.SIGKILL
    assert kill_tree(None) is None


def test_kill_tree_never_signals_the_callers_own_process_group():
    """A child that shares nvsh's group (pi, acp, agy today) is killed alone."""
    proc = _spawn(_IGNORES_SIGTERM)
    assert os.getpgid(proc.pid) == os.getpgrp()
    assert kill_tree(proc, grace=0.2) == -signal.SIGKILL


def test_subprocess_agent_spawns_its_child_in_a_new_session():
    script = "import os\nprint(os.getsid(0) == os.getpid())\n"
    events = _collect(_ScriptedAgent(script))
    assert [e.text for e in events if e.kind is EventKind.TEXT_DELTA] == ["True"]


def test_subprocess_agent_cancel_kills_the_whole_tree():
    agent = _ScriptedAgent(_SPAWNS_GRANDCHILD)
    agent.start()
    events = agent.run(_request(), AgentContext())
    first = next(events)
    grandchild = int(first.text)
    child = agent._proc.pid
    agent.cancel()
    events.close()
    assert _wait_gone([child, grandchild]) == []


def test_force_stop_defaults_to_cancel_then_close():
    calls: list[str] = []

    class _Recorder(_ScriptedAgent):
        def cancel(self) -> None:
            calls.append("cancel")

        def close(self) -> None:
            calls.append("close")

    _Recorder("").force_stop()
    assert calls == ["cancel", "close"]


def test_force_stop_kills_a_subprocess_agents_tree():
    agent = _ScriptedAgent(_SPAWNS_GRANDCHILD)
    agent.start()
    agent._proc, grandchild = _spawn_tree()
    agent.force_stop()
    assert _wait_gone([agent._proc.pid, grandchild]) == []


def test_stop_paths_never_open_harness_settings_files():
    """Stops are signals and protocol messages only (c14/h13): no stop path
    calls any file-opening API, so no harness settings/trust file is touched."""
    import ast
    import inspect
    import textwrap

    import nvsh.agent._subprocess as sp
    from nvsh.agent.base import NvshAgent

    file_apis = {"open", "read_text", "write_text", "read_bytes", "write_bytes", "unlink"}
    for func in (
        sp.kill_tree,
        sp._own_group,
        sp._signal,
        sp.escalate_close,
        SubprocessAgent.cancel,
        SubprocessAgent.close,
        SubprocessAgent._terminate_if_running,
        NvshAgent.force_stop,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                assert name not in file_apis, (func.__qualname__, name)
