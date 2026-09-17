"""``NVSH_FAKE_IGNORE_CANCEL`` / ``NVSH_FAKE_GRANDCHILD`` on every fake (t1).

Task t1 of the reliable-agent-stop plan gives eight fake harnesses (pi,
pi_scripted, codex, codex-app-server, acp, agy, claude, qwen) two opt-in
knobs so later tasks can prove a real adapter's ``force_stop()``/``kill_tree``
actually ends a turn a harness's own cancel protocol failed to stop:

* ``NVSH_FAKE_IGNORE_CANCEL=1`` -- the fake receives its family's cancel
  message (pi/pi_scripted: ``abort``; codex-app-server: ``turn/interrupt``;
  acp: ``session/cancel``; claude/qwen: a stream-json interrupt
  control_request) and does nothing: no response, no further output, no
  exit. codex (the ``exec`` fallback) and agy have no protocol-level cancel
  at all -- for them "ignoring cancel" means the turn never ends on its own
  once the scripted transcript is exhausted.
* ``NVSH_FAKE_GRANDCHILD=1`` -- the fake starts a ``sleep 600`` child at
  launch and writes ``{"harness": <pid>, "grandchild": <pid>}`` to
  ``$NVSH_FAKE_PID_FILE``.

Every case here uses both knobs together and checks that 3 seconds after
the fake received its cancel message (where one exists), the harness pid
and the grandchild pid are both still alive, and the pid file carries both.

A last group proves the flags are opt-in: with ``NVSH_FAKE_IGNORE_CANCEL``
unset, each fake's pre-existing default-behaviour tests
(``tests/test_agent_conformance.py`` and friends) keep passing unchanged --
that is covered by the full ``pytest -n auto`` run this task's acceptance
criteria require, not re-proven line by line here.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

from tests._fake_adapters import reap_fake_pids  # noqa: F401 - fixture, used below

# Teardown kills the fake harness/grandchild pids each test's fakes recorded
# (task t23), so a failing or respawning case leaks no ``sleep 600``.
pytestmark = pytest.mark.usefixtures("reap_fake_pids")

FAKES = Path(__file__).parent / "fakes"

#: How long a "still alive" check waits after the cancel message before
#: sampling liveness -- matches the plan's acceptance wording exactly.
_ALIVE_AFTER = 3.0

#: Bound on waiting for the pid file / a line of output, so a genuine
#: regression (the fake crashes, or unexpectedly responds/exits) fails the
#: test instead of hanging the suite.
_WAIT_TIMEOUT = 5.0


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _base_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "NVSH_FAKE_IGNORE_CANCEL": "1",
            "NVSH_FAKE_GRANDCHILD": "1",
            "NVSH_FAKE_PID_FILE": str(tmp_path / "pids.json"),
        }
    )
    env.update(extra)
    return env


def _spawn(fake: str, argv_extra: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(  # nosec B603 - fixed argv, test fixture
        [sys.executable, str(FAKES / fake), *argv_extra],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )


def _write_line(proc: subprocess.Popen, obj: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()


def _read_line(proc: subprocess.Popen) -> str:
    """Read one stdout line.

    Every line this suite reads is one a fake writes immediately in
    response to the line just sent it, so a plain ``readline()`` resolves
    promptly in the passing case; a hung/broken fake is caught by the
    liveness assertions below and by pytest's own suite-level timeout, not
    by a bespoke deadline here.
    """
    assert proc.stdout is not None
    return proc.stdout.readline()


def _wait_for_pid_file(path: Path, timeout: float = _WAIT_TIMEOUT) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written")


@pytest.fixture
def cleanup() -> Iterator[list[subprocess.Popen]]:
    procs: list[subprocess.Popen] = []
    yield procs
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _assert_pid_file_and_liveness(tmp_path: Path, proc: subprocess.Popen) -> dict:
    pids = _wait_for_pid_file(tmp_path / "pids.json")
    assert pids["harness"] == proc.pid
    grandchild_pid = pids["grandchild"]

    time.sleep(_ALIVE_AFTER)

    assert proc.poll() is None, "harness fake exited after receiving its cancel message"
    assert _is_alive(proc.pid), "harness pid is not alive"
    assert _is_alive(grandchild_pid), "grandchild pid is not alive"
    return pids


# ---------------------------------------------------------------------------
# pi family: `abort`
# ---------------------------------------------------------------------------


def test_fake_pi_ignores_abort(tmp_path: Path, cleanup: list[subprocess.Popen]) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    env = _base_env(tmp_path)
    proc = _spawn("pi", ["--mode", "rpc", "--session-dir", str(session_dir)], env)
    cleanup.append(proc)

    _write_line(proc, {"type": "prompt", "id": 1, "message": "hi"})
    ack = json.loads(_read_line(proc))
    assert ack["command"] == "prompt"
    assert ack["success"] is True
    _read_line(proc)  # the text_delta handle_prompt() streams

    _write_line(proc, {"type": "abort", "id": 2})

    _assert_pid_file_and_liveness(tmp_path, proc)


def test_fake_pi_scripted_ignores_abort(tmp_path: Path, cleanup: list[subprocess.Popen]) -> None:
    env = _base_env(tmp_path, NVSH_TEST_PI_SCRIPT="[]")
    proc = _spawn("pi_scripted", [], env)
    cleanup.append(proc)

    _write_line(proc, {"type": "prompt", "id": 1, "message": "hi"})
    ack = json.loads(_read_line(proc))
    assert ack["command"] == "prompt"
    assert ack["success"] is True

    _write_line(proc, {"type": "abort", "id": 2})

    _assert_pid_file_and_liveness(tmp_path, proc)


# ---------------------------------------------------------------------------
# codex family: exec fallback has no protocol cancel; app-server: `turn/interrupt`
# ---------------------------------------------------------------------------


def test_fake_codex_exec_never_ends_on_its_own(
    tmp_path: Path, cleanup: list[subprocess.Popen]
) -> None:
    events = tmp_path / "events.json"
    events.write_text(json.dumps([{"kind": "status", "text": "working"}]), encoding="utf-8")
    env = _base_env(tmp_path, NVSH_FAKE_EVENTS=str(events))
    proc = _spawn("codex", [], env)
    cleanup.append(proc)

    _read_line(proc)  # the task_started status line

    _assert_pid_file_and_liveness(tmp_path, proc)


def test_fake_codex_app_server_ignores_turn_interrupt(
    tmp_path: Path, cleanup: list[subprocess.Popen]
) -> None:
    env = _base_env(tmp_path)
    proc = _spawn("codex-app-server", ["app-server"], env)
    cleanup.append(proc)

    _write_line(proc, {"id": 1, "method": "initialize", "params": {}})
    reply = json.loads(_read_line(proc))
    assert reply["id"] == 1
    assert "result" in reply

    _write_line(
        proc,
        {"id": 2, "method": "turn/interrupt", "params": {"threadId": "t", "turnId": "u"}},
    )

    _assert_pid_file_and_liveness(tmp_path, proc)


# ---------------------------------------------------------------------------
# acp family: `session/cancel`
# ---------------------------------------------------------------------------


def test_fake_acp_ignores_session_cancel(tmp_path: Path, cleanup: list[subprocess.Popen]) -> None:
    # An empty script means session/prompt would normally resolve
    # immediately; a pending permission keeps the turn open so
    # session/cancel has something to (fail to) end.
    script = json.dumps(
        [
            {
                "permission": {
                    "toolCall": {"toolCallId": "c1", "status": "pending", "title": "shell"},
                    "options": [
                        {"optionId": "proceed_once", "name": "Allow", "kind": "allow_once"}
                    ],
                }
            }
        ]
    )
    env = _base_env(tmp_path, NVSH_TEST_ACP_SCRIPT=script)
    proc = _spawn("acp", [], env)
    cleanup.append(proc)

    _write_line(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    _read_line(proc)
    _write_line(proc, {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {}})
    _read_line(proc)
    _write_line(
        proc,
        {"jsonrpc": "2.0", "id": 3, "method": "session/prompt", "params": {"sessionId": "s"}},
    )
    _read_line(proc)  # the session/request_permission the script asks

    _write_line(
        proc,
        {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": "s"}},
    )

    _assert_pid_file_and_liveness(tmp_path, proc)


# ---------------------------------------------------------------------------
# agy: no protocol cancel -- the turn just never ends on its own
# ---------------------------------------------------------------------------


def test_fake_agy_cold_never_ends_on_its_own(
    tmp_path: Path, cleanup: list[subprocess.Popen]
) -> None:
    events = tmp_path / "events.json"
    events.write_text(json.dumps({"stdout": ['{"line": 1}'], "exit_code": 0}), encoding="utf-8")
    env = _base_env(tmp_path, NVSH_FAKE_EVENTS=str(events))
    proc = _spawn("agy", [], env)
    cleanup.append(proc)

    _read_line(proc)

    _assert_pid_file_and_liveness(tmp_path, proc)


def test_fake_agy_warm_never_ends_on_its_own(
    tmp_path: Path, cleanup: list[subprocess.Popen]
) -> None:
    events = tmp_path / "events.json"
    events.write_text(
        json.dumps({"turns": [{"stdout": ['{"line": 1}'], "exit_code": 0}]}), encoding="utf-8"
    )
    env = _base_env(tmp_path, NVSH_FAKE_EVENTS=str(events))
    proc = _spawn("agy", ["--input-format", "stream-json"], env)
    cleanup.append(proc)

    _write_line(proc, {"turn": 1})
    _read_line(proc)

    _assert_pid_file_and_liveness(tmp_path, proc)


# ---------------------------------------------------------------------------
# stream-json family (claude, qwen-p): an interrupt control_request
# ---------------------------------------------------------------------------

_INTERRUPT = {"type": "control_request", "request_id": "int-1", "request": {"subtype": "interrupt"}}


def test_fake_claude_ignores_interrupt(tmp_path: Path, cleanup: list[subprocess.Popen]) -> None:
    events = tmp_path / "events.json"
    events.write_text(json.dumps([{"kind": "status", "text": "working"}]), encoding="utf-8")
    env = _base_env(tmp_path, NVSH_FAKE_EVENTS=str(events))
    proc = _spawn("claude", [], env)
    cleanup.append(proc)

    _read_line(proc)  # the system/status line

    _write_line(proc, _INTERRUPT)

    _assert_pid_file_and_liveness(tmp_path, proc)


def test_fake_qwen_ignores_interrupt(tmp_path: Path, cleanup: list[subprocess.Popen]) -> None:
    events = tmp_path / "events.json"
    events.write_text(json.dumps([{"kind": "status", "text": "working"}]), encoding="utf-8")
    env = _base_env(tmp_path, NVSH_FAKE_EVENTS=str(events))
    proc = _spawn("qwen", [], env)
    cleanup.append(proc)

    _read_line(proc)  # the system status line

    _write_line(proc, _INTERRUPT)

    _assert_pid_file_and_liveness(tmp_path, proc)
