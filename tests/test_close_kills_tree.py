"""A normal ``close()`` takes the harness's whole process tree with it (t23, d14).

Every persistent adapter spawns its harness with ``start_new_session=True``,
so the harness leads a process group holding any tool grandchild it started.
A harness that exits cleanly on stdin EOF used to be the end of
``escalate_close``: the leader was reaped and the grandchild -- the fakes'
``NVSH_FAKE_GRANDCHILD=1`` ``sleep 600`` -- lived on, orphaned. These cases
run each persistent adapter (pi, codex app-server, acp, agy warm and cold)
through an ordinary ``run()`` then ``close()`` and assert the grandchild is
gone within 3s.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess  # nosec B404 - fixed argv, test fixture
import sys
from pathlib import Path

import pytest

from nvsh.agent import AgentContext, AgentRequest, EventKind, RequestKind
from nvsh.agent import _subprocess as sp
from nvsh.agent.acp import AcpAgent
from nvsh.agent.agy import AgyAgent
from nvsh.agent.codex import CodexAgent
from nvsh.agent.pi import PiAgent
from tests import _fake_adapters
from tests.test_agent_agy import TEXT_TURN_STDOUT
from tests.test_agent_subprocess import _pid_alive, _wait_gone


def _grandchild_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return _fake_adapters.fake_env(
        HOME=str(home),
        XDG_STATE_HOME=str(tmp_path / "state"),
        NVSH_FAKE_GRANDCHILD="1",
        NVSH_FAKE_PID_FILE=str(tmp_path / "pids.json"),
        **extra,
    )


def _pi(tmp_path: Path):
    return (
        PiAgent(pi_path="pi", env=_grandchild_env(tmp_path)),
        _fake_adapters.conformance_context(),
    )


def _codex(tmp_path: Path):
    agent = CodexAgent({}, binary="codex-app-server", env=_grandchild_env(tmp_path))
    return agent, _fake_adapters.conformance_context()


def _acp(tmp_path: Path):
    agent = AcpAgent(["acp"], "fake", env=_grandchild_env(tmp_path, NVSH_TEST_ACP_SCRIPT="[]"))
    return agent, _fake_adapters.conformance_context()


def _agy(tmp_path: Path, *, warm: bool):
    turn = {"stdout": TEXT_TURN_STDOUT, "exit_code": 0}
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"turns": [turn]} if warm else turn), encoding="utf-8")
    agent = AgyAgent(warm=warm, env=_grandchild_env(tmp_path, NVSH_FAKE_EVENTS=str(path)))
    # An empty context (cwd="") keeps warm agy from rebinding to a new tree.
    return agent, AgentContext()


FACTORIES = {
    "pi": _pi,
    "codex": _codex,
    "acp": _acp,
    "agy-warm": lambda tmp_path: _agy(tmp_path, warm=True),
    "agy-cold": lambda tmp_path: _agy(tmp_path, warm=False),
}


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_close_after_run_leaves_no_grandchild(tmp_path: Path, name: str) -> None:
    agent, context = FACTORIES[name](tmp_path)
    request = AgentRequest(kind=RequestKind.EXPLICIT, prompt="say OK")
    pids: dict = {}
    try:
        agent.start()
        events = []
        for event in agent.run(request, context):
            events.append(event)
            if event.kind is EventKind.PROPOSAL:
                _fake_adapters.answer_proposal(agent, event)
        assert events, f"{name}: {events}"
        assert events[-1].kind is EventKind.DONE, f"{name}: {events}"
        pids = json.loads((tmp_path / "pids.json").read_text(encoding="utf-8"))
        if name != "agy-cold":  # a cold turn reaps its own tree at turn end (below)
            assert _pid_alive(pids["grandchild"]), f"{name}: grandchild never started"
    finally:
        agent.close()
    try:
        assert _wait_gone([pids["harness"], pids["grandchild"]], within=3.0) == []
    finally:
        _fake_adapters.kill_fake_pids(tmp_path / "pids.json")


def test_cold_agy_turn_end_reaps_its_grandchild_before_close(tmp_path: Path) -> None:
    """Each cold agy turn is its own process tree: it is reaped when the turn ends."""
    agent, context = _agy(tmp_path, warm=False)
    try:
        events = list(agent.run(AgentRequest(kind=RequestKind.EXPLICIT, prompt="say OK"), context))
        assert events[-1].kind is EventKind.DONE, events
        pids = json.loads((tmp_path / "pids.json").read_text(encoding="utf-8"))
        assert _wait_gone([pids["harness"], pids["grandchild"]], within=3.0) == []
    finally:
        agent.close()
        _fake_adapters.kill_fake_pids(tmp_path / "pids.json")


# -- escalate_close: the group reap itself --------------------------------

_EOF_EXITS_LEAVING_GRANDCHILD = (
    "import subprocess, sys\n"
    "child = subprocess.Popen(['sleep', '600'])\n"
    "print(child.pid, flush=True)\n"
    "sys.stdin.read()\n"
)


def _spawn(script: str, *, new_session: bool) -> tuple[subprocess.Popen, int]:
    proc = subprocess.Popen(  # nosec B603 - fixed argv, test fixture
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=new_session,
    )
    assert proc.stdout is not None
    return proc, int(proc.stdout.readline().strip())


def test_escalate_close_reaps_the_group_when_the_leader_exits_on_eof() -> None:
    proc, grandchild = _spawn(_EOF_EXITS_LEAVING_GRANDCHILD, new_session=True)
    assert sp.escalate_close(proc, wait=2.0, grace=0.5) == 0
    assert _wait_gone([proc.pid, grandchild], within=3.0) == []


def test_escalate_close_reaps_the_group_of_an_already_reaped_leader() -> None:
    proc, grandchild = _spawn(_EOF_EXITS_LEAVING_GRANDCHILD, new_session=True)
    assert proc.stdin is not None
    proc.stdin.close()
    proc.wait(timeout=5)
    assert _pid_alive(grandchild)
    sp.escalate_close(proc, wait=0.5, grace=0.5)
    assert _wait_gone([grandchild], within=3.0) == []


def test_escalate_close_never_signals_the_callers_own_group(monkeypatch) -> None:
    """A child sharing nvsh's group: its leftover grandchild is not ours to reap
    through a group signal, and nvsh's own group is never signalled."""
    proc, grandchild = _spawn(_EOF_EXITS_LEAVING_GRANDCHILD, new_session=False)
    signalled: list[int] = []
    real_killpg = os.killpg

    def spy(pgid: int, sig: int) -> None:
        signalled.append(pgid)
        real_killpg(pgid, sig)

    monkeypatch.setattr(sp.os, "killpg", spy)
    try:
        assert sp.escalate_close(proc, wait=2.0, grace=0.2) == 0
        assert os.getpgrp() not in signalled
    finally:
        os.kill(grandchild, signal.SIGKILL)
