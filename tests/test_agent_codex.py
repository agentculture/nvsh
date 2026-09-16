"""CodexAgent over ``codex app-server``, plus the ``codex exec --json`` fallback.

The end-to-end tests drive the real adapter against ``tests/fakes/codex-app-server``,
which replays a redacted transcript recorded from codex-cli 0.147.0 -- so what is
exercised here is the adapter's actual JSON-RPC client (threads, reader thread,
request/response correlation, approval round trip, interrupt), not a mock of it.
The mapping tests call the parsing helpers directly with wire objects whose field
names come from ``codex app-server generate-json-schema``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from nvsh.agent.base import AgentContext, AgentEvent, AgentRequest, EventKind, RequestKind
from nvsh.agent.codex import (
    APPROVAL_POLICY,
    BANNED_TOKENS,
    SANDBOX_MODE,
    CodexAgent,
    CodexRpcError,
    approval_fields,
)

FAKES_DIR = Path(__file__).parent / "fakes"
FAKE_BINARY = "codex-app-server"


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env["NVSH_FAKE_CODEX_APPROVAL_TIMEOUT"] = "10"
    env.update(extra)
    return env


def _agent(**kwargs) -> CodexAgent:
    kwargs.setdefault("binary", FAKE_BINARY)
    return CodexAgent({}, **kwargs)


def _request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, command="apt upgrade", exit_code=1)


def _context() -> AgentContext:
    return AgentContext(platform="dgx-spark", output="mount: /boot is read-only", cwd="/work")


def _drive(agent: CodexAgent, approve: bool | None = True) -> list[AgentEvent]:
    """Run one turn, answering the approval request the transcript raises."""
    events: list[AgentEvent] = []
    agent.start()
    try:
        for event in agent.run(_request(), _context()):
            events.append(event)
            if event.kind == EventKind.PROPOSAL and approve is not None:
                agent.respond_approval(event.args["request_id"], approve)
    finally:
        agent.close()
    return events


def _sent(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _by_method(sent: list[dict], method: str) -> list[dict]:
    return [line for line in sent if line.get("method") == method]


# ---------------------------------------------------------------------------
# argv and thread/start params: what nvsh is allowed to say
# ---------------------------------------------------------------------------


def test_model_and_effort_travel_as_config_overrides() -> None:
    agent = _agent(model="gpt-5.6-sol", effort="xhigh")
    argv = agent.app_server_argv()
    assert argv[0] == FAKE_BINARY
    assert argv[-1] == "app-server"
    assert "-c" in argv
    assert "model=gpt-5.6-sol" in argv
    assert "model_reasoning_effort=xhigh" in argv


def test_model_and_effort_are_passed_verbatim() -> None:
    # Opaque strings (decision c24): nvsh never validates or rewrites them.
    agent = _agent(model="Some/Weird-Model:v2", effort="ultra-maximal")
    argv = agent.app_server_argv()
    assert "model=Some/Weird-Model:v2" in argv
    assert "model_reasoning_effort=ultra-maximal" in argv


def test_unset_model_and_effort_add_no_flags() -> None:
    assert _agent().app_server_argv() == [FAKE_BINARY, "app-server"]


def test_extra_args_are_appended_before_the_subcommand() -> None:
    agent = _agent(extra_args=["--enable", "some_feature"])
    assert agent.app_server_argv() == [FAKE_BINARY, "--enable", "some_feature", "app-server"]


def test_config_table_supplies_model_effort_and_extra_args() -> None:
    agent = CodexAgent(
        {"model": "m", "effort": "e", "extra_args": ["--x"], "approval": "harness"},
        binary=FAKE_BINARY,
    )
    assert agent.app_server_argv() == [
        FAKE_BINARY,
        "-c",
        "model=m",
        "-c",
        "model_reasoning_effort=e",
        "--x",
        "app-server",
    ]
    assert agent.capabilities().approval == "harness"


def test_app_server_argv_never_starts_the_daemon() -> None:
    assert "daemon" not in _agent(model="m", effort="e").app_server_argv()


def test_exec_fallback_argv_carries_the_same_overrides() -> None:
    agent = _agent(model="m", effort="e")
    argv = agent._argv(_request(), _context())
    assert argv[:6] == [FAKE_BINARY, "-c", "model=m", "-c", "model_reasoning_effort=e", "exec"]
    assert argv[6] == "--json"
    assert "apt upgrade" in argv[7]


def test_thread_start_asks_on_request_with_a_read_only_sandbox() -> None:
    params = _agent().thread_start_params(_context())
    assert params["approvalPolicy"] == APPROVAL_POLICY == "on-request"
    assert params["sandbox"] == SANDBOX_MODE == "read-only"
    assert params["cwd"] == "/work"


@pytest.mark.parametrize("banned", BANNED_TOKENS)
def test_banned_tokens_appear_in_neither_argv_nor_params(banned: str) -> None:
    """``never``, ``danger-full-access`` and ``--full-auto`` are unreachable.

    Checked element-wise rather than over a joined blob: the ``exec``
    fallback's last argv element is the *prompt*, whose English prose
    legitimately contains the word "never" ("Never propose ``sudo``
    unless..."), and a substring match over that would be testing the
    system brief's wording rather than codex's policy flags.
    """
    agent = _agent(model="m", effort="e", extra_args=["--enable", "x"])
    flags = agent.app_server_argv() + agent._argv(_request(), _context())[:-1]
    for element in flags:
        assert banned not in element, element
    for value in agent.thread_start_params(_context()).values():
        assert banned not in json.dumps(value), value


# ---------------------------------------------------------------------------
# The recorded transcript, end to end
# ---------------------------------------------------------------------------


def test_replayed_turn_streams_thinking_text_tools_and_a_proposal(tmp_path: Path) -> None:
    sent = tmp_path / "sent.jsonl"
    agent = _agent(env=_env(NVSH_FAKE_CODEX_COMMANDS=str(sent)))
    events = _drive(agent, approve=True)

    kinds = [event.kind for event in events]
    assert EventKind.THINKING in kinds
    assert EventKind.TEXT_DELTA in kinds
    assert EventKind.PROPOSAL in kinds
    assert EventKind.TOOL_CALL in kinds
    assert EventKind.TOOL_RESULT in kinds
    assert kinds[-1] == EventKind.DONE

    thinking = "".join(e.text for e in events if e.kind == EventKind.THINKING)
    assert "read-only mount" in thinking
    assert "/boot is mounted ro" in thinking

    # The proposal carries codex's own command and reason, never prompt text.
    proposal = next(e.proposal for e in events if e.kind == EventKind.PROPOSAL)
    assert proposal.command == "/bin/bash -lc 'mount -o remount,rw /boot'"
    assert "remounting /boot" in proposal.rationale.lower()

    tool_call = next(e for e in events if e.kind == EventKind.TOOL_CALL)
    assert tool_call.tool == "commandExecution"
    assert tool_call.args["command"] == proposal.command
    tool_result = next(e for e in events if e.kind == EventKind.TOOL_RESULT)
    assert tool_result.result["status"] == "completed"
    assert tool_result.result["exitCode"] == 0


def test_the_client_speaks_the_recorded_request_sequence(tmp_path: Path) -> None:
    sent = tmp_path / "sent.jsonl"
    agent = _agent(model="m", env=_env(NVSH_FAKE_CODEX_COMMANDS=str(sent)))
    _drive(agent, approve=True)
    lines = _sent(sent)

    methods = [line.get("method") for line in lines if line.get("method")]
    assert methods[:3] == ["initialize", "thread/start", "turn/start"]

    initialize = _by_method(lines, "initialize")[0]
    assert initialize["params"]["clientInfo"]["name"] == "nvsh"

    thread_start = _by_method(lines, "thread/start")[0]["params"]
    assert thread_start["approvalPolicy"] == "on-request"
    assert thread_start["sandbox"] == "read-only"

    turn_start = _by_method(lines, "turn/start")[0]["params"]
    assert turn_start["threadId"] == "th-fake-0000-0001"
    assert turn_start["input"][0]["type"] == "text"
    assert "apt upgrade" in turn_start["input"][0]["text"]


def test_approval_is_answered_with_accept_and_decline(tmp_path: Path) -> None:
    approved = tmp_path / "yes.jsonl"
    _drive(_agent(env=_env(NVSH_FAKE_CODEX_COMMANDS=str(approved))), approve=True)
    decisions = [line["result"]["decision"] for line in _sent(approved) if "result" in line]
    assert decisions == ["accept"]

    declined = tmp_path / "no.jsonl"
    events = _drive(_agent(env=_env(NVSH_FAKE_CODEX_COMMANDS=str(declined))), approve=False)
    decisions = [line["result"]["decision"] for line in _sent(declined) if "result" in line]
    assert decisions == ["decline"]
    result = next(e for e in events if e.kind == EventKind.TOOL_RESULT)
    assert result.result["status"] == "declined"


def test_legacy_exec_command_approval_uses_the_older_vocabulary(tmp_path: Path) -> None:
    sent = tmp_path / "sent.jsonl"
    env = _env(NVSH_FAKE_CODEX_COMMANDS=str(sent), NVSH_FAKE_CODEX_LEGACY_APPROVAL="1")
    events = _drive(_agent(env=env), approve=True)

    proposal = next(e.proposal for e in events if e.kind == EventKind.PROPOSAL)
    # The legacy method carries argv as a list; it is joined, never dropped.
    assert proposal.command == "/bin/bash -lc mount -o remount,rw /boot"
    decisions = [line["result"]["decision"] for line in _sent(sent) if "result" in line]
    assert decisions == ["approved"]


def test_an_unanswered_approval_is_declined_on_close(tmp_path: Path) -> None:
    sent = tmp_path / "sent.jsonl"
    agent = _agent(env=_env(NVSH_FAKE_CODEX_COMMANDS=str(sent)))
    agent.start()
    try:
        for event in agent.run(_request(), _context()):
            if event.kind == EventKind.PROPOSAL:
                break
    finally:
        agent.close()
    decisions = [line["result"]["decision"] for line in _sent(sent) if "result" in line]
    assert decisions == ["decline"]


def test_cancel_interrupts_the_turn(tmp_path: Path) -> None:
    sent = tmp_path / "sent.jsonl"
    agent = _agent(env=_env(NVSH_FAKE_CODEX_COMMANDS=str(sent)))
    seen: list[AgentEvent] = []
    agent.start()
    try:
        for event in agent.run(_request(), _context()):
            seen.append(event)
            if event.kind == EventKind.THINKING:
                agent.cancel()
    finally:
        agent.close()
    interrupts = _by_method(_sent(sent), "turn/interrupt")
    assert interrupts, _sent(sent)
    assert interrupts[0]["params"] == {
        "threadId": "th-fake-0000-0001",
        "turnId": "turn-fake-0000-0001",
    }
    # Nothing is yielded after cancel() (the contract the conformance suite
    # holds every adapter to).
    assert seen[-1].kind == EventKind.THINKING


def test_steer_sends_turn_steer_with_the_active_turn_id(tmp_path: Path) -> None:
    sent = tmp_path / "sent.jsonl"
    agent = _agent(env=_env(NVSH_FAKE_CODEX_COMMANDS=str(sent)))
    steered: list[bool] = []
    agent.start()
    try:
        for event in agent.run(_request(), _context()):
            if event.kind == EventKind.TEXT_DELTA and not steered:
                steered.append(agent.steer("check /etc/fstab too"))
            if event.kind == EventKind.PROPOSAL:
                agent.respond_approval(event.args["request_id"], False)
    finally:
        agent.close()
    assert steered == [True]
    steer = _by_method(_sent(sent), "turn/steer")[0]["params"]
    assert steer["threadId"] == "th-fake-0000-0001"
    assert steer["expectedTurnId"] == "turn-fake-0000-0001"
    assert steer["input"] == [{"type": "text", "text": "check /etc/fstab too"}]


def test_steer_outside_a_turn_reports_false() -> None:
    assert _agent().steer("anything") is False


def test_resume_rejoins_a_stored_thread(tmp_path: Path) -> None:
    sent = tmp_path / "sent.jsonl"
    agent = _agent(env=_env(NVSH_FAKE_CODEX_COMMANDS=str(sent)))
    try:
        assert agent.resume("th-fake-0000-0001") == "th-fake-0000-0001"
        assert agent.thread_id == "th-fake-0000-0001"
    finally:
        agent.close()
    resume = _by_method(_sent(sent), "thread/resume")[0]["params"]
    assert resume["threadId"] == "th-fake-0000-0001"
    assert resume["approvalPolicy"] == "on-request"
    assert resume["sandbox"] == "read-only"


# ---------------------------------------------------------------------------
# The exec --json fallback
# ---------------------------------------------------------------------------


def test_initialize_failure_falls_back_to_codex_exec_json() -> None:
    agent = _agent(env=_env(NVSH_FAKE_CODEX_NO_APP_SERVER="1"))
    events = _drive(agent, approve=None)
    assert [e.kind for e in events] == [
        EventKind.STATUS,
        EventKind.TEXT_DELTA,
        EventKind.DONE,
    ]
    assert events[1].text == "from codex exec"
    assert agent.ensure_app_server() is False


def test_a_missing_binary_falls_back_and_then_reports_the_failure() -> None:
    agent = _agent(binary="nvsh-no-such-codex-binary")
    events = _drive(agent, approve=None)
    assert events[-1].kind == EventKind.ERROR
    assert "nvsh-no-such-codex-binary" in events[-1].error


def test_app_server_can_be_switched_off_entirely() -> None:
    agent = _agent(app_server=False)
    assert agent.ensure_app_server() is False


def test_resume_without_an_app_server_raises() -> None:
    agent = _agent(app_server=False)
    with pytest.raises(CodexRpcError):
        agent.resume("th-fake-0000-0001")


# ---------------------------------------------------------------------------
# Notification mapping, straight from the schema's field names
# ---------------------------------------------------------------------------


def _map(agent: CodexAgent, obj: dict) -> AgentEvent | None:
    return agent._map(obj)


def test_reasoning_deltas_map_to_thinking() -> None:
    agent = _agent()
    for method in ("item/reasoning/textDelta", "item/reasoning/summaryTextDelta"):
        event = _map(agent, {"method": method, "params": {"delta": "hm"}})
        assert event == AgentEvent(kind=EventKind.THINKING, text="hm")


def test_agent_message_delta_maps_to_text_delta() -> None:
    event = _map(_agent(), {"method": "item/agentMessage/delta", "params": {"delta": "hi"}})
    assert event == AgentEvent(kind=EventKind.TEXT_DELTA, text="hi")


def test_non_command_items_are_not_tool_events() -> None:
    agent = _agent()
    obj = {"method": "item/started", "params": {"item": {"type": "userMessage", "id": "u"}}}
    assert _map(agent, obj) is None


def test_turn_completed_failed_maps_to_error() -> None:
    obj = {
        "method": "turn/completed",
        "params": {"turn": {"status": "failed", "error": {"message": "at capacity"}}},
    }
    event = _map(_agent(), obj)
    assert event.kind == EventKind.ERROR
    assert event.error == "at capacity"


def test_turn_completed_interrupted_maps_to_done() -> None:
    obj = {"method": "turn/completed", "params": {"turn": {"status": "interrupted"}}}
    assert _map(_agent(), obj).kind == EventKind.DONE


def test_error_notification_maps_to_error() -> None:
    obj = {"method": "error", "params": {"error": {"message": "stream disconnected"}}}
    event = _map(_agent(), obj)
    assert event.kind == EventKind.ERROR
    assert event.error == "stream disconnected"


def test_bookkeeping_notifications_are_dropped() -> None:
    agent = _agent()
    for method in ("thread/tokenUsage/updated", "turn/started", "serverRequest/resolved"):
        assert _map(agent, {"method": method, "params": {}}) is None


def test_an_unknown_notification_surfaces_as_status() -> None:
    event = _map(_agent(), {"method": "some/new/thing", "params": {}})
    assert event == AgentEvent(kind=EventKind.STATUS, text="some/new/thing")


def test_a_response_is_not_an_event() -> None:
    assert _map(_agent(), {"id": 7, "result": {}}) is None


def test_an_unknown_server_request_surfaces_as_status_not_a_proposal() -> None:
    obj = {"id": 3, "method": "item/tool/requestUserInput", "params": {}}
    event = _map(_agent(), obj)
    assert event.kind == EventKind.STATUS


def test_approval_fields_reads_string_and_argv_commands() -> None:
    assert approval_fields({"command": "ls -l", "reason": "why"}) == ("ls -l", "why")
    assert approval_fields({"command": ["ls", "-l"]}) == ("ls -l", "")
    # A request carrying no command yields no command: approving it runs nothing.
    assert approval_fields({"reason": "just asking"}) == ("", "just asking")


# ---------------------------------------------------------------------------
# Self-reported capabilities
# ---------------------------------------------------------------------------


def test_capabilities_report_the_app_server_surface() -> None:
    caps = _agent(env=_env()).capabilities()
    assert caps.streaming is True
    assert caps.thinking is True
    assert caps.effort is True
    assert caps.tool_calling is True
    assert caps.cancellation is True
    assert caps.persistent_session is True
    assert caps.local_model is False
    assert caps.approval == "nvsh"
    assert caps.unmediated_file_access is True
    assert caps.path.endswith(FAKE_BINARY)


def test_close_is_idempotent() -> None:
    agent = _agent(env=_env())
    agent.start()
    list(agent.run(_request(), _context()))[:1]
    agent.close()
    agent.close()


# ---------------------------------------------------------------------------
# force_stop: kill_tree on an app-server that ignores turn/interrupt
# (plan reliable-agent-stop, task t5)
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """True while *pid* is a live (non-zombie) process.

    Mirrors ``tests/test_agent_subprocess.py``'s helper: a zombie still
    answers ``os.kill(pid, 0)`` (the kernel keeps the pid entry until it is
    reaped), so a "no pid alive" check reads ``/proc/<pid>/stat`` and treats
    state ``Z``/``X`` as dead.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            stat = handle.read()
    except OSError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def test_force_stop_kills_the_app_server_pid_when_it_ignores_interrupt(
    tmp_path: Path,
) -> None:
    sent = tmp_path / "sent.jsonl"
    env = _env(NVSH_FAKE_CODEX_COMMANDS=str(sent), NVSH_FAKE_IGNORE_CANCEL="1")
    agent = _agent(env=env)
    agent.start()
    events = agent.run(_request(), _context())
    # Drive far enough that the app-server is up, a thread/turn exist, and
    # the turn is actually in flight -- exactly the moment an operator would
    # hit "stop" on a hung turn.
    for event in events:
        if event.kind == EventKind.THINKING:
            break

    proc = agent._rpc
    assert proc is not None and proc.poll() is None
    pid = proc.pid

    agent.force_stop()
    events.close()

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.05)
    assert not _pid_alive(pid), "codex app-server pid survived force_stop() past 3s"

    # The app-server handle and session are gone: the next run() must not
    # write into a dead pipe or resume a thread that no longer exists.
    assert agent._rpc is None
    assert agent.thread_id is None


def test_run_after_force_stop_starts_a_fresh_app_server_with_new_session_status(
    tmp_path: Path,
) -> None:
    sent = tmp_path / "sent.jsonl"
    env = _env(NVSH_FAKE_CODEX_COMMANDS=str(sent), NVSH_FAKE_IGNORE_CANCEL="1")
    agent = _agent(env=env)
    agent.start()
    events = agent.run(_request(), _context())
    for event in events:
        if event.kind == EventKind.THINKING:
            break
    agent.force_stop()
    events.close()

    # A fresh run() must succeed end to end: new handshake, new thread, new
    # turn, reported with a 'new session' status ahead of the turn's events.
    second = _drive(agent, approve=True)
    kinds = [event.kind for event in second]
    assert kinds[0] == EventKind.STATUS and second[0].text == "new session"
    assert kinds[-1] == EventKind.DONE

    # initialize/thread/start/turn/start all ran twice -- once for the
    # killed app-server, once for the fresh one force_stop() forced -- proof
    # it is a genuinely new process and thread, not a resumed dead one.
    # (NVSH_FAKE_CODEX_COMMANDS appends across both fake processes.)
    methods = [line.get("method") for line in _sent(sent) if line.get("method")]
    assert methods.count("initialize") == 2
    assert methods.count("thread/start") == 2
    assert methods.count("turn/start") == 2


def test_force_stop_on_an_agent_that_never_started_does_not_raise() -> None:
    _agent(env=_env()).force_stop()
