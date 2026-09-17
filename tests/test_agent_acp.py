"""AcpAgent: the generic ACP client, its qwen/kiro entries, and its guards.

Everything but the last test runs offline against ``tests/fakes/acp``, a
scripted fake ACP agent replaying a transcript recorded from a live
``qwen --acp`` (0.23.3) session, permission request included. The final test
is a live smoke against the real ``qwen --acp`` on this machine and is
skipped unless ``NVSH_LIVE_QWEN=1``, the same way the live pi tests are
gated.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from nvsh.agent import acp
from nvsh.agent.acp import AcpAgent, AcpError, build
from nvsh.agent.base import AgentContext, AgentEvent, AgentRequest, EventKind, RequestKind
from tests import test_agent_conformance as conformance
from tests._fake_adapters import reap_fake_pids  # noqa: F401 - fixture, used below

# Teardown kills the fake harness/grandchild pids each test's fakes recorded
# (task t23), so a failing or respawning case leaks no ``sleep 600``.
pytestmark = pytest.mark.usefixtures("reap_fake_pids")

FAKES_DIR = Path(__file__).parent / "fakes"


def _fake_env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env.update(extra)
    return env


def _request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, command="ls /nope", exit_code=2)


def _context() -> AgentContext:
    return AgentContext(platform="platform: dgx-spark", output="ls: /nope: No such file")


def _fake_agent(**kwargs) -> AcpAgent:
    """An AcpAgent pointed at ``tests/fakes/acp`` instead of a real harness."""
    env = kwargs.pop("env", None) or _fake_env()
    kwargs.setdefault("thinking", True)
    return AcpAgent(["acp"], kwargs.pop("name", "fake"), env=env, **kwargs)


# -- entries and argv ------------------------------------------------------


def test_kiro_entry_argv_carries_model_and_extra_args():
    agent = build("kiro", {"model": "claude-sonnet-4.5", "extra_args": ["--no-interactive"]})
    assert agent.build_argv() == [
        "kiro-cli",
        "acp",
        "--model",
        "claude-sonnet-4.5",
        "--no-interactive",
    ]


def test_kiro_argv_never_carries_trust_all_tools():
    argv = build("kiro", {"model": "glm-5"}).build_argv()
    assert "--trust-all-tools" not in argv
    with pytest.raises(ValueError, match="bypasses approval"):
        build("kiro", {"extra_args": ["--trust-all-tools"]})


def test_qwen_entry_argv_has_no_model_flag():
    # qwen selects its model in-session (session/set_config_option), so the
    # model never reaches argv.
    assert build("qwen", {"model": "worker"}).build_argv() == ["qwen", "--acp"]


def test_unknown_entry_is_refused():
    with pytest.raises(ValueError, match="unknown ACP harness"):
        build("nope")


# -- decision c53: plan by default, harness approval opt-in ----------------


def test_qwen_defaults_to_plan_mode_without_tool_calling():
    agent = build("qwen")
    caps = agent.capabilities()
    assert agent._mode == "plan"
    assert caps.tool_calling is False
    assert caps.approval == "nvsh"
    assert caps.thinking is True
    assert caps.local_model is True


def test_qwen_harness_approval_opts_into_default_mode():
    agent = build("qwen", {"approval": "harness"})
    caps = agent.capabilities()
    assert agent._mode == "default"
    assert caps.approval == "harness"
    assert caps.tool_calling is True


def test_kiro_reports_an_approval_channel_and_no_thinking():
    caps = build("kiro").capabilities()
    assert caps.tool_calling is True
    assert caps.thinking is False
    assert caps.approval == "nvsh"
    # kiro's modes are the operator's own agent configs: nvsh sets none.
    assert build("kiro")._mode is None


def test_every_entry_reports_unmediated_file_access_over_acp():
    for name in acp.ENTRIES:
        caps = build(name).capabilities()
        assert caps.unmediated_file_access is True
        assert caps.path == "acp"
        assert caps.persistent_session is True


def test_no_entry_can_be_configured_into_a_bypass_mode():
    for name in acp.ENTRIES:
        for approval in ("nvsh", "harness"):
            agent = build(name, {"approval": approval})
            assert (agent._mode or "") not in acp.FORBIDDEN_MODES


@pytest.mark.parametrize("mode", ["auto", "auto-edit", "yolo"])
def test_bypass_modes_are_refused_at_construction(mode):
    with pytest.raises(ValueError, match="without asking"):
        AcpAgent(["qwen", "--acp"], "qwen", mode=mode)


def test_initialize_timeout_is_env_overridable():
    assert acp.initialize_timeout({}) == acp._INITIALIZE_TIMEOUT_SECONDS
    assert acp.initialize_timeout({acp.INITIALIZE_TIMEOUT_ENV: "2.5"}) == 2.5
    # Junk and non-positive values fall back rather than disabling the bound.
    assert acp.initialize_timeout({acp.INITIALIZE_TIMEOUT_ENV: "x"}) == (
        acp._INITIALIZE_TIMEOUT_SECONDS
    )
    assert acp.initialize_timeout({acp.INITIALIZE_TIMEOUT_ENV: "0"}) == (
        acp._INITIALIZE_TIMEOUT_SECONDS
    )


# -- the recorded transcript -----------------------------------------------


def _drive(agent: AcpAgent, answer: dict | None = None) -> list[AgentEvent]:
    """Run one turn, answering the transcript's permission with ``answer``."""
    events: list[AgentEvent] = []
    try:
        for event in agent.run(_request(), _context()):
            events.append(event)
            if event.kind is EventKind.PROPOSAL and answer is not None:
                agent.respond_ui(event.args["request_id"], **answer)
    finally:
        agent.close()
    return events


def test_recorded_transcript_maps_to_nvsh_events():
    agent = _fake_agent()
    events = _drive(agent, answer={"value": "once"})
    kinds = [event.kind for event in events]

    assert EventKind.THINKING in kinds
    assert EventKind.TEXT_DELTA in kinds
    assert kinds[-1] is EventKind.DONE
    assert kinds.index(EventKind.TOOL_CALL) < kinds.index(EventKind.PROPOSAL)
    assert kinds.index(EventKind.PROPOSAL) < kinds.index(EventKind.TOOL_RESULT)

    thinking = "".join(e.text for e in events if e.kind is EventKind.THINKING)
    assert "CPU cores" in thinking
    text = "".join(e.text for e in events if e.kind is EventKind.TEXT_DELTA)
    assert "20 CPU cores" in text

    tool_call = next(e for e in events if e.kind is EventKind.TOOL_CALL)
    assert tool_call.tool == "run_shell_command"
    tool_result = next(e for e in events if e.kind is EventKind.TOOL_RESULT)
    assert "Output: 20" in str(tool_result.result)


def test_permission_request_becomes_a_proposal_carrying_the_command():
    agent = _fake_agent()
    events = _drive(agent, answer={"value": "once"})
    proposal = next(e for e in events if e.kind is EventKind.PROPOSAL).proposal
    assert proposal.command == "nproc"
    assert proposal.rationale == "Count CPU cores with nproc"


def test_chatty_updates_never_reach_the_panel():
    events = _drive(_fake_agent(), answer={"value": "once"})
    # usage_update / available_commands_update carry nothing an operator can
    # act on, and the vendor request in the transcript is not an event.
    assert [e for e in events if e.kind is EventKind.STATUS] == []


def _answers(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_allow_selects_allow_once_never_allow_always(tmp_path):
    log = tmp_path / "frames.jsonl"
    agent = _fake_agent(env=_fake_env(NVSH_TEST_ACP_COMMANDS=str(log)))
    _drive(agent, answer={"value": "once"})

    outcomes = [f["result"]["outcome"] for f in _answers(log) if "result" in f]
    assert outcomes == [{"outcome": "selected", "optionId": "proceed_once"}]


def test_deny_selects_reject_once(tmp_path):
    log = tmp_path / "frames.jsonl"
    agent = _fake_agent(env=_fake_env(NVSH_TEST_ACP_COMMANDS=str(log)))
    _drive(agent, answer={"value": "deny", "reason": "not now"})

    outcomes = [f["result"]["outcome"] for f in _answers(log) if "result" in f]
    assert outcomes == [{"outcome": "selected", "optionId": "cancel"}]


def test_cancelled_dialog_is_denied_too(tmp_path):
    log = tmp_path / "frames.jsonl"
    agent = _fake_agent(env=_fake_env(NVSH_TEST_ACP_COMMANDS=str(log)))
    _drive(agent, answer={"cancelled": True})

    outcomes = [f["result"]["outcome"] for f in _answers(log) if "result" in f]
    assert outcomes == [{"outcome": "selected", "optionId": "cancel"}]


def test_handshake_advertises_no_filesystem_and_sets_the_mode(tmp_path):
    log = tmp_path / "frames.jsonl"
    agent = build("qwen", env=_fake_env(NVSH_TEST_ACP_COMMANDS=str(log)))
    # Same entry semantics, but launched through the fake binary.
    agent._command = ["acp"]
    _drive(agent, answer={"value": "once"})

    frames = _answers(log)
    initialize = next(f for f in frames if f.get("method") == "initialize")
    assert initialize["params"]["clientCapabilities"] == {
        "fs": {"readTextFile": False, "writeTextFile": False}
    }
    assert "terminal" not in initialize["params"]["clientCapabilities"]
    assert initialize["params"]["protocolVersion"] == acp.PROTOCOL_VERSION

    new_session = next(f for f in frames if f.get("method") == "session/new")
    assert new_session["params"]["mcpServers"] == []

    set_mode = next(f for f in frames if f.get("method") == "session/set_mode")
    assert set_mode["params"]["modeId"] == "plan"


def test_model_and_effort_are_selected_in_session(tmp_path):
    log = tmp_path / "frames.jsonl"
    agent = build(
        "qwen",
        {"model": "associate", "effort": "high"},
        env=_fake_env(NVSH_TEST_ACP_COMMANDS=str(log)),
    )
    agent._command = ["acp"]
    _drive(agent, answer={"value": "once"})

    options = {
        f["params"]["configId"]: f["params"]["value"]
        for f in _answers(log)
        if f.get("method") == "session/set_config_option"
    }
    assert options == {"model": "associate(openai)", "reasoning_effort": "high"}


def test_unknown_model_leaves_the_harness_default_alone(tmp_path):
    log = tmp_path / "frames.jsonl"
    agent = build(
        "qwen", {"model": "no-such-model"}, env=_fake_env(NVSH_TEST_ACP_COMMANDS=str(log))
    )
    agent._command = ["acp"]
    _drive(agent, answer={"value": "once"})
    assert not [f for f in _answers(log) if f.get("method") == "session/set_config_option"]


def test_vendor_requests_are_answered_method_not_found(tmp_path):
    log = tmp_path / "frames.jsonl"
    agent = _fake_agent(env=_fake_env(NVSH_TEST_ACP_COMMANDS=str(log)))
    _drive(agent, answer={"value": "once"})

    errors = [f for f in _answers(log) if "error" in f]
    assert [f["error"]["code"] for f in errors] == [-32601]
    assert errors[0]["id"] == 4242


def test_session_resume_reuses_a_stored_session(tmp_path):
    log = tmp_path / "frames.jsonl"
    agent = _fake_agent(env=_fake_env(NVSH_TEST_ACP_COMMANDS=str(log)))
    try:
        agent.resume("acp-stored-session-0002")
        assert agent.session_id == "acp-stored-session-0002"
    finally:
        agent.close()

    resume = next(f for f in _answers(log) if f.get("method") == "session/resume")
    assert resume["params"]["sessionId"] == "acp-stored-session-0002"
    assert resume["params"]["mcpServers"] == []


# -- force_stop: kill_tree beats a harness that ignores session/cancel -----
# (plan reliable-agent-stop, task t6)

#: A script that pauses on a single permission request and never resolves
#: the turn on its own -- exactly what a still-open dialog needs to prove
#: force_stop() denies it rather than leaving it hanging.
_PAUSED_ON_PERMISSION_SCRIPT = json.dumps(
    [
        {
            "permission": {
                "toolCall": {"toolCallId": "c1", "status": "pending", "title": "shell"},
                "options": [{"optionId": "proceed_once", "name": "Allow", "kind": "allow_once"}],
            }
        }
    ]
)


def _pid_alive(pid: int) -> bool:
    """True while *pid* is a live (non-zombie) process.

    Mirrors ``tests/test_agent_subprocess.py``'s helper: a zombie (state
    ``Z``) is dead for this check's purposes even though ``os.kill(pid, 0)``
    would still succeed on it.
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


def _wait_gone(pids: list[int], within: float = 3.0) -> list[int]:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if _pid_alive(pid)]
        if not alive:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if _pid_alive(pid)]


def _wait_for_pid_file(path: Path, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written")


def _ignoring_env(log: Path, pid_file: Path, **extra: str) -> dict[str, str]:
    return _fake_env(
        NVSH_FAKE_IGNORE_CANCEL="1",
        NVSH_FAKE_GRANDCHILD="1",
        NVSH_FAKE_PID_FILE=str(pid_file),
        NVSH_TEST_ACP_SCRIPT=_PAUSED_ON_PERMISSION_SCRIPT,
        NVSH_TEST_ACP_COMMANDS=str(log),
        **extra,
    )


def test_force_stop_kills_the_process_tree_when_the_harness_ignores_cancel(tmp_path):
    """Acceptance: force_stop() leaves no pid alive within 3s.

    The fake ignores ``session/cancel`` and never resolves the turn on its
    own, and it starts a ``sleep 600`` grandchild at launch (its own
    process, outside nvsh's process group) to prove the whole tree -- not
    just the harness itself -- is killed.
    """
    log = tmp_path / "frames.jsonl"
    pid_file = tmp_path / "pids.json"
    agent = _fake_agent(env=_ignoring_env(log, pid_file))

    events = agent.run(_request(), _context())
    proposal = next(e for e in events if e.kind is EventKind.PROPOSAL)
    assert proposal.args["request_id"] in agent._pending

    pids = _wait_for_pid_file(pid_file)
    harness_pid, grandchild_pid = pids["harness"], pids["grandchild"]
    assert _pid_alive(harness_pid)
    assert _pid_alive(grandchild_pid)

    # A spy on the raw frame writer, not the fake's own record of what it
    # managed to read: force_stop() kills the process right after writing
    # these frames, so whether the fake's reader thread got scheduled in
    # time to log them before SIGTERM lands is a race nothing here should
    # depend on. What must be true unconditionally is that nvsh *attempted*
    # to send them.
    sent: list[dict] = []
    original_send = agent._send

    def _spy_send(obj: dict) -> bool:
        sent.append(obj)
        return original_send(obj)

    agent._send = _spy_send  # type: ignore[method-assign]

    agent.force_stop()
    events.close()

    assert _wait_gone([harness_pid, grandchild_pid]) == []
    # The open dialog was denied, not just abandoned.
    assert agent._pending == {}

    assert any(
        obj.get("result", {}).get("outcome") == {"outcome": "cancelled"} for obj in sent
    ), "the pending permission dialog was not denied"
    assert any(
        obj.get("method") == "session/cancel" for obj in sent
    ), "session/cancel was never sent, even best-effort"


def test_run_after_force_stop_reinitialises_a_new_session(tmp_path):
    """Acceptance: the next run() after force_stop() re-initialises and succeeds."""
    log = tmp_path / "frames.jsonl"
    pid_file = tmp_path / "pids.json"
    agent = _fake_agent(env=_ignoring_env(log, pid_file))

    events = agent.run(_request(), _context())
    next(e for e in events if e.kind is EventKind.PROPOSAL)
    first_pid = agent._proc.pid

    agent.force_stop()
    events.close()

    assert _wait_gone([first_pid]) == []
    assert agent._proc is None
    assert agent._session_id == ""

    # A plain, immediately-resolving script for the second turn: this test
    # is about re-initialisation succeeding, not about the cancel dance again.
    agent._env["NVSH_TEST_ACP_SCRIPT"] = "[]"
    second_events = list(agent.run(_request(), _context()))

    assert second_events[-1].kind is EventKind.DONE
    assert agent._proc is not None
    assert agent._proc.pid != first_pid
    assert agent._session_id

    initialize_calls = [f for f in _answers(log) if f.get("method") == "initialize"]
    new_session_calls = [f for f in _answers(log) if f.get("method") == "session/new"]
    assert len(initialize_calls) == 2, "the second run() must re-run the ACP handshake"
    assert len(new_session_calls) == 2

    agent.close()


# -- bounded initialize ----------------------------------------------------


def test_initialize_that_never_answers_yields_one_error_and_closes():
    agent = _fake_agent(env=_fake_env(NVSH_TEST_ACP_NO_INIT="1"), initialize_timeout_seconds=0.5)
    events = list(agent.run(_request(), _context()))
    agent.close()

    assert [e.kind for e in events] == [EventKind.ERROR]
    assert "did not answer initialize within 0.5s" in events[0].error
    assert agent._proc is None


def test_a_missing_binary_is_an_error_event_not_a_traceback():
    agent = AcpAgent(["nvsh-no-such-acp-binary"], "missing", env=_fake_env())
    events = list(agent.run(_request(), _context()))
    assert [e.kind for e in events] == [EventKind.ERROR]
    assert "failed to start" in events[0].error


def test_start_raises_rather_than_hanging_when_initialize_is_unanswered():
    agent = _fake_agent(env=_fake_env(NVSH_TEST_ACP_NO_INIT="1"), initialize_timeout_seconds=0.5)
    with pytest.raises(AcpError):
        agent.start()
    agent.close()


# -- conformance -----------------------------------------------------------


def _script_to_directives(script) -> list[dict]:
    """The conformance suite's AgentEvent script as fake-ACP directives."""
    directives: list[dict] = []
    for event in script:
        if event.kind is EventKind.STATUS:
            directives.append({"update": {"sessionUpdate": event.text}})
        elif event.kind is EventKind.TEXT_DELTA:
            directives.append(
                {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": event.text},
                    }
                }
            )
        elif event.kind is EventKind.ERROR:
            directives.append({"error": event.error})
        elif event.kind is EventKind.DONE:
            break  # the fake answers the prompt when its script runs out
        else:  # pragma: no cover - the suite scripts no other kinds
            raise NotImplementedError(f"no fake-ACP directive for {event.kind}")
    return directives


def AcpAgentViaFake(script):  # noqa: N802 - factory name doubles as the pytest id
    env = _fake_env(NVSH_TEST_ACP_SCRIPT=json.dumps(_script_to_directives(script)))
    return AcpAgent(["acp"], "fake", env=env)


@pytest.mark.parametrize(
    "case",
    [
        conformance.test_streaming_order,
        conformance.test_cancel_mid_stream,
        conformance.test_error_propagation,
        conformance.test_capability_report,
        conformance.test_teardown,
    ],
    ids=lambda case: case.__name__,
)
def test_acp_passes_the_conformance_suite(case):
    """Every shared conformance case, run against AcpAgent via the fake.

    The suite's own ``ADAPTERS`` list is a different task's file; running its
    cases here keeps AcpAgent under exactly the same contract without
    reaching into it.
    """
    case(AcpAgentViaFake)


# -- live smoke ------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NVSH_LIVE_QWEN") != "1",
    reason="live qwen --acp smoke; set NVSH_LIVE_QWEN=1 to run",
)
def test_live_qwen_acp_streams_thinking_and_tool_calls():
    """A real ``qwen --acp`` turn, in the default (plan) mode nvsh ships.

    Asserts only what the adapter is responsible for -- that a live harness's
    thought chunks and tool calls arrive as THINKING and TOOL_CALL -- never
    what the model happens to say.
    """
    agent = build("qwen")
    request = AgentRequest(
        kind=RequestKind.EXPLICIT,
        prompt="How many CPU cores does this machine have? Read /proc/cpuinfo to find out.",
    )
    kinds = set()
    try:
        for event in agent.run(request, AgentContext(platform="platform: dgx-spark")):
            kinds.add(event.kind)
            if event.kind is EventKind.PROPOSAL:
                agent.respond_ui(event.args["request_id"], value="once")
            if {EventKind.THINKING, EventKind.TOOL_CALL} <= kinds:
                agent.cancel()
                break
            assert event.kind is not EventKind.ERROR, event.error
    finally:
        agent.close()

    assert EventKind.THINKING in kinds
    assert EventKind.TOOL_CALL in kinds
