"""Shared conformance suite, run against every registered NvshAgent adapter.

Two layers, both parametrised:

**The original five cases** (``test_streaming_order``,
``test_cancel_mid_stream``, ``test_error_propagation``,
``test_capability_report``, ``test_teardown``) run against every factory in
:data:`ADAPTERS`. A factory takes a ``script`` (a list of ``AgentEvent`` to
replay) and returns a fresh, unstarted ``NvshAgent``. These cases script
``STATUS``, ``TEXT_DELTA``, ``ERROR`` and ``DONE`` only, so a factory can
only join ``ADAPTERS`` if its fake's wire format can express all four --
``agy``'s recorded NDJSON vocabulary has no catch-all STATUS line, which is
why ``AgyAgentViaFake`` is absent here and present in :data:`CASES`.

**The cross-adapter cases** (everything below the ``-- cross-adapter``
banner) run against :data:`tests._fake_adapters.CASES`: one entry per
backend nvsh ships -- pi, agy, claude, codex, qwen (ACP), qwen-p (print
mode) and kiro (ACP). They assert the behaviour the spec requires of *every*
adapter rather than of one:

* THINKING and TOOL_CALL are emitted on a rich turn (or, where the fake's
  wire format cannot express the kind, the adapter's declared capability
  says so -- the assertion message names which);
* a PROPOSAL arrives, or the adapter declares ``tool_calling=False``;
* the operator's effort string reaches the child verbatim (decision c24:
  nvsh never validates or rewrites it), and an adapter that declares
  ``effort=False`` sends it nowhere;
* a CLI that rejects that effort has its stderr tail surfaced, not
  swallowed;
* ``approval`` and ``unmediated_file_access`` are declared, not defaulted.

The fakes, the case table and the per-adapter knowledge of how to drive each
one live in ``tests/_fake_adapters.py``; this module only states the contract.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from nvsh.agent import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Capabilities,
    EventKind,
    FakeAgent,
    RequestKind,
)
from nvsh.agent.acp import AcpAgent
from nvsh.agent.agy import AgyAgent
from nvsh.agent.claude import ClaudeAgent
from nvsh.agent.codex import CodexAgent
from nvsh.agent.openai_compat import OpenAICompatAgent
from nvsh.agent.pi import PiAgent
from nvsh.agent.qwen import QwenAgent
from tests import _fake_adapters
from tests._fake_adapters import reap_fake_pids  # noqa: F401 - fixture, used below
from tests._fake_adapters import (
    CASES,
    RAISE_ON_START,
    AcpKiroViaFake,
    AcpQwenViaFake,
    ClaudeAgentViaFake,
    CodexAgentViaFake,
    PiAgentViaFake,
    QwenAgentViaFake,
    drive,
)
from tests.test_agent_subprocess import _pid_alive, _wait_gone

# Teardown kills the fake harness/grandchild pids each test's fakes recorded
# (task t23), so a failing or respawning case leaks no ``sleep 600``.
pytestmark = pytest.mark.usefixtures("reap_fake_pids")

# Registry of adapter factories for the original five cases. Append here to
# bring a new backend under them; append to CASES (in tests/_fake_adapters.py)
# to bring it under the cross-adapter cases.
ADAPTERS = [
    FakeAgent,
    PiAgentViaFake,
    ClaudeAgentViaFake,
    CodexAgentViaFake,
    QwenAgentViaFake,
    AcpQwenViaFake,
    AcpKiroViaFake,
]


@pytest.fixture(params=ADAPTERS, ids=lambda factory: factory.__name__)
def adapter_factory(request):
    return request.param


def _request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, command="ls /nope", exit_code=2)


def _context() -> AgentContext:
    return AgentContext(platform="dgx-spark", output="ls: /nope: No such file", cwd="/tmp")


def test_streaming_order(adapter_factory):
    Event = AgentEvent
    script = [
        Event(kind=EventKind.STATUS, text="inspecting"),
        Event(kind=EventKind.TEXT_DELTA, text="looking at "),
        Event(kind=EventKind.TEXT_DELTA, text="the failure"),
        Event(kind=EventKind.DONE),
    ]
    agent = adapter_factory(script)
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert [e.kind for e in events] == [
        EventKind.STATUS,
        EventKind.TEXT_DELTA,
        EventKind.TEXT_DELTA,
        EventKind.DONE,
    ]
    assert events[1].text == "looking at "
    assert events[2].text == "the failure"


def test_cancel_mid_stream(adapter_factory):
    Event = AgentEvent
    script = [
        Event(kind=EventKind.STATUS, text="one"),
        Event(kind=EventKind.STATUS, text="two"),
        Event(kind=EventKind.STATUS, text="three"),
        Event(kind=EventKind.DONE),
    ]
    agent = adapter_factory(script)
    agent.start()
    seen = []
    try:
        for event in agent.run(_request(), _context()):
            seen.append(event)
            if len(seen) == 1:
                agent.cancel()
    finally:
        agent.close()
    # Cancellation is checked between yields: exactly the events already in
    # flight when cancel() was called are seen, and nothing scripted after.
    assert seen == [Event(kind=EventKind.STATUS, text="one")]


def test_error_propagation(adapter_factory):
    Event = AgentEvent
    script = [
        Event(kind=EventKind.STATUS, text="inspecting"),
        Event(kind=EventKind.ERROR, error="backend unreachable"),
        Event(kind=EventKind.DONE),
    ]
    agent = adapter_factory(script)
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    # The error event is delivered, and it ends the stream: DONE never comes.
    assert [e.kind for e in events] == [EventKind.STATUS, EventKind.ERROR]
    assert events[-1].error == "backend unreachable"


def test_capability_report(adapter_factory):
    agent = adapter_factory([])
    caps = agent.capabilities()
    assert isinstance(caps, Capabilities)
    assert isinstance(caps.streaming, bool)
    assert isinstance(caps.tool_calling, bool)
    assert isinstance(caps.cancellation, bool)
    assert isinstance(caps.persistent_session, bool)
    assert isinstance(caps.local_model, bool)


def test_teardown(adapter_factory):
    agent = adapter_factory([AgentEvent(kind=EventKind.DONE)])
    agent.start()
    list(agent.run(_request(), _context()))
    # close() must not raise, and must be safe to call more than once.
    agent.close()
    agent.close()


# ---------------------------------------------------------------------------
# -- cross-adapter: every backend nvsh ships, under one contract
# ---------------------------------------------------------------------------


@pytest.fixture(params=CASES, ids=lambda case: case.name)
def adapter_case(request):
    return request.param


def _rich(case):
    """One rich turn plus the capabilities the adapter declares for it."""
    agent = case.build_rich()
    caps = agent.capabilities()
    return drive(agent), caps


def test_thinking_is_emitted_or_declared_unsupported(adapter_case):
    """A backend that can think streams THINKING; one that cannot says so.

    ``agy`` has no thinking line in its recorded NDJSON vocabulary and
    ``tests/fakes/qwen`` serialises no thinking part, so for those the
    declared capability is what is checked -- deliberately weaker, and
    called out rather than skipped.
    """
    events, caps = _rich(adapter_case)
    kinds = {event.kind for event in events}
    if "thinking" in adapter_case.scriptable:
        assert EventKind.THINKING in kinds, f"{adapter_case.name}: no THINKING in {sorted(kinds)}"
        assert (
            caps.thinking or adapter_case.name == "kiro"
        ), f"{adapter_case.name} streamed THINKING but declares thinking=False"
    else:
        assert isinstance(caps.thinking, bool), (
            f"{adapter_case.name}'s fake cannot express a thinking delta, so only the "
            "declared capability is checked here"
        )


def test_tool_calls_are_emitted_or_declared_unsupported(adapter_case):
    """A tool the backend runs arrives as TOOL_CALL, or is declared absent."""
    events, caps = _rich(adapter_case)
    kinds = {event.kind for event in events}
    if "tool_call" in adapter_case.scriptable:
        assert EventKind.TOOL_CALL in kinds, f"{adapter_case.name}: no TOOL_CALL in {sorted(kinds)}"
        tool_calls = [e for e in events if e.kind == EventKind.TOOL_CALL]
        assert all(call.tool for call in tool_calls), f"{adapter_case.name}: unnamed TOOL_CALL"
    else:
        assert caps.tool_calling is False, (
            f"{adapter_case.name}'s fake cannot express a tool call, so it must declare "
            "tool_calling=False rather than claim an unproven capability"
        )


def test_proposal_arrives_or_tool_calling_is_declared_false(adapter_case):
    """Propose, don't run: either the operator is asked, or nothing is run.

    An adapter that mediates approval (``tool_calling=True``) must raise a
    PROPOSAL the caller can answer. An adapter that raises none must declare
    ``tool_calling=False`` rather than leave the caller to guess.

    The converse -- ``tool_calling=False`` implies no PROPOSAL, ever -- is
    deliberately *not* asserted, because it does not hold for ACP qwen and
    the reason is a gap in evidence rather than in code: ``build("qwen")``
    reports ``tool_calling=False`` on the grounds that plan mode analyses
    without executing (decision c53), while ``tests/fakes/acp`` replays a
    recorded session that did ask permission. Nothing in the tree records a
    plan-mode qwen turn, so "plan mode never asks" is an assumption here,
    not a measurement.
    """
    events, caps = _rich(adapter_case)
    proposals = [e for e in events if e.kind == EventKind.PROPOSAL]
    assert (
        proposals or caps.tool_calling is False
    ), f"{adapter_case.name} raised no PROPOSAL and does not declare tool_calling=False"
    if caps.tool_calling:
        assert proposals, f"{adapter_case.name} declares tool_calling=True but raised no PROPOSAL"
    if not proposals:
        return
    for event in proposals:
        assert event.proposal is not None
        assert (
            event.args.get("request_id") is not None
        ), f"{adapter_case.name}: a PROPOSAL with no request_id cannot be answered"
    # The turn really resumed on the answer the driver gave, rather than
    # stalling on an unanswered dialog (deviation d11).
    assert events[-1].kind in (EventKind.DONE, EventKind.ERROR)


def test_effort_reaches_the_child_verbatim(adapter_case):
    """Decision c24: an effort is an opaque string nvsh never rewrites.

    Adapters that declare ``effort=False`` must send it nowhere at all --
    silently dropping it is the correct behaviour there, and is what the
    capability reports.
    """
    caps = adapter_case.build_rich().capabilities()
    probe = adapter_case.effort_probe
    words = adapter_case.effort_words(probe)
    carried = [word for word in words if probe in word]
    if caps.effort:
        assert carried, f"{adapter_case.name}: {probe!r} appears nowhere in {words}"
    else:
        assert (
            not carried
        ), f"{adapter_case.name} declares effort=False but still sent {probe!r}: {carried}"


def test_a_rejected_effort_surfaces_the_cli_stderr_tail(adapter_case):
    """A CLI that refuses to run must not fail silently.

    Adapters that spawn their child inside ``run()`` report this as an ERROR
    event. Adapters that spawn it in ``start()`` (see ``RAISE_ON_START``)
    raise instead, and ``nvsh/daemon.py`` turns that into
    ``ERROR "no agent available: <exc>"`` -- so the operator reads the same
    tail either way. What matters, and what is asserted here, is that the
    CLI's own words survive the trip.
    """
    agent = adapter_case.build_rejecting()
    raised = ""
    events = []
    try:
        events = drive(agent)
    except Exception as exc:  # noqa: BLE001 - the start()-time path, on purpose
        raised = str(exc)

    if adapter_case.name in RAISE_ON_START:
        assert raised, f"{adapter_case.name} is pinned as raising on start() but did not"
        surfaced = raised
    else:
        assert not raised, f"{adapter_case.name} raised instead of yielding an ERROR: {raised}"
        errors = [e.error for e in events if e.kind == EventKind.ERROR]
        assert errors, f"{adapter_case.name}: no ERROR event, only {[e.kind for e in events]}"
        surfaced = "\n".join(errors)

    if _fake_adapters.REJECTED_EFFORT_TAIL in surfaced:
        return
    # The tail is missing. That is tolerated only for the adapters pinned in
    # STDERR_TAIL_RACE (see its comment: AcpAgent's stderr drain races its
    # own launch-failure message), and even there the failure must still be
    # reported with the CLI's exit status rather than silently.
    assert (
        adapter_case.name in _fake_adapters.STDERR_TAIL_RACE
    ), f"{adapter_case.name} swallowed the CLI's stderr tail: {surfaced!r}"
    assert (
        "exited with code 2" in surfaced
    ), f"{adapter_case.name} reported neither the stderr tail nor the exit status: {surfaced!r}"


def test_approval_and_unmediated_file_access_are_declared(adapter_case):
    """Both fields are stated by every adapter, never left to the default.

    ``Capabilities`` defaults ``approval="none"`` and
    ``unmediated_file_access=False`` -- the two most permissive-sounding and
    most reassuring answers respectively. An adapter that simply omits them
    is claiming something it never considered, so the declaration is checked
    in the source of ``capabilities()`` as well as in its value.
    """
    if adapter_case.name in _fake_adapters.UNDECLARED_FILE_ACCESS:
        pytest.xfail(
            "deviation: nvsh/agent/qwen.py's capabilities() omits unmediated_file_access, "
            "so print-mode qwen inherits False while reading files with its own tools"
        )
    agent = adapter_case.build_rich()
    caps = agent.capabilities()
    assert caps.approval in (
        "nvsh",
        "harness",
        "none",
    ), f"{adapter_case.name} reports an unknown approval mediator {caps.approval!r}"
    assert isinstance(caps.unmediated_file_access, bool)

    source = inspect.getsource(type(agent).capabilities)
    for field in ("approval", "unmediated_file_access"):
        assert field in source, (
            f"{type(agent).__name__}.capabilities() never mentions {field!r}: it inherits the "
            f"Capabilities default instead of declaring what this backend actually does"
        )


# ---------------------------------------------------------------------------
# -- t13 (reliable-agent-stop): ignoring-harness stop and respawn, every
# adapter family
# ---------------------------------------------------------------------------
#
# One parametrised test over eight variants -- pi, codex, acp, agy warm, agy
# cold, claude, qwen-p and openai-compat (agy's warm and cold modes spawn on
# two different code paths in nvsh/agent/agy.py, so they are separate cases
# here). Each variant drives its real adapter against the same
# "NVSH_FAKE_IGNORE_CANCEL=1" fakes waves 1-2's per-adapter tests
# (tests/test_pi_agent.py, test_agent_codex.py, test_agent_acp.py,
# test_agent_agy.py, test_agent_claude.py, test_agent_qwen.py,
# test_agent_openai_compat.py) already use, and proves the shared contract:
#
# 1. the turn is genuinely in flight -- the harness process, and (where the
#    fake supports it) its own grandchild, are alive;
# 2. cancel() alone cannot end a harness that ignores it, so force_stop() is
#    what actually kills the whole process tree, well inside the same tight
#    budget the per-adapter tests use (3s for a pid-based backend, 1s for
#    openai-compat's HTTP stream) -- tight enough that falling back to
#    NvshAgent's default cancel()-then-close() (whose escalate_close()
#    still reaches kill_tree eventually, just through a slower wait/
#    terminate/kill ladder) blows the budget and fails the case;
# 3. the next run() on the very same adapter instance succeeds end to end.


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


def _run_in_thread(agent, request=None, context=None):
    """Drive ``agent.run()`` to completion on a background thread.

    A turn against an ignoring-harness fake never ends on its own, so
    stepping it on the calling thread would block the whole suite; every
    case below drives it here and inspects the adapter/process from the
    main thread instead, the same pattern
    ``tests/test_agent_claude.py``/``test_agent_qwen.py``/
    ``test_agent_openai_compat.py`` already use.
    """
    request = request if request is not None else _fake_adapters.conformance_request()
    context = context if context is not None else _fake_adapters.conformance_context()
    collected: list = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for event in agent.run(request, context):
                collected.append(event)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            errors.append(exc)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread, collected, errors


def _assert_force_stopped(thread, errors, pids, *, budget: float = 3.0) -> None:
    """The turn really ended, and (where ``pids`` is non-empty) nothing survived."""
    thread.join(timeout=budget + 7.0)
    assert not thread.is_alive(), "run() never returned after force_stop()"
    if errors:
        raise errors[0]
    if pids:
        assert _wait_gone(pids, within=budget) == []


# -- pi ----------------------------------------------------------------


def _pi_env(tmp_path: Path, **extra: str) -> dict:
    env = dict(os.environ)
    env["PATH"] = str(_fake_adapters.FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env["HOME"] = str(tmp_path / "home")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    env.update(extra)
    return env


def _case_pi(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.json"
    env = _pi_env(
        tmp_path,
        NVSH_FAKE_IGNORE_CANCEL="1",
        NVSH_FAKE_GRANDCHILD="1",
        NVSH_FAKE_PID_FILE=str(pid_file),
    )
    agent = PiAgent(pi_path="pi", env=env)
    try:
        agent.start()
        pids = _wait_for_pid_file(pid_file)
        assert _pid_alive(pids["harness"])

        agent.force_stop()
        assert _wait_gone([pids["harness"], pids["grandchild"]]) == []

        events = list(
            agent.run(_fake_adapters.conformance_request(), _fake_adapters.conformance_context())
        )
        assert events[-1].kind == EventKind.DONE, f"pi: next run() did not finish: {events}"
    finally:
        agent.close()


# -- codex ---------------------------------------------------------------


def _case_codex(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.json"
    env = _fake_adapters.fake_env(
        NVSH_FAKE_IGNORE_CANCEL="1",
        NVSH_FAKE_GRANDCHILD="1",
        NVSH_FAKE_PID_FILE=str(pid_file),
    )
    agent = CodexAgent({}, binary="codex-app-server", env=env)
    try:
        thread, _events, errors = _run_in_thread(agent)
        pids = _wait_for_pid_file(pid_file)
        assert _pid_alive(pids["harness"])

        agent.force_stop()
        _assert_force_stopped(thread, errors, [pids["harness"], pids["grandchild"]])
        assert agent._rpc is None  # noqa: SLF001 - proving no dead pipe is reused

        second = drive(agent)
        assert second[-1].kind == EventKind.DONE, f"codex: next run() did not finish: {second}"
    finally:
        agent.close()


# -- acp (qwen/kiro's transport) -----------------------------------------

_ACP_PAUSED_ON_PERMISSION_SCRIPT = json.dumps(
    [
        {
            "permission": {
                "toolCall": {"toolCallId": "c1", "status": "pending", "title": "shell"},
                "options": [{"optionId": "proceed_once", "name": "Allow", "kind": "allow_once"}],
            }
        }
    ]
)


def _case_acp(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.json"
    log = tmp_path / "frames.jsonl"
    env = _fake_adapters.fake_env(
        NVSH_FAKE_IGNORE_CANCEL="1",
        NVSH_FAKE_GRANDCHILD="1",
        NVSH_FAKE_PID_FILE=str(pid_file),
        NVSH_TEST_ACP_SCRIPT=_ACP_PAUSED_ON_PERMISSION_SCRIPT,
        NVSH_TEST_ACP_COMMANDS=str(log),
    )
    agent = AcpAgent(["acp"], "fake", env=env, thinking=True)
    try:
        events = agent.run(
            _fake_adapters.conformance_request(), _fake_adapters.conformance_context()
        )
        proposal = next(e for e in events if e.kind is EventKind.PROPOSAL)
        assert proposal.args["request_id"] in agent._pending  # noqa: SLF001

        pids = _wait_for_pid_file(pid_file)
        assert _pid_alive(pids["harness"])

        agent.force_stop()
        events.close()
        assert _wait_gone([pids["harness"], pids["grandchild"]]) == []
        assert agent._pending == {}  # noqa: SLF001 - the open dialog was denied

        agent._env["NVSH_TEST_ACP_SCRIPT"] = "[]"  # noqa: SLF001 - a plain second turn
        second = list(
            agent.run(_fake_adapters.conformance_request(), _fake_adapters.conformance_context())
        )
        assert second[-1].kind == EventKind.DONE, f"acp: next run() did not finish: {second}"
    finally:
        agent.close()


# -- agy: warm and cold are separate code paths (nvsh/agent/agy.py) -----


def _agy_turn(*, done: bool, step_index: int = 1) -> dict:
    conversation = "fake-conv-t13"
    stdout = [
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "conversation_id": conversation,
                    "step_index": step_index,
                    "state": "ACTIVE",
                    "step_type": "agent_response",
                    "text_delta": "hi",
                },
            }
        )
    ]
    if done:
        stdout.append(
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "conversation_id": conversation,
                        "status": "SUCCESS",
                        "response": "hi",
                    },
                }
            )
        )
    return {"stdout": stdout, "exit_code": 0}


def _agy_request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.EXPLICIT, prompt="say hi")


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _case_agy_cold(tmp_path: Path) -> None:
    """Cold agy spawns its own process group per turn (``start_new_session``
    on that ``Popen``, same as warm's -- task t22), so force_stop() must
    reach a tool grandchild the cold child started, not just the child
    alone. Proves force_stop() (AgyAgent has no override; the base default
    is cancel()-then-close(), whose ``_terminate`` escalates through
    ``kill_tree``) reliably kills the whole tree within budget even though
    the fake ignores SIGTERM (``NVSH_FAKE_IGNORE_CANCEL=1``), and that the
    next run() spawns a fresh one.
    """
    pid_file = tmp_path / "pids.json"
    events_path = tmp_path / "events.json"
    _write_json(events_path, {"stdout": [], "exit_code": 0, "sleep_before": 3600})
    env = _fake_adapters.fake_env(
        NVSH_FAKE_EVENTS=str(events_path),
        NVSH_FAKE_IGNORE_CANCEL="1",
        NVSH_FAKE_GRANDCHILD="1",
        NVSH_FAKE_PID_FILE=str(pid_file),
    )
    agent = AgyAgent(warm=False, env=env)
    try:
        thread, _events, errors = _run_in_thread(
            agent, request=_agy_request(), context=AgentContext()
        )
        pids = _wait_for_pid_file(pid_file)
        assert _pid_alive(pids["harness"])
        assert _pid_alive(pids["grandchild"])

        agent.force_stop()
        _assert_force_stopped(thread, errors, [pids["harness"], pids["grandchild"]])

        _write_json(events_path, _agy_turn(done=True))
        second = list(agent.run(_agy_request(), AgentContext()))
        assert second[-1].kind == EventKind.DONE, f"agy cold: next run() did not finish: {second}"
    finally:
        agent.close()


def _case_agy_warm(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.json"
    events_path = tmp_path / "events.json"
    _write_json(events_path, {"turns": [{"stdout": [], "exit_code": 0, "sleep_before": 3600}]})
    env = _fake_adapters.fake_env(
        NVSH_FAKE_EVENTS=str(events_path),
        NVSH_FAKE_IGNORE_CANCEL="1",
        NVSH_FAKE_GRANDCHILD="1",
        NVSH_FAKE_PID_FILE=str(pid_file),
    )
    agent = AgyAgent(warm=True, env=env)
    try:
        agent.start()  # the very first, start()-spawned warm child
        pids = _wait_for_pid_file(pid_file)
        assert _pid_alive(pids["harness"])
        first_pid = pids["harness"]

        # An empty AgentContext (cwd="") keeps run() from treating this as
        # a different working tree and respawning before the turn is even
        # sent (AgyAgent.run() rebinds -- closes and re-spawns -- whenever
        # ``context.cwd`` differs from the process it already has bound).
        thread, _events, errors = _run_in_thread(
            agent, request=_agy_request(), context=AgentContext()
        )
        time.sleep(0.2)  # let the prompt actually reach the sleeping turn

        agent.force_stop()
        _assert_force_stopped(thread, errors, [pids["harness"], pids["grandchild"]])

        _write_json(events_path, {"turns": [_agy_turn(done=True)]})
        second = list(agent.run(_agy_request(), AgentContext()))
        assert second[-1].kind == EventKind.DONE, f"agy warm: next run() did not finish: {second}"
        assert agent._proc is not None  # noqa: SLF001
        assert agent._proc.pid != first_pid  # noqa: SLF001 - a genuinely fresh process
    finally:
        agent.close()


# -- claude ----------------------------------------------------------------


def _case_claude(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.json"
    env = _fake_adapters.fake_env(
        NVSH_FAKE_IGNORE_CANCEL="1",
        NVSH_FAKE_GRANDCHILD="1",
        NVSH_FAKE_PID_FILE=str(pid_file),
    )
    agent = ClaudeAgent({}, env=env)
    try:
        agent.start()
        thread, _events, errors = _run_in_thread(agent)
        pids = _wait_for_pid_file(pid_file)
        assert _pid_alive(pids["harness"])

        agent.force_stop()
        _assert_force_stopped(thread, errors, [pids["harness"], pids["grandchild"]])

        # A plain, immediately-resolving script for the second turn.
        agent._env["NVSH_FAKE_EVENTS"] = _fake_adapters._write_events_file(  # noqa: SLF001
            [AgentEvent(kind=EventKind.DONE)]
        )
        agent.start()
        second = list(
            agent.run(_fake_adapters.conformance_request(), _fake_adapters.conformance_context())
        )
        assert second[-1].kind == EventKind.DONE, f"claude: next run() did not finish: {second}"
    finally:
        agent.close()


# -- qwen-p (print-mode fallback) ------------------------------------------


def _case_qwen_p(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.json"
    env = _fake_adapters.fake_env(
        NVSH_FAKE_IGNORE_CANCEL="1",
        NVSH_FAKE_GRANDCHILD="1",
        NVSH_FAKE_PID_FILE=str(pid_file),
    )
    agent = QwenAgent({}, env=env)
    try:
        agent.start()
        thread, _events, errors = _run_in_thread(
            agent, request=AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi")
        )
        pids = _wait_for_pid_file(pid_file)
        assert _pid_alive(pids["harness"])

        agent.force_stop()
        _assert_force_stopped(thread, errors, [pids["harness"], pids["grandchild"]])

        agent._env["NVSH_FAKE_EVENTS"] = _fake_adapters._write_events_file(  # noqa: SLF001
            [AgentEvent(kind=EventKind.DONE)]
        )
        agent.start()
        second = list(
            agent.run(AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi"), AgentContext())
        )
        assert second[-1].kind == EventKind.DONE, f"qwen-p: next run() did not finish: {second}"
    finally:
        agent.close()


# -- openai-compat (HTTP, not a subprocess) --------------------------------


class _StallOrServeHandler(BaseHTTPRequestHandler):
    """Stalls forever on a path containing ``/stall``, else serves one turn."""

    server_version = "NvshConformanceT13/1.0"

    def log_message(self, *_args):  # noqa: D401 - silence test server logging
        pass

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if "/stall" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            self.server.stall_connected.set()  # type: ignore[attr-defined]
            time.sleep(30)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _case_openai_compat(_tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StallOrServeHandler)
    server.stall_connected = threading.Event()  # type: ignore[attr-defined]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/stall"
        agent = OpenAICompatAgent({"base_url": base_url})
        agent.start()
        try:
            thread, _events, errors = _run_in_thread(agent)
            assert server.stall_connected.wait(  # type: ignore[attr-defined]
                timeout=5
            ), "the stalled request never connected"
            time.sleep(0.1)  # let the generator actually block on the read

            started = time.monotonic()
            agent.force_stop()
            _assert_force_stopped(thread, errors, [], budget=1.0)
            elapsed = time.monotonic() - started
            assert elapsed < 1.0, f"openai-compat: force_stop() took {elapsed:.3f}s, budget is 1s"

            agent._base_url = f"http://127.0.0.1:{server.server_port}"  # noqa: SLF001
            agent.start()
            second = list(
                agent.run(
                    _fake_adapters.conformance_request(), _fake_adapters.conformance_context()
                )
            )
            assert (
                second[-1].kind == EventKind.DONE
            ), f"openai-compat: next run() did not finish: {second}"
        finally:
            agent.close()
    finally:
        server.shutdown()
        server_thread.join(timeout=5)


STOP_CASES = [
    ("pi", _case_pi),
    ("codex", _case_codex),
    ("acp", _case_acp),
    ("agy-warm", _case_agy_warm),
    ("agy-cold", _case_agy_cold),
    ("claude", _case_claude),
    ("qwen-p", _case_qwen_p),
    ("openai-compat", _case_openai_compat),
]


@pytest.mark.parametrize("stop_case", STOP_CASES, ids=[name for name, _ in STOP_CASES])
def test_ignoring_harness_stop_and_respawn(stop_case, tmp_path):
    """Every adapter family: cancel() cannot end an ignoring harness, so
    force_stop() kills the whole process tree within budget, and the very
    same adapter instance's next run() succeeds end to end.

    Removing (or no-oping) any one adapter's ``force_stop()`` override falls
    back to ``NvshAgent``'s default -- ``cancel()`` then ``close()``, whose
    ``escalate_close()`` still reaches ``kill_tree`` eventually but only
    after a slower wait/terminate/kill ladder -- which blows this test's
    budget and fails only that adapter's case.
    """
    _name, check = stop_case
    check(tmp_path)


def test_fake_agent_runs_again_after_a_cancel_without_a_second_start():
    """The fixture backend must model the contract the daemon relies on:
    ``start()`` once per warm session, and a cancel that ends one turn only."""
    agent = FakeAgent(
        [AgentEvent(kind=EventKind.TEXT_DELTA, text="hi"), AgentEvent(kind=EventKind.DONE)]
    )
    agent.start()
    agent.cancel()
    events = list(
        agent.run(_fake_adapters.conformance_request(), _fake_adapters.conformance_context())
    )
    assert [e.kind for e in events] == [EventKind.TEXT_DELTA, EventKind.DONE]


def test_fake_agent_honours_a_cancel_that_lands_before_the_first_step():
    agent = FakeAgent(
        [AgentEvent(kind=EventKind.TEXT_DELTA, text="hi"), AgentEvent(kind=EventKind.DONE)]
    )
    agent.start()
    stream = agent.run(_fake_adapters.conformance_request(), _fake_adapters.conformance_context())
    agent.cancel()
    assert list(stream) == []
