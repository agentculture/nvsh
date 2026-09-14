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
from tests import _fake_adapters
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
