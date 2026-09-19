"""Tier residency in the per-user daemon (task t12).

Covers the spec's daemon-residency claim and its honesty condition -- the
resident models live in the daemon, are built on first use rather than at
daemon start, and are unloaded after an idle period; the one-shot client
asks the daemon rather than loading a model per call, and falls straight
through to the full agent when the daemon is unavailable -- plus the
child-process-isolation claim's second half: Tier 1 selection happens
*before* the daemon's turn lock is taken, so a Tier 1 answer never waits
behind another shell's full-agent turn.

Every tier here is a :class:`~nvsh.tiers.fake.FakeTier` script: inert data.
Nothing in this file runs a command, and the daemon is never given an agent
to run one with.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

from nvsh import client_transport
from nvsh import daemon as daemon_mod
from nvsh.agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    EventKind,
    RequestKind,
    request_to_dict,
)
from nvsh.config import Config
from nvsh.platform._model import Platform
from nvsh.tiers import manager as manager_mod
from nvsh.tiers.base import Decline, DeclineReason, TierDecision
from nvsh.tiers.fake import FakeTier
from nvsh.tiers.manager import TierManager
from nvsh.tiers.records import TierRecords, default_records_path

_PLATFORM = Platform(kind="jetson")

#: A read-only pick that grounds without a runner (no arguments) and renders
#: to a static system command on every platform (``free -m``).
_MEMORY = TierDecision(operation="memory_stats", args={}, confidence=0.9, read_only=True)


# --- helpers ---------------------------------------------------------------


class _Clock:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _env(tmp_path: Path) -> dict[str, str]:
    run = tmp_path / "run"
    state = tmp_path / "state"
    run.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    return {"XDG_RUNTIME_DIR": str(run), "XDG_STATE_HOME": str(state), "HOME": str(tmp_path)}


def _config(**overrides: object) -> Config:
    config = Config()
    config.tiers = {**config.tiers, "enabled": True, **overrides}
    return config


def _request(prompt: str = "how much memory is free") -> AgentRequest:
    return AgentRequest(kind=RequestKind.EXPLICIT, prompt=prompt)


def _manager(
    tmp_path: Path,
    script: list[object] | None = None,
    *,
    clock: _Clock | None = None,
    builds: list[FakeTier] | None = None,
    config: Config | None = None,
) -> TierManager:
    """A manager whose Tier 1 is a scripted fake, recorded in *builds*."""
    made = builds if builds is not None else []

    def factory() -> FakeTier:
        tier = FakeTier(list(script if script is not None else [_MEMORY] * 8))
        made.append(tier)
        return tier

    return TierManager(
        config if config is not None else _config(),
        _PLATFORM,
        clock=clock if clock is not None else _Clock(),
        env=_env(tmp_path),
        tier1_factory=factory,
    )


def _answer(manager: TierManager, request: AgentRequest | None = None) -> manager_mod.TierAnswer:
    """Run one tier request to completion and return its answer."""
    session = manager.open(request if request is not None else _request(), AgentContext())
    for _ in session:
        pass
    return session.answer


def _events(manager: TierManager) -> list[AgentEvent]:
    session = manager.open(_request(), AgentContext())
    return list(session)


def _records(tmp_path: Path) -> list[dict]:
    return TierRecords(default_records_path(_env(tmp_path))).read_all()


def _daemon(
    tmp_path: Path, manager: TierManager, *, idle_timeout: float = 60.0
) -> daemon_mod.Daemon:
    return daemon_mod.Daemon(
        _config(),
        env=_env(tmp_path),
        idle_timeout=idle_timeout,
        tier_manager=manager,
    )


def _tier_message(shell: str = "A") -> dict:
    return {
        "shell": shell,
        "kind": daemon_mod.TIER_KIND,
        "request": request_to_dict(_request()),
        "context": {},
    }


def _outcome_frame(events: list[AgentEvent]) -> AgentEvent:
    frames = [event for event in events if daemon_mod.TIER_OUTCOME_KEY in (event.args or {})]
    assert len(frames) == 1, [event.args for event in events]
    return frames[0]


def _serve(daemon: daemon_mod.Daemon) -> Iterator[None]:
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    for _ in range(500):
        if daemon_mod.is_running(daemon.env):
            break
        time.sleep(0.01)
    yield
    daemon.shutdown()
    thread.join(10)


# --- the manager builds nothing until it is asked --------------------------


def test_manager_builds_nothing_at_construction(tmp_path: Path) -> None:
    builds: list[FakeTier] = []
    _manager(tmp_path, builds=builds)
    assert builds == []


def test_the_first_request_builds_tier_one(tmp_path: Path) -> None:
    builds: list[FakeTier] = []
    _answer(_manager(tmp_path, builds=builds))
    assert len(builds) == 1


def test_a_second_request_reuses_the_loaded_tier(tmp_path: Path) -> None:
    builds: list[FakeTier] = []
    manager = _manager(tmp_path, builds=builds)
    _answer(manager)
    _answer(manager)
    assert len(builds) == 1


def test_disabled_tiers_build_nothing(tmp_path: Path) -> None:
    builds: list[FakeTier] = []
    _answer(_manager(tmp_path, builds=builds, config=_config(enabled=False)))
    assert builds == []


def test_disabled_tiers_answer_escalate(tmp_path: Path) -> None:
    answer = _answer(_manager(tmp_path, config=_config(enabled=False)))
    assert answer.outcome == manager_mod.ESCALATE


# --- what one routed request produces --------------------------------------


def test_a_handled_request_names_the_tier_that_answered(tmp_path: Path) -> None:
    assert _answer(_manager(tmp_path)).tier == "fake"


def test_a_handled_request_yields_the_rendered_proposal(tmp_path: Path) -> None:
    proposals = [
        event.proposal for event in _events(_manager(tmp_path)) if event.proposal is not None
    ]
    assert [proposal.command for proposal in proposals] == ["free -m"]


def test_a_handled_request_carries_a_route_id(tmp_path: Path) -> None:
    assert _answer(_manager(tmp_path)).route_id != ""


def test_a_declining_tier_escalates_with_its_reason(tmp_path: Path) -> None:
    script: list[object] = [Decline(DeclineReason.TIER_UNAVAILABLE, "no engine")]
    assert _answer(_manager(tmp_path, script)).declines == (("fake", "tier_unavailable"),)


def test_a_tier_factory_that_raises_escalates(tmp_path: Path) -> None:
    def factory() -> FakeTier:
        raise RuntimeError("cactus-needle is not installed")

    manager = TierManager(
        _config(), _PLATFORM, clock=_Clock(), env=_env(tmp_path), tier1_factory=factory
    )
    session = manager.open(_request(), AgentContext())
    list(session)
    assert session.answer.outcome == manager_mod.ESCALATE


def test_a_tier_factory_that_raises_costs_one_status_line(tmp_path: Path) -> None:
    def factory() -> FakeTier:
        raise RuntimeError("cactus-needle is not installed")

    manager = TierManager(
        _config(), _PLATFORM, clock=_Clock(), env=_env(tmp_path), tier1_factory=factory
    )
    kinds = [event.kind for event in manager.open(_request(), AgentContext())]
    assert kinds == [EventKind.STATUS]


# --- idle unload -----------------------------------------------------------


def test_sweep_keeps_a_tier_inside_the_idle_window(tmp_path: Path) -> None:
    clock = _Clock()
    builds: list[FakeTier] = []
    manager = _manager(tmp_path, clock=clock, builds=builds)
    _answer(manager)
    clock.advance(10.0)
    manager.sweep()
    assert builds[0].closed is False


def test_sweep_unloads_a_tier_after_idle_unload_seconds(tmp_path: Path) -> None:
    clock = _Clock()
    builds: list[FakeTier] = []
    manager = _manager(tmp_path, clock=clock, builds=builds)
    _answer(manager)
    clock.advance(901.0)
    manager.sweep()
    assert builds[0].closed is True


def test_a_request_after_an_idle_unload_rebuilds_the_tier(tmp_path: Path) -> None:
    clock = _Clock()
    builds: list[FakeTier] = []
    manager = _manager(tmp_path, clock=clock, builds=builds)
    _answer(manager)
    clock.advance(901.0)
    manager.sweep()
    _answer(manager)
    assert len(builds) == 2


def test_idle_unload_seconds_of_zero_never_unloads(tmp_path: Path) -> None:
    clock = _Clock()
    builds: list[FakeTier] = []
    manager = _manager(tmp_path, clock=clock, builds=builds, config=_config(idle_unload_seconds=0))
    _answer(manager)
    clock.advance(100000.0)
    manager.sweep()
    assert builds[0].closed is False


def test_close_closes_the_loaded_tier(tmp_path: Path) -> None:
    builds: list[FakeTier] = []
    manager = _manager(tmp_path, builds=builds)
    _answer(manager)
    manager.close()
    assert builds[0].closed is True


# --- the follow-up decision record -----------------------------------------


def test_record_decision_writes_the_operator_decision(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    answer = _answer(manager)
    manager.record_decision(answer.route_id, "approved")
    decisions = [entry.get("operator_decision") for entry in _records(tmp_path)]
    assert "approved" in decisions


def test_record_decision_is_refused_for_an_unknown_route(tmp_path: Path) -> None:
    assert _manager(tmp_path).record_decision("nope", "approved") is False


def test_record_decision_is_accepted_only_once(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    answer = _answer(manager)
    manager.record_decision(answer.route_id, "approved")
    assert manager.record_decision(answer.route_id, "approved") is False


# --- the daemon's wiring ---------------------------------------------------


def test_a_fresh_daemon_reports_no_loaded_tiers(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    assert daemon.state()["tiers"]["loaded"] is False


def test_a_tier_request_loads_the_tier(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    list(daemon.handle_message(_tier_message()))
    assert daemon.state()["tiers"]["loaded"] is True


def test_a_tier_request_never_builds_an_agent(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    list(daemon.handle_message(_tier_message()))
    assert daemon.state()["agents"] == 0


def test_a_tier_request_ends_with_the_outcome_frame_then_done(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    events = list(daemon.handle_message(_tier_message()))
    assert [event.kind for event in events[-2:]] == [EventKind.STATUS, EventKind.DONE]


def test_the_outcome_frame_names_the_tier_that_handled_it(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    frame = _outcome_frame(list(daemon.handle_message(_tier_message())))
    assert frame.args[daemon_mod.TIER_OUTCOME_KEY] == daemon_mod.TIER_HANDLED


def test_an_escalating_tier_request_reports_escalate(tmp_path: Path) -> None:
    script: list[object] = [Decline(DeclineReason.TIER_UNAVAILABLE, "no engine")]
    daemon = _daemon(tmp_path, _manager(tmp_path, script))
    frame = _outcome_frame(list(daemon.handle_message(_tier_message())))
    assert frame.args[daemon_mod.TIER_OUTCOME_KEY] == daemon_mod.TIER_ESCALATE


def test_a_disabled_daemon_answers_a_tier_request_with_escalate(tmp_path: Path) -> None:
    manager = _manager(tmp_path, config=_config(enabled=False))
    daemon = _daemon(tmp_path, manager)
    frame = _outcome_frame(list(daemon.handle_message(_tier_message())))
    assert frame.args[daemon_mod.TIER_OUTCOME_KEY] == daemon_mod.TIER_ESCALATE


def test_the_daemon_and_the_manager_agree_on_the_outcome_names() -> None:
    assert (daemon_mod.TIER_HANDLED, daemon_mod.TIER_ESCALATE) == (
        manager_mod.HANDLED,
        manager_mod.ESCALATE,
    )


def test_a_tier_answer_returns_while_another_shell_holds_the_run_lock(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with daemon._run_lock:
            held.set()
            release.wait(30.0)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(10.0), "the holding thread never took the run lock"
    frame = _outcome_frame(list(daemon.handle_message(_tier_message("B"))))
    release.set()
    holder.join(10.0)
    assert frame.args[daemon_mod.TIER_OUTCOME_KEY] == daemon_mod.TIER_HANDLED


def test_a_tier_decision_control_records_the_operator_decision(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    frame = _outcome_frame(list(daemon.handle_message(_tier_message())))
    events = list(
        daemon.handle_message(
            {
                "shell": "A",
                "kind": daemon_mod.TIER_DECISION_KIND,
                "route_id": frame.args["route_id"],
                "decision": "approved",
            }
        )
    )
    assert [event.kind for event in events] == [EventKind.STATUS, EventKind.DONE]


def test_a_tier_decision_for_an_unknown_route_is_an_error(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    events = list(
        daemon.handle_message(
            {"shell": "A", "kind": daemon_mod.TIER_DECISION_KIND, "route_id": "9", "decision": "x"}
        )
    )
    assert [event.kind for event in events] == [EventKind.ERROR]


def test_daemon_shutdown_closes_the_tiers(tmp_path: Path) -> None:
    builds: list[FakeTier] = []
    daemon = _daemon(tmp_path, _manager(tmp_path, builds=builds))
    list(daemon.handle_message(_tier_message()))
    daemon.shutdown()
    assert builds[0].closed is True


def test_the_watchdog_unloads_idle_tiers(tmp_path: Path) -> None:
    builds: list[FakeTier] = []

    def factory() -> FakeTier:
        tier = FakeTier([_MEMORY])
        builds.append(tier)
        return tier

    # The real clock here, not the hand-moved one: this test is about the
    # daemon's own watchdog noticing, so it has to be the wall clock.
    manager = TierManager(
        _config(idle_unload_seconds=1),
        _PLATFORM,
        env=_env(tmp_path),
        tier1_factory=factory,
    )
    daemon = _daemon(tmp_path, manager)
    server = _serve(daemon)
    next(server)
    list(daemon.handle_message(_tier_message()))
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and not builds[0].closed:
        time.sleep(0.05)
    closed = builds[0].closed
    next(server, None)
    assert closed is True


# --- the client's half of the exchange -------------------------------------


def test_ask_tiers_over_the_socket_reports_the_tier_that_answered(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    server = _serve(daemon)
    next(server)
    reply = client_transport.ask_tiers(_request(), AgentContext(), env=daemon.env)
    next(server, None)
    assert (reply.outcome, reply.tier) == (daemon_mod.TIER_HANDLED, "fake")


def test_ask_tiers_over_the_socket_streams_the_proposal(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    server = _serve(daemon)
    next(server)
    reply = client_transport.ask_tiers(_request(), AgentContext(), env=daemon.env)
    next(server, None)
    commands = [event.proposal.command for event in reply.events if event.proposal is not None]
    assert commands == ["free -m"]


def test_tier_decision_over_the_socket_records_the_operator_decision(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, _manager(tmp_path))
    server = _serve(daemon)
    next(server)
    reply = client_transport.ask_tiers(_request(), AgentContext(), env=daemon.env)
    reported = client_transport.tier_decision(reply.route_id, "approved", env=daemon.env)
    next(server, None)
    assert reported is True


def test_ask_tiers_escalates_with_no_daemon_listening(tmp_path: Path) -> None:
    reply = client_transport.ask_tiers(_request(), AgentContext(), env=_env(tmp_path))
    assert reply.outcome == daemon_mod.TIER_ESCALATE


def test_ask_tiers_with_no_daemon_starts_none(tmp_path: Path) -> None:
    env = _env(tmp_path)
    client_transport.ask_tiers(_request(), AgentContext(), env=env)
    assert daemon_mod.is_running(env) is False


_NO_MODEL_SCRIPT = """
import json, sys
from nvsh import client_transport
from nvsh.agent.base import AgentRequest, RequestKind

env = {"XDG_RUNTIME_DIR": sys.argv[1], "XDG_STATE_HOME": sys.argv[1], "HOME": sys.argv[1]}
reply = client_transport.ask_tiers(
    AgentRequest(kind=RequestKind.EXPLICIT, prompt="how much memory is free"), env=env
)
loaded = sorted(name for name in sys.modules if name.startswith("nvsh.tiers"))
print(json.dumps({"outcome": reply.outcome, "tiers": loaded}))
"""


@pytest.mark.parametrize("key", ["outcome", "tiers"])
def test_ask_tiers_with_no_daemon_loads_no_tier_module(tmp_path: Path, key: str) -> None:
    proc = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-c", _NO_MODEL_SCRIPT, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    reported = json.loads(proc.stdout.strip().splitlines()[-1])
    expected = {"outcome": daemon_mod.TIER_ESCALATE, "tiers": []}
    assert reported[key] == expected[key]
