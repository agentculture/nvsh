"""Tests for the client's tier step (task t13).

The client asks the daemon's resident local tiers *before* the full agent,
renders a tier answer through the same panel and approval path an agent's
proposal takes, names the tier in the panel's header slot, and lets the
operator send a declined request on to the full agent.
"""

from __future__ import annotations

import io
import types

import pytest

from nvsh import client as client_mod
from nvsh import client_transport
from nvsh import panel as panel_mod
from nvsh.agent.base import AgentEvent, EventKind, Proposal, ProposalKind, RequestKind
from nvsh.daemon import TIER_ESCALATE, TIER_HANDLED


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    state = tmp_path / "state"
    config = tmp_path / "config"
    state.mkdir()
    config.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("NVSH_NO_DAEMON", "1")
    monkeypatch.setattr(
        client_mod, "_platform_block", lambda: "platform: dgx-spark\n  gb10: yes  [file: /etc/x]"
    )
    return types.SimpleNamespace(state=state, config=config, tmp=tmp_path)


def _tiers(xdg, enabled: bool) -> None:
    """Write the operator's ``[tiers] enabled`` setting."""
    nvsh_dir = xdg.config / "nvsh"
    nvsh_dir.mkdir(parents=True, exist_ok=True)
    (nvsh_dir / "config.toml").write_text(
        f"[tiers]\nenabled = {'true' if enabled else 'false'}\n", encoding="utf-8"
    )


def _args(tmp_path, line: str = "ls /nope", exit_code: int = 2):
    return types.SimpleNamespace(
        exit=exit_code, pipestatus=str(exit_code), line=line, cwd=str(tmp_path), log="", json=False
    )


def _panel(stdin_text: str = ""):
    return panel_mod.Panel(out=io.StringIO(), in_=io.StringIO(stdin_text), env={}, isatty=False)


def _stub_send(calls, events=None):
    def send(request, context=None, **kwargs):
        calls.append((request, context, kwargs))
        yield from (
            events
            or [AgentEvent(kind=EventKind.TEXT_DELTA, text="hi"), AgentEvent(kind=EventKind.DONE)]
        )

    return send


def _stub_tiers(asked, reply=None):
    answer = reply if reply is not None else client_transport.TierReply(outcome=TIER_ESCALATE)

    def ask_tiers(request, context=None, **kwargs):
        asked.append((request, context, kwargs))
        return answer

    return ask_tiers


def _handled(tier: str = "needle", events=(), route_id: str = "7", **extra):
    return client_transport.TierReply(
        outcome=TIER_HANDLED, tier=tier, route_id=route_id, events=tuple(events), **extra
    )


def _text_events(text: str = "the fan is at 40%"):
    return [
        AgentEvent(kind=EventKind.TEXT_DELTA, text=text, args={"tier": "needle"}),
        AgentEvent(kind=EventKind.DONE),
    ]


def _proposal_events(command: str, kind: ProposalKind = ProposalKind.FIX):
    proposal = Proposal(command=command, rationale="the tier's pick", kind=kind)
    return [
        AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal, args={"tier": "needle"}),
        AgentEvent(kind=EventKind.DONE),
    ]


# --- criterion 1: when the tiers are consulted at all --------------------


def test_disabled_tiers_are_never_consulted(xdg, monkeypatch):
    _tiers(xdg, False)
    asked: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers(asked))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    assert asked == []


def test_disabled_tiers_are_not_consulted_by_ask(xdg, monkeypatch):
    _tiers(xdg, False)
    asked: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers(asked))
    client_mod.ask("why is it hot?", panel=_panel())
    assert asked == []


def test_ctrl_g_consults_the_tiers_when_enabled(xdg, monkeypatch):
    _tiers(xdg, True)
    asked: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers(asked))
    client_mod.ask("why is it hot?", panel=_panel())
    assert len(asked) == 1


def test_slash_ask_consults_the_tiers_when_enabled(xdg, monkeypatch):
    _tiers(xdg, True)
    asked: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers(asked))
    client_mod.handle_slash("/ask why is it hot?", panel=_panel())
    assert len(asked) == 1


def test_a_failure_is_offered_to_the_tiers_unchanged(xdg, monkeypatch):
    _tiers(xdg, True)
    asked: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers(asked))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    assert asked[0][0].kind is RequestKind.FAILURE


def test_an_explicit_target_sends_the_request_past_both_tiers(xdg, monkeypatch):
    _tiers(xdg, True)
    asked: list = []
    calls: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    monkeypatch.setattr(client_transport, "one_shot", _stub_send(calls))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers(asked))
    monkeypatch.setattr("nvsh.agent.registry.installed", lambda name, config=None: True)
    client_mod.ask("why is it hot?", agent="claude", panel=_panel())
    assert asked == []


def test_a_follow_up_turn_does_not_consult_the_tiers_again(xdg, monkeypatch):
    _tiers(xdg, True)
    (xdg.config / "nvsh" / "approved.toml").write_text(
        'user_patterns = [\n  "echo *",\n]\n', encoding="utf-8"
    )
    asked: list = []
    calls: list = []
    streams = [
        _proposal_events("echo inspector-ran", ProposalKind.INSPECT),
        [AgentEvent(kind=EventKind.TEXT_DELTA, text="done"), AgentEvent(kind=EventKind.DONE)],
    ]

    def send(request, context=None, **kwargs):
        calls.append((request, context, kwargs))
        yield from streams[min(len(calls) - 1, len(streams) - 1)]

    monkeypatch.setattr(client_transport, "send", send)
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers(asked))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    assert len(asked) == 1, "the follow-up belongs to the full agent's conversation"


# --- criterion 2: the panel header names who answered --------------------


def test_the_header_names_the_tier_that_answered(xdg, monkeypatch):
    _tiers(xdg, True)
    calls: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    monkeypatch.setattr(
        client_transport, "ask_tiers", _stub_tiers([], _handled(events=_text_events()))
    )
    panel = _panel()
    client_mod.ask("why is it hot?", panel=panel)
    assert "\nneedle\n" in panel.out.getvalue()


def test_a_tier_that_answered_never_reaches_the_full_agent(xdg, monkeypatch):
    _tiers(xdg, True)
    calls: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    monkeypatch.setattr(
        client_transport, "ask_tiers", _stub_tiers([], _handled(events=_text_events()))
    )
    client_mod.ask("why is it hot?", panel=_panel())
    assert calls == []


def test_the_header_names_the_escalation(xdg, monkeypatch):
    _tiers(xdg, True)
    reply = client_transport.TierReply(
        outcome=TIER_ESCALATE, declines=(("needle", "low_confidence"),)
    )
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([], reply))
    panel = _panel()
    client_mod.ask("why is it hot?", panel=panel)
    assert "needle -> " in panel.out.getvalue()


def test_the_header_is_unchanged_when_no_tier_was_consulted(xdg, monkeypatch):
    _tiers(xdg, False)
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    panel = _panel()
    client_mod.ask("why is it hot?", panel=panel)
    assert "->" not in panel.out.getvalue()


def test_an_unavailable_tier_says_so_once(xdg, monkeypatch):
    _tiers(xdg, True)
    notice = "needle unavailable: 512 MB available is below the 1024 MB floor"
    reply = client_transport.TierReply(
        outcome=TIER_ESCALATE,
        events=(
            AgentEvent(kind=EventKind.STATUS, text=notice),
            AgentEvent(kind=EventKind.STATUS, text=notice),
            AgentEvent(kind=EventKind.DONE),
        ),
    )
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([], reply))
    panel = _panel()
    client_mod.ask("why is it hot?", panel=panel)
    assert panel.out.getvalue().count(notice) == 1


def test_an_unavailable_tier_is_not_an_error(xdg, monkeypatch):
    _tiers(xdg, True)
    reply = client_transport.TierReply(
        outcome=TIER_ESCALATE,
        events=(AgentEvent(kind=EventKind.STATUS, text="needle unavailable: no model"),),
    )
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([], reply))
    panel = _panel()
    client_mod.ask("why is it hot?", panel=panel)
    assert "error" not in panel.out.getvalue().lower()


# --- the escalation carries the tiers' read-only work --------------------


def test_the_escalated_request_carries_the_tiers_inspection_results(xdg, monkeypatch):
    _tiers(xdg, True)
    reply = client_transport.TierReply(
        outcome=TIER_ESCALATE,
        escalation_context=(("memory_stats", "MemAvailable:  2000 kB"),),
    )
    calls: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([], reply))
    client_mod.ask("why is it hot?", panel=_panel())
    assert "memory_stats: MemAvailable: 2000 kB" in calls[0][1].output


def test_an_empty_escalation_context_leaves_the_context_alone(xdg, monkeypatch):
    _tiers(xdg, True)
    calls: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([]))
    client_mod.ask("why is it hot?", panel=_panel())
    assert client_mod.ESCALATION_HEADER not in calls[0][1].output


# --- criterion 3: declining a tier proposal ------------------------------


def test_a_tier_proposal_is_approved_on_the_usual_keypress(xdg, monkeypatch):
    _tiers(xdg, True)
    marker = xdg.tmp / "ran"
    reply = _handled(events=_proposal_events(f"touch {marker}"))
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([], reply))
    client_mod.ask("fix it", panel=_panel("\n"))
    assert marker.exists()


def test_declining_a_tier_proposal_can_send_the_same_request_on(xdg, monkeypatch):
    _tiers(xdg, True)
    reply = _handled(events=_proposal_events("touch /nope/nope"))
    calls: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([], reply))
    panel = _panel("q\n")
    monkeypatch.setattr(panel, "confirm", lambda question: True)
    client_mod.ask("fix it", panel=panel)
    assert [call[0].ask for call in calls] == ["fix it"]


def test_the_escalated_request_after_a_decline_names_the_tier(xdg, monkeypatch):
    _tiers(xdg, True)
    reply = _handled(events=_proposal_events("touch /nope/nope"))
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([], reply))
    panel = _panel("q\n")
    monkeypatch.setattr(panel, "confirm", lambda question: True)
    client_mod.ask("fix it", panel=panel)
    assert "needle -> " in panel.out.getvalue()


def test_declining_without_agreeing_never_reaches_the_full_agent(xdg, monkeypatch):
    _tiers(xdg, True)
    reply = _handled(events=_proposal_events("touch /nope/nope"))
    calls: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([], reply))
    # A panel that cannot prompt answers the offer "no" without asking.
    client_mod.ask("fix it", panel=_panel("q\n"))
    assert calls == []


def test_a_declined_tier_proposal_that_stops_there_exits_declined(xdg, monkeypatch):
    _tiers(xdg, True)
    reply = _handled(events=_proposal_events("touch /nope/nope"))
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "ask_tiers", _stub_tiers([], reply))
    assert client_mod.ask("fix it", panel=_panel("q\n")) == 3


# --- the operator's answer is reported back ------------------------------


def _record_decisions(reported):
    def tier_decision(route_id, decision, **kwargs):
        reported.append((route_id, decision))
        return True

    return tier_decision


def test_an_approved_tier_proposal_is_reported(xdg, monkeypatch):
    _tiers(xdg, True)
    marker = xdg.tmp / "ran"
    reported: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(
        client_transport,
        "ask_tiers",
        _stub_tiers([], _handled(events=_proposal_events(f"touch {marker}"))),
    )
    monkeypatch.setattr(client_transport, "tier_decision", _record_decisions(reported))
    client_mod.ask("fix it", panel=_panel("\n"))
    assert reported == [("7", client_mod.TIER_APPROVED)]


def test_a_declined_tier_proposal_is_reported(xdg, monkeypatch):
    _tiers(xdg, True)
    reported: list = []
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(
        client_transport,
        "ask_tiers",
        _stub_tiers([], _handled(events=_proposal_events("touch /nope/nope"))),
    )
    monkeypatch.setattr(client_transport, "tier_decision", _record_decisions(reported))
    client_mod.ask("fix it", panel=_panel("q\n"))
    assert reported == [("7", client_mod.TIER_DECLINED)]


def test_a_failure_to_report_never_breaks_the_turn(xdg, monkeypatch):
    _tiers(xdg, True)
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(
        client_transport, "ask_tiers", _stub_tiers([], _handled(events=_text_events()))
    )

    def boom(route_id, decision, **kwargs):
        raise OSError("the daemon went away")

    monkeypatch.setattr(client_transport, "tier_decision", boom)
    assert client_mod.ask("why is it hot?", panel=_panel()) == 0
