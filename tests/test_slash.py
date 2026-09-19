"""Tests for nvsh.slash (task t14): registry, platform filtering, completion,
dispatch routing and /undo/-approve behaviour.
"""

from __future__ import annotations

import io
import json
import types

import pytest

from nvsh import client as client_mod
from nvsh import client_transport
from nvsh import slash as slash_mod
from nvsh.agent.base import AgentEvent, EventKind, RequestKind
from nvsh.panel import Panel


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
    return types.SimpleNamespace(state=state, config=config, tmp=tmp_path)


def _panel(stdin_text: str = ""):
    return Panel(out=io.StringIO(), in_=io.StringIO(stdin_text), env={}, isatty=False)


# --- registry --------------------------------------------------------------


def test_registry_resolves_every_command_by_name():
    for name in (
        "ask",
        "fix",
        "explain",
        "retry",
        "context",
        "agent",
        "help",
        "undo",
        "approve",
        "doctor",
        "power",
        "clocks",
    ):
        cmd = slash_mod.resolve(name)
        assert cmd is not None
        assert cmd.name == name


def test_resolve_unknown_is_none():
    assert slash_mod.resolve("nope") is None


def test_remember_good_family_is_not_registered():
    for name in ("remember-good", "diff-good", "restore-good"):
        assert slash_mod.resolve(name) is None


# --- platform filtering ------------------------------------------------


def test_power_and_clocks_hidden_on_dgx_spark():
    names = {cmd.name for cmd in slash_mod.visible_commands("dgx-spark")}
    assert "power" not in names
    assert "clocks" not in names
    assert "ask" in names


def test_power_and_clocks_visible_on_jetson():
    names = {cmd.name for cmd in slash_mod.visible_commands("jetson")}
    assert "power" in names
    assert "clocks" in names


def test_dispatch_reports_hidden_command_as_unknown(xdg):
    p = _panel()
    rc = slash_mod.dispatch("/power", platform_kind="dgx-spark", panel=p)
    assert rc == 1
    assert "unknown" in p.out.getvalue().lower()


def test_dispatch_runs_stub_on_jetson(xdg):
    p = _panel()
    rc = slash_mod.dispatch("/power", platform_kind="jetson", panel=p)
    assert rc == 0
    assert "not implemented" in p.out.getvalue()


# --- completion ----------------------------------------------------------


def test_complete_with_no_words_is_the_full_palette():
    items = slash_mod.complete([], "dgx-spark")
    values = {item.value for item in items}
    assert "/ask" in values
    assert "/doctor" in values
    assert "/power" not in values  # jetson-only, hidden on dgx-spark


def test_complete_full_palette_is_jetson_aware():
    items = slash_mod.complete([], "jetson")
    values = {item.value for item in items}
    assert "/power" in values
    assert "/clocks" in values


def test_complete_doctor_arguments():
    items = slash_mod.complete(["/doctor", "--st"], "dgx-spark")
    values = {item.value for item in items}
    assert "--json" in values
    assert "--strict" in values


def test_complete_agent_offers_use_and_adapter_names():
    items = slash_mod.complete(["/agent"], "dgx-spark")
    values = {item.value for item in items}
    assert "list" in values
    assert "use" in values
    assert "pi" in values


def test_complete_approve_offers_verbs():
    items = slash_mod.complete(["/approve"], "dgx-spark")
    values = {item.value for item in items}
    assert {"list", "add", "remove", "--session"} <= values


def test_complete_unknown_command_is_empty():
    assert slash_mod.complete(["/nope"], "dgx-spark") == []


def test_complete_command_with_no_provider_is_empty():
    # /help has no completion provider of its own.
    assert slash_mod.complete(["/help", "wh"], "dgx-spark") == []


# --- dispatch routing ------------------------------------------------------


def _stub_send(calls, events=None):
    def send(request, context=None, **kwargs):
        calls.append((request, context, kwargs))
        yield from (
            events
            or [AgentEvent(kind=EventKind.TEXT_DELTA, text="hi"), AgentEvent(kind=EventKind.DONE)]
        )

    return send


def test_dispatch_ask_routes_to_client_ask_with_slash_kind(xdg, monkeypatch):
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    rc = slash_mod.dispatch("/ask why", platform_kind="dgx-spark", panel=_panel())
    assert rc == 0
    assert len(calls) == 1
    request = calls[0][0]
    assert request.kind is RequestKind.SLASH
    assert "why" in request.prompt


def test_dispatch_fix_routes_to_client_fix(xdg, monkeypatch):
    client_mod.save_last_failure(
        types.SimpleNamespace(exit=2, pipestatus="2", line="ls /nope", cwd=str(xdg.tmp), log="")
    )
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    rc = slash_mod.dispatch("/fix", platform_kind="dgx-spark", panel=_panel())
    assert rc == 0
    assert calls[0][0].kind.value == "failure"


def test_dispatch_unknown_command_is_a_user_error(xdg):
    p = _panel()
    rc = slash_mod.dispatch("/nope", platform_kind="dgx-spark", panel=p)
    assert rc == 1
    assert "unknown" in p.out.getvalue().lower()


def test_dispatch_result_reports_whether_the_line_was_handled(xdg):
    """d5: 'handled' is what lets `nvsh slash` exit 0 for a verb that ran."""
    handled = slash_mod.dispatch_result("/help", platform_kind="dgx-spark", panel=_panel())
    assert handled.handled is True
    assert handled.exit_code == 0

    unknown = slash_mod.dispatch_result("/nope", platform_kind="dgx-spark", panel=_panel())
    assert unknown.handled is False
    assert unknown.exit_code == 1

    empty = slash_mod.dispatch_result("", platform_kind="dgx-spark", panel=_panel())
    assert empty.handled is False

    hidden = slash_mod.dispatch_result("/power", platform_kind="dgx-spark", panel=_panel())
    assert hidden.handled is False


def test_dispatch_result_keeps_a_handled_verbs_own_exit_code(xdg):
    ran = slash_mod.dispatch_result("/agent use bogus", platform_kind="dgx-spark", panel=_panel())
    assert ran.handled is True
    assert ran.exit_code == 1


def test_dispatch_empty_line_is_a_user_error(xdg):
    p = _panel()
    assert slash_mod.dispatch("/", platform_kind="dgx-spark", panel=p) == 1
    assert slash_mod.dispatch("", platform_kind="dgx-spark", panel=p) == 1


def test_dispatch_help_lists_visible_commands(xdg):
    p = _panel()
    rc = slash_mod.dispatch("/help", platform_kind="dgx-spark", panel=p)
    assert rc == 0
    out = p.out.getvalue()
    assert "/ask" in out
    assert "/power" not in out


# --- /undo: never runs anything on the machine ------------------------


def test_undo_never_calls_the_executor_with_no_daemon(xdg, monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "_run_command", lambda *a, **k: calls.append((a, k)))
    p = _panel()
    rc = slash_mod.dispatch("/undo", platform_kind="dgx-spark", panel=p)
    assert rc == 0
    assert calls == []


def test_undo_clears_pending_proposal_with_no_daemon(xdg):
    path = client_mod.last_failure_path()
    client_mod._write_private_json(
        path, {"failure_id": "abc", "line": "ls /nope", "pending_proposal": {"command": "ls"}}
    )
    rc = slash_mod.dispatch("/undo", platform_kind="dgx-spark", panel=_panel())
    assert rc == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "pending_proposal" not in data
    assert data["failure_id"] == "abc"  # the rest of the failure record survives


def test_undo_reports_the_daemon_rewind_and_leaves_the_state_file_alone(xdg, monkeypatch):
    """With a daemon answering, /undo says so and touches no local state."""
    path = client_mod.last_failure_path()
    client_mod._write_private_json(
        path, {"failure_id": "abc", "pending_proposal": {"command": "ls"}}
    )
    monkeypatch.setattr(
        client_transport, "control", lambda *a, **k: [AgentEvent(kind=EventKind.DONE)]
    )
    p = _panel()
    rc = slash_mod.dispatch("/undo", platform_kind="dgx-spark", panel=p)
    assert rc == 0
    assert "undid the last agent turn" in p.out.getvalue()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pending_proposal"] == {"command": "ls"}


def test_undo_with_no_recorded_failure_still_never_executes(xdg, monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "_run_command", lambda *a, **k: calls.append((a, k)))
    rc = slash_mod.dispatch("/undo", platform_kind="dgx-spark", panel=_panel())
    assert rc == 0
    assert calls == []


# --- /approve: add/remove/list over a tmp XDG_CONFIG_HOME ---------------


def test_approve_add_list_remove_roundtrip(xdg):
    p1 = _panel()
    rc = slash_mod.dispatch("/approve add 'kubectl get *'", platform_kind="dgx-spark", panel=p1)
    assert rc == 0

    p2 = _panel()
    rc = slash_mod.dispatch("/approve list", platform_kind="dgx-spark", panel=p2)
    assert rc == 0
    assert "kubectl get *" in p2.out.getvalue()

    p3 = _panel()
    rc = slash_mod.dispatch("/approve remove 'kubectl get *'", platform_kind="dgx-spark", panel=p3)
    assert rc == 0

    p4 = _panel()
    slash_mod.dispatch("/approve list", platform_kind="dgx-spark", panel=p4)
    assert "kubectl get *" not in p4.out.getvalue()


def test_approve_add_session_scope_is_listed_under_session_not_user(xdg):
    """d15: a session approval survives into the next process, under its own scope.

    It used to live in one ``Approvals`` instance's memory, so the very next
    ``/approve list`` -- a fresh ``Approvals.load()`` -- had already
    forgotten it, and the operator was asked again immediately. It now lives
    in the login session's runtime dir, so it is listed; what must still
    never happen is it reaching ``approved.toml``.
    """
    p1 = _panel()
    slash_mod.dispatch(
        "/approve add 'apt install *' --session", platform_kind="dgx-spark", panel=p1
    )
    p2 = _panel()
    slash_mod.dispatch("/approve list", platform_kind="dgx-spark", panel=p2)
    user_block, session_block = p2.out.getvalue().split("session:", 1)
    assert "apt install *" not in user_block
    assert "apt install *" in session_block
    assert not (xdg.config / "nvsh" / "approved.toml").exists()


def test_approve_add_refuses_dangerous_pattern(xdg):
    p = _panel()
    rc = slash_mod.dispatch("/approve add 'sudo *'", platform_kind="dgx-spark", panel=p)
    assert rc == 1


# --- /agent --------------------------------------------------------------


def test_agent_list_reports_pi(xdg):
    p = _panel()
    rc = slash_mod.dispatch("/agent list", platform_kind="dgx-spark", panel=p)
    assert rc == 0
    assert "pi" in p.out.getvalue()


def test_agent_use_unknown_name_is_a_user_error(xdg):
    p = _panel()
    rc = slash_mod.dispatch("/agent use bogus", platform_kind="dgx-spark", panel=p)
    assert rc == 1


# --- d16: /steer -----------------------------------------------------------


def test_steer_is_registered_and_visible_everywhere():
    cmd = slash_mod.resolve("steer")
    assert cmd is not None
    assert cmd.safety == slash_mod.SAFETY_AGENT
    assert cmd.visible_on("generic")
    assert "/steer" in [item.value for item in slash_mod.complete([], "generic")]


def test_dispatch_steer_passes_the_whole_line_as_the_text(xdg, monkeypatch):
    seen = []
    monkeypatch.setattr(client_mod, "steer", lambda text, **kw: seen.append(text) or 0)
    assert slash_mod.dispatch('/steer just run "free -h"', platform_kind="generic") == 0
    assert seen == ['just run "free -h"']


# --- d23: /ask --agent <name> ---------------------------------------------


def test_complete_full_palette_offers_the_agent_marks():
    values = {item.value for item in slash_mod.complete([], "dgx-spark")}
    assert "@pi" in values
    assert "@qwen" in values
    assert "@openai-compat" in values


# --- task t6: @target grammar (backend[/model[/effort]] and aliases) ------


def test_agent_mark_items_default_first_then_aliases_then_adapters(xdg):
    (xdg.config / "nvsh").mkdir()
    (xdg.config / "nvsh" / "config.toml").write_text(
        '[aliases]\ndefault = "pi"\nreviewer = "claude/opus/high"\nlocal = "pi"\n'
    )
    values = [item.value for item in slash_mod.agent_mark_items()]
    assert values[0] == "@default"
    # aliases (minus 'default') come next, alphabetically, ahead of any
    # adapter not itself an alias name.
    alias_names = ["@local", "@reviewer"]
    assert values[1 : 1 + len(alias_names)] == alias_names
    for name in ("@pi", "@qwen", "@claude", "@codex", "@openai-compat"):
        assert name in values
    # no duplicates: 'pi' and 'claude' are both an alias target's backend
    # and a registered adapter name, but each mark appears once.
    assert len(values) == len(set(values))


def test_agent_mark_items_tolerates_missing_config(xdg):
    # No config.toml written at all: 'default' still appears, no exception.
    values = [item.value for item in slash_mod.agent_mark_items()]
    assert values[0] == "@default"


def test_agent_mark_items_tolerates_invalid_config(xdg):
    (xdg.config / "nvsh").mkdir()
    (xdg.config / "nvsh" / "config.toml").write_text("not valid toml [[[")
    values = [item.value for item in slash_mod.agent_mark_items()]
    assert values[0] == "@default"


def test_split_agent_flag_returns_the_full_target_string():
    for target in ("claude/sonnet/medium", "reviewer", "local", "default", "pi"):
        name, rest = slash_mod.split_agent_flag(f"--agent {target} why is the gpu slow")
        assert name == target
        assert rest == "why is the gpu slow"


def test_split_agent_flag_equals_form_full_target_string():
    name, rest = slash_mod.split_agent_flag("--agent=claude/sonnet/medium why is the gpu slow")
    assert name == "claude/sonnet/medium"
    assert rest == "why is the gpu slow"


def test_complete_ask_offers_the_agent_flag_and_names():
    values = {item.value for item in slash_mod.complete(["/ask"], "dgx-spark")}
    assert "--agent" in values
    assert "pi" in values


def test_dispatch_ask_agent_flag_is_split_off_the_question(xdg, monkeypatch):
    seen = {}

    def fake_ask(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["agent"] = kwargs.get("agent")
        return 0

    monkeypatch.setattr(client_mod, "ask", fake_ask)
    assert slash_mod.dispatch("/ask --agent pi how much ram?", platform_kind="dgx-spark") == 0
    assert seen == {"prompt": "how much ram?", "agent": "pi"}


def test_dispatch_ask_agent_equals_form(xdg, monkeypatch):
    seen = {}

    def fake_ask(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["agent"] = kwargs.get("agent")
        return 0

    monkeypatch.setattr(client_mod, "ask", fake_ask)
    assert slash_mod.dispatch("/ask --agent=qwen why?", platform_kind="dgx-spark") == 0
    assert seen == {"prompt": "why?", "agent": "qwen"}


def test_dispatch_ask_without_the_flag_keeps_the_whole_question(xdg, monkeypatch):
    seen = {}

    def fake_ask(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["agent"] = kwargs.get("agent")
        return 0

    monkeypatch.setattr(client_mod, "ask", fake_ask)
    assert slash_mod.dispatch("/ask --agent is not a flag here", platform_kind="dgx-spark") == 0
    assert seen["agent"] == "is"  # '--agent <word>' always binds the next word


def test_ask_with_an_unavailable_agent_prints_one_line(xdg, monkeypatch):
    from nvsh.agent import registry

    monkeypatch.setattr(registry, "installed", lambda name, which=None, config=None: name == "pi")
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "one_shot", _stub_send([]))
    p = _panel()
    rc = slash_mod.dispatch("/ask --agent qwen why?", platform_kind="dgx-spark", panel=p)
    lines = [ln for ln in p.out.getvalue().splitlines() if ln.strip()]
    assert rc == 1
    assert len(lines) == 1, lines
    assert lines[0].startswith("nvsh: @qwen is not available: ")


def test_ask_with_an_unknown_agent_names_the_known_ones(xdg, monkeypatch):
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "one_shot", _stub_send([]))
    p = _panel()
    slash_mod.dispatch("/ask --agent nope why?", platform_kind="dgx-spark", panel=p)
    out = p.out.getvalue()
    assert "nvsh: @nope is not available: " in out
    assert "pi" in out


# --- PR #8 review: malformed quoting is an input error, not a guess ---------


def test_unterminated_quote_is_rejected_not_reparsed(xdg, monkeypatch):
    """The dispatcher contract: a parse failure is reported, never guessed at.

    Re-splitting on whitespace would hand ``/ask "question`` to the agent as
    if it had parsed.
    """
    sent = []
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    monkeypatch.setattr(client_transport, "one_shot", lambda *a, **k: sent.append(a) or iter(()))
    p = _panel()
    result = slash_mod.dispatch_result('/ask "question', platform_kind="dgx-spark", panel=p)
    assert result.handled is False
    assert result.exit_code == 1
    assert not sent
    out = p.out.getvalue()
    assert "quot" in out.lower() or "parse" in out.lower()


def test_malformed_quoting_names_the_line_and_suggests_help(xdg):
    p = _panel()
    slash_mod.dispatch_result("/fix 'oops", platform_kind="dgx-spark", panel=p)
    assert "/help" in p.out.getvalue()
