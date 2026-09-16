"""CLI-level tests for the doctor extensions (task t17).

Covers the acceptance criteria that need the full ``_diagnose()``/``cmd_doctor``
wiring: the wheel-install branch still runs every new check, and the
``--prompt-command``/``--bind-p``/``--keymap`` flags reach the in-shell checks.
``tests/test_doctor_checks.py`` covers the pure functions in isolation and
``tests/test_doctor_resident_prompt.py`` pins the pre-existing checks this
task must not disturb.
"""

from __future__ import annotations

from nvsh import doctor_checks
from nvsh.cli._commands import doctor as doctor_mod

_NEW_CHECK_IDS = {
    "platform_detected",
    "agent_configured",
    "agent_reachable",
    "default_target_not_demo",
    "hook_sourced",
    "hook_first_in_prompt_command",
    "bindings_present",
    "capture_active",
    "daemon_status",
    "terminfo_present",
}


def test_wheel_install_branch_still_runs_every_new_check(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NVSH_HOOK_VERSION", raising=False)

    report = doctor_mod._diagnose()

    ids = {c["id"] for c in report["checks"]}
    assert _NEW_CHECK_IDS <= ids
    # The old single "source_checkout" info check is still there too.
    assert "source_checkout" in ids


def test_source_checkout_case_reports_the_hook_checks_as_info_when_unhooked(monkeypatch, tmp_path):
    """Wheel-install, no hooked shell: the hook-only checks degrade to info,
    per the "cannot run outside a hooked shell" rule, and never block
    healthy on their own. Backend reachability etc. are real checks and may
    legitimately warn/error depending on the machine -- that is not this
    test's concern (see tests/test_doctor_checks.py for those in isolation).
    """
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NVSH_HOOK_VERSION", raising=False)
    monkeypatch.delenv("NVSH_LOG", raising=False)

    report = doctor_mod._diagnose()

    hook_only_ids = {
        "hook_sourced",
        "hook_first_in_prompt_command",
        "bindings_present",
    }
    for check in report["checks"]:
        if check["id"] in hook_only_ids:
            assert check["passed"] is False
            assert check["severity"] == "info", check


def test_hook_first_in_prompt_command_flows_through_prompt_command_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    report = doctor_mod._diagnose(
        prompt_command_text='declare -a PROMPT_COMMAND=([0]="__ghostty_hook" [1]="__nvsh_hook")'
    )

    check = next(c for c in report["checks"] if c["id"] == "hook_first_in_prompt_command")
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert report["healthy"] is False


def test_bindings_present_flows_through_bind_p_and_keymap_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    # The real three-dump payload: bind -p, then the marker, then bind -s and
    # bind -X (which quotes the shell command name).
    bind_p_text = (
        '"\\C-a": beginning-of-line\n'
        + doctor_checks.BIND_SECTION_MARKER
        + '\n"\\C-m": "\\C-x\\C-n\\C-j"\n'
        + '"\\C-x\\C-g": "__nvsh_ctrl_g"\n"\\C-g": "\\C-x\\C-g\\C-j"\n'
        + '"\\C-x\\C-n": "__nvsh_enter"\n'
    )
    report = doctor_mod._diagnose(bind_p_text=bind_p_text, keymap="vi-insert")

    check = next(c for c in report["checks"] if c["id"] == "bindings_present")
    assert check["passed"] is True, check


def test_healthy_ignores_info_severity_failures(monkeypatch, tmp_path):
    """The general rule: an info-severity failed check never flips healthy."""
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setattr(
        doctor_mod,
        "doctor_checks",
        _StubChecks(
            [{"id": "x", "passed": False, "severity": "info", "message": "", "remediation": ""}]
        ),
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    report = doctor_mod._diagnose()
    assert report["healthy"] is True


class _StubChecks:
    def __init__(self, checks):
        self._checks = checks

    def collect_checks(self, **_kwargs):
        return self._checks


def _text_report(capsys):
    """Render ``cmd_doctor``'s text output over a stubbed check list."""
    import argparse

    args = argparse.Namespace(json=False, prompt_command=None, bind_p=None, keymap=None)
    doctor_mod.cmd_doctor(args)
    return capsys.readouterr().out


def test_text_output_never_renders_FAIL_for_an_info_check(monkeypatch, tmp_path, capsys):
    """d4c: the summary said healthy while four checks printed [FAIL]."""
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setattr(
        doctor_mod,
        "doctor_checks",
        _StubChecks(
            [
                {
                    "id": "bindings_present",
                    "passed": False,
                    "severity": "info",
                    "message": "not running in a hooked shell",
                    "remediation": "run /doctor from a hooked shell",
                }
            ]
        ),
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    out = _text_report(capsys)

    assert "nvsh doctor: healthy" in out
    assert "[FAIL]" not in out, out
    assert "[info] bindings_present:" in out


def test_text_output_renders_FAIL_only_when_the_report_is_unhealthy(monkeypatch, tmp_path, capsys):
    """The invariant behind d4c: [FAIL] on screen <-> healthy=false in JSON."""
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setattr(
        doctor_mod,
        "doctor_checks",
        _StubChecks(
            [
                {
                    "id": "info_fail",
                    "passed": False,
                    "severity": "info",
                    "message": "m",
                    "remediation": "",
                },
                {
                    "id": "warn_fail",
                    "passed": False,
                    "severity": "warning",
                    "message": "m",
                    "remediation": "",
                },
            ]
        ),
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    out = _text_report(capsys)

    assert "nvsh doctor: unhealthy" in out
    assert "[info] info_fail:" in out
    assert "[FAIL] warn_fail:" in out


def test_unhooked_shell_text_report_has_no_FAIL_lines(monkeypatch, tmp_path, capsys):
    """The thor/orin symptom, end to end: a plain shell reports healthy and
    prints no [FAIL] line for the checks that cannot run outside a hook."""
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NVSH_HOOK_VERSION", raising=False)
    monkeypatch.delenv("NVSH_LOG", raising=False)

    report = doctor_mod._diagnose()
    out = _text_report(capsys)

    fail_lines = [ln for ln in out.splitlines() if ln.startswith("[FAIL]")]
    assert bool(fail_lines) is (report["healthy"] is False), (report["healthy"], fail_lines)
    for check_id in ("hook_sourced", "hook_first_in_prompt_command", "bindings_present"):
        assert f"[FAIL] {check_id}:" not in out, out


def test_cmd_doctor_registers_new_flags():
    import argparse

    sub = argparse.ArgumentParser().add_subparsers()
    doctor_mod.register(sub)
    parser = sub.choices["doctor"]
    args = parser.parse_args(
        ["--json", "--prompt-command", "x", "--bind-p", "y", "--keymap", "emacs"]
    )
    assert args.prompt_command == "x"
    assert args.bind_p == "y"
    assert args.keymap == "emacs"


def test_cmd_doctor_registers_apply_flag():
    import argparse

    sub = argparse.ArgumentParser().add_subparsers()
    doctor_mod.register(sub)
    parser = sub.choices["doctor"]
    args = parser.parse_args(["--apply"])
    assert args.apply is True
    args = parser.parse_args([])
    assert args.apply is False


# --- --apply / agent_turn_not_hung (task t19) -------------------------------


class _RecordingTransport:
    """Records every ``status``/``kill_active`` call it is given (task t19).

    Used to assert plain ``nvsh doctor`` (no ``--apply``) never sends a
    mutating control message: only ``status`` calls should ever appear.
    """

    def __init__(self, running: bool = False, active_turn=None, target=None):
        self.calls: list[tuple[str, dict]] = []
        self.running = running
        self.active_turn = active_turn
        self.target = target

    def status(self, *, env=None, timeout=5.0):
        self.calls.append(("status", {"timeout": timeout}))
        return {"running": self.running, "active_turn": self.active_turn, "target": self.target}

    def kill_active(self, *, confirmed=False, env=None):
        self.calls.append(("kill_active", {"confirmed": confirmed}))
        return "killed"


def test_plain_doctor_never_sends_a_mutating_control_message(monkeypatch, tmp_path):
    """Acceptance criterion: plain doctor never sends a mutating control message."""
    fake = _RecordingTransport(running=True, active_turn={"shell": "1", "elapsed": 5.0})
    monkeypatch.setattr(doctor_checks, "client_transport", fake)
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    report = doctor_mod._diagnose()

    kinds = {kind for kind, _ in fake.calls}
    assert kinds, "the agent_turn_not_hung check never probed the daemon at all"
    assert kinds == {"status"}, f"a mutating control message was sent: {fake.calls}"
    assert any(c["id"] == "agent_turn_not_hung" for c in report["checks"])


def test_confirm_live_owner_kill_refuses_without_a_tty(monkeypatch):
    monkeypatch.setattr(doctor_mod, "_is_interactive", lambda: False)
    assert doctor_mod._confirm_live_owner_kill("A", 12.0) is False


def test_confirm_live_owner_kill_asks_and_names_the_owner(monkeypatch):
    monkeypatch.setattr(doctor_mod, "_is_interactive", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "y")
    assert doctor_mod._confirm_live_owner_kill("shellA", 42.0) is True
    assert "shellA" in prompts[0]


def test_apply_agent_turn_not_hung_kills_dead_owner_without_confirmation(tmp_path):
    """Acceptance: --apply on a dead-owner hung turn kills it."""
    from nvsh.agent.audit import AuditLog

    calls = []

    def fake_status(*, env, timeout):
        return {"active_turn": {"shell": "999", "elapsed": 400.0}, "target": {"backend": "demo"}}

    def fake_kill_active(*, confirmed, env):
        calls.append(("kill_active", confirmed))
        return "killed"

    def fake_confirm(owner, elapsed):
        raise AssertionError("a dead owner must never need a confirm prompt")

    audit = AuditLog(path=tmp_path / "audit.jsonl")

    result = doctor_mod._apply_agent_turn_not_hung(
        env={},
        status=fake_status,
        kill_active=fake_kill_active,
        pid_gone=lambda shell: True,
        confirm=fake_confirm,
        audit=audit,
    )

    assert result["outcome"] == "killed"
    assert calls == [("kill_active", True)]
    entries = audit.read_all()
    assert entries[-1]["event"] == "stop"
    assert entries[-1]["kind"] == "doctor_apply"
    assert entries[-1]["shell"] == "999"
    assert entries[-1]["outcome"] == "killed"


def test_apply_agent_turn_not_hung_refuses_live_owner_without_confirmation(tmp_path):
    """Acceptance: --apply on a live owner's turn without confirm changes nothing."""
    from nvsh.agent.audit import AuditLog

    def fake_status(*, env, timeout):
        return {"active_turn": {"shell": "A", "elapsed": 10.0}, "target": None}

    def fake_kill_active(*, confirmed, env):
        raise AssertionError("kill_active must never be sent when confirmation was refused")

    audit = AuditLog(path=tmp_path / "audit.jsonl")

    result = doctor_mod._apply_agent_turn_not_hung(
        env={},
        status=fake_status,
        kill_active=fake_kill_active,
        pid_gone=lambda shell: False,
        confirm=lambda owner, elapsed: False,
        audit=audit,
    )

    assert result["outcome"] == "refused"
    entries = audit.read_all()
    assert entries[-1]["kind"] == "doctor_apply"
    assert entries[-1]["shell"] == "A"
    assert entries[-1]["outcome"] == "refused"


def test_apply_agent_turn_not_hung_kills_live_owner_once_confirmed(tmp_path):
    from nvsh.agent.audit import AuditLog

    calls = []

    def fake_status(*, env, timeout):
        return {"active_turn": {"shell": "A", "elapsed": 10.0}, "target": None}

    def fake_kill_active(*, confirmed, env):
        calls.append(confirmed)
        return "killed"

    audit = AuditLog(path=tmp_path / "audit.jsonl")

    result = doctor_mod._apply_agent_turn_not_hung(
        env={},
        status=fake_status,
        kill_active=fake_kill_active,
        pid_gone=lambda shell: False,
        confirm=lambda owner, elapsed: True,
        audit=audit,
    )

    assert result["outcome"] == "killed"
    assert calls == [True]


def test_apply_agent_turn_not_hung_is_a_noop_when_nothing_is_active(tmp_path):
    from nvsh.agent.audit import AuditLog

    def fake_status(*, env, timeout):
        return {"active_turn": None, "target": None}

    def boom(**_kwargs):
        raise AssertionError("nothing to kill")

    audit = AuditLog(path=tmp_path / "audit.jsonl")

    result = doctor_mod._apply_agent_turn_not_hung(
        env={},
        status=fake_status,
        kill_active=boom,
        pid_gone=boom,
        confirm=boom,
        audit=audit,
    )
    assert result["outcome"] == "idle"
    assert audit.read_all() == []


def test_apply_then_recheck_passes_for_a_dead_owner_turn(tmp_path):
    """Acceptance: doctor --apply kills a dead-owner turn and a re-run passes."""
    from nvsh.agent.audit import AuditLog

    state = {"active": {"shell": "999", "elapsed": 400.0}}

    def fake_status(*, env, timeout):
        return {"active_turn": state["active"], "target": None}

    def fake_kill_active(*, confirmed, env):
        state["active"] = None
        return "killed"

    def pid_gone(shell):
        return True

    before = doctor_checks.check_agent_turn_not_hung(
        state["active"], threshold=300.0, pid_gone=pid_gone
    )
    assert before["passed"] is False

    audit = AuditLog(path=tmp_path / "audit.jsonl")
    result = doctor_mod._apply_agent_turn_not_hung(
        env={},
        status=fake_status,
        kill_active=fake_kill_active,
        pid_gone=pid_gone,
        confirm=lambda *_: False,
        audit=audit,
    )
    assert result["outcome"] == "killed"

    after = doctor_checks.check_agent_turn_not_hung(
        fake_status(env={}, timeout=1.0)["active_turn"], threshold=300.0, pid_gone=pid_gone
    )
    assert after["passed"] is True


def test_cmd_doctor_apply_fixes_only_a_failing_agent_turn_not_hung_check(
    monkeypatch, tmp_path, capsys
):
    import argparse
    import json

    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setattr(
        doctor_mod,
        "doctor_checks",
        _StubChecks(
            [
                {
                    "id": "agent_turn_not_hung",
                    "passed": False,
                    "severity": "warning",
                    "message": "hung",
                    "remediation": "nvsh doctor --apply",
                }
            ]
        ),
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    calls = []

    def fake_apply(*, env):
        calls.append(env)
        return {"outcome": "killed", "message": "shell 1's turn: killed"}

    monkeypatch.setattr(doctor_mod, "_apply_agent_turn_not_hung", fake_apply)

    args = argparse.Namespace(json=True, prompt_command=None, bind_p=None, keymap=None, apply=True)
    doctor_mod.cmd_doctor(args)
    payload = json.loads(capsys.readouterr().out)

    assert len(calls) == 1
    assert payload["apply"]["outcome"] == "killed"


def test_cmd_doctor_without_apply_never_calls_the_fix(monkeypatch, tmp_path, capsys):
    import argparse

    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setattr(
        doctor_mod,
        "doctor_checks",
        _StubChecks(
            [
                {
                    "id": "agent_turn_not_hung",
                    "passed": False,
                    "severity": "warning",
                    "message": "hung",
                    "remediation": "nvsh doctor --apply",
                }
            ]
        ),
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def boom(**_kwargs):
        raise AssertionError("plain doctor must never call the fix")

    monkeypatch.setattr(doctor_mod, "_apply_agent_turn_not_hung", boom)

    args = argparse.Namespace(
        json=False, prompt_command=None, bind_p=None, keymap=None, apply=False
    )
    doctor_mod.cmd_doctor(args)
    capsys.readouterr()


def test_cmd_doctor_apply_is_a_noop_when_the_check_already_passes(monkeypatch, tmp_path, capsys):
    import argparse

    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: None)
    monkeypatch.setattr(
        doctor_mod,
        "doctor_checks",
        _StubChecks(
            [
                {
                    "id": "agent_turn_not_hung",
                    "passed": True,
                    "severity": "info",
                    "message": "no active agent turn",
                    "remediation": "",
                }
            ]
        ),
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def boom(**_kwargs):
        raise AssertionError("nothing to fix; the fix must not run")

    monkeypatch.setattr(doctor_mod, "_apply_agent_turn_not_hung", boom)

    args = argparse.Namespace(json=False, prompt_command=None, bind_p=None, keymap=None, apply=True)
    doctor_mod.cmd_doctor(args)
    capsys.readouterr()
