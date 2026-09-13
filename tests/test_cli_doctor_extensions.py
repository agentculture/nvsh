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
        + '"\\C-g": "__nvsh_ctrl_g"\n"\\C-x\\C-n": "__nvsh_enter"\n'
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
