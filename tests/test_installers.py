"""Tests for nvsh.installers — deviation d1: 'nvsh setup' detects and offers to
install missing helper tools (uv, tmux, node/npm, pi).

Acceptance criteria covered:
- TOOLS covers pi, node (pi's prerequisite), uv, tmux.
- missing_tools() reports only tools whose binary is absent from PATH.
- plan_installs() computes the exact planned command for each fleet machine
  (orin: apt-get+snap present, nothing else; thor: node/npm/tmux present, no
  pi; spark: everything present), including uv's snap-vs-curl branching.
- The curl-pipe-sh uv installer is only ever printed, never executed.
- run_install() never calls ``run`` unless confirm() returns True (or the
  step is executable and confirmed), never uses shell=True, keeps sudo
  commands attached to the terminal (no output capture), and records every
  attempt (run or declined) to the audit log.
"""

from __future__ import annotations

import inspect
import subprocess

import pytest

from nvsh import installers
from nvsh.agent.audit import AuditLog


def _which_factory(present: set[str]):
    def _which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return _which


ORIN_WHICH = _which_factory({"apt-get", "snap"})
THOR_WHICH = _which_factory({"apt-get", "snap", "curl", "node", "npm", "tmux", "uv"})
SPARK_WHICH = _which_factory({"apt-get", "snap", "curl", "node", "npm", "tmux", "pi", "uv"})


# --------------------------------------------------------------------------
# registry shape
# --------------------------------------------------------------------------


def test_tools_registry_has_expected_names():
    assert {tool.name for tool in installers.TOOLS} == {"pi", "node", "uv", "tmux"}


def test_tool_spec_has_purpose_and_binary():
    for tool in installers.TOOLS:
        assert isinstance(tool.purpose, str)
        assert tool.purpose
        assert isinstance(tool.binary, str)
        assert tool.binary
        assert callable(tool.install_commands)


# --------------------------------------------------------------------------
# missing_tools
# --------------------------------------------------------------------------


def test_missing_tools_orin_reports_all_four():
    missing = installers.missing_tools(which=ORIN_WHICH)
    assert {tool.name for tool in missing} == {"pi", "node", "uv", "tmux"}


def test_missing_tools_thor_reports_only_pi():
    missing = installers.missing_tools(which=THOR_WHICH)
    assert {tool.name for tool in missing} == {"pi"}


def test_missing_tools_spark_reports_nothing():
    missing = installers.missing_tools(which=SPARK_WHICH)
    assert missing == []


# --------------------------------------------------------------------------
# plan_installs — exact commands per machine
# --------------------------------------------------------------------------


def test_plan_installs_orin_exact_commands():
    missing = installers.missing_tools(which=ORIN_WHICH)
    plan = installers.plan_installs(missing, which=ORIN_WHICH)
    by_tool = {step.tool: step for step in plan}

    node_step = by_tool["node"]
    assert node_step.argv == ("sudo", "apt-get", "install", "-y", "nodejs", "npm")
    assert node_step.needs_sudo is True
    assert node_step.executable is True

    tmux_step = by_tool["tmux"]
    assert tmux_step.argv == ("sudo", "apt-get", "install", "-y", "tmux")
    assert tmux_step.needs_sudo is True
    assert tmux_step.executable is True

    uv_step = by_tool["uv"]
    assert uv_step.argv == ("sudo", "snap", "install", "astral-uv", "--classic")
    assert uv_step.needs_sudo is True
    assert uv_step.executable is True

    # pi needs npm, and orin has no npm -- no runnable command, node must
    # come first.
    pi_step = by_tool["pi"]
    assert pi_step.argv is None
    assert pi_step.executable is False


def test_plan_installs_thor_pi_only():
    missing = installers.missing_tools(which=THOR_WHICH)
    plan = installers.plan_installs(missing, which=THOR_WHICH)
    assert len(plan) == 1
    pi_step = plan[0]
    assert pi_step.tool == "pi"
    assert pi_step.argv == ("npm", "install", "-g", "@earendil-works/pi-coding-agent")
    assert pi_step.needs_sudo is False
    assert pi_step.executable is True


def test_plan_installs_spark_is_empty():
    missing = installers.missing_tools(which=SPARK_WHICH)
    plan = installers.plan_installs(missing, which=SPARK_WHICH)
    assert plan == []


def test_uv_falls_back_to_curl_print_only_when_no_snap():
    which = _which_factory({"apt-get", "curl"})
    missing = installers.missing_tools(which=which)
    plan = installers.plan_installs(missing, which=which)
    uv_step = next(step for step in plan if step.tool == "uv")
    assert uv_step.argv is None
    assert uv_step.executable is False
    assert uv_step.shell_line == "curl -LsSf https://astral.sh/uv/install.sh | sh"


def test_uv_has_no_plan_when_neither_snap_nor_curl_present():
    which = _which_factory({"apt-get"})
    missing = installers.missing_tools(which=which)
    plan = installers.plan_installs(missing, which=which)
    uv_step = next(step for step in plan if step.tool == "uv")
    assert uv_step.argv is None
    assert uv_step.executable is False
    assert "no known installer" in uv_step.shell_line


def test_node_and_tmux_have_no_plan_without_apt_get():
    which = _which_factory(set())
    missing = installers.missing_tools(which=which)
    plan = installers.plan_installs(missing, which=which)
    node_step = next(step for step in plan if step.tool == "node")
    tmux_step = next(step for step in plan if step.tool == "tmux")
    assert node_step.argv is None
    assert node_step.executable is False
    assert tmux_step.argv is None
    assert tmux_step.executable is False


# --------------------------------------------------------------------------
# run_install
# --------------------------------------------------------------------------


def _npm_step():
    return installers.InstallStep(
        tool="pi",
        argv=("npm", "install", "-g", "@earendil-works/pi-coding-agent"),
        shell_line="npm install -g @earendil-works/pi-coding-agent",
        needs_sudo=False,
        executable=True,
    )


def _sudo_step():
    return installers.InstallStep(
        tool="tmux",
        argv=("sudo", "apt-get", "install", "-y", "tmux"),
        shell_line="sudo apt-get install -y tmux",
        needs_sudo=True,
        executable=True,
    )


def _curl_step():
    return installers.InstallStep(
        tool="uv",
        argv=None,
        shell_line="curl -LsSf https://astral.sh/uv/install.sh | sh",
        needs_sudo=False,
        executable=False,
    )


def _fake_run(returncode=0):
    calls = []

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(args=argv, returncode=returncode, stdout="ok", stderr="")

    run.calls = calls
    return run


def test_run_install_never_calls_run_when_confirm_declines(tmp_path):
    run = _fake_run()
    result = installers.run_install(
        _npm_step(),
        run=run,
        confirm=lambda _msg: False,
        audit=AuditLog(path=tmp_path / "audit.jsonl"),
    )
    assert run.calls == []
    assert result.ran is False
    assert result.returncode is None


def test_run_install_calls_run_with_list_argv_when_confirmed(tmp_path):
    run = _fake_run()
    result = installers.run_install(
        _npm_step(),
        run=run,
        confirm=lambda _msg: True,
        audit=AuditLog(path=tmp_path / "audit.jsonl"),
    )
    assert len(run.calls) == 1
    argv, kwargs = run.calls[0]
    assert argv == ["npm", "install", "-g", "@earendil-works/pi-coding-agent"]
    assert isinstance(argv, list)
    assert kwargs.get("shell") is not True
    assert result.ran is True
    assert result.returncode == 0


def test_run_install_sudo_step_does_not_capture_output(tmp_path):
    run = _fake_run()
    installers.run_install(
        _sudo_step(),
        run=run,
        confirm=lambda _msg: True,
        audit=AuditLog(path=tmp_path / "audit.jsonl"),
    )
    _argv, kwargs = run.calls[0]
    assert kwargs.get("capture_output") is not True
    assert "stdout" not in kwargs


def test_run_install_never_executes_print_only_step_even_with_yes(tmp_path):
    run = _fake_run()
    result = installers.run_install(
        _curl_step(),
        run=run,
        confirm=lambda _msg: True,
        audit=AuditLog(path=tmp_path / "audit.jsonl"),
    )
    assert run.calls == []
    assert result.ran is False


def test_run_install_never_uses_shell_true_in_source():
    source = inspect.getsource(installers.run_install)
    assert "shell=True" not in source


def test_run_install_records_audit_entry_on_confirmed_run(tmp_path):
    run = _fake_run()
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    installers.run_install(_npm_step(), run=run, confirm=lambda _msg: True, audit=audit)
    entries = audit.read_all()
    assert len(entries) == 1
    assert entries[0]["event"] == "install"
    assert entries[0]["decision"] is True


def test_run_install_records_audit_entry_on_decline(tmp_path):
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    installers.run_install(_npm_step(), run=_fake_run(), confirm=lambda _msg: False, audit=audit)
    entries = audit.read_all()
    assert len(entries) == 1
    assert entries[0]["decision"] is False


@pytest.mark.parametrize("returncode", [0, 1])
def test_run_install_result_returncode_matches(tmp_path, returncode):
    run = _fake_run(returncode=returncode)
    result = installers.run_install(
        _npm_step(),
        run=run,
        confirm=lambda _msg: True,
        audit=AuditLog(path=tmp_path / "audit.jsonl"),
    )
    assert result.returncode == returncode


# --- PR #8 review: the pi step is recomputed after node/npm land ------------


def test_pi_step_is_recomputed_after_node_is_installed(monkeypatch):
    """A fresh machine (no node, no npm, no pi) must still get pi installed.

    The whole plan used to be computed up front, so pi was permanently
    marked non-executable ("npm not found") even though the node step
    installed npm moments later.
    """
    from nvsh.cli._commands import setup as setup_mod

    present: set[str] = set()

    def which(name):
        return f"/usr/bin/{name}" if name in present else None

    ran: list[str] = []

    def fake_run_install(step, **kwargs):
        if not step.executable:
            return installers.InstallResult(tool=step.tool, ran=False, returncode=None)
        ran.append(step.tool)
        if step.tool == "node":
            present.update({"node", "npm"})  # apt-get installed nodejs + npm
        return installers.InstallResult(tool=step.tool, ran=True, returncode=0)

    monkeypatch.setattr(setup_mod.installers, "run_install", fake_run_install)
    present.add("apt-get")
    missing = installers.missing_tools(which=which)
    assert [t.name for t in missing] == ["node", "pi", "uv", "tmux"]

    rows, any_ran = setup_mod._process_installs(
        missing, offer_only=False, confirm=lambda _m: True, which=which
    )
    assert any_ran is True
    pi_row = next(row for row in rows if row["tool"] == "pi")
    assert pi_row["executable"] is True, pi_row
    assert pi_row["ran"] is True
    assert "npm" in pi_row["command"]
    assert "pi" in ran


def test_offer_only_still_reports_the_unmet_dependency(monkeypatch):
    """With nothing installed, the offer text still says npm is missing."""
    from nvsh.cli._commands import setup as setup_mod

    def which(name):
        return "/usr/bin/apt-get" if name == "apt-get" else None

    missing = installers.missing_tools(which=which)
    rows, any_ran = setup_mod._process_installs(missing, offer_only=True, confirm=None, which=which)
    assert any_ran is False
    pi_row = next(row for row in rows if row["tool"] == "pi")
    assert pi_row["executable"] is False
    assert "npm not found" in pi_row["command"]


# --------------------------------------------------------------------------
# missing_tools(chosen=...) -- offers scoped to the harness pick (t3)
# --------------------------------------------------------------------------


def test_missing_tools_default_chosen_is_unscoped_and_backward_compatible():
    """No ``chosen`` (the default) behaves exactly like before this param."""
    missing = installers.missing_tools(which=ORIN_WHICH)
    assert {tool.name for tool in missing} == {"pi", "node", "uv", "tmux"}
    # Explicitly passing chosen=None must match the bare default.
    assert installers.missing_tools(which=ORIN_WHICH, chosen=None) == missing


def test_missing_tools_chosen_claude_never_offers_pi_or_node_when_claude_on_path():
    which = _which_factory({"claude"})
    missing = installers.missing_tools(which=which, chosen="claude")
    names = {tool.name for tool in missing}
    assert "pi" not in names
    assert "node" not in names


def test_missing_tools_chosen_claude_offers_node_when_claude_and_npm_both_missing():
    which = _which_factory(set())
    missing = installers.missing_tools(which=which, chosen="claude")
    names = {tool.name for tool in missing}
    assert "node" in names
    # pi is still never offered for a non-pi pick.
    assert "pi" not in names


def test_missing_tools_chosen_claude_never_offers_node_when_npm_present():
    which = _which_factory({"npm"})
    missing = installers.missing_tools(which=which, chosen="claude")
    assert "node" not in {tool.name for tool in missing}


def test_missing_tools_chosen_agy_never_offers_node_since_agy_needs_no_node():
    which = _which_factory(set())
    missing = installers.missing_tools(which=which, chosen="agy")
    assert "node" not in {tool.name for tool in missing}
    assert "pi" not in {tool.name for tool in missing}


def test_missing_tools_chosen_pi_keeps_todays_node_and_pi_behaviour():
    missing = installers.missing_tools(which=ORIN_WHICH, chosen="pi")
    assert {tool.name for tool in missing} == {"pi", "node", "uv", "tmux"}

    missing_thor = installers.missing_tools(which=THOR_WHICH, chosen="pi")
    assert {tool.name for tool in missing_thor} == {"pi"}


def test_missing_tools_uv_and_tmux_offered_regardless_of_chosen():
    which = _which_factory(set())
    for chosen in (None, "claude", "pi", "agy", "codex", "qwen", "kiro", "openai-compat"):
        missing = installers.missing_tools(which=which, chosen=chosen)
        names = {tool.name for tool in missing}
        assert "uv" in names
        assert "tmux" in names


# --------------------------------------------------------------------------
# harness_install_step -- per-harness install specs (t3)
# --------------------------------------------------------------------------


def test_harness_install_step_claude_is_executable_npm_step():
    which = _which_factory({"npm"})
    step = installers.harness_install_step("claude", which=which)
    assert step.argv == ("npm", "install", "-g", "@anthropic-ai/claude-code")
    assert step.executable is True
    assert step.needs_sudo is False


def test_harness_install_step_codex_is_executable_npm_step():
    which = _which_factory({"npm"})
    step = installers.harness_install_step("codex", which=which)
    assert step.argv == ("npm", "install", "-g", "@openai/codex")
    assert step.executable is True


def test_harness_install_step_qwen_is_executable_npm_step():
    which = _which_factory({"npm"})
    step = installers.harness_install_step("qwen", which=which)
    assert step.argv == ("npm", "install", "-g", "@qwen-code/qwen-code")
    assert step.executable is True


def test_harness_install_step_pi_reuses_existing_pi_command():
    which = _which_factory({"npm"})
    step = installers.harness_install_step("pi", which=which)
    assert step.argv == ("npm", "install", "-g", "@earendil-works/pi-coding-agent")
    assert step.executable is True


def test_harness_install_step_pi_without_npm_is_not_executable():
    which = _which_factory(set())
    step = installers.harness_install_step("pi", which=which)
    assert step.argv is None
    assert step.executable is False
    assert "npm not found" in step.shell_line


def test_harness_install_step_npm_harness_without_npm_is_not_executable():
    which = _which_factory(set())
    step = installers.harness_install_step("claude", which=which)
    assert step.argv is None
    assert step.executable is False
    assert "npm not found" in step.shell_line


@pytest.mark.parametrize("name", ["agy", "kiro", "openai-compat"])
def test_harness_install_step_no_known_installer_harnesses(name):
    which = _which_factory({"npm"})
    step = installers.harness_install_step(name, which=which)
    assert step.argv is None
    assert step.executable is False
    assert "no known installer" in step.shell_line


def test_harness_install_step_never_executable_without_argv():
    """A non-executable step must never carry an argv a caller could run."""
    which = _which_factory({"npm"})
    for name in ["agy", "kiro", "openai-compat"]:
        step = installers.harness_install_step(name, which=which)
        assert step.executable is False
        assert step.argv is None
