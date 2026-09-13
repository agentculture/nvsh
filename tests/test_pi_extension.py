"""Tests for the pi approval extension (task t11).

Acceptance criteria covered:

* Node-free static checks on ``nvsh/agent/pi_ext/approval.ts`` -- it is a
  pure forwarder to ``nvsh approve check/add`` and never re-implements
  policy (t5 owns that).
* the ``.ts`` file ships inside the wheel (pyproject.toml hatch include).
* ``default_approval_extension_path()`` is absolute and points at a file
  that exists on disk once this task lands.
* a Python-level walk through the same three scenarios the extension
  forwards to ("nvidia-smi -L" -> user, "apt install foo" -> ask then
  session-approved then ask again for a fresh ``Approvals``, "sudo rm -rf
  /x" -> always ask), recorded through the new ``nvsh approve audit`` verb.
* an optional live test (skipped unless ``NVSH_LIVE_PI=1`` and a real ``pi``
  binary is on PATH) that loads the extension and confirms pi does not
  report a load error for it.

The live, model-driven flow through the real extension (nvidia-smi runs,
apt install asks/approves-for-session/asks again on a fresh daemon, sudo rm
always asks) is NOT exercised here -- that requires a tool-calling model
turn and is t19's job on real hardware. This file only proves the extension
*source* forwards correctly and that the Python policy it forwards to
behaves as specified.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTENSION_PATH = REPO_ROOT / "nvsh" / "agent" / "pi_ext" / "approval.ts"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _read_extension() -> str:
    return EXTENSION_PATH.read_text(encoding="utf-8")


# -- static, Node-free checks on the .ts source -----------------------------


def test_extension_file_exists():
    assert EXTENSION_PATH.is_file()


def test_extension_under_120_lines():
    lines = _read_extension().splitlines()
    assert len(lines) < 120, f"approval.ts has {len(lines)} lines, must be under 120"


def test_extension_registers_tool_call_handler():
    text = _read_extension()
    assert 'pi.on("tool_call"' in text


def test_extension_forwards_to_nvsh_approve_check():
    text = _read_extension()
    assert "approve" in text and "check" in text
    # argv built as an array, not a shell string.
    assert "spawnSync" in text
    assert '"approve"' in text
    assert '"check"' in text


def test_extension_offers_the_four_choices():
    text = _read_extension()
    for choice in ("once", "session", "user", "deny"):
        assert f'"{choice}"' in text, f"missing choice {choice!r}"


def test_extension_select_lists_the_four_choices_in_order():
    """d15: the panel's keys map onto these, so the order is part of the contract."""
    text = _read_extension()
    start = text.index("await ctx.ui.select(")
    options = text[text.index("[", start) : text.index("]", start) + 1]
    found = [c for c in ("once", "session", "user", "deny") if f'"{c}"' in options]
    assert found == ["once", "session", "user", "deny"], options
    order = [options.index(f'"{c}"') for c in found]
    assert order == sorted(order), options


def test_extension_widens_the_user_scope_to_first_word_star():
    """d15: `user` approves the command *class*, not just this exact line."""
    text = _read_extension()
    assert "command.trim().split(" in text
    assert "`${firstWord} *`" in text


def test_extension_adds_the_exact_line_for_the_session_scope():
    """Session scope approves the line the model actually proposed, unwidened."""
    text = _read_extension()
    branch = text[text.index('if (choice === "session")') :]
    branch = branch[: branch.index('audit(command, "session")')]
    assert '"--session"' in branch
    assert "command," in branch
    assert "firstWord" not in branch


def test_extension_blocks_with_reason_shape():
    text = _read_extension()
    assert "block: true" in text
    assert "reason" in text


def test_extension_blocks_non_bash_tools():
    text = _read_extension()
    assert "bash" in text
    assert "nvsh v1 allows only the bash tool" in text


def test_extension_has_no_non_node_imports():
    text = _read_extension()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("import"):
            continue
        assert (
            "node:" in stripped or "ExtensionAPI" in stripped or "@earendil-works" in stripped
        ), f"non-node, non-type import found: {stripped!r}"
        if "from" in stripped:
            module = stripped.split("from", 1)[1].strip().strip(";").strip("'\"")
            assert module.startswith("node:") or module.startswith(
                "@earendil-works/"
            ), f"dependency-free extension must not import npm package {module!r}"


def test_extension_calls_approve_audit():
    text = _read_extension()
    assert "audit" in text


# -- d8: the extension forwards only the raw command argument -------------


def _select_argument(text: str) -> str:
    """Return the source of ``ctx.ui.select(...)``'s first argument."""
    start = text.index("await ctx.ui.select(") + len("await ctx.ui.select(")
    return text[start : text.index(",", start)].strip()


def _assignment_of(text: str, name: str) -> str:
    """Return the source of ``const <name> = ...;``."""
    start = text.index(f"const {name} = ")
    return text[start : text.index(");", start)]


def test_extension_never_embeds_a_human_prompt_in_the_select_payload():
    """d8: the panel text must never ride along as (or inside) the command."""
    text = _read_extension()
    assert "run this command?" not in text
    assert "command: ${" not in text


def test_extension_select_payload_is_a_json_envelope_with_the_raw_command():
    text = _read_extension()
    argument = _select_argument(text)
    # A bare identifier, never an inline template literal of prose.
    assert argument.isidentifier(), f"select payload is not a plain variable: {argument!r}"
    envelope = _assignment_of(text, argument)
    assert "JSON.stringify" in envelope, f"select payload is not a JSON envelope: {envelope!r}"
    assert '"approval"' in envelope
    # The command argument is forwarded by reference, unmodified.
    assert "command," in envelope or "command: command" in envelope
    assert "${" not in envelope, "no interpolated prose may enter the envelope"


def test_extension_checks_and_audits_the_same_raw_command():
    """check/add/audit all receive the bare `command` variable, not a message."""
    text = _read_extension()
    assert 'const command = String((event.input && event.input.command) || "");' in text
    assert "checkCommand(command)" in text
    for call in ("audit(command,", "audit(command,"):
        assert call in text


# -- wheel packaging ----------------------------------------------------


def test_pi_ext_ships_in_the_wheel_glob():
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    wheel = pyproject.split("[tool.hatch.build.targets.wheel]", 1)[1].split("\n[", 1)[0]
    assert "nvsh/agent/pi_ext/*.ts" in wheel


# -- default_approval_extension_path() -----------------------------------


def test_default_approval_extension_path_absolute_and_exists():
    from nvsh.agent.pi import default_approval_extension_path

    path = default_approval_extension_path()
    assert path.is_absolute()
    assert path.is_file()
    assert path.name == "approval.ts"


# -- Python-level policy scenarios (what the extension forwards to) --------


@pytest.fixture()
def policy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runtime = tmp_path / "run"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    return tmp_path


def _run(args):
    from nvsh.cli import main

    return main(args)


def test_scenario_nvidia_smi_is_user_approved(policy_env, capsys):
    rc = _run(["approve", "check", "nvidia-smi -L", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "user"


def test_scenario_apt_install_asks_then_session_then_a_fresh_runtime_dir_asks_again(
    policy_env, capsys, monkeypatch
):
    rc = _run(["approve", "check", "apt install foo", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "ask"

    rc = _run(["approve", "add", "apt install foo", "--session", "--json"])
    assert rc == 0
    capsys.readouterr()

    # d15: `nvsh approve add --session` runs in a throwaway subprocess, so
    # the pattern is written to the login session's runtime dir. A later
    # `nvsh approve check` -- another process entirely -- must see it.
    rc = _run(["approve", "check", "apt install foo", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "session"

    from nvsh.approvals import Approvals

    assert Approvals.load().decide("apt install foo") == "session"

    # ... and it dies with the runtime dir (logout, or a wiped /run/user).
    fresh_runtime = policy_env / "run2"
    fresh_runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(fresh_runtime))
    assert Approvals.load().decide("apt install foo") == "ask"


def test_scenario_sudo_rm_always_asks_even_after_user_attempt(policy_env, capsys):
    rc = _run(["approve", "check", "sudo rm -rf /x", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "ask"

    rc = _run(["approve", "add", "sudo *", "--json"])
    assert rc == 1
    err = json.loads(capsys.readouterr().err)
    assert "sudo" in err["message"]

    rc = _run(["approve", "check", "sudo rm -rf /x", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "ask"


def test_scenario_all_three_recorded_via_approve_audit(policy_env, capsys):
    from nvsh.agent.audit import AuditLog

    for cmd, decision in (
        ("nvidia-smi -L", "user"),
        ("apt install foo", "ask"),
        ("sudo rm -rf /x", "ask"),
    ):
        rc = _run(
            [
                "approve",
                "audit",
                "--tool",
                "bash",
                "--command",
                cmd,
                "--decision",
                decision,
                "--json",
            ]
        )
        assert rc == 0
        capsys.readouterr()

    entries = AuditLog().read_all()
    commands = [e["proposal"]["command"] for e in entries]
    assert "nvidia-smi -L" in commands
    assert "apt install foo" in commands
    assert "sudo rm -rf /x" in commands
    decisions = {e["proposal"]["command"]: e["decision"] for e in entries}
    assert decisions["nvidia-smi -L"] == "user"
    assert decisions["apt install foo"] == "ask"
    assert decisions["sudo rm -rf /x"] == "ask"


# -- optional live smoke test ---------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("NVSH_LIVE_PI") == "1"),
    reason="set NVSH_LIVE_PI=1 and have the real pi binary on PATH to run this",
)
def test_live_pi_loads_extension_without_error(tmp_path):
    pi_path = shutil.which("pi")
    if pi_path is None:
        pytest.skip("no real pi binary on PATH")

    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)

    nvsh_bin = shutil.which("nvsh") or sys.executable
    env.setdefault("NVSH_BIN", nvsh_bin if nvsh_bin != sys.executable else "nvsh")

    proc = subprocess.Popen(
        [pi_path, "--mode", "rpc", "--no-session", "-e", str(EXTENSION_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        proc.stdin.write((json.dumps({"type": "get_state", "id": "probe"}) + "\n").encode())
        proc.stdin.flush()
        line = proc.stdout.readline()
        assert line, "no output from real pi rpc process"
        obj = json.loads(line)
        assert obj.get("type") == "response"
    finally:
        proc.terminate()
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
        assert b"approval.ts" not in stderr or b"error" not in stderr.lower()
