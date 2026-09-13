"""Behavioural tests for the bash hook core (``nvsh/shell/hook.bash``).

Every test drives a real ``bash --norc --noprofile -i`` on a pty (stdlib
:mod:`pty`) and feeds CR-terminated keystrokes, because ``PROMPT_COMMAND``
only runs when an interactive bash draws a prompt. ``NVSH_CAPTURE=0`` keeps
the sessions out of a nested ``script(1)``.

The hook calls a *fake* ``nvsh`` executable placed first on ``PATH``: it
appends its argv to one file and one line per invocation to a counter file,
which is how "a successful command forks no nvsh process" is asserted.
"""

from __future__ import annotations

import os
import pty
import re
import select
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK = REPO_ROOT / "nvsh" / "shell" / "hook.bash"
BASH = shutil.which("bash") or "/bin/bash"
GHOSTTY_BASH = Path("/usr/share/ghostty/shell-integration/bash/ghostty.bash")

#: How much slower a prompt may get with the hook installed, per command.
#: Overridable so a loaded CI box can relax the bound; the real measured
#: median is always printed by the latency test.
LATENCY_TOLERANCE_MS = float(os.environ.get("NVSH_LATENCY_TOLERANCE_MS", "5"))

PROMPT_MARK = re.compile(r"@(\d+\.\d+)@")


# --------------------------------------------------------------------------
# pty session helper
# --------------------------------------------------------------------------


def _run_bash(commands, env, *, timeout=180.0, cwd=None):
    """Feed ``commands`` to an interactive bash on a pty; return its output."""

    master, slave = pty.openpty()
    proc = subprocess.Popen(  # noqa: S603
        [BASH, "--norc", "--noprofile", "-i"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        cwd=str(cwd) if cwd else None,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave)
    payload = "".join(f"{line}\r" for line in [*commands, "exit"]).encode()
    os.set_blocking(master, False)
    out = bytearray()
    sent = 0
    deadline = time.monotonic() + timeout
    try:
        while True:
            if time.monotonic() > deadline:
                proc.kill()
                raise AssertionError(f"bash session timed out; output so far:\n{out.decode()}")
            want_write = [master] if sent < len(payload) else []
            readable, writable, _ = select.select([master], want_write, [], 0.2)
            if writable:
                try:
                    sent += os.write(master, payload[sent : sent + 4096])
                except BlockingIOError:
                    pass
                except OSError:
                    sent = len(payload)
            if readable:
                try:
                    chunk = os.read(master, 65536)
                except BlockingIOError:
                    chunk = b""
                except OSError:
                    break
                if not chunk:
                    break
                out += chunk
            elif proc.poll() is not None and sent >= len(payload):
                break
    finally:
        os.close(master)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()
            proc.wait(timeout=5)
    return out.decode("utf-8", "replace")


def _base_env(tmp_path, **extra):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "TERM": "dumb",
        "LANG": "C.UTF-8",
        "PS1": "@${EPOCHREALTIME}@\\n",
        "NVSH_CAPTURE": "0",
        "HISTFILE": str(tmp_path / ".bash_history"),
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
    }
    env.update({k: v for k, v in extra.items() if v is not None})
    return env


@pytest.fixture()
def fake_nvsh(tmp_path):
    """A fake ``nvsh`` first on PATH that records argv and invocation count."""

    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "argv.log"
    counter = tmp_path / "count.log"
    script = bindir / "nvsh"
    script.write_text(
        "#!/bin/sh\n"
        '{ for a in "$@"; do printf "%s\\t" "$a"; done; printf "\\n"; } '
        '>> "$NVSH_FAKE_ARGV"\n'
        'echo x >> "$NVSH_FAKE_COUNT"\n'
        "exit 0\n"
    )
    script.chmod(0o755)
    counter.write_text("")
    argv_log.write_text("")

    class Fake:
        bin = bindir
        argv_file = argv_log
        count_file = counter

        @property
        def calls(self):
            return [
                line.rstrip("\t").split("\t")
                for line in argv_log.read_text().splitlines()
                if line.strip()
            ]

        @property
        def count(self):
            return len([ln for ln in counter.read_text().splitlines() if ln.strip()])

        def env(self, tmp, **extra):
            return _base_env(
                tmp,
                PATH=f"{bindir}:{os.environ.get('PATH', '')}",
                NVSH_FAKE_ARGV=str(argv_log),
                NVSH_FAKE_COUNT=str(counter),
                **extra,
            )

    return Fake()


def _source(extra=""):
    return f"source {HOOK}{extra}"


def _tagged(output, tag):
    """Last line carrying ``tag``, sliced from the tag.

    The pty stream interleaves OSC 133 markers with command output, so the
    tag is not necessarily at the start of the line.
    """

    lines = [ln for ln in output.splitlines() if tag in ln]
    assert lines, f"no {tag} line in session output:\n{output}"
    line = lines[-1]
    return line[line.index(tag) :]


def _argv_dict(call):
    """Turn ``['hook', '--exit', '2', ...]`` into a flag -> value mapping."""

    flags = {}
    for idx, token in enumerate(call):
        if token.startswith("--") and idx + 1 < len(call):
            flags[token] = call[idx + 1]
    return flags


# --------------------------------------------------------------------------
# kill switch
# --------------------------------------------------------------------------


def test_nvsh_disable_makes_the_file_a_noop(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path, NVSH_DISABLE="1", NVSH_CAPTURE="1")
    out = _run_bash(
        [
            _source(),
            "declare -p PROMPT_COMMAND 2>&1 | sed 's/^/PC:/'",
            "if declare -F __nvsh_hook >/dev/null 2>&1; then F=y; else F=n; fi",
            'echo "HOOKFN=$F"',
            "echo LOG:[${NVSH_LOG:-}]",
            "echo WRAPPED:[${NVSH_WRAPPED:-}]",
            "false",
        ],
        env,
    )
    assert "HOOKFN=n" in out
    assert "HOOKFN=y" not in out
    assert "LOG:[]" in out
    assert "WRAPPED:[]" in out
    assert "__nvsh_hook" not in _tagged(out, "PC:")
    assert fake_nvsh.count == 0


# --------------------------------------------------------------------------
# PROMPT_COMMAND composition
# --------------------------------------------------------------------------


def test_hook_is_first_element_of_prompt_command(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path)
    out = _run_bash(
        [_source(), "declare -p PROMPT_COMMAND | sed 's/^/PC:/'"],
        env,
    )
    line = _tagged(out, "PC:")
    assert 'declare -a PROMPT_COMMAND=([0]="__nvsh_hook"' in line


def test_existing_string_prompt_command_is_preserved(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path)
    out = _run_bash(
        [
            "PROMPT_COMMAND='echo legacy >/dev/null'",
            _source(),
            "declare -p PROMPT_COMMAND | sed 's/^/PC:/'",
        ],
        env,
    )
    line = _tagged(out, "PC:")
    assert "declare -a PROMPT_COMMAND=" in line
    assert line.index("__nvsh_hook") < line.index("echo legacy")


def test_double_source_installs_the_hook_once(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path)
    out = _run_bash(
        [_source(), _source(), "declare -p PROMPT_COMMAND | sed 's/^/PC:/'"],
        env,
    )
    line = _tagged(out, "PC:")
    assert line.count("__nvsh_hook") == 1


def test_unload_removes_only_the_nvsh_element(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path)
    out = _run_bash(
        [
            "PROMPT_COMMAND='echo legacy >/dev/null'",
            _source(),
            "declare -p PROMPT_COMMAND | sed 's/^/BEFORE:/'",
            "__nvsh_hook_unload",
            "declare -p PROMPT_COMMAND | sed 's/^/PC:/'",
            "false",
        ],
        env,
    )
    assert "__nvsh_hook" in _tagged(out, "BEFORE:")
    line = _tagged(out, "PC:")
    assert "__nvsh_hook" not in line
    assert "echo legacy" in line
    assert fake_nvsh.count == 0


@pytest.mark.skipif(not GHOSTTY_BASH.exists(), reason="ghostty shell integration not installed")
def test_composes_with_ghostty_integration(tmp_path, fake_nvsh):
    env = fake_nvsh.env(
        tmp_path,
        TERM_PROGRAM="ghostty",
        GHOSTTY_RESOURCES_DIR="/usr/share/ghostty",
        TERM="xterm-256color",
    )
    out = _run_bash(
        [
            f"source {GHOSTTY_BASH}",
            _source(),
            "declare -p PROMPT_COMMAND | tr -d '\\n' | sed 's/^/PC:/'",
            "echo",
            "trap -p DEBUG | tr -d '\\n' | sed 's/^/DBG:/'",
            "echo",
        ],
        env,
    )
    line = _tagged(out, "PC:")
    assert line.count("__ghostty_hook") == 1
    assert line.index("__nvsh_hook") < line.index("__ghostty_hook")
    debug = "\n".join(ln for ln in out.splitlines() if "DBG:" in ln)
    assert "__nvsh" not in debug


def test_no_debug_trap_is_installed(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path)
    out = _run_bash(
        [_source(), "trap -p DEBUG | sed 's/^/DBG:/'", "echo DBGEND"],
        env,
    )
    debug = "\n".join(ln for ln in out.splitlines() if "DBG:" in ln)
    assert "__nvsh" not in debug


@pytest.mark.skipif(not GHOSTTY_BASH.exists(), reason="ghostty shell integration not installed")
def test_hook_sources_ghostty_itself_when_term_program_is_ghostty(tmp_path, fake_nvsh):
    env = fake_nvsh.env(
        tmp_path,
        TERM_PROGRAM="ghostty",
        GHOSTTY_RESOURCES_DIR="/usr/share/ghostty",
        TERM="xterm-256color",
    )
    out = _run_bash(
        [
            _source(),
            "if declare -F __ghostty_hook >/dev/null; then G=y; else G=n; fi",
            'echo "GHOSTTY=$G"',
        ],
        env,
    )
    assert "GHOSTTY=y" in out


def test_emits_own_osc133_markers_without_ghostty(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path)
    out = _run_bash([_source(), "echo hello-osc"], env)
    assert "133;C" in out
    assert "133;D" in out


# --------------------------------------------------------------------------
# status / PIPESTATUS capture
# --------------------------------------------------------------------------


def _last_debug_record(tmp_path, fake_nvsh, commands, **extra):
    debug = tmp_path / "hook-debug.log"
    env = fake_nvsh.env(tmp_path, NVSH_HOOK_DEBUG_FILE=str(debug), **extra)
    _run_bash([_source(), *commands], env)
    lines = [ln for ln in debug.read_text().splitlines() if ln.strip()]
    assert lines, "hook wrote no debug records"
    status, _, pipestatus = lines[-1].partition("\t")
    return status, pipestatus


def test_pipestatus_false_pipe_true(tmp_path, fake_nvsh):
    status, pipestatus = _last_debug_record(tmp_path, fake_nvsh, ["false | true"])
    assert pipestatus == "1 0"
    assert status == "0"


def test_pipestatus_three_stage_pipeline(tmp_path, fake_nvsh):
    status, pipestatus = _last_debug_record(tmp_path, fake_nvsh, ["true | false | true"])
    assert pipestatus == "0 1 0"
    assert status == "0"


def test_pipefail_makes_the_recorded_status_one(tmp_path, fake_nvsh):
    status, pipestatus = _last_debug_record(
        tmp_path, fake_nvsh, ["set -o pipefail", "false | true"]
    )
    assert status == "1"
    assert pipestatus == "1 0"


# --------------------------------------------------------------------------
# pre-filter and the Python entrypoint contract
# --------------------------------------------------------------------------


def test_failure_calls_the_python_entrypoint_with_the_argv_contract(tmp_path, fake_nvsh):
    workdir = tmp_path / "work"
    workdir.mkdir()
    env = fake_nvsh.env(tmp_path, NVSH_LOG="/tmp/nvsh-test.log")  # noqa: S108
    _run_bash([_source(), "ls /nvsh-no-such-dir"], env, cwd=workdir)
    assert fake_nvsh.count == 1
    call = fake_nvsh.calls[0]
    assert call[0] == "hook"
    flags = _argv_dict(call)
    assert flags["--exit"] == "2"
    assert flags["--pipestatus"] == "2"
    assert flags["--line"] == "ls /nvsh-no-such-dir"
    assert flags["--cwd"] == str(workdir)
    assert flags["--log"] == "/tmp/nvsh-test.log"  # noqa: S108


@pytest.mark.parametrize("command", ["true", "(exit 130)", "(exit 141)"])
def test_prefilter_skips_non_error_exit_codes(tmp_path, fake_nvsh, command):
    env = fake_nvsh.env(tmp_path)
    _run_bash([_source(), command], env)
    assert fake_nvsh.count == 0


def test_prefilter_skips_interactive_programs(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path, NVSH_INTERACTIVE_PROGRAMS="vim nosuchprog")
    _run_bash([_source(), "nosuchprog"], env)
    assert fake_nvsh.count == 0


def test_prefilter_does_not_refire_on_a_redrawn_prompt(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path)
    _run_bash([_source(), "ls /nvsh-no-such-dir", "__nvsh_hook", "__nvsh_hook"], env)
    assert fake_nvsh.count == 1


def test_hook_does_not_fire_inside_a_sourced_script(tmp_path, fake_nvsh):
    script = tmp_path / "inner.sh"
    script.write_text("ls /nvsh-no-such-dir\n__nvsh_hook\n")
    env = fake_nvsh.env(tmp_path)
    _run_bash([_source(), f"source {script}"], env)
    # The failing command inside the sourced script must not reach nvsh,
    # and the prompt that follows the `source` sees the same HISTCMD line.
    assert fake_nvsh.count == 0


def test_thousand_successful_commands_fork_no_nvsh_process(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path)
    _run_bash([_source(), *(["true"] * 1000)], env, timeout=300.0)
    assert fake_nvsh.count == 0


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------


def _prompt_intervals(output):
    stamps = [float(m) for m in PROMPT_MARK.findall(output)]
    return [b - a for a, b in zip(stamps, stamps[1:]) if 0 <= b - a < 1.0]


def test_prompt_latency_overhead_is_small(tmp_path, fake_nvsh, capsys):
    count = 400
    env = fake_nvsh.env(tmp_path)
    baseline_out = _run_bash([":", *(["true"] * count)], env, timeout=300.0)
    hooked_out = _run_bash([_source(), *(["true"] * count)], env, timeout=300.0)
    base = _prompt_intervals(baseline_out)
    hooked = _prompt_intervals(hooked_out)
    assert len(base) > count // 2 and len(hooked) > count // 2
    base_median = statistics.median(base) * 1000
    hook_median = statistics.median(hooked) * 1000
    delta = hook_median - base_median
    with capsys.disabled():
        print(
            f"\nprompt latency: baseline median {base_median:.3f} ms, "
            f"hooked median {hook_median:.3f} ms, delta {delta:.3f} ms "
            f"(tolerance {LATENCY_TOLERANCE_MS} ms, n={len(hooked)})"
        )
    assert delta < LATENCY_TOLERANCE_MS


# --------------------------------------------------------------------------
# the interactive-program default list has exactly one source of truth
# --------------------------------------------------------------------------


def test_hook_default_interactive_programs_match_triggers_module():
    from nvsh.triggers import INTERACTIVE_PROGRAMS

    text = HOOK.read_text()
    matches = re.findall(
        r"^__NVSH_INTERACTIVE_PROGRAMS_DEFAULT='([^']*)'$", text, flags=re.MULTILINE
    )
    assert len(matches) == 1, "the default list must live in exactly one place in hook.bash"
    assert set(matches[0].split()) == set(INTERACTIVE_PROGRAMS)


# --------------------------------------------------------------------------
# capture (the bash side of t8's session log)
# --------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("script") is None, reason="util-linux script(1) not installed")
def test_capture_execs_script_and_exports_the_log(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path, NVSH_CAPTURE="1")
    env.pop("TMUX", None)
    out = _run_bash(
        [
            _source(),
            'echo "WRAPPED=${NVSH_WRAPPED:-none}"',
            'echo "LOGDIR=$(dirname "${NVSH_LOG:-/none}")"',
            'stat -c "MODE=%a" "${NVSH_LOG:-/none}"',
        ],
        env,
    )
    assert "WRAPPED=1" in out
    assert f"LOGDIR={tmp_path / 'run' / 'nvsh'}" in out
    assert "MODE=600" in out


@pytest.mark.skipif(shutil.which("script") is None, reason="util-linux script(1) not installed")
def test_capture_is_skipped_when_disabled(tmp_path, fake_nvsh):
    env = fake_nvsh.env(tmp_path, NVSH_CAPTURE="0")
    out = _run_bash([_source(), 'echo "WRAPPED=${NVSH_WRAPPED:-none}"'], env)
    assert "WRAPPED=none" in out
