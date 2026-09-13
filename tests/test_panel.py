"""Tests for nvsh.panel (task t13): styling, streaming, keys, Ctrl+C.

Covers the acceptance criteria for the panel: hard-coded SGR only (never
tput, never curses), readable output under TERM=dumb / NO_COLOR / non-tty /
TERM=xterm-ghostty-without-terminfo, the four keys (Enter/e/d/Esc), a
proposal shown as the exact command, and a Ctrl+C that cancels, restores
the terminal and returns within a second.
"""

from __future__ import annotations

import io
import os
import pty
import signal
import subprocess  # nosec B404 - fixed argv, no shell=True
import sys
import threading
import time
from pathlib import Path

import pytest

from nvsh import panel as panel_mod
from nvsh.agent.base import AgentEvent, EventKind, Proposal, ProposalKind

REPO_ROOT = Path(__file__).resolve().parents[1]


def _panel(out=None, in_=None, env=None, isatty=False):
    return panel_mod.Panel(
        out=out if out is not None else io.StringIO(),
        in_=in_,
        env=env if env is not None else {},
        isatty=isatty,
    )


# --- style ---------------------------------------------------------------


def test_style_enabled_only_on_a_tty_without_no_color():
    assert panel_mod.style({"TERM": "xterm-256color"}, True).enabled is True
    assert panel_mod.style({"TERM": "xterm-256color"}, False).enabled is False
    assert panel_mod.style({"TERM": "dumb"}, True).enabled is False
    assert panel_mod.style({"TERM": "xterm", "NO_COLOR": "1"}, True).enabled is False


def test_style_is_hard_coded_sgr_and_empty_when_disabled():
    on = panel_mod.style({"TERM": "xterm"}, True)
    off = panel_mod.style({"TERM": "dumb"}, True)
    assert on.bold.startswith("\x1b[") and on.reset == "\x1b[0m"
    assert off.bold == "" and off.reset == "" and off.red == ""


def test_panel_module_never_imports_curses_or_calls_tput():
    source = Path(panel_mod.__file__).read_text(encoding="utf-8")
    assert "curses" not in source
    assert "terminfo" not in source
    # Nothing is shelled out to (no tput, no infocmp): no subprocess at all.
    assert "subprocess" not in source


# --- streaming -----------------------------------------------------------


def test_first_text_delta_is_on_screen_before_the_second_arrives():
    out = io.StringIO()
    p = _panel(out=out)
    seen: list[str] = []

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="first")
        seen.append(out.getvalue())
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" second")
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(events())
    assert seen and "first" in seen[0]
    assert "second" not in seen[0]
    assert result.text == "first second"
    assert result.done is True
    assert result.interrupted is False


def test_header_names_the_command_and_exit_code():
    out = io.StringIO()
    p = _panel(out=out)
    p.header("ls /nope", 2)
    assert "nvsh: ls /nope failed (exit 2)" in out.getvalue()


def test_error_event_is_rendered_and_reported():
    out = io.StringIO()
    p = _panel(out=out)
    result = p.stream(iter([AgentEvent(kind=EventKind.ERROR, error="no agent available")]))
    assert "no agent available" in out.getvalue()
    assert result.error == "no agent available"


def test_status_events_render_without_a_spinner():
    out = io.StringIO()
    p = _panel(out=out)
    p.stream(
        iter([AgentEvent(kind=EventKind.STATUS, text="one-shot pi"), AgentEvent(EventKind.DONE)])
    )
    text = out.getvalue()
    assert "one-shot pi" in text
    for spinner in ("|/-\\", "\r|", "⠋"):
        assert spinner not in text


def test_empty_status_events_render_nothing():
    """d11: a backend may emit a STATUS with no text; the panel stays quiet."""
    out = io.StringIO()
    p = _panel(out=out)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.STATUS, text=""),
                AgentEvent(kind=EventKind.TEXT_DELTA, text="hello"),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    assert out.getvalue() == "hello\n"


def test_tool_call_says_which_tool_is_running():
    out = io.StringIO()
    p = _panel(out=out)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.TOOL_CALL, tool="bash", args={"command": "nvidia-smi"}),
                AgentEvent(kind=EventKind.TOOL_RESULT, tool="bash", result={"output": "ok"}),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    text = out.getvalue()
    assert "... running tool: bash" in text
    assert "... tool bash finished" in text


def test_proposal_events_are_collected_and_handed_to_on_proposal():
    p = _panel()
    proposal = Proposal(command="df -h", rationale="check disk", kind=ProposalKind.INSPECT)
    seen = []
    p.stream(
        iter([AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal), AgentEvent(EventKind.DONE)]),
        on_proposal=lambda prop, event: seen.append(prop),
    )
    assert seen == [proposal]


# --- terminal matrix -----------------------------------------------------


@pytest.mark.parametrize(
    "env,isatty",
    [
        ({"TERM": "dumb"}, True),
        ({"TERM": "xterm", "NO_COLOR": "1"}, True),
        ({"TERM": "xterm-256color"}, False),
        ({"TERM": "xterm-ghostty", "TERMINFO": "/nonexistent-empty-dir"}, False),
    ],
)
def test_terminal_matrix_renders_readable_text_with_no_escapes(env, isatty):
    out = io.StringIO()
    p = _panel(out=out, env=env, isatty=isatty)
    p.header("ls /nope", 2)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.TEXT_DELTA, text="the path does not exist"),
                AgentEvent(kind=EventKind.DONE),
            ]
        )
    )
    p.render_proposal(
        Proposal(command="mkdir -p /nope", rationale="create it", kind=ProposalKind.FIX)
    )
    text = out.getvalue()
    assert "\x1b[" not in text
    assert "ls /nope" in text
    assert "the path does not exist" in text
    assert "mkdir -p /nope" in text


def test_ghostty_without_terminfo_still_runs_under_a_real_subprocess(tmp_path):
    env = dict(os.environ)
    env.update(
        {
            "TERM": "xterm-ghostty",
            "TERMINFO": str(tmp_path / "empty"),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    (tmp_path / "empty").mkdir()
    script = (
        "import sys;"
        "from nvsh.panel import Panel;"
        "from nvsh.agent.base import AgentEvent, EventKind;"
        "p = Panel(out=sys.stdout);"
        "p.header('ls /nope', 2);"
        "p.stream(iter([AgentEvent(kind=EventKind.TEXT_DELTA, text='ok'),"
        " AgentEvent(kind=EventKind.DONE)]))"
    )
    proc = subprocess.run(  # nosec B603 - fixed argv
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
    assert proc.stderr == ""


# --- proposal rendering and keys ----------------------------------------


def test_proposal_is_rendered_verbatim_inside_an_ascii_box_when_style_is_off():
    out = io.StringIO()
    p = _panel(out=out)
    p.render_proposal(
        Proposal(command="sudo rm -rf /var/tmp/x", rationale="why", kind=ProposalKind.FIX)
    )
    text = out.getvalue()
    assert "sudo rm -rf /var/tmp/x" in text
    assert "\x1b[" not in text
    assert all(ord(ch) < 128 for ch in text)


def test_show_proposal_non_tty_reads_a_line():
    proposal = Proposal(command="df -h", rationale="disk", kind=ProposalKind.INSPECT)
    cases = {"\n": "approve", "e\n": "explain", "d\n": "details", "q\n": "ignore", "": "ignore"}
    for typed, expected in cases.items():
        p = _panel(out=io.StringIO(), in_=io.StringIO(typed), isatty=False)
        assert p.show_proposal(proposal) == expected


@pytest.mark.parametrize(
    "key,expected",
    [
        (b"\r", "approve"),
        (b"\n", "approve"),
        (b"e", "explain"),
        (b"d", "details"),
        (b"\x1b", "ignore"),
        (b"\x03", "ignore"),
    ],
)
def test_show_proposal_reads_single_keys_on_a_tty(key, expected):
    """One raw keypress on a real pty decides the proposal.

    The key is typed *after* the panel has entered raw mode, because the
    panel deliberately flushes pending input first (``tty.setraw`` uses
    ``TCSAFLUSH``): a keystroke queued before the proposal appeared must
    never be able to approve a command the operator has not seen.
    """
    master, slave = pty.openpty()
    typist = threading.Timer(0.2, lambda: os.write(master, key))
    typist.start()
    try:
        with os.fdopen(slave, "rb", buffering=0) as tty_in:
            p = panel_mod.Panel(out=io.StringIO(), in_=tty_in, env={}, isatty=True)
            proposal = Proposal("df -h", "disk", ProposalKind.INSPECT)
            assert p.show_proposal(proposal) == expected
    finally:
        typist.cancel()
        os.close(master)


# --- Ctrl+C --------------------------------------------------------------


def test_sigint_during_stream_cancels_and_reports_interrupted():
    out = io.StringIO()
    p = _panel(out=out)
    cancelled: list[int] = []

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        os.kill(os.getpid(), signal.SIGINT)
        for _ in range(200):
            time.sleep(0.01)
            yield AgentEvent(kind=EventKind.TEXT_DELTA, text=".")

    result = p.stream(events(), cancel=lambda: cancelled.append(1))
    assert result.interrupted is True
    assert cancelled == [1]
    assert "interrupted" in out.getvalue().lower()
    # The handler must be uninstalled again.
    assert signal.getsignal(signal.SIGINT) is not None


def test_ctrl_c_returns_to_a_prompt_within_one_second(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys, time\n"
        "from nvsh.panel import Panel\n"
        "from nvsh.agent.base import AgentEvent, EventKind\n"
        "cancelled = []\n"
        "def events():\n"
        "    yield AgentEvent(kind=EventKind.TEXT_DELTA, text='working')\n"
        "    while True:\n"
        "        time.sleep(0.05)\n"
        "        yield AgentEvent(kind=EventKind.TEXT_DELTA, text='.')\n"
        "p = Panel(out=sys.stdout, env={'NO_COLOR': '1'}, isatty=False)\n"
        "sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
        "res = p.stream(events(), cancel=lambda: cancelled.append(1))\n"
        "open(sys.argv[1], 'w').write('cancelled' if cancelled else 'no')\n"
        "sys.exit(130 if res.interrupted else 0)\n",
        encoding="utf-8",
    )
    marker = tmp_path / "marker"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(  # nosec B603 - fixed argv
        [sys.executable, str(driver), str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "READY"
        started = time.monotonic()
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        elapsed = time.monotonic() - started
    finally:
        if proc.poll() is None:  # pragma: no cover - only on failure
            proc.kill()
    assert proc.returncode == 130
    assert elapsed < 1.0, f"took {elapsed:.3f}s"
    assert marker.read_text(encoding="utf-8") == "cancelled"
