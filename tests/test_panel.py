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
from nvsh.agent.base import AgentEvent, EventKind, Proposal, ProposalKind, Target

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
    assert on.bold.startswith("\x1b[")
    assert on.reset == "\x1b[0m"
    assert off.bold == ""
    assert off.reset == ""
    assert off.red == ""


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
    assert seen
    assert "first" in seen[0]
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
    # d19: the command itself, not only the tool name (the exit code is
    # unknown here, so the finished line keeps the old wording).
    assert "... running: nvidia-smi" in text
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


def test_a_proposal_event_without_a_proposal_renders_nothing_and_keeps_going():
    """A malformed event must never end the stream or reach ``on_proposal``."""
    out = io.StringIO()
    p = _panel(out=out)
    seen: list[Proposal] = []

    def events():
        yield AgentEvent(kind=EventKind.PROPOSAL, proposal=None)
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="still here")
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(events(), on_proposal=lambda proposal, event: seen.append(proposal))
    assert seen == []
    assert result.proposals == []
    assert result.text == "still here"
    assert result.done is True


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


# --- waiting indicator (d13) --------------------------------------------


def _slow_events(delay: float):
    """An event source whose first event only arrives after ``delay``."""

    def gen():
        time.sleep(delay)
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="the answer")
        yield AgentEvent(kind=EventKind.DONE)

    return gen()


def test_waiting_line_repaints_in_place_with_an_elapsed_count_on_a_tty():
    out = io.StringIO()
    p = _panel(out=out, env={"TERM": "xterm-256color"}, isatty=True)
    result = p.stream(_slow_events(2.5))
    text = out.getvalue()
    assert "waiting for the agent (1s)" in text
    assert "waiting for the agent (2s)" in text
    # repainted in place: carriage return + hard-coded erase-to-end-of-line
    assert "\r\x1b[2K" in text
    assert text.index("waiting for the agent") < text.index("the answer")
    assert result.text == "the answer"
    assert result.done is True


def test_waiting_line_is_one_plain_line_and_never_repaints_off_a_tty():
    out = io.StringIO()
    p = _panel(out=out, isatty=False)
    p.stream(_slow_events(2.5))
    text = out.getvalue()
    assert text.count("... waiting for the agent") == 1
    assert "(1s)" not in text
    assert "\r" not in text
    assert "\x1b[" not in text
    assert "the answer" in text


def test_no_waiting_line_when_the_first_event_arrives_immediately():
    out = io.StringIO()
    p = _panel(out=out, env={"TERM": "xterm-256color"}, isatty=True)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.TEXT_DELTA, text="instant"),
                AgentEvent(kind=EventKind.DONE),
            ]
        )
    )
    assert "waiting" not in out.getvalue()


def test_waiting_line_is_cleared_before_the_first_text_reaches_the_screen():
    out = io.StringIO()
    p = _panel(out=out, env={"TERM": "xterm-256color"}, isatty=True)
    p.stream(_slow_events(1.4))
    text = out.getvalue()
    head, _, tail = text.rpartition("\r\x1b[2K")
    assert "waiting for the agent" in head
    assert tail.startswith("the answer")


# --- keypress acknowledgement (d13) --------------------------------------


@pytest.mark.parametrize(
    "typed,ack",
    [
        ("\n", "nvsh: running"),
        ("e\n", "nvsh: explaining"),
        ("d\n", "nvsh: details"),
        ("q\n", "nvsh: ignored"),
        ("", "nvsh: ignored"),
    ],
)
def test_show_proposal_acknowledges_the_keypress_immediately(typed, ack):
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO(typed), isatty=False)
    p.show_proposal(Proposal(command="df -h", rationale="disk", kind=ProposalKind.INSPECT))
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert lines[-1].startswith(ack), lines
    # the client still prints its own outcome line afterwards; no duplication here
    assert out.getvalue().count("nvsh: ") == 1


# --- fallback notice (d13) -----------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        "one-shot pi: daemon did not start within 10s",
        "one-shot fake: daemon refused the connection",
        "daemon did not start within 10s",
        "daemon refused the connection",
        "daemon connection lost: [Errno 32] Broken pipe",
    ],
)
def test_fallback_status_is_a_visible_notice_keeping_the_original_text(status):
    out = io.StringIO()
    p = _panel(out=out, isatty=False)
    p.stream(iter([AgentEvent(kind=EventKind.STATUS, text=status), AgentEvent(EventKind.DONE)]))
    text = out.getvalue()
    assert "nvsh: falling back" in text
    assert status in text
    assert not text.startswith("... ")


def test_fallback_notice_is_not_dim_when_style_is_on():
    out = io.StringIO()
    p = _panel(out=out, env={"TERM": "xterm-256color"}, isatty=True)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.STATUS, text="one-shot pi: daemon refused"),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    notice = [ln for ln in out.getvalue().splitlines() if "falling back" in ln][0]
    assert "\x1b[2m" not in notice
    assert "\x1b[" in notice


def test_ordinary_status_stays_dim_and_empty_status_renders_nothing():
    out = io.StringIO()
    p = _panel(out=out, env={"TERM": "xterm-256color"}, isatty=True)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.STATUS, text="checking disk"),
                AgentEvent(kind=EventKind.STATUS, text=""),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    text = out.getvalue()
    assert "\x1b[2m... checking disk\x1b[0m" in text
    assert text.count("...") == 1


# --- approve for this session / for this user (d15) ----------------------


def _proposal() -> Proposal:
    return Proposal(command="df -h", rationale="disk", kind=ProposalKind.INSPECT)


def test_legend_offers_session_and_user_keys_on_one_80_column_line():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("q\n"), isatty=False)
    p.show_proposal(_proposal())
    legend = [ln for ln in out.getvalue().splitlines() if ln.startswith("[Enter]")][0]
    assert len(legend) <= 80, f"legend is {len(legend)} columns: {legend!r}"
    for token in ("[Enter] run", "[s/S]", "[u/U]", "[d] details", "[t] tell", "[Esc] ignore"):
        assert token in legend, legend
    assert "session" in legend
    assert "user" in legend
    # order: run, session, user, explain, details, tell, ignore
    positions = [
        legend.index(t) for t in ("[Enter]", "[s/S]", "[u/U]", "[e]", "[d]", "[t]", "[Esc]")
    ]
    assert positions == sorted(positions), legend


@pytest.mark.parametrize(
    "typed,expected",
    [
        ("s\n", panel_mod.APPROVE_SESSION),
        ("S\n", panel_mod.APPROVE_SESSION_SPECIFIC),
        ("u\n", panel_mod.APPROVE_USER),
        ("U\n", panel_mod.APPROVE_USER_SPECIFIC),
    ],
)
def test_show_proposal_reads_the_scope_keys(typed, expected):
    p = _panel(out=io.StringIO(), in_=io.StringIO(typed), isatty=False)
    assert p.show_proposal(_proposal()) == expected


@pytest.mark.parametrize(
    "typed,ack",
    [
        ("s\n", "nvsh: running (approved for this session) ..."),
        ("u\n", "nvsh: running (approved for this user) ..."),
    ],
)
def test_scope_keys_acknowledge_with_the_scope_named(typed, ack):
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO(typed), isatty=False)
    p.show_proposal(_proposal())
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert lines[-1] == ack, lines


@pytest.mark.parametrize("key,expected", [(b"s", "session"), (b"u", "user")])
def test_scope_keys_are_read_as_single_keypresses_on_a_tty(key, expected):
    master, slave = pty.openpty()
    typist = threading.Timer(0.2, lambda: os.write(master, key))
    typist.start()
    try:
        with os.fdopen(slave, "rb", buffering=0) as tty_in:
            p = panel_mod.Panel(out=io.StringIO(), in_=tty_in, env={}, isatty=True)
            assert p.show_proposal(_proposal()) == expected
    finally:
        typist.cancel()
        os.close(master)


def test_guard_refuses_a_scope_key_with_one_line_and_no_acknowledgement():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("s\n"), isatty=False)
    choice = p.show_proposal(
        _proposal(), guard=lambda scope: "patterns starting with 'rm' are never approved"
    )
    assert choice == panel_mod.REFUSED
    text = out.getvalue()
    refusals = [ln for ln in text.splitlines() if "cannot approve" in ln]
    assert len(refusals) == 1, text
    assert "session" in refusals[0]
    assert "patterns starting with 'rm' are never approved" in refusals[0]
    assert "nvsh: running" not in text


def test_guard_is_not_consulted_for_the_run_once_key():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("\n"), isatty=False)
    seen = []

    def guard(scope):
        seen.append(scope)
        return "nope"

    assert p.show_proposal(_proposal(), guard=guard) == panel_mod.APPROVE
    assert seen == []


# --- d16: [t] tell -------------------------------------------------------


def test_legend_offers_the_tell_key_on_one_80_column_line():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("q\n"), isatty=False)
    p.show_proposal(_proposal())
    legend = [ln for ln in out.getvalue().splitlines() if ln.startswith("[Enter]")][0]
    assert len(legend) <= 80, f"legend is {len(legend)} columns: {legend!r}"
    for token in ("[Enter] run", "[s/S] +session", "[u/U] +user", "[e] why"):
        assert token in legend, legend
    for token in ("[d] details", "[t] tell", "[Esc] ignore"):
        assert token in legend, legend
    positions = [
        legend.index(t) for t in ("[Enter]", "[s/S]", "[u/U]", "[e]", "[d]", "[t]", "[Esc]")
    ]
    assert positions == sorted(positions), legend


@pytest.mark.parametrize("typed", ["t\n", "T\n"])
def test_show_proposal_reads_the_tell_key_without_acknowledging_a_run(typed):
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO(typed), isatty=False)
    assert p.show_proposal(_proposal()) == panel_mod.TELL
    assert "nvsh: running" not in out.getvalue()


def test_read_tell_prompts_and_returns_the_typed_line():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("just run free -h\n"), isatty=False)
    assert p.read_tell() == "just run free -h"
    assert "nvsh> " in out.getvalue()


def test_read_tell_treats_an_empty_line_as_a_cancel():
    p = _panel(out=io.StringIO(), in_=io.StringIO("\n"), isatty=False)
    assert p.read_tell() == ""


def test_tell_key_then_a_line_is_read_in_cooked_mode_on_a_real_tty():
    """``t`` is one raw keypress; the sentence after it is a cooked line read."""
    master, slave = pty.openpty()

    def typist():
        # After the panel has entered raw mode: tty.setraw uses TCSAFLUSH, so
        # a key queued before the proposal appeared would be discarded.
        time.sleep(0.3)
        os.write(master, b"t")
        time.sleep(0.3)
        os.write(master, b"just run free -h\n")

    thread = threading.Thread(target=typist, daemon=True)
    try:
        with os.fdopen(slave, "rb", buffering=0) as tty_in:
            out = io.StringIO()
            p = panel_mod.Panel(out=out, in_=tty_in, env={}, isatty=True)
            thread.start()
            assert p.show_proposal(_proposal()) == panel_mod.TELL
            assert p.read_tell() == "just run free -h"
    finally:
        thread.join(timeout=5)
        os.close(master)


def test_read_tell_on_a_tty_leaves_the_terminal_in_cooked_mode():
    import termios

    master, slave = pty.openpty()
    typist = threading.Timer(0.2, lambda: os.write(master, b"hello\n"))
    typist.start()
    try:
        with os.fdopen(slave, "rb", buffering=0) as tty_in:
            p = panel_mod.Panel(out=io.StringIO(), in_=tty_in, env={}, isatty=True)
            before = termios.tcgetattr(tty_in.fileno())
            assert p.read_tell() == "hello"
            assert termios.tcgetattr(tty_in.fileno()) == before
    finally:
        typist.cancel()
        os.close(master)


# --- d18: details --------------------------------------------------------


def test_detail_proposal_prints_kind_command_rationale_and_every_extra():
    out = io.StringIO()
    p = _panel(out=out)
    p.detail_proposal(
        Proposal(command="free -h", rationale="check memory", kind=ProposalKind.INSPECT),
        details={
            "approved": "no",
            "session pattern": "free -h",
            "user pattern": "free *",
            "backend": "pi",
            "conversation": "daemon, shell 4242",
            "output": "812 bytes (redacted)",
        },
    )
    text = out.getvalue()
    for line in (
        "kind: inspect",
        "command: free -h",
        "rationale: check memory",
        "approved: no",
        "session pattern: free -h",
        "user pattern: free *",
        "backend: pi",
        "conversation: daemon, shell 4242",
        "output: 812 bytes (redacted)",
    ):
        assert line in text, text


def test_detail_proposal_without_extras_is_the_old_three_lines():
    out = io.StringIO()
    p = _panel(out=out)
    p.detail_proposal(Proposal(command="free -h", rationale="mem", kind=ProposalKind.INSPECT))
    assert out.getvalue().splitlines() == [
        "kind: inspect",
        "command: free -h",
        "rationale: mem",
    ]


# --- d19 (panel half): the running command and the exit code -------------


def test_tool_call_shows_the_bare_command_not_only_the_tool_name():
    out = io.StringIO()
    p = _panel(out=out)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.TOOL_CALL, tool="bash", args={"command": "nvidia-smi"}),
                AgentEvent(
                    kind=EventKind.TOOL_RESULT, tool="bash", result={"output": "", "exitCode": 0}
                ),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    text = out.getvalue()
    assert "... running: nvidia-smi" in text
    assert "... finished (exit 0)" in text
    assert "running tool: bash" not in text


def test_tool_call_without_a_command_still_names_the_tool():
    out = io.StringIO()
    p = _panel(out=out)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.TOOL_CALL, tool="read_file", args={"path": "/etc/x"}),
                AgentEvent(kind=EventKind.TOOL_RESULT, tool="read_file", result={"output": "x"}),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    text = out.getvalue()
    assert "... running tool: read_file" in text
    assert "... tool read_file finished" in text


def test_a_long_running_command_is_truncated_to_one_hundred_columns():
    out = io.StringIO()
    p = _panel(out=out)
    command = "echo " + "x" * 300
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.TOOL_CALL, tool="bash", args={"command": command}),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    line = [ln for ln in out.getvalue().splitlines() if ln.startswith("... running: ")][0]
    body = line[len("... running: ") :]
    assert len(body) == 100, line
    assert body.endswith("…")


def test_a_multiline_running_command_is_shown_on_one_line():
    out = io.StringIO()
    p = _panel(out=out)
    p.stream(
        iter(
            [
                AgentEvent(
                    kind=EventKind.TOOL_CALL, tool="bash", args={"command": "echo a\necho b"}
                ),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    lines = [ln for ln in out.getvalue().splitlines() if "running" in ln]
    assert lines == ["... running: echo a echo b"]


@pytest.mark.parametrize("key", ["exitCode", "exit_code", "exit", "returncode"])
def test_a_nonzero_exit_code_is_reported_however_the_backend_spells_it(key):
    out = io.StringIO()
    p = _panel(out=out)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.TOOL_RESULT, tool="bash", result={key: 127}),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    assert "... finished (exit 127)" in out.getvalue()


def test_running_and_finished_helpers_are_callable_for_locally_run_commands():
    out = io.StringIO()
    p = _panel(out=out)
    p.running("free -h")
    p.finished(0)
    assert out.getvalue() == "... running: free -h\n... finished (exit 0)\n"


# --- d16/d24: say what each scope key would store ------------------------


_SCOPES = {
    "session": "this exact line",
    "session-specific": "'whatis ls *'",
    "user": "'whatis *'",
    "user-specific": "'whatis ls *'",
}


def test_scope_line_says_what_each_key_would_store_before_the_legend():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("q\n"), isatty=False)
    p.show_proposal(Proposal("whatis ls", "look it up", ProposalKind.INSPECT), scopes=_SCOPES)
    lines = out.getvalue().splitlines()
    scope_lines = [ln for ln in lines if ln.startswith("[s] ") or ln.startswith("[u] ")]
    assert len(scope_lines) == 2, lines
    session_line, user_line = scope_lines
    assert session_line == "[s] this exact line  [S] 'whatis ls *'  (this session)"
    assert user_line == "[u] 'whatis *'  [U] 'whatis ls *'  (persisted for you)"
    for line in scope_lines:
        assert len(line) <= 80, f"{len(line)} columns: {line!r}"
    legend_at = lines.index(panel_mod.LEGEND)
    assert max(lines.index(ln) for ln in scope_lines) < legend_at


def test_a_long_scope_line_is_truncated_to_80_columns():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("q\n"), isatty=False)
    wide = {
        "session": "each stage exactly",
        "session-specific": "'" + "x" * 90 + " *'",
        "user": "'" + "y" * 90 + " *'",
        "user-specific": "'" + "z" * 90 + " *'",
    }
    p.show_proposal(_proposal(), scopes=wide)
    for line in out.getvalue().splitlines():
        assert len(line) <= 80, f"{len(line)} columns: {line!r}"


def test_a_refused_scope_is_named_as_unavailable_instead_of_promised():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("q\n"), isatty=False)
    p.show_proposal(
        Proposal("sudo nvpmodel -m 0", "power", ProposalKind.FIX),
        guard=lambda scope: None if scope.startswith("session") else "never pre-approved",
        scopes=_SCOPES,
    )
    text = out.getvalue()
    assert "[u]/[U] not available: never pre-approved" in text
    assert "[s] this exact line" in text


def test_no_scope_line_when_the_caller_passes_no_scopes():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("q\n"), isatty=False)
    p.show_proposal(_proposal())
    lines = out.getvalue().splitlines()
    assert [ln for ln in lines if ln.startswith("[s] ")] == []


@pytest.mark.parametrize(
    "typed,ack",
    [
        ("s\n", "nvsh: running; 'whatis ls' approved for this session"),
        ("S\n", "nvsh: running; 'whatis ls *' approved for this session"),
        ("u\n", "nvsh: running; 'whatis *' approved for this user"),
        ("U\n", "nvsh: running; 'whatis ls *' approved for this user"),
    ],
)
def test_scope_acks_name_the_stored_pattern(typed, ack):
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO(typed), isatty=False)
    patterns = {
        "session": "'whatis ls'",
        "session-specific": "'whatis ls *'",
        "user": "'whatis *'",
        "user-specific": "'whatis ls *'",
    }
    p.show_proposal(
        Proposal("whatis ls", "look", ProposalKind.INSPECT),
        scopes=_SCOPES,
        patterns=patterns,
    )
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert lines[-1] == ack, lines


def test_a_scope_ack_falls_back_to_the_scope_phrase_without_patterns():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("u\n"), isatty=False)
    p.show_proposal(Proposal("whatis ls", "look", ProposalKind.INSPECT), scopes=_SCOPES)
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert lines[-1] == "nvsh: running; 'whatis *' approved for this user"


# --- d22: the header says which backend is being asked -------------------


def test_header_names_the_backend_it_is_forwarding_to():
    out = io.StringIO()
    p = _panel(out=out)
    p.header("ls /nope", 2, backend_label="pi/associate")
    assert out.getvalue() == "nvsh: ls /nope failed (exit 2), forwarding to pi/associate\n"


def test_header_without_a_backend_label_is_the_bare_failure_line():
    out = io.StringIO()
    p = _panel(out=out)
    p.header("ls /nope", 2)
    assert out.getvalue() == "nvsh: ls /nope failed (exit 2)\n"


def test_header_ask_form_names_the_backend_and_the_question():
    out = io.StringIO()
    p = _panel(out=out)
    p.header("", 0, backend_label="pi/associate", ask="why is memory high?")
    assert out.getvalue() == "nvsh: asking pi/associate: why is memory high?\n"


def test_header_ask_form_without_a_label_still_reads_as_a_sentence():
    out = io.StringIO()
    p = _panel(out=out)
    p.header("", 0, ask="why is memory high?")
    assert out.getvalue() == "nvsh: asking: why is memory high?\n"


# --- t18: THINKING rendering ----------------------------------------------


def test_thinking_is_a_dimmed_run_closed_before_the_first_text_delta():
    out = io.StringIO()
    p = _panel(out=out, env={"TERM": "xterm-256color"}, isatty=True)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.THINKING, text="considering "),
                AgentEvent(kind=EventKind.THINKING, text="the failure"),
                AgentEvent(kind=EventKind.TEXT_DELTA, text="here is the answer"),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    text = out.getvalue()
    dim_run = "\x1b[2mconsidering the failure\x1b[0m\n"
    assert dim_run in text
    assert text.index(dim_run) < text.index("here is the answer")


def test_thinking_run_is_closed_before_a_tool_call_and_before_a_proposal_and_done():
    out = io.StringIO()
    p = _panel(out=out, env={"TERM": "xterm-256color"}, isatty=True)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.THINKING, text="let me check"),
                AgentEvent(kind=EventKind.TOOL_CALL, tool="bash", args={"command": "df -h"}),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    text = out.getvalue()
    assert "\x1b[2mlet me check\x1b[0m\n" in text
    assert text.index("\x1b[0m\n") < text.index("... running: df -h")


def test_thinking_with_nothing_after_it_is_still_closed_at_stream_end():
    out = io.StringIO()
    p = _panel(out=out, env={"TERM": "xterm-256color"}, isatty=True)
    p.stream(iter([AgentEvent(kind=EventKind.THINKING, text="hm"), AgentEvent(EventKind.DONE)]))
    assert "\x1b[2mhm\x1b[0m\n" in out.getvalue()


@pytest.mark.parametrize(
    "env,isatty",
    [
        ({"TERM": "dumb"}, True),
        ({"TERM": "xterm", "NO_COLOR": "1"}, True),
        ({"TERM": "xterm-256color"}, False),
    ],
)
def test_thinking_prints_plain_prefixed_lines_with_no_sgr(env, isatty):
    out = io.StringIO()
    p = _panel(out=out, env=env, isatty=isatty)
    p.stream(
        iter(
            [
                AgentEvent(kind=EventKind.THINKING, text="considering the failure"),
                AgentEvent(kind=EventKind.TEXT_DELTA, text="here is the answer"),
                AgentEvent(EventKind.DONE),
            ]
        )
    )
    text = out.getvalue()
    assert "\x1b[" not in text
    assert "thinking: considering the failure" in text
    assert "here is the answer" in text


def test_thinking_deltas_arrive_as_they_are_written_not_buffered():
    out = io.StringIO()
    p = _panel(out=out, env={"TERM": "xterm-256color"}, isatty=True)
    seen: list[str] = []

    def events():
        yield AgentEvent(kind=EventKind.THINKING, text="first")
        seen.append(out.getvalue())
        yield AgentEvent(kind=EventKind.THINKING, text=" second")
        yield AgentEvent(EventKind.DONE)

    p.stream(events())
    assert seen
    assert "first" in seen[0]
    assert "second" not in seen[0]


# --- t18: target header line -----------------------------------------------


def test_no_target_header_line_when_no_target_is_given():
    """Backward compatible: a panel built with no target renders nothing new."""
    out = io.StringIO()
    p = _panel(out=out)
    p.stream(iter([AgentEvent(kind=EventKind.TEXT_DELTA, text="hi"), AgentEvent(EventKind.DONE)]))
    assert "·" not in out.getvalue()


def test_target_header_line_names_harness_model_effort_path_and_warmth():
    out = io.StringIO()
    p = panel_mod.Panel(
        out=out,
        in_=io.StringIO(""),
        env={},
        isatty=False,
        target=Target(backend="pi", model="associate", effort="high"),
        path="/usr/bin/pi",
        warm=True,
    )
    p.stream(iter([AgentEvent(EventKind.DONE)]))
    first_line = out.getvalue().splitlines()[0]
    assert first_line == "pi/associate/high · /usr/bin/pi · warm"


def test_target_header_line_says_one_shot_when_not_warm():
    out = io.StringIO()
    p = _panel(out=out)
    p.set_target(Target(backend="fake"), path="/bin/fake", warm=False)
    p.stream(iter([AgentEvent(EventKind.DONE)]))
    assert out.getvalue().splitlines()[0] == "fake · /bin/fake · one-shot"


def test_target_header_line_omits_missing_model_and_effort():
    out = io.StringIO()
    p = _panel(out=out)
    p.set_target(Target(backend="claude"), path="/bin/claude", warm=True)
    p.stream(iter([AgentEvent(EventKind.DONE)]))
    assert out.getvalue().splitlines()[0] == "claude · /bin/claude · warm"


# --- d26: numbered stages, and picking which of them an approval covers ---


_MULTI_SCOPES = {
    "session": "exact",
    "session-specific": "'ls /tmp/git *' | 'grep -i *'",
    "user": "'ls *' | 'grep *'",
    "user-specific": "'ls /tmp/git *' | 'grep -i *'",
}
_MULTI_STAGES = ["'ls /tmp/git'", "'grep -i orin'"]
_MULTI_STAGE_PATTERNS = {
    "session": ["'ls /tmp/git'", "'grep -i orin'"],
    "session-specific": ["'ls /tmp/git *'", "'grep -i *'"],
    "user": ["'ls *'", "'grep *'"],
    "user-specific": ["'ls /tmp/git *'", "'grep -i *'"],
}


def _multi_proposal():
    return Proposal("ls /tmp/git | grep -i orin", "look", ProposalKind.INSPECT)


def _show_multi(typed):
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO(typed), isatty=False)
    choice = p.show_proposal(
        _multi_proposal(),
        scopes=_MULTI_SCOPES,
        patterns={k: " ".join(v) for k, v in _MULTI_STAGE_PATTERNS.items()},
        stages=_MULTI_STAGES,
        stage_patterns=_MULTI_STAGE_PATTERNS,
    )
    return p, out.getvalue(), choice


def test_multi_stage_scope_lines_number_the_stages_and_name_each_pattern():
    _p, text, _choice = _show_multi("q\n")
    lines = text.splitlines()
    assert "stages: 1 'ls /tmp/git'  2 'grep -i orin'" in lines
    assert "[s] exact  [S] 'ls /tmp/git *' | 'grep -i *'  (this session)" in lines
    assert "[u] 'ls *' | 'grep *'  [U] 'ls /tmp/git *' | 'grep -i *'  (persisted for you)" in lines
    stages_at = lines.index("stages: 1 'ls /tmp/git'  2 'grep -i orin'")
    assert stages_at < lines.index(panel_mod.LEGEND)


def test_a_long_stages_line_is_clipped_to_80_columns():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("q\n"), isatty=False)
    p.show_proposal(
        _multi_proposal(),
        scopes=_MULTI_SCOPES,
        stages=["'" + "x" * 60 + "'", "'" + "y" * 60 + "'"],
    )
    for line in out.getvalue().splitlines():
        assert len(line) <= 80, f"{len(line)} columns: {line!r}"


def test_an_unapprovable_stage_is_rendered_as_not_approvable():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("q\n"), isatty=False)
    p.show_proposal(
        _multi_proposal(),
        scopes=_MULTI_SCOPES,
        stages=["'ls /tmp'", "(not approvable)"],
    )
    assert "stages: 1 'ls /tmp'  2 (not approvable)" in out.getvalue().splitlines()


@pytest.mark.parametrize(
    "typed,picked",
    [
        ("all\n", [1, 2]),
        ("\n", [1, 2]),
        ("1\n", [1]),
        ("2\n", [2]),
        ("1,2\n", [1, 2]),
        ("1 2\n", [1, 2]),
        ("junk\nall\n", [1, 2]),
        ("junk\n2\n", [2]),
        ("junk\njunk\n", [1, 2]),
    ],
)
def test_the_stages_prompt_parses_what_the_operator_typed(typed, picked):
    p, text, choice = _show_multi("s\n" + typed)
    assert choice == panel_mod.APPROVE_SESSION
    assert "stages [all,1,2]: " in text
    assert p.stage_choice == picked


def test_a_single_stage_command_never_asks_which_stage():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("s\n"), isatty=False)
    p.show_proposal(
        Proposal("whatis ls", "look it up", ProposalKind.INSPECT),
        scopes=_SCOPES,
        stages=["'whatis ls'"],
    )
    assert "stages [" not in out.getvalue()
    assert p.stage_choice is None


def test_a_non_scope_key_never_asks_which_stage():
    _p, text, choice = _show_multi("\n")
    assert choice == panel_mod.APPROVE
    assert "stages [" not in text


def test_the_ack_names_only_the_stored_stage_patterns():
    _p, text, _choice = _show_multi("s\n2\n")
    assert "nvsh: running; 'grep -i orin' approved for this session (stage 2 of 2)" in text


def test_the_ack_says_stages_plural_when_a_subset_of_three_is_stored():
    out = io.StringIO()
    p = _panel(out=out, in_=io.StringIO("u\n1,3\n"), isatty=False)
    p.show_proposal(
        Proposal("a | b | c", "x", ProposalKind.INSPECT),
        scopes={"user": "'a *' | 'b *' | 'c *'"},
        stages=["'a'", "'b'", "'c'"],
        stage_patterns={"user": ["'a *'", "'b *'", "'c *'"]},
    )
    assert "'a *' 'c *' approved for this user (stages 1,3 of 3)" in out.getvalue()


def test_the_ack_keeps_the_d24_wording_when_every_stage_is_stored():
    _p, text, _choice = _show_multi("u\nall\n")
    assert "nvsh: running; 'ls *' 'grep *' approved for this user\n" in text
