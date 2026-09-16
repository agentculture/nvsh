"""Tests for the panel's stop state (task t14).

Covers the acceptance criteria: a lone Esc on a pty during streaming ends
exactly like SIGINT (``StreamResult(interrupted=True)``); the first press
only calls ``cancel`` and prints the stopping line while the panel stays
up, and a second press calls ``force_stop`` exactly once; Esc is seen even
while the event source yields nothing; and an arrow key -- during streaming
or at the proposal -- neither interrupts nor counts as Esc and leaves no
stray bytes behind.
"""

from __future__ import annotations

import io
import os
import pty
import select
import signal
import threading
import time

import pytest

from nvsh import panel as panel_mod
from nvsh.agent.base import AgentEvent, EventKind, Proposal, ProposalKind

TTY_ENV = {"TERM": "xterm-256color", "NO_COLOR": "1"}
STOPPING = "stopping… press again to kill"


@pytest.fixture
def pty_pair():
    master, slave = pty.openpty()
    tty_in = os.fdopen(slave, "rb", buffering=0)
    try:
        yield master, tty_in
    finally:
        tty_in.close()
        try:
            os.close(master)
        except OSError:
            pass


def _panel(tty_in, out=None):
    return panel_mod.Panel(
        out=out if out is not None else io.StringIO(), in_=tty_in, env=TTY_ENV, isatty=True
    )


def _leftover(fd: int) -> bool:
    return bool(select.select([fd], [], [], 0.1)[0])


def _press(kind: str, master: int) -> None:
    if kind == "esc":
        os.write(master, b"\x1b")
    else:
        os.kill(os.getpid(), signal.SIGINT)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --- one press: polite stop, panel stays up -------------------------------


def _one_press_run(
    kind: str, master: int, tty_in
) -> tuple[panel_mod.StreamResult, str, list, list]:
    out = io.StringIO()
    p = _panel(tty_in, out)
    cancelled: list[int] = []
    killed: list[int] = []
    stopped = threading.Event()

    def cancel():
        cancelled.append(1)
        stopped.set()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press(kind, master)
        assert stopped.wait(5)
        # The panel stays up after the first press: this still renders.
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" winding down")

    result = p.stream(events(), cancel=cancel, force_stop=lambda: killed.append(1))
    return result, out.getvalue(), cancelled, killed


def test_lone_esc_during_streaming_is_identical_to_sigint(pty_pair):
    master, tty_in = pty_pair
    by_esc, esc_out, esc_cancel, esc_kill = _one_press_run("esc", master, tty_in)
    by_int, int_out, int_cancel, int_kill = _one_press_run("sigint", master, tty_in)
    assert by_esc.interrupted is True
    assert by_esc == by_int
    assert by_esc.text == "working winding down"
    assert esc_cancel == int_cancel == [1]
    assert esc_kill == int_kill == []
    for text in (esc_out, int_out):
        assert STOPPING in text
        assert "nvsh: interrupted" in text
        assert text.index(STOPPING) < text.index("winding down")
    assert esc_out == int_out


@pytest.mark.parametrize("kind", ["esc", "sigint"])
def test_a_single_press_never_calls_force_stop(pty_pair, kind):
    master, tty_in = pty_pair
    result, _out, cancelled, killed = _one_press_run(kind, master, tty_in)
    assert result.interrupted is True
    assert cancelled == [1]
    assert killed == []


# --- second press: kill ----------------------------------------------------


@pytest.mark.parametrize("first,second", [("esc", "esc"), ("sigint", "esc"), ("esc", "sigint")])
def test_second_press_calls_force_stop_exactly_once(pty_pair, first, second):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    cancelled: list[int] = []
    killed: list[int] = []
    gone = threading.Event()

    def force_stop():
        killed.append(1)
        gone.set()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press(first, master)
        assert _wait_for(lambda: cancelled)
        time.sleep(0.15)  # a separate keypress, not one ESC ESC burst
        _press(second, master)
        # A harness that ignores the polite cancel: silent until killed.
        gone.wait(10)

    started = time.monotonic()
    result = p.stream(events(), cancel=lambda: cancelled.append(1), force_stop=force_stop)
    assert time.monotonic() - started < 5
    assert result.interrupted is True
    assert cancelled == [1]
    assert killed == [1]
    assert out.getvalue().count(STOPPING) == 1


def test_second_press_without_force_stop_still_ends_the_stream(pty_pair):
    master, tty_in = pty_pair
    p = _panel(tty_in)
    cancelled: list[int] = []
    release = threading.Event()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("esc", master)
        assert _wait_for(lambda: cancelled)
        time.sleep(0.15)
        _press("esc", master)
        release.wait(10)

    started = time.monotonic()
    try:
        result = p.stream(events(), cancel=lambda: cancelled.append(1))
    finally:
        release.set()
    assert time.monotonic() - started < 5
    assert result.interrupted is True
    assert cancelled == [1]


# --- silent source ---------------------------------------------------------


def test_esc_prints_the_stopping_line_within_1s_while_the_source_is_silent(pty_pair):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    killed = threading.Event()
    timing: dict[str, float] = {}

    def events():
        # Nothing for 30s unless the harness is killed.
        killed.wait(30)
        return
        yield  # pragma: no cover - makes this a generator

    def operator():
        time.sleep(0.3)
        os.write(master, b"\x1b")
        pressed = time.monotonic()
        if _wait_for(lambda: STOPPING in out.getvalue(), timeout=5):
            timing["stopping"] = time.monotonic() - pressed
        time.sleep(0.15)
        os.write(master, b"\x1b")

    thread = threading.Thread(target=operator, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        result = p.stream(events(), cancel=lambda: None, force_stop=killed.set)
    finally:
        killed.set()
        thread.join(5)
    assert "stopping" in timing, out.getvalue()
    assert timing["stopping"] < 1.0, timing
    assert result.interrupted is True
    assert time.monotonic() - started < 10


# --- arrow keys ------------------------------------------------------------


def test_arrow_key_during_streaming_is_not_esc_and_leaves_no_bytes(pty_pair):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    cancelled: list[int] = []

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        os.write(master, b"\x1b[A")
        time.sleep(0.4)
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" done")
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(events(), cancel=lambda: cancelled.append(1))
    assert result.interrupted is False
    assert result.done is True
    assert result.text == "working done"
    assert cancelled == []
    assert "stopping" not in out.getvalue()
    assert not _leftover(tty_in.fileno())


def test_arrow_key_at_the_proposal_is_not_esc_and_leaves_no_bytes(pty_pair):
    master, tty_in = pty_pair
    p = _panel(tty_in)

    def typist():
        # After raw mode is entered (tty.setraw flushes earlier input).
        time.sleep(0.3)
        os.write(master, b"\x1b[B")
        time.sleep(0.3)
        os.write(master, b"\r")

    thread = threading.Thread(target=typist, daemon=True)
    thread.start()
    try:
        choice = p.show_proposal(Proposal("df -h", "disk", ProposalKind.INSPECT))
    finally:
        thread.join(5)
    assert choice == panel_mod.APPROVE
    assert not _leftover(tty_in.fileno())


def test_lone_esc_at_the_proposal_still_ignores(pty_pair):
    master, tty_in = pty_pair
    p = _panel(tty_in)
    typist = threading.Timer(0.3, lambda: os.write(master, b"\x1b"))
    typist.start()
    try:
        assert p.show_proposal(Proposal("df -h", "disk", ProposalKind.INSPECT)) == "ignore"
    finally:
        typist.cancel()
    assert not _leftover(tty_in.fileno())


def test_proposal_key_read_does_not_spin_on_a_hung_up_tty():
    master, slave = pty.openpty()
    with os.fdopen(slave, "rb", buffering=0) as tty_in:
        p = _panel(tty_in)
        os.close(master)
        started = time.monotonic()
        assert p.show_proposal(Proposal("df -h", "disk", ProposalKind.INSPECT)) == "ignore"
        assert time.monotonic() - started < 2


def test_raw_key_read_returns_empty_at_eof_instead_of_spinning():
    read_end, write_end = os.pipe()
    os.close(write_end)
    try:
        started = time.monotonic()
        assert panel_mod._read_raw_key(read_end) == ""
        assert time.monotonic() - started < 1
    finally:
        os.close(read_end)


# --- the proposal suspends the watcher -------------------------------------


def test_keys_at_a_proposal_mid_stream_reach_the_proposal_not_the_watcher(pty_pair):
    master, tty_in = pty_pair
    p = _panel(tty_in)
    choices: list[str] = []
    cancelled: list[int] = []

    def on_proposal(proposal, _event):
        threading.Timer(0.3, lambda: os.write(master, b"\x1b")).start()
        choices.append(p.show_proposal(proposal))

    def events():
        yield AgentEvent(kind=EventKind.PROPOSAL, proposal=Proposal("df -h", "d", ProposalKind.FIX))
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(events(), on_proposal=on_proposal, cancel=lambda: cancelled.append(1))
    assert choices == ["ignore"]
    assert cancelled == []
    assert result.interrupted is False
    assert result.done is True
