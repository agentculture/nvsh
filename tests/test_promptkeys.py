"""Timed single-key reads for the stop-choice prompt, exercised on a real pty.

Every test drives :func:`nvsh.promptkeys.read_choice` through an
``os.openpty()`` pair rather than a stubbed reader, because the behaviour
that matters (escape-sequence draining, the ``tcflush`` of typeahead, the
termios restore) only exists against a terminal.

Keys must be written *after* the read has flushed pending input, otherwise
the flush would eat them. Instead of sleeping and hoping, the helpers below
wrap ``promptkeys._flush_pending`` and let the writer thread wait on the
event it sets, so the ordering is deterministic on a loaded machine. The
``before_read`` callback -- the real prompt's legend hook -- is the same
instant seen from the other side, and the tests that care about a fast
answer write from there.
"""

from __future__ import annotations

import os
import pty
import select
import signal
import subprocess
import sys
import threading
import time

import pytest

from nvsh import keys as keys_mod
from nvsh import promptkeys

# Generous everywhere: these run under `pytest -n auto` on loaded Jetsons.
_PATIENT = 10.0
_WAIT = 5.0


def _cbreak(fd: int):
    """Put ``fd`` in the cbreak the panel already holds while it streams.

    Returns the saved attributes. Without this a pty slave stays canonical,
    so bytes written before the prompt opens are invisible to ``select``
    until a newline arrives -- which is not how the real prompt is entered.
    """
    import termios

    saved = termios.tcgetattr(fd)
    attrs = termios.tcgetattr(fd)
    attrs[3] &= ~(termios.ICANON | termios.ECHO)
    attrs[6][termios.VMIN] = 1
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return saved


@pytest.fixture
def tty_pair():
    master, slave = pty.openpty()
    _cbreak(slave)
    try:
        yield master, slave
    finally:
        _close_pair(master, slave)


@pytest.fixture
def plain_tty_pair():
    """A pty left in its default cooked state, as a shell hands one over.

    The signal tests need this rather than :func:`tty_pair`: what they assert
    is that the terminal is *given back*, which is only visible when what it
    is given back to differs from what ``read_choice`` sets.
    """
    master, slave = pty.openpty()
    try:
        yield master, slave
    finally:
        _close_pair(master, slave)


def _close_pair(*fds: int) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _after_flush(monkeypatch, action) -> threading.Thread:
    """Run ``action()`` once :func:`read_choice` has flushed typeahead."""
    flushed = threading.Event()
    real = promptkeys._flush_pending

    def spy(fd: int) -> None:
        real(fd)
        flushed.set()

    monkeypatch.setattr(promptkeys, "_flush_pending", spy)

    def run() -> None:
        if flushed.wait(_WAIT):
            action()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _write_after_flush(monkeypatch, master: int, payload: bytes) -> threading.Thread:
    return _after_flush(monkeypatch, lambda: os.write(master, payload))


def _pending(fd: int) -> bool:
    return bool(select.select([fd], [], [], 0)[0])


# -- criterion 1: the public contract -----------------------------------------


def test_allowed_key_is_returned(tty_pair, monkeypatch):
    master, slave = tty_pair
    _write_after_flush(monkeypatch, master, b"s")
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == "s"


def test_unlisted_key_is_ignored_and_the_read_goes_on(tty_pair, monkeypatch):
    master, slave = tty_pair
    _write_after_flush(monkeypatch, master, b"xq9t")
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == "t"


def test_uppercase_answers_a_lowercase_key(tty_pair, monkeypatch):
    master, slave = tty_pair
    _write_after_flush(monkeypatch, master, b"S")
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == "s"


def test_lone_esc_returns_esc(tty_pair, monkeypatch):
    master, slave = tty_pair
    _write_after_flush(monkeypatch, master, b"\x1b")
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == promptkeys.ESC
    assert promptkeys.ESC == keys_mod.ESC


def test_ctrl_c_returns_interrupt(tty_pair, monkeypatch):
    master, slave = tty_pair
    _write_after_flush(monkeypatch, master, b"\x03")
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == promptkeys.INTERRUPT


def test_csi_sequence_neither_answers_nor_leaves_bytes_behind(tty_pair, monkeypatch):
    master, slave = tty_pair
    _write_after_flush(monkeypatch, master, b"\x1b[A\x1b[1;5Ds")
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == "s"
    assert not _pending(slave)


def test_ss3_sequence_neither_answers_nor_leaves_bytes_behind(tty_pair, monkeypatch):
    master, slave = tty_pair
    _write_after_flush(monkeypatch, master, b"\x1bOPt")
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == "t"
    assert not _pending(slave)


def test_deadline_is_monotonic_across_ignored_keys(tty_pair, monkeypatch):
    """Ignored keys must not extend the window: the deadline is absolute."""
    master, slave = tty_pair
    stop = threading.Event()

    def noise() -> None:
        # Self-limiting: a regression that reset the deadline on every byte
        # must fail this test, not hang the suite.
        until = time.monotonic() + _PATIENT
        while not stop.is_set() and time.monotonic() < until:
            try:
                os.write(master, b"x")
            except OSError:
                return
            time.sleep(0.05)

    _after_flush(monkeypatch, noise)
    started = time.monotonic()
    try:
        assert promptkeys.read_choice(slave, "ts", 2.0) == promptkeys.TIMEOUT
    finally:
        stop.set()
    assert 1.0 <= time.monotonic() - started <= 4.0


# -- criterion 2: typeahead is discarded before the read starts ----------------


def test_two_esc_within_20ms_leave_the_read_unanswered(tty_pair):
    master, slave = tty_pair
    os.write(master, b"\x1b")
    os.write(master, b"\x1b")
    # Deterministic: wait until the bytes have actually reached the slave's
    # input queue, so the flush under test has something to discard.
    deadline = time.monotonic() + _WAIT
    while not _pending(slave) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _pending(slave), "pty did not deliver the typeahead"
    assert promptkeys.read_choice(slave, "ts", 2.0) == promptkeys.TIMEOUT


def test_typeahead_of_an_allowed_key_is_discarded(tty_pair):
    master, slave = tty_pair
    os.write(master, b"s")
    deadline = time.monotonic() + _WAIT
    while not _pending(slave) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _pending(slave)
    assert promptkeys.read_choice(slave, "ts", 2.0) == promptkeys.TIMEOUT


# -- the prompt is drawn between the flush and the read ------------------------


def test_the_prompt_is_drawn_after_the_flush_so_neither_key_is_confused(tty_pair):
    """One test, both halves of the ordering (c33 and the Qodo finding).

    ``s`` is typed before the prompt exists and must die with the rest of the
    typeahead; ``t`` is typed by ``before_read``, which is the instant the
    legend reaches the screen, and must answer.
    """
    master, slave = tty_pair
    os.write(master, b"s")
    deadline = time.monotonic() + _WAIT
    while not _pending(slave) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _pending(slave), "pty did not deliver the typeahead"
    assert promptkeys.read_choice(slave, "ts", _PATIENT, lambda: os.write(master, b"t")) == "t"


def test_a_key_typed_the_instant_the_prompt_appears_is_never_lost(tty_pair):
    """200 prompts, one write each, zero losses.

    Before the flush moved ahead of the legend a key written the moment the
    prompt appeared could be discarded by the flush that followed it, which
    is why the pty tests used to re-type their answers. One write is now
    enough, every time.
    """
    master, slave = tty_pair
    for _ in range(200):
        key = promptkeys.read_choice(slave, "ts", _PATIENT, lambda: os.write(master, b"s"))
        assert key == "s"
        assert not _pending(slave)


# -- criterion 3: termios, EOF and the timeout --------------------------------


def test_termios_is_restored_on_the_normal_path(tty_pair, monkeypatch):
    import termios

    master, slave = tty_pair
    before = termios.tcgetattr(slave)
    _write_after_flush(monkeypatch, master, b"s")
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == "s"
    assert termios.tcgetattr(slave) == before


def test_termios_is_restored_when_the_read_raises(tty_pair, monkeypatch):
    import termios

    master, slave = tty_pair
    before = termios.tcgetattr(slave)

    def boom(fd, timeout):
        raise RuntimeError("boom")

    monkeypatch.setattr(promptkeys, "_read_byte", boom)
    _write_after_flush(monkeypatch, master, b"s")
    with pytest.raises(RuntimeError):
        promptkeys.read_choice(slave, "ts", _PATIENT)
    assert termios.tcgetattr(slave) == before


def test_eof_returns_eof_at_once_without_spinning(tty_pair, monkeypatch):
    master, slave = tty_pair
    _after_flush(monkeypatch, lambda: os.close(master))
    started = time.monotonic()
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == promptkeys.EOF
    assert time.monotonic() - started < _PATIENT / 2


def test_a_hung_up_tty_does_not_spin(tty_pair, monkeypatch):
    """A closed master must not be read in a tight loop until the timeout."""
    master, slave = tty_pair
    reads = 0
    real = promptkeys._read_byte

    def counted(fd, timeout):
        nonlocal reads
        reads += 1
        return real(fd, timeout)

    monkeypatch.setattr(promptkeys, "_read_byte", counted)
    _after_flush(monkeypatch, lambda: os.close(master))
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == promptkeys.EOF
    assert reads <= 2


def test_no_key_returns_timeout_within_a_second(tty_pair):
    _master, slave = tty_pair
    started = time.monotonic()
    assert promptkeys.read_choice(slave, "ts", 2.0) == promptkeys.TIMEOUT
    elapsed = time.monotonic() - started
    assert 1.0 <= elapsed <= 3.0


def test_select_failure_reports_eof_and_restores_termios(tty_pair, monkeypatch):
    import termios

    _master, slave = tty_pair
    before = termios.tcgetattr(slave)

    def broken(*_a, **_k):
        raise OSError(9, "bad fd")

    monkeypatch.setattr(promptkeys.select, "select", broken)
    assert promptkeys.read_choice(slave, "ts", _PATIENT) == promptkeys.EOF
    assert termios.tcgetattr(slave) == before


def test_works_on_a_plain_pipe_without_termios():
    """No tty: the read degrades to select() and never raises."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"t")
        assert promptkeys.read_choice(read_fd, "ts", _PATIENT) == "t"
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_a_closed_pipe_is_eof():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    try:
        assert promptkeys.read_choice(read_fd, "ts", _PATIENT) == promptkeys.EOF
    finally:
        os.close(read_fd)


def test_return_constants_are_distinct_and_never_collide_with_a_key():
    sentinels = {promptkeys.ESC, promptkeys.INTERRUPT, promptkeys.EOF, promptkeys.TIMEOUT}
    assert len(sentinels) == 4
    assert all(len(value) > 1 for value in sentinels)


# -- termios survives SIGHUP/SIGTERM (h23) ------------------------------------

#: A child that sits in :func:`read_choice` on its own tty. It announces
#: nothing: the parent watches the tty's own flags instead (``_await_cbreak``),
#: which is both deterministic and independent of this module's API, so the
#: test reads the same against the code before the fix.
_WAITER = """
import sys
sys.path.insert(0, sys.argv[1])
from nvsh import promptkeys
promptkeys.read_choice(0, "ts", 60.0)
"""

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _lflags(fd: int) -> int:
    import termios

    return termios.tcgetattr(fd)[3]


def _await_cbreak(fd: int) -> bool:
    """Wait until someone is holding ``fd`` the way the prompt holds it."""
    import termios

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not _lflags(fd) & (termios.ICANON | termios.ISIG):
            return True
        time.sleep(0.01)
    return False


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_a_termination_signal_gives_the_terminal_back(plain_tty_pair, signum):
    """A real process, a real tty, a real signal (review finding 2 / h23).

    The child dies *from the signal* -- the previous disposition is
    delegated to, not swallowed -- and the tty it was holding in cbreak is
    canonical, echoing and signal-generating again afterwards. Without the
    handlers this fails on the flags: the child dies with the terminal it
    borrowed still in cbreak.
    """
    import termios

    _master, slave = plain_tty_pair
    before = termios.tcgetattr(slave)
    child = subprocess.Popen(  # a fixed argv, no shell
        [sys.executable, "-c", _WAITER, _REPO],
        stdin=slave,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _await_cbreak(slave), "the child never took the terminal"
        os.kill(child.pid, signum)
        assert child.wait(timeout=30) == -signum, "the signal's own disposition was not honoured"
    finally:
        if child.poll() is None:  # pragma: no cover - only if the assert above failed
            child.kill()
            child.wait(timeout=30)
    after = _lflags(slave)
    assert after & termios.ICANON, "canonical mode was not restored"
    assert after & termios.ECHO, "echo was not restored"
    assert after & termios.ISIG, "signal generation was not restored"
    assert termios.tcgetattr(slave) == before, "the terminal came back changed"


def test_off_the_main_thread_the_read_works_without_the_safety_net(tty_pair):
    """``signal.signal`` is main-thread only; the read must not raise.

    Installing handlers is refused off the main thread, so the prompt runs
    there without the SIGHUP/SIGTERM net -- reading keys exactly as it does
    anywhere else, which is what the panel's worker threads need.
    """
    master, slave = tty_pair
    answer: dict[str, object] = {}

    def run() -> None:
        try:
            answer["key"] = promptkeys.read_choice(
                slave, "ts", _PATIENT, lambda: os.write(master, b"s")
            )
        except BaseException as exc:  # noqa: BLE001 - the point of the test
            answer["raised"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(_WAIT * 2)
    assert not thread.is_alive()
    assert answer == {"key": "s"}
    assert signal.getsignal(signal.SIGTERM) is not None


# -- criterion 4: nvsh/keys.py is reused, not re-implemented ------------------


def test_the_timed_read_reuses_the_key_module():
    assert promptkeys._after_esc is keys_mod._after_esc
    assert promptkeys._read_byte is keys_mod._read_byte
