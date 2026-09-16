"""Tests for nvsh.keys (task t11): cbreak hold, lone-Esc disambiguation, restore.

Covers the acceptance criteria: a lone ESC on a pty reads as ``'esc'`` within
100ms, an arrow-key sequence (``ESC [ A``) is drained whole and reads as
nothing, a child holding a KeyWatcher killed by SIGHUP/SIGTERM leaves the pty
back in ICANON|ECHO, and the watcher is a no-op off a tty, under TERM=dumb and
under NVSH_DISABLE=1.
"""

from __future__ import annotations

import os
import pty
import select
import signal
import subprocess  # nosec B404 - fixed argv, no shell=True
import sys
import termios
import time
from pathlib import Path

import pytest

from nvsh import keys

REPO_ROOT = Path(__file__).resolve().parents[1]
TTY_ENV = {"TERM": "xterm-256color"}


@pytest.fixture
def pty_pair():
    master, slave = pty.openpty()
    try:
        yield master, slave
    finally:
        for fd in (master, slave):
            try:
                os.close(fd)
            except OSError:
                pass


def _readable(fd: int) -> bool:
    return bool(select.select([fd], [], [], 0)[0])


def _lflag(fd: int) -> int:
    return termios.tcgetattr(fd)[3]


# ---------------------------------------------------------------------------
# lone Esc vs escape sequences
# ---------------------------------------------------------------------------
def test_lone_esc_is_esc_within_100ms(pty_pair):
    master, slave = pty_pair
    with keys.KeyWatcher(fd=slave, env=TTY_ENV) as watcher:
        assert watcher.active
        os.write(master, b"\x1b")
        start = time.monotonic()
        result = watcher.poll(1.0)
        elapsed = time.monotonic() - start
    assert result == "esc"
    assert elapsed < 0.1


def test_arrow_sequence_is_not_esc_and_is_drained(pty_pair):
    master, slave = pty_pair
    with keys.KeyWatcher(fd=slave, env=TTY_ENV) as watcher:
        os.write(master, b"\x1b[A")
        assert watcher.poll(0.2) is None
        assert not _readable(slave)


def test_ss3_and_long_csi_sequences_are_drained(pty_pair):
    master, slave = pty_pair
    with keys.KeyWatcher(fd=slave, env=TTY_ENV) as watcher:
        os.write(master, b"\x1bOP\x1b[15~\x1b[1;5C")
        assert watcher.poll(0.2) is None
        assert not _readable(slave)


def test_typeahead_is_discarded_and_esc_after_it_still_seen(pty_pair):
    master, slave = pty_pair
    with keys.KeyWatcher(fd=slave, env=TTY_ENV) as watcher:
        os.write(master, b"ls -la\n")
        assert watcher.poll(0.1) is None
        assert not _readable(slave)
        os.write(master, b"xy\x1b")
        assert watcher.poll(0.5) == "esc"


def test_poll_times_out_with_no_input(pty_pair):
    _master, slave = pty_pair
    with keys.KeyWatcher(fd=slave, env=TTY_ENV) as watcher:
        start = time.monotonic()
        assert watcher.poll(0.05) is None
        assert time.monotonic() - start < 0.5


def test_cbreak_keeps_isig_and_restores_on_exit(pty_pair):
    _master, slave = pty_pair
    before = termios.tcgetattr(slave)
    with keys.KeyWatcher(fd=slave, env=TTY_ENV):
        lflag = _lflag(slave)
        assert not lflag & termios.ICANON
        assert not lflag & termios.ECHO
        assert lflag & termios.ISIG
    assert termios.tcgetattr(slave) == before


def test_signal_handlers_restored_on_exit(pty_pair):
    _master, slave = pty_pair
    prev_hup = signal.getsignal(signal.SIGHUP)
    prev_term = signal.getsignal(signal.SIGTERM)
    with keys.KeyWatcher(fd=slave, env=TTY_ENV):
        assert signal.getsignal(signal.SIGHUP) is not prev_hup
    assert signal.getsignal(signal.SIGHUP) is prev_hup
    assert signal.getsignal(signal.SIGTERM) is prev_term


# ---------------------------------------------------------------------------
# read_choice_key
# ---------------------------------------------------------------------------
def test_read_choice_key_disambiguates(pty_pair):
    master, slave = pty_pair
    with keys.KeyWatcher(fd=slave, env=TTY_ENV):
        os.write(master, b"\x1b")
        assert keys.read_choice_key(slave) == "esc"
        os.write(master, b"\x1b[B")
        assert keys.read_choice_key(slave) == ""
        assert not _readable(slave)
        os.write(master, b"e")
        assert keys.read_choice_key(slave) == "e"
        os.write(master, b"\r")
        assert keys.read_choice_key(slave) in ("\r", "\n")


def test_read_choice_key_eof_is_empty():
    r, w = os.pipe()
    os.close(w)
    try:
        assert keys.read_choice_key(r) == ""
    finally:
        os.close(r)


# ---------------------------------------------------------------------------
# no-op cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "env",
    [
        {"TERM": "dumb"},
        {"TERM": "xterm-256color", "NVSH_DISABLE": "1"},
    ],
)
def test_noop_on_dumb_or_disabled(pty_pair, env):
    master, slave = pty_pair
    before = termios.tcgetattr(slave)
    prev_hup = signal.getsignal(signal.SIGHUP)
    with keys.KeyWatcher(fd=slave, env=env) as watcher:
        assert not watcher.active
        assert termios.tcgetattr(slave) == before
        assert signal.getsignal(signal.SIGHUP) is prev_hup
        os.write(master, b"\x1b")
        assert watcher.poll(0.1) is None
    assert termios.tcgetattr(slave) == before


def test_noop_when_not_a_tty():
    r, w = os.pipe()
    try:
        with keys.KeyWatcher(fd=r, env=TTY_ENV) as watcher:
            assert not watcher.active
            os.write(w, b"\x1b")
            assert watcher.poll(0.1) is None
        # the byte was never read
        assert os.read(r, 1) == b"\x1b"
    finally:
        os.close(r)
        os.close(w)


def test_disable_zero_does_not_disable(pty_pair):
    _master, slave = pty_pair
    with keys.KeyWatcher(fd=slave, env={"TERM": "xterm", "NVSH_DISABLE": "0"}) as watcher:
        assert watcher.active


# ---------------------------------------------------------------------------
# SIGHUP / SIGTERM restore the terminal
# ---------------------------------------------------------------------------
_CHILD = """
import os, sys, time
from nvsh import keys
with keys.KeyWatcher(fd=0, env={"TERM": "xterm-256color"}) as w:
    os.write(2, b"ready\\n" if w.active else b"inactive\\n")
    while True:
        w.poll(0.05)
"""


@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGTERM])
def test_signal_kill_restores_icanon_echo(pty_pair, sig):
    _master, slave = pty_pair
    child = subprocess.Popen(  # nosec B603 - fixed argv
        [sys.executable, "-c", _CHILD],
        stdin=slave,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )
    try:
        line = child.stderr.readline()
        assert line == b"ready\n"
        assert not _lflag(slave) & termios.ICANON
        child.send_signal(sig)
        rc = child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        child.stderr.close()
    # previous disposition (SIG_DFL) was re-raised: the child died of the signal
    assert rc == -sig
    lflag = _lflag(slave)
    assert lflag & termios.ICANON
    assert lflag & termios.ECHO
