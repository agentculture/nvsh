"""Terminal key watcher: see a lone Esc while the agent works, safely.

While the panel streams, nvsh holds stdin in cbreak mode so a lone Esc can
stop the agent exactly like Ctrl+C. Holding the terminal is the dangerous
part (hook constraint: never lock the operator out), so this module:

* arms only when stdin is a tty, ``TERM`` is not ``dumb`` and ``NVSH_DISABLE``
  is unset (or ``0``, matching ``nvsh/shell/hook.bash``); otherwise
  :class:`KeyWatcher` is a no-op that never touches termios or signals;
* keeps ``ISIG`` on, so Ctrl+C still arrives as SIGINT, not as a byte;
* restores the saved termios on every exit path it can see: the context
  manager exit, and SIGHUP/SIGTERM, whose handlers restore the terminal and
  then re-raise the previous disposition;
* tells a lone ``0x1b`` (no further byte within 50ms) from a CSI/SS3 escape
  sequence (arrow and function keys), which is drained whole and ignored;
* discards every other byte: typeahead typed while the panel streams is
  consumed and never executed (kernel 6.2+ cannot push bytes back).

Stdlib only: this runs on the failure path of an interactive shell.
"""

from __future__ import annotations

import os
import select
import signal
import time
from collections.abc import Mapping

ESC = "esc"
ESC_TIMEOUT = 0.05

_ESC_BYTE = 0x1B
_CSI_INTRO = ord("[")
_SS3_INTRO = ord("O")
_SIGNALS = (signal.SIGHUP, signal.SIGTERM)


def _enabled(fd: int, env: Mapping[str, str]) -> bool:
    disable = env.get("NVSH_DISABLE", "")
    if disable and disable != "0":
        return False
    if env.get("TERM", "") == "dumb":
        return False
    try:
        return os.isatty(fd)
    except OSError:
        return False


def _read_byte(fd: int, timeout: float | None) -> int | None:
    """One byte from ``fd``, or ``None`` on timeout/EOF/error."""
    try:
        if timeout is not None and not select.select([fd], [], [], max(timeout, 0))[0]:
            return None
        data = os.read(fd, 1)
    except (OSError, ValueError):
        return None
    return data[0] if data else None


def _after_esc(fd: int) -> str | None:
    """Classify what follows an ESC byte already read.

    Returns ``'esc'`` for a lone Esc (nothing within 50ms, or a second ESC)
    and ``None`` once an escape sequence has been drained whole.
    """
    nxt = _read_byte(fd, ESC_TIMEOUT)
    if nxt is None or nxt == _ESC_BYTE:
        return ESC
    if nxt == _CSI_INTRO:
        # parameter/intermediate bytes 0x20-0x3f, final byte 0x40-0x7e
        while True:
            byte = _read_byte(fd, ESC_TIMEOUT)
            if byte is None or 0x40 <= byte <= 0x7E:
                return None
    if nxt == _SS3_INTRO:
        _read_byte(fd, ESC_TIMEOUT)
        return None
    return None  # Alt+key: ESC followed by a plain byte


def read_choice_key(fd: int) -> str:
    """Block for one keypress on ``fd`` (already in raw/cbreak mode).

    Returns ``'esc'`` for a lone Esc, ``''`` for a drained escape sequence
    (arrow/function key) or EOF, and the decoded character otherwise.
    """
    byte = _read_byte(fd, None)
    if byte is None:
        return ""
    if byte == _ESC_BYTE:
        return _after_esc(fd) or ""
    return bytes([byte]).decode("utf-8", errors="replace")


class KeyWatcher:
    """Hold ``fd`` in cbreak for a stream and report lone Esc presses."""

    def __init__(self, fd: int = 0, env: Mapping[str, str] | None = None) -> None:
        self.fd = fd
        self._env = os.environ if env is None else env
        self._saved: list | None = None
        self._prev_handlers: dict[int, object] = {}
        self.active = False

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> KeyWatcher:
        if not _enabled(self.fd, self._env):
            return self
        try:
            import termios

            saved = termios.tcgetattr(self.fd)
            attrs = termios.tcgetattr(self.fd)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            attrs[3] |= termios.ISIG
            attrs[6][termios.VMIN] = 1
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        except Exception:  # noqa: BLE001 - not a usable terminal: stay a no-op
            return self
        self._saved = saved
        self.active = True
        self._install_handlers()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._restore_termios()
        self._restore_handlers()

    # -- polling --------------------------------------------------------------
    def poll(self, timeout: float) -> str | None:
        """Wait up to ``timeout`` seconds; ``'esc'`` on a lone Esc, else ``None``.

        Every byte that arrives in the window is consumed: escape sequences
        are drained whole and all other bytes are discarded.
        """
        if not self.active:  # no-op watcher: still honour the wait, never spin a loop
            if timeout > 0:
                time.sleep(timeout)
            return None
        deadline = time.monotonic() + timeout
        while self.active:
            remaining = deadline - time.monotonic()
            try:
                ready = select.select([self.fd], [], [], max(remaining, 0))[0]
            except (OSError, ValueError):
                return None
            if not ready:
                return None
            byte = _read_byte(self.fd, 0)
            if byte is None:  # EOF or a hung-up tty: never spin
                return None
            if byte == _ESC_BYTE and _after_esc(self.fd) == ESC:
                return ESC
        return None

    # -- restore paths --------------------------------------------------------
    def _restore_termios(self, when: str = "TCSADRAIN") -> None:
        saved, self._saved = self._saved, None
        self.active = False
        if saved is None:
            return
        try:
            import termios

            termios.tcsetattr(self.fd, getattr(termios, when), saved)
        except Exception:  # noqa: BLE001 - teardown must never raise at the prompt
            return

    def _install_handlers(self) -> None:
        for sig in _SIGNALS:
            try:
                self._prev_handlers[sig] = signal.signal(sig, self._on_signal)
            except (ValueError, OSError):  # not the main thread: exit path only
                continue

    def _restore_handlers(self) -> None:
        prev_handlers, self._prev_handlers = self._prev_handlers, {}
        for sig, prev in prev_handlers.items():
            try:
                if signal.getsignal(sig) == self._on_signal:
                    signal.signal(sig, prev)  # type: ignore[arg-type]
            except (ValueError, OSError, TypeError):
                continue

    def _on_signal(self, signum: int, frame: object) -> None:
        # TCSANOW: a hung-up tty must not make the handler wait on output drain
        self._restore_termios("TCSANOW")
        prev = self._prev_handlers.get(signum, signal.SIG_DFL)
        self._restore_handlers()
        if prev is None:
            prev = signal.SIG_DFL
        if callable(prev):
            prev(signum, frame)
            return
        if prev == signal.SIG_IGN:
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
