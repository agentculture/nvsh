"""A timed single-key read for nvsh's inline prompts.

The stop-choice prompt (``[t] steer  [s] stop  [Esc] keep going``) needs
something neither existing reader offers: :func:`nvsh.panel._read_raw_key`
blocks forever, and :meth:`nvsh.keys.KeyWatcher.poll` is timed but throws
away every byte except a lone Esc. This module adds the missing piece
*beside* them -- ``nvsh/keys.py`` keeps no functional diff (spec boundary
c7/h7) and this module imports its helpers rather than re-implementing Esc
disambiguation.

:func:`read_choice` therefore:

* **discards typeahead first** (``termios.tcflush(TCIFLUSH)``), so a
  double-tapped Esc or a key repeat that arrives before the prompt is drawn
  opens the prompt and leaves it open rather than instantly answering it
  (c33/h24);
* **draws the prompt itself**, through the ``before_read`` callback, in the
  one window where that is safe: after the flush and before the first
  ``select()``. The legend therefore reaches the screen while the fd is
  already in cbreak, so a key typed the instant the operator sees it is
  never echoed, never flushed and never lost -- the caller must not print
  the legend before calling, or the flush would eat the answer to it;
* holds the fd in cbreak with ``ISIG`` *off*, so Ctrl+C arrives as the byte
  ``0x03`` and answers the prompt as "stop" instead of killing the client;
* drains CSI/SS3 escape sequences whole through :func:`nvsh.keys._after_esc`,
  so an arrow key neither answers the prompt nor leaves bytes behind;
* waits on ``select()`` against a *monotonic deadline*, so ignored keys
  cannot extend the window;
* never spins: EOF or a hung-up tty returns :data:`EOF` at once;
* restores termios on every exit path, including an exception **and
  SIGHUP/SIGTERM** (h23). The signal half is not re-implemented here: the
  private :class:`_PromptTerminal` inherits :class:`nvsh.keys.KeyWatcher`'s
  handler install/restore/delegate machinery and only replaces how the
  terminal is entered (``ISIG`` off, no ``TERM``/tty gate), which keeps
  ``nvsh/keys.py`` at no functional diff (c7/h7). Off the main thread
  ``signal.signal`` is refused; the inherited installer swallows that, so a
  ``read_choice`` on a worker thread reads keys without the safety net
  rather than raising.

Stdlib only, and safe on a platform without ``termios``: the import is
guarded the way ``nvsh/panel.py`` guards it, and a fd that is not a terminal
degrades to a plain ``select()``-driven byte read.
"""

from __future__ import annotations

import select
import time
from collections.abc import Callable, Iterable

# ESC is re-exported: callers of this module need only one import, and the
# value stays identical to nvsh.keys.ESC.
from nvsh.keys import ESC, KeyWatcher, _after_esc, _read_byte

__all__ = ["ESC", "INTERRUPT", "EOF", "TIMEOUT", "read_choice"]

#: Ctrl+C typed at the prompt (byte ``0x03``; cbreak here keeps ``ISIG`` off).
INTERRUPT = "interrupt"

#: End of input: a closed or hung-up terminal. Means "keep going", never stop.
EOF = "eof"

#: Nobody answered within ``timeout`` seconds.
TIMEOUT = "timeout"

_ESC_BYTE = 0x1B
_CTRL_C_BYTE = 0x03


def read_choice(
    fd: int,
    keys: Iterable[str],
    timeout: float | None = None,
    before_read: Callable[[], object] | None = None,
) -> str:
    """Wait up to ``timeout`` seconds for one of ``keys`` on ``fd``.

    Returns the matched key (lower-cased, so ``S`` answers ``s``), or one of
    :data:`ESC`, :data:`INTERRUPT`, :data:`EOF` and :data:`TIMEOUT`. Keys
    that are not in ``keys`` are consumed and ignored, and the deadline keeps
    running while they are. ``timeout=None`` waits indefinitely.

    Pending input is discarded before the first read, so only keys typed
    *after* the prompt is on screen can answer it (c33). ``before_read`` is
    how the prompt gets on screen: it is called once, after that flush and
    before the first wait, and anything it raises propagates with the
    terminal already restored. Every key typed from the moment it returns is
    kept, so a caller must draw its legend there rather than before the
    call.
    """
    allowed = frozenset(keys)
    with _PromptTerminal(fd):
        _flush_pending(fd)
        if before_read is not None:
            before_read()
        return _wait_for_choice(fd, allowed, timeout)


def _wait_for_choice(fd: int, allowed: frozenset[str], timeout: float | None) -> str:
    """Read until a key answers, the deadline passes or the fd ends."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        byte = _next_byte(fd, deadline)
        if isinstance(byte, str):  # TIMEOUT or EOF, never a key
            return byte
        answer = _classify(fd, byte, allowed)
        if answer is not None:
            return answer


def _next_byte(fd: int, deadline: float | None) -> int | str:
    """One byte, or :data:`TIMEOUT`/:data:`EOF` when none can come."""
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        return TIMEOUT
    try:
        ready = select.select([fd], [], [], remaining)[0]
    except (OSError, ValueError):  # closed or invalid fd: nothing to read
        return EOF
    if not ready:
        return TIMEOUT
    byte = _read_byte(fd, 0)
    if byte is None:  # EOF or a hung-up tty: return at once, never spin
        return EOF
    return byte


def _classify(fd: int, byte: int, allowed: frozenset[str]) -> str | None:
    """What ``byte`` answers, or ``None`` when it answers nothing."""
    if byte == _CTRL_C_BYTE:
        return INTERRUPT
    if byte == _ESC_BYTE:
        # A CSI/SS3 sequence is drained whole: it neither answers nor lingers.
        return ESC if _after_esc(fd) == ESC else None
    char = bytes([byte]).decode("utf-8", errors="replace")
    if char in allowed:
        return char
    if char.lower() in allowed:
        return char.lower()
    return None


# -- terminal helpers: termios only, no capability lookup ---------------------


class _PromptTerminal(KeyWatcher):
    """The prompt's hold on ``fd``: cbreak with ``ISIG`` off, safely released.

    Everything but the entry is :class:`nvsh.keys.KeyWatcher`'s: ``__exit__``
    restores termios and the previous SIGHUP/SIGTERM dispositions, and
    ``_on_signal`` restores termios with ``TCSANOW`` (a hung-up tty must not
    make a handler wait on an output drain) before delegating to whatever was
    installed before -- so a SIGHUP or SIGTERM during the 30 s prompt still
    ends the process the way it would have, with the operator's terminal
    already given back (h23).

    The two differences from the watcher are deliberate: ``ISIG`` is off (the
    prompt wants Ctrl+C as the byte ``0x03``, not a signal), and there is no
    ``TERM``/``isatty`` gate, because ``read_choice`` must also work on a
    plain pipe -- where ``_enter_cbreak`` returns ``None`` and this becomes
    the no-op the watcher would have been.
    """

    def __enter__(self) -> "_PromptTerminal":
        saved = _enter_cbreak(self.fd)
        if saved is None:  # not a terminal: read the bytes, touch nothing
            return self
        self._saved = saved
        self.active = True
        self._install_handlers()
        return self


def _termios():
    try:
        import termios
    except ImportError:  # pragma: no cover - POSIX only
        return None
    return termios


def _enter_cbreak(fd: int):
    """Non-canonical, no echo, ``ISIG`` off. Returns the saved attrs or ``None``."""
    termios = _termios()
    if termios is None:  # pragma: no cover - POSIX only
        return None
    try:
        saved = termios.tcgetattr(fd)
        attrs = termios.tcgetattr(fd)
        attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:  # noqa: BLE001 - not a terminal: read the bytes anyway
        return None
    return saved


def _flush_pending(fd: int) -> None:
    """Drop typeahead typed before the prompt was drawn (c33)."""
    termios = _termios()
    if termios is None:  # pragma: no cover - POSIX only
        return
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
    except Exception:  # noqa: BLE001 - a pipe has nothing to flush
        return
