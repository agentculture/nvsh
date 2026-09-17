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
* holds the fd in cbreak with ``ISIG`` *off*, so Ctrl+C arrives as the byte
  ``0x03`` and answers the prompt as "stop" instead of killing the client;
* drains CSI/SS3 escape sequences whole through :func:`nvsh.keys._after_esc`,
  so an arrow key neither answers the prompt nor leaves bytes behind;
* waits on ``select()`` against a *monotonic deadline*, so ignored keys
  cannot extend the window;
* never spins: EOF or a hung-up tty returns :data:`EOF` at once;
* restores termios on every exit path, including an exception.

Stdlib only, and safe on a platform without ``termios``: the import is
guarded the way ``nvsh/panel.py`` guards it, and a fd that is not a terminal
degrades to a plain ``select()``-driven byte read.
"""

from __future__ import annotations

import select
import time
from collections.abc import Iterable

# ESC is re-exported: callers of this module need only one import, and the
# value stays identical to nvsh.keys.ESC.
from nvsh.keys import ESC, _after_esc, _read_byte

__all__ = ["ESC", "INTERRUPT", "EOF", "TIMEOUT", "read_choice"]

#: Ctrl+C typed at the prompt (byte ``0x03``; cbreak here keeps ``ISIG`` off).
INTERRUPT = "interrupt"

#: End of input: a closed or hung-up terminal. Means "keep going", never stop.
EOF = "eof"

#: Nobody answered within ``timeout`` seconds.
TIMEOUT = "timeout"

_ESC_BYTE = 0x1B
_CTRL_C_BYTE = 0x03


def read_choice(fd: int, keys: Iterable[str], timeout: float | None = None) -> str:
    """Wait up to ``timeout`` seconds for one of ``keys`` on ``fd``.

    Returns the matched key (lower-cased, so ``S`` answers ``s``), or one of
    :data:`ESC`, :data:`INTERRUPT`, :data:`EOF` and :data:`TIMEOUT`. Keys
    that are not in ``keys`` are consumed and ignored, and the deadline keeps
    running while they are. ``timeout=None`` waits indefinitely.

    Pending input is discarded before the first read, so only keys typed
    *after* the prompt is on screen can answer it.
    """
    allowed = frozenset(keys)
    saved = _enter_cbreak(fd)
    try:
        _flush_pending(fd)
        return _wait_for_choice(fd, allowed, timeout)
    finally:
        _restore(fd, saved)


def _wait_for_choice(fd: int, allowed: frozenset[str], timeout: float | None) -> str:
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        remaining = None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
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
        if byte == _CTRL_C_BYTE:
            return INTERRUPT
        if byte == _ESC_BYTE:
            if _after_esc(fd) == ESC:
                return ESC
            continue  # CSI/SS3 drained whole: neither answers nor lingers
        char = bytes([byte]).decode("utf-8", errors="replace")
        if char in allowed:
            return char
        if char.lower() in allowed:
            return char.lower()


# -- terminal helpers: termios only, no capability lookup ---------------------


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


def _restore(fd: int, saved) -> None:
    if saved is None:
        return
    termios = _termios()
    if termios is None:  # pragma: no cover - POSIX only
        return
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    except Exception:  # noqa: BLE001 - teardown must never raise at the prompt
        return
