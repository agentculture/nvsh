#!/usr/bin/env python3
"""Record and replay a terminal session as an asciicast v2 file.

A stdlib stand-in for ``asciinema``, which is not installed on any
of the three verification machines (Jetson AGX Orin, Jetson Thor, DGX Spark)
and cannot be installed on an air-gapped box. The output is the real
asciicast v2 format, so ``asciinema play`` reads these files too; ``play``
here exists so a machine with no asciinema can still watch a recording.

    record-cast.py record OUT.cast -- bash -i
    record-cast.py record OUT.cast --feed 'ls /nope\\r|60' -- bash -i
    record-cast.py play OUT.cast --speed 2

``--feed`` drives the session without a human at the keyboard (the same
``keys|seconds`` steps the verification drivers use: ``\\r`` Enter, ``\\t``
Tab, ``^G`` Ctrl+G, ``^C`` Ctrl+C, ``^[`` Esc). Without it, stdin is passed
through and the recording ends when the command exits.

``--replace-from FILE`` reads ``OLD=NEW`` lines and rewrites every recorded
chunk before it is written, which is how a recording made against a private
endpoint is scrubbed without the secret ever reaching the ``.cast`` file or
this script's argv.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import select
import shlex
import signal
import struct
import sys
import termios
import time
import tty
from pathlib import Path

#: Env vars (or, with a trailing ``*``, prefixes) a ``--clean-env`` child
#: inherits from the recording operator's environment. Everything else is
#: dropped -- including every ``NVSH_*`` behaviour knob and every ``XDG_*``
#: directory, which a caller that wants them passes explicitly with
#: ``--env`` -- so two runs on different machines/shells start from the
#: same env.
_CLEAN_ENV_ALLOW = ("PATH", "TERM", "HOME", "LANG", "LC_*", "COLUMNS", "LINES")

#: What ``--feed`` accepts as key escapes.
_KEY_ESCAPES = (
    ("\\r", "\r"),
    ("\\n", "\n"),
    ("\\t", "\t"),
    ("^G", "\x07"),
    ("^C", "\x03"),
    ("^D", "\x04"),
    ("^[", "\x1b"),
)


def _decode_keys(raw: str) -> str:
    for token, char in _KEY_ESCAPES:
        raw = raw.replace(token, char)
    return raw


def _load_replacements(path: Path | None) -> list[tuple[bytes, bytes]]:
    """``OLD=NEW`` lines -> byte pairs applied to every recorded chunk."""
    if path is None:
        return []
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        old, _, new = line.partition("=")
        if old:
            pairs.append((old.encode(), new.encode()))
    return pairs


def _scrub(data: bytes, pairs: list[tuple[bytes, bytes]]) -> bytes:
    for old, new in pairs:
        data = data.replace(old, new)
    return data


def _holdback(data: bytes, pairs: list[tuple[bytes, bytes]]) -> int:
    """How many trailing bytes of *data* could be the start of a token.

    A pty read can split a hostname or an address across two chunks, and a
    per-chunk replace would then miss it. Any suffix of *data* that is a
    proper prefix of some token is held back and re-scanned together with
    the next chunk (see :class:`_Scrubber`).
    """
    longest = 0
    for old, _new in pairs:
        for size in range(min(len(old) - 1, len(data)), 0, -1):
            if data.endswith(old[:size]):
                longest = max(longest, size)
                break
    return longest


class _Scrubber:
    """Streaming ``OLD=NEW`` replacement that survives chunk boundaries."""

    def __init__(self, pairs: list[tuple[bytes, bytes]]) -> None:
        self._pairs = pairs
        self._pending = b""

    def feed(self, chunk: bytes) -> bytes:
        if not self._pairs:
            return chunk
        data = _scrub(self._pending + chunk, self._pairs)
        keep = _holdback(data, self._pairs)
        self._pending = data[len(data) - keep :] if keep else b""
        return data[: len(data) - keep] if keep else data

    def flush(self) -> bytes:
        data, self._pending = _scrub(self._pending, self._pairs), b""
        return data


def _window_size(default_cols: int, default_rows: int) -> tuple[int, int]:
    try:
        size = os.get_terminal_size(sys.stdin.fileno())
    except (OSError, ValueError):
        return default_cols, default_rows
    return size.columns or default_cols, size.lines or default_rows


def _set_window_size(fd: int, cols: int, rows: int) -> None:
    """Set the pty's window size via ``TIOCSWINSZ`` (``struct winsize``)."""
    packed = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)


def _build_env(base: dict[str, str], clean: bool, overrides: list[str] | None) -> dict[str, str]:
    """The child's environment: ``base`` (or an allowlisted subset of it if
    ``clean`` is set) plus ``NAME=VALUE`` items from ``overrides``."""
    if clean:
        env = {}
        for key, value in base.items():
            for allowed in _CLEAN_ENV_ALLOW:
                if allowed.endswith("*"):
                    if key.startswith(allowed[:-1]):
                        env[key] = value
                        break
                elif key == allowed:
                    env[key] = value
                    break
    else:
        env = dict(base)
    for item in overrides or []:
        key, _, value = item.partition("=")
        if key:
            env[key] = value
    return env


def record(args: argparse.Namespace) -> int:
    """Run ``args.command`` on a pty, writing an asciicast v2 file."""
    cols, rows = args.cols, args.rows
    if not args.feed:
        cols, rows = _window_size(cols, rows)
    replacements = _load_replacements(Path(args.replace_from) if args.replace_from else None)

    env = _build_env(dict(os.environ), args.clean_env, args.env)
    env["COLUMNS"] = str(cols)
    env["LINES"] = str(rows)

    out = Path(args.out).open("w", encoding="utf-8")
    header_timestamp = int(time.time()) if args.timestamp is None else args.timestamp
    header = {
        "version": 2,
        "width": cols,
        "height": rows,
        "timestamp": header_timestamp,
        "env": {"SHELL": env.get("SHELL", ""), "TERM": env.get("TERM", "")},
    }
    if args.title:
        header["title"] = args.title
    out.write(json.dumps(header) + "\n")
    out.flush()

    pid, fd = pty.fork()
    if pid == 0:  # child
        try:
            # The pty's controlling tty is already fd 0/1/2 in the child
            # (that's what pty.fork() sets up); set its window size before
            # exec so the child's first read of it (``stty size``, a shell
            # prompt redraw, curses init) sees --cols/--rows rather than
            # whatever size the pty was allocated with.
            _set_window_size(0, cols, rows)
            os.execvpe(args.command[0], list(args.command), env)
        except OSError:
            pass
        os._exit(127)

    started = time.monotonic()
    saved = None
    passthrough = not args.feed and sys.stdin.isatty()
    if passthrough:
        saved = termios.tcgetattr(sys.stdin.fileno())
        tty.setraw(sys.stdin.fileno())

    scrubber = _Scrubber(replacements)

    def emit(data: bytes, *, final: bool = False) -> None:
        data = scrubber.feed(data) + (scrubber.flush() if final else b"")
        if not data:
            return
        stamp = round(time.monotonic() - started, 6)
        out.write(json.dumps([stamp, "o", data.decode("utf-8", "replace")]) + "\n")
        out.flush()

    def pump(seconds: float) -> bool:
        """Copy the pty's output for ``seconds``; False once it closes."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            sources = [fd, sys.stdin.fileno()] if passthrough else [fd]
            readable, _, _ = select.select(sources, [], [], 0.05)
            if passthrough and sys.stdin.fileno() in readable:
                os.write(fd, os.read(sys.stdin.fileno(), 4096))
            if fd in readable:
                try:
                    chunk = os.read(fd, 65536)
                except OSError as exc:
                    print(f"record-cast: pty closed ({exc})", file=sys.stderr)
                    return False
                if not chunk:
                    print("record-cast: pty reached EOF", file=sys.stderr)
                    return False
                emit(chunk)
        return True

    try:
        pump(args.settle)
        for step in args.feed or []:
            keys, _, wait = step.rpartition("|")
            for char in _decode_keys(keys):
                os.write(fd, char.encode())
                time.sleep(args.key_delay)
            if not pump(float(wait or 1)):
                break
        if not args.feed:
            while pump(3600.0):
                pass
        else:
            os.write(fd, b"exit\r")
            pump(args.settle)
    finally:
        if saved is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except OSError:
            pass
        emit(b"", final=True)  # the scrubber's held-back tail
        out.close()
    return 0


def play(args: argparse.Namespace) -> int:
    """Replay a ``.cast`` file onto stdout, honouring its timings."""
    lines = Path(args.file).read_text(encoding="utf-8").splitlines()
    if not lines:
        print(f"empty cast file: {args.file}", file=sys.stderr)
        return 1
    previous = 0.0
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            stamp, kind, data = json.loads(line)
        except (ValueError, TypeError):
            continue
        if kind != "o":
            continue
        delay = min(max(stamp - previous, 0.0) / args.speed, args.idle_limit)
        time.sleep(delay)
        previous = stamp
        sys.stdout.write(data)
        sys.stdout.flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record-cast.py", description="Record/replay a terminal session as asciicast v2."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    rec = sub.add_parser("record", help="record a session")
    rec.add_argument("out", help="output .cast path")
    rec.add_argument("--feed", action="append", help="scripted step: 'keys|seconds'")
    rec.add_argument("--env", action="append", help="extra child env: NAME=VALUE")
    rec.add_argument(
        "--clean-env",
        action="store_true",
        help="start the child from an allowlist (PATH, TERM, HOME, LANG, "
        "XDG_*, NVSH_*, COLUMNS, LINES) instead of the full parent env",
    )
    rec.add_argument("--replace-from", help="file of OLD=NEW scrub rules")
    rec.add_argument("--title", help="asciicast title")
    rec.add_argument("--cols", type=int, default=100)
    rec.add_argument("--rows", type=int, default=34)
    rec.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="pin the header's unix timestamp (default: current time)",
    )
    rec.add_argument("--settle", type=float, default=1.5, help="seconds to pump before/after")
    rec.add_argument("--key-delay", type=float, default=0.03, help="seconds between keystrokes")
    rec.add_argument("command", nargs="*", help="command to run (after --)")
    rec.set_defaults(func=record)

    rep = sub.add_parser("play", help="replay a .cast file")
    rep.add_argument("file")
    rep.add_argument("--speed", type=float, default=1.0)
    rep.add_argument("--idle-limit", type=float, default=2.0)
    rep.set_defaults(func=play)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Everything after the first bare "--" is the command, untouched by
    # argparse (a recorded command has options of its own).
    command: list[str] = []
    if "--" in raw:
        cut = raw.index("--")
        raw, command = raw[:cut], raw[cut + 1 :]
    parser = build_parser()
    args = parser.parse_args(raw)
    if args.mode == "record":
        command = command or list(args.command or [])
        if not command:
            parser.error("record needs a command: record-cast.py record OUT.cast -- bash -i")
        args.command = command
        print(f"recording {shlex.join(command)} -> {args.out}", file=sys.stderr)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
