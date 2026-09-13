#!/usr/bin/env python3
"""Record and replay a terminal session as an asciicast v2 file.

A ~200-line stdlib stand-in for ``asciinema``, which is not installed on any
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
import json
import os
import pty
import select
import shlex
import signal
import sys
import termios
import time
import tty
from pathlib import Path

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


def _window_size(default_cols: int, default_rows: int) -> tuple[int, int]:
    try:
        size = os.get_terminal_size(sys.stdin.fileno())
    except (OSError, ValueError):
        return default_cols, default_rows
    return size.columns or default_cols, size.lines or default_rows


def record(args: argparse.Namespace) -> int:
    """Run ``args.command`` on a pty, writing an asciicast v2 file."""
    cols, rows = args.cols, args.rows
    if not args.feed:
        cols, rows = _window_size(cols, rows)
    replacements = _load_replacements(Path(args.replace_from) if args.replace_from else None)

    env = dict(os.environ)
    env["COLUMNS"] = str(cols)
    env["LINES"] = str(rows)
    for item in args.env or []:
        key, _, value = item.partition("=")
        if key:
            env[key] = value

    out = Path(args.out).open("w", encoding="utf-8")
    header = {
        "version": 2,
        "width": cols,
        "height": rows,
        "timestamp": int(time.time()),
        "env": {"SHELL": env.get("SHELL", ""), "TERM": env.get("TERM", "")},
    }
    if args.title:
        header["title"] = args.title
    out.write(json.dumps(header) + "\n")
    out.flush()

    pid, fd = pty.fork()
    if pid == 0:  # child
        try:
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

    def emit(data: bytes) -> None:
        data = _scrub(data, replacements)
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
    rec.add_argument("--replace-from", help="file of OLD=NEW scrub rules")
    rec.add_argument("--title", help="asciicast title")
    rec.add_argument("--cols", type=int, default=100)
    rec.add_argument("--rows", type=int, default=34)
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
