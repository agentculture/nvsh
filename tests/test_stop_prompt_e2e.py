"""End-to-end: every outcome of the stop-choice prompt, from one first press.

Spec ``docs/specs/2026-09-17-stop-choice-prompt.md`` promises (honesty
condition h1) that *all four* outcomes -- steer, stop, keep going and the
timeout -- are reachable from a single first Ctrl+C on a pty, and (c1/h1)
that **nothing reaches the harness between the press and the operator's
key**. tests/test_panel_stop.py proves that against a stubbed panel and
tests/test_client_stop.py against stub transports; this file proves it with
the real client process, the real daemon, the real adapters and a real
terminal, the way tests/test_agent_stop.py proves the two-press stop.

The rig is tests/test_agent_stop.py's: the same :class:`Terminal` (an
interactive ``bash`` whose controlling tty is a pty), the same
``PROMPT``/legend matchers and the same single-write :func:`_type_choice`,
which needs no retry because :func:`nvsh.promptkeys.read_choice` discards
typeahead *before* it draws the legend. Only the *harnesses*
differ, because that test's fakes deliberately stall a turn forever and
these cases need a turn that also **ends**:

* ``openai-compat`` talks to :class:`_ScriptedServer` below -- an SSE server
  in this file that records every request body, reports when a stream was
  dropped, and releases the rest of a turn when the test says so. It is the
  family the plan singles out for stop-and-correct (no mid-turn channel) and
  the only one whose wire the test can hold open and then finish at will.
* ``claude`` is tests/fakes/claude, re-reading ``$NVSH_FAKE_EVENTS`` on
  every process: the subprocess family for the same stop-and-correct case.
* ``pi`` is :data:`_PI_STEER_FAKE`, a purpose-built ``pi --mode rpc`` written
  into the rig's ``bin/``. No committed fake can do what the steer case
  needs -- tests/fakes/pi's ``NVSH_FAKE_STALL_TURN`` stalls *every* prompt,
  including the steering one, and tests/fakes/pi_scripted replays its script
  only on the first prompt -- so neither can show a turn that stalls, takes a
  mid-turn steer and then finishes. This one does exactly that and nothing
  else, and records every command it received.

"No adapter call between the press and the key" is asserted from what the
harness recorded (the pi command log, the server's request/drop log, the
claude pid file), never inferred from the panel's output, and each outcome's
audit.jsonl lines are checked as well.

The one deliberate exception to "no sleeps" is the pause in the last test,
which proves spec assumption c35: a client that reads nothing from the
daemon socket for 5s loses no events. There the wait *is* the subject, and
the test measures how many bytes the harness got out while it lasted --
452306 against a 212992-byte socket buffer, when this was written -- rather
than assuming the buffer was exceeded.

Timings are deadlines, never bare sleeps (plan risk r5): these run under
``pytest -n auto`` on loaded boxes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from nvsh import client_transport
from nvsh.client import STOP_AND_CORRECT_LABEL, STOPPED_NOTICE
from nvsh.panel import STEER_LABEL, STOPPING_TEXT
from tests.test_agent_stop import pytestmark  # noqa: F401 - needs bash and a pty, same as the rig
from tests.test_agent_stop import (  # the rig, reused verbatim
    PAUSED_TAIL,
    PROMPT,
    SETUP_TIMEOUT,
    Terminal,
    _kill_daemon,
    _kill_quietly,
    _legend_in,
    _pid_alive,
    _read_pid_file,
    _type_choice,
)

FAKES_DIR = Path(__file__).parent / "fakes"

#: The request the operator's shell makes in every case here.
QUESTION = "why did the build fail?"

#: What the operator types at the ``nvsh> `` correction line.
CORRECTION = "look at the linker, not the compiler"

#: Generous bounds for everything that is not one of the spec's thresholds.
PROMPT_WITHIN = 1.0
FINISH_WITHIN = 25.0


# -- a scripted, recording SSE server (openai-compat) ---------------------------


@dataclass
class _Script:
    """What the server does for one incoming ``/chat/completions`` request."""

    before: list[str] = field(default_factory=list)
    #: Held here (writing SSE keep-alive comments) until the test releases it.
    release: threading.Event | None = None
    after: list[str] = field(default_factory=list)
    finish: bool = True


def _sse(text: str) -> bytes:
    chunk = json.dumps({"choices": [{"delta": {"content": text}}]})
    return f"data: {chunk}\n\n".encode()


class _ScriptedHandler(BaseHTTPRequestHandler):
    server_version = "NvshStopPromptE2E/1.0"

    def log_message(self, *_args):  # noqa: D401 - silence test server logging
        pass

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        script = self.server.take(body)  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.flush()
        if not self._send(script.before):
            return
        if script.release is not None and not self._hold(script.release):
            return
        if not self._send(script.after):
            return
        if script.finish:
            self._send_raw(b"data: [DONE]\n\n")

    def _send(self, chunks: list[str]) -> bool:
        for text in chunks:
            if not self._send_raw(_sse(text)):
                return False
        return True

    def _hold(self, release: threading.Event) -> bool:
        closing = self.server.closing  # type: ignore[attr-defined]
        while not release.is_set() and not closing.is_set():
            if not self._send_raw(b": keep-alive\n\n"):
                return False
            release.wait(0.05)
        return not closing.is_set()

    def _send_raw(self, payload: bytes) -> bool:
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except OSError:
            self.server.dropped.append(time.monotonic())  # type: ignore[attr-defined]
            return False
        self.server.sent_bytes[-1] += len(payload)  # type: ignore[attr-defined]
        return True


class _ScriptedServer(ThreadingHTTPServer):
    """Records every request; answers the n-th one with the n-th script."""

    daemon_threads = True

    def configure(self, scripts: list[_Script]) -> None:
        self.scripts = scripts
        self.requests: list[dict] = []
        self.dropped: list[float] = []
        self.sent_bytes: list[int] = [0]
        self.closing = threading.Event()
        self._lock = threading.Lock()

    def take(self, body: str) -> _Script:
        with self._lock:
            index = len(self.requests)
            self.requests.append({"body": body, "at": time.monotonic()})
            self.sent_bytes.append(0)
            return self.scripts[min(index, len(self.scripts) - 1)]

    def prompt(self, index: int) -> str:
        """The user text of the index-th request, as the adapter sent it."""
        payload = json.loads(self.requests[index]["body"])
        return "\n".join(
            str(message.get("content", ""))
            for message in payload["messages"]
            if message.get("role") == "user"
        )


# -- a pi that stalls, takes one mid-turn steer, then finishes -------------------

_PI_STEER_FAKE = '''#!/usr/bin/env python3
"""A ``pi --mode rpc`` whose turn stalls until a mid-turn steer arrives.

Written by tests/test_stop_prompt_e2e.py because no committed fake can do
this: tests/fakes/pi's NVSH_FAKE_STALL_TURN stalls the steering prompt too,
and tests/fakes/pi_scripted replays its script only on the *first* prompt.
Every command received is appended to $NVSH_E2E_PI_LOG, so the test can
assert from the harness's own record that nothing reached it between the
operator's press and the operator's key.
"""
import json
import os
import sys
import uuid
from pathlib import Path


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()


def log(cmd):
    with open(os.environ["NVSH_E2E_PI_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(cmd) + "\\n")


def session_file():
    argv = sys.argv
    if "--session-dir" not in argv:
        return ""
    directory = Path(argv[argv.index("--session-dir") + 1])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (str(uuid.uuid4()) + ".jsonl")
    path.write_text("", encoding="utf-8")
    return str(path)


def delta(text):
    emit({"type": "message_update",
          "assistantMessageEvent": {"type": "text_delta", "delta": text}})


def main():
    state = {"file": session_file()}
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue
        log(cmd)
        kind = cmd.get("type")
        if kind == "prompt":
            emit({"id": cmd.get("id"), "type": "response",
                  "command": "prompt", "success": True})
            if cmd.get("streamingBehavior") == "steer":
                delta(" STEERED-WITH[" + str(cmd.get("message", "")) + "]")
                emit({"type": "agent_end", "messages": [], "willRetry": False})
            else:
                delta("FIRST-TURN-TEXT")
        elif kind == "abort":
            emit({"id": cmd.get("id"), "type": "response",
                  "command": "abort", "success": True})
            emit({"type": "agent_end", "messages": [],
                  "willRetry": False, "aborted": True})
        else:
            data = {}
            if kind == "get_state":
                data = {"isStreaming": False, "sessionFile": state["file"]}
            elif kind in ("new_session", "switch_session"):
                if kind == "new_session":
                    state["file"] = session_file()
                data = {"cancelled": False}
            emit({"id": cmd.get("id"), "type": "response",
                  "command": kind, "success": True, "data": data})


main()
'''


# -- the rig --------------------------------------------------------------------


@dataclass
class Rig:
    terminal: Terminal
    env: dict[str, str]
    tmp: Path
    state: Path
    server: _ScriptedServer | None = None
    pi_log: Path | None = None
    claude_events: Path | None = None
    pid_file: Path | None = None

    def audit(self) -> list[dict]:
        path = self.state / "nvsh" / "audit.jsonl"
        if not path.exists():
            return []
        lines = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                lines.append(json.loads(raw))
        return [line for line in lines if line.get("event") == "stop"]

    def stops(self, kind: str) -> list[dict]:
        return [line for line in self.audit() if line.get("kind") == kind]

    def daemon_log(self) -> str:
        path = self.state / "nvsh" / "daemon.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def pi_commands(self) -> list[dict]:
        if self.pi_log is None or not self.pi_log.exists():
            return []
        return [
            json.loads(raw)
            for raw in self.pi_log.read_text(encoding="utf-8").splitlines()
            if raw.strip()
        ]


def _config_toml(adapter: str, base_url: str) -> str:
    lines = ["[aliases]", f'default = "{adapter}"', ""]
    if base_url:
        lines += [f'[agents."{adapter}"]', f'base_url = "{base_url}"', ""]
    return "\n".join(lines)


@pytest.fixture
def rig_factory(tmp_path):
    """The tests/test_agent_stop.py rig, with this file's harnesses wired in."""
    made: list[Rig] = []
    # A short runtime dir: the daemon's unix socket path must fit in 108 bytes.
    runtime = Path(tempfile.mkdtemp(prefix="nvsh-t9-"))

    def build(adapter: str, *, scripts: list[_Script] | None = None) -> Rig:
        room = tmp_path / f"rig{len(made)}"
        bindir = room / "bin"
        bindir.mkdir(parents=True)
        home = room / "home"
        home.mkdir()
        config_home = room / "config"
        (config_home / "nvsh").mkdir(parents=True)
        state = room / "state"

        rig = Rig(terminal=None, env={}, tmp=room, state=state)  # type: ignore[arg-type]
        extra: dict[str, str] = {}
        base_url = ""
        if adapter == "openai-compat":
            server = _ScriptedServer(("127.0.0.1", 0), _ScriptedHandler)
            server.configure(scripts or [_Script()])
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            rig.server = server
        elif adapter == "pi":
            script = bindir / "pi"
            script.write_text(_PI_STEER_FAKE, encoding="utf-8")
            script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            rig.pi_log = room / "pi-commands.jsonl"
            extra["NVSH_E2E_PI_LOG"] = str(rig.pi_log)
        elif adapter == "claude":
            (bindir / "claude").symlink_to(FAKES_DIR / "claude")
            rig.claude_events = room / "claude-events.json"
            rig.pid_file = room / "pids.json"
            extra.update(
                {
                    "NVSH_FAKE_EVENTS": str(rig.claude_events),
                    "NVSH_FAKE_IGNORE_CANCEL": "1",
                    "NVSH_FAKE_GRANDCHILD": "1",
                    "NVSH_FAKE_PID_FILE": str(rig.pid_file),
                }
            )
        else:  # pragma: no cover - a typo in a test, not a behaviour
            raise AssertionError(f"unknown adapter for this rig: {adapter}")

        (config_home / "nvsh" / "config.toml").write_text(
            _config_toml(adapter, base_url), encoding="utf-8"
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("NVSH_", "PYTEST_", "CLAUDE_CODE_")) and key != "CLAUDECODE"
        }
        env.update(
            {
                "PATH": str(bindir) + os.pathsep + env.get("PATH", ""),
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(config_home),
                "XDG_STATE_HOME": str(state),
                "XDG_RUNTIME_DIR": str(runtime),
                "PS1": PROMPT,
                "PROMPT_COMMAND": "",
                "HISTFILE": "/dev/null",
                "TERM": "xterm",
                "NO_COLOR": "1",
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
                **extra,
            }
        )
        rig.env = env
        rig.terminal = Terminal(env, room)
        made.append(rig)
        return rig

    yield build

    for rig in made:
        rig.terminal.close()
        try:
            client_transport.stop(env=rig.env)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        if rig.pid_file is not None:
            for pid in (_read_pid_file(rig.pid_file) or {}).values():
                _kill_quietly(int(pid))
        if rig.server is not None:
            rig.server.closing.set()
            rig.server.shutdown()
            rig.server.server_close()
    _kill_daemon(runtime)
    shutil.rmtree(runtime, ignore_errors=True)


# -- driving one turn -----------------------------------------------------------


#: The line bash runs: the real ``/ask`` dispatch, in a real child process on
#: the pty, whose **own exit status is the turn's**. ``nvsh slash`` itself
#: always exits 0 on purpose (deviation d5: a non-zero status on the hidden
#: ``nvsh slash`` line would make the bash hook diagnose nvsh's own panel),
#: and it reports the turn's real code only in ``--json`` -- which disables
#: the stop prompt (d2). So the status is taken from the same
#: ``DispatchResult.exit_code`` that ``--json`` prints, one product call in.
#:
#: ``{patch}`` is where the timeout case shortens
#: :data:`nvsh.panel.STOP_PROMPT_TIMEOUT`: it is a plain module constant with
#: no environment hook, and the plan forbids adding one, so the constant is
#: rebound in the child before the client is entered. Nothing else changes.
_CLIENT = (
    "import sys; {patch}from nvsh.slash import dispatch_result; "
    'sys.exit(dispatch_result("/ask {question}", platform_kind="generic").exit_code)'
)


def _ask(rig: Rig, question: str = QUESTION, *, patch: str = "") -> None:
    """Wait for bash, then run the real client with its exit status echoed."""
    term = rig.terminal
    assert term.wait_for(lambda: term.prompts() >= 1, time.monotonic() + SETUP_TIMEOUT), term.buffer
    body = _CLIENT.format(patch=patch, question=question)
    term.type(f"{sys.executable} -c '{body}'; echo \"EXIT=$?\"\n")


def _await(term: Terminal, needle: str, budget: float = SETUP_TIMEOUT) -> None:
    assert term.wait_for(
        lambda: needle in term.buffer, time.monotonic() + budget
    ), f"never saw {needle!r}:\n{term.buffer}"


def _press(term: Terminal, label: str) -> int:
    """The operator's first Ctrl+C: the legend within 1s, with its ``[t]`` label.

    Returns the buffer offset the legend was found after, so a caller can
    assert on what came later without matching the legend again.
    """
    mark = len(term.buffer)
    first = time.monotonic()
    term.type("\x03")
    assert term.wait_for(
        lambda: _legend_in(term.buffer[mark:]), first + PROMPT_WITHIN
    ), f"no choice prompt within {PROMPT_WITHIN}s:\n{term.buffer}"
    assert f"[t] {label}  {PAUSED_TAIL}" in term.buffer[mark:], term.buffer[mark:]
    assert STOPPING_TEXT not in term.buffer[mark:], "the first press stopped the turn"
    # Every case here rides the warm daemon session; a one-shot fallback
    # would be a different code path than the one under test.
    assert "falling back" not in term.buffer, term.buffer
    assert "one-shot" not in term.buffer, term.buffer
    return mark


def _correct(term: Terminal, text: str = CORRECTION) -> None:
    """Answer the prompt with ``[t]`` and type one correction line."""
    _type_choice(term, "t", lambda: "nvsh> " in term.buffer, budget=5.0)
    _await(term, "nvsh> ")
    term.type(text + "\n")


def _exit_code(term: Terminal, budget: float = FINISH_WITHIN) -> int:
    assert term.wait_for(
        lambda: "EXIT=" in term.buffer.split("EXIT=$?", 1)[-1],
        time.monotonic() + budget,
    ), f"the client never exited:\n{term.buffer}"
    tail = term.buffer.split("EXIT=$?", 1)[-1]
    digits = tail.split("EXIT=", 1)[1].strip().split()[0]
    return int(digits)


# -- (a) steer: pi takes the correction mid-turn and the turn ends at 0 ----------


def test_steer_from_one_press_reaches_the_harness_mid_turn_and_exits_zero(rig_factory):
    rig = rig_factory("pi")
    term = rig.terminal
    _ask(rig)
    _await(term, "FIRST-TURN-TEXT")

    before = rig.pi_commands()
    mark = _press(term, STEER_LABEL)
    # c1/h1: between the press and the key the harness heard nothing new --
    # asserted from pi's own command log, not from the panel's output.
    assert rig.pi_commands() == before, rig.pi_commands()
    assert not [cmd for cmd in before if cmd.get("type") == "abort"]

    _correct(term)
    assert term.wait_for(
        lambda: "STEERED-WITH[" in term.buffer[mark:], time.monotonic() + FINISH_WITHIN
    ), term.buffer
    assert CORRECTION in term.buffer[mark:]
    assert _exit_code(term) == 0
    assert STOPPING_TEXT not in term.buffer[mark:]

    steers = [cmd for cmd in rig.pi_commands() if cmd.get("streamingBehavior") == "steer"]
    assert len(steers) == 1, rig.pi_commands()
    assert steers[0]["message"] == CORRECTION
    # Nothing cancelled this turn *before* the correction reached it. (A late
    # ``abort`` can still follow: PiAgent tidies up its event generator when
    # the finished turn's stream is closed, which is the adapter's own
    # housekeeping, not an operator stop -- the audit log below has no
    # ``cancel`` line at all.)
    commands = rig.pi_commands()
    kinds = [cmd.get("type") for cmd in commands]
    steer_at = next(
        index for index, cmd in enumerate(commands) if cmd.get("streamingBehavior") == "steer"
    )
    assert "abort" not in kinds[:steer_at], commands

    lines = rig.stops("steer")
    assert [line["outcome"] for line in lines] == ["delivered"], lines
    assert lines[0]["origin"] == "stop_prompt"
    assert lines[0]["correction_chars"] == len(CORRECTION)
    assert CORRECTION not in json.dumps(lines)
    assert rig.stops("cancel") == []
    assert rig.stops("keep_going") == []


# -- (b) stop & correct: the turn is cancelled and the correction resent ---------


def test_stop_and_correct_on_openai_compat_resends_a_self_contained_request(rig_factory):
    release = threading.Event()
    rig = rig_factory(
        "openai-compat",
        scripts=[
            _Script(before=["FIRST-TURN-TEXT"], release=release, finish=False),
            _Script(before=["SECOND-TURN-TEXT"]),
        ],
    )
    term = rig.terminal
    _ask(rig)
    _await(term, "FIRST-TURN-TEXT")

    mark = _press(term, STOP_AND_CORRECT_LABEL)
    # Nothing reached the harness: still one request, and its stream is open.
    assert len(rig.server.requests) == 1, rig.server.requests
    assert rig.server.dropped == [], rig.server.dropped

    _correct(term)
    assert term.wait_for(
        lambda: len(rig.server.requests) == 2, time.monotonic() + FINISH_WITHIN
    ), rig.server.requests
    # Not a threshold: a closed peer is only noticed on the next write.
    assert term.wait_for(
        lambda: bool(rig.server.dropped), time.monotonic() + 10.0
    ), "the first turn's stream was never cancelled"
    release.set()

    follow_up = rig.server.prompt(1)
    assert QUESTION in follow_up
    assert STOPPED_NOTICE in follow_up
    assert CORRECTION in follow_up

    _await(term, "SECOND-TURN-TEXT", FINISH_WITHIN)
    assert _exit_code(term) == 0
    assert STOPPING_TEXT in term.buffer[mark:]

    queued = rig.stops("steer")
    assert [line["outcome"] for line in queued] == ["queued"], queued
    assert queued[0]["origin"] == "stop_prompt"
    assert queued[0]["correction_chars"] == len(CORRECTION)
    cancels = rig.stops("cancel")
    assert len(cancels) == 1, cancels
    assert cancels[0]["origin"] == "stop_prompt"
    assert rig.stops("force_kill") == []


def test_stop_and_correct_on_a_subprocess_family_resends_the_correction(rig_factory):
    rig = rig_factory("claude")
    rig.claude_events.write_text(
        json.dumps([{"kind": "text_delta", "text": "FIRST-TURN-TEXT"}]), encoding="utf-8"
    )
    term = rig.terminal
    _ask(rig)
    _await(term, "FIRST-TURN-TEXT")
    # The first process has already read its script, so the next one -- the
    # follow-up's -- can be given a script that finishes.
    rig.claude_events.write_text(
        json.dumps([{"kind": "text_delta", "text": "SECOND-TURN-TEXT"}, {"kind": "done"}]),
        encoding="utf-8",
    )
    stdin_log = rig.tmp / "claude-stdin.jsonl"

    pids = list((_read_pid_file(rig.pid_file) or {}).values())
    assert pids
    assert all(_pid_alive(int(pid)) for pid in pids)
    mark = _press(term, STOP_AND_CORRECT_LABEL)
    assert all(_pid_alive(int(pid)) for pid in pids), "the press reached the harness"
    assert not stdin_log.exists() or stdin_log.stat().st_size == 0

    _correct(term)
    _await(term, "SECOND-TURN-TEXT", FINISH_WITHIN)
    assert _exit_code(term) == 0
    assert STOPPING_TEXT in term.buffer[mark:]
    assert rig.stops("cancel")[0]["origin"] == "stop_prompt"
    assert [line["outcome"] for line in rig.stops("steer")] == ["queued"]


# -- (c) stop --------------------------------------------------------------------


def test_stop_from_one_press_and_s_exits_130(rig_factory):
    release = threading.Event()
    rig = rig_factory(
        "openai-compat",
        scripts=[_Script(before=["FIRST-TURN-TEXT"], release=release, finish=False)],
    )
    term = rig.terminal
    _ask(rig)
    _await(term, "FIRST-TURN-TEXT")

    mark = _press(term, STOP_AND_CORRECT_LABEL)
    assert len(rig.server.requests) == 1
    assert rig.server.dropped == []

    _type_choice(term, "s", lambda: STOPPING_TEXT in term.buffer[mark:], budget=5.0)
    _await(term, STOPPING_TEXT)
    assert _exit_code(term) == 130, term.buffer
    assert len(rig.server.requests) == 1, "a stop must not resend anything"
    # Not a threshold: the server only notices the closed socket on its next
    # write, so this gets a generous bound, as tests/test_agent_stop.py does.
    assert term.wait_for(
        lambda: bool(rig.server.dropped), time.monotonic() + 10.0
    ), "the stop never reached the harness"
    release.set()

    cancels = rig.stops("cancel")
    assert len(cancels) == 1, cancels
    assert "origin" not in cancels[0], cancels[0]
    assert rig.stops("keep_going") == []
    assert rig.stops("steer") == []


# -- (d) keep going ---------------------------------------------------------------


def test_keep_going_from_one_press_and_esc_renders_the_rest_and_exits_zero(rig_factory):
    release = threading.Event()
    rig = rig_factory(
        "openai-compat",
        scripts=[
            _Script(
                before=["FIRST-TURN-TEXT"],
                release=release,
                after=[f"REST-{index:02d} " for index in range(20)],
            )
        ],
    )
    term = rig.terminal
    _ask(rig)
    _await(term, "FIRST-TURN-TEXT")

    mark = _press(term, STOP_AND_CORRECT_LABEL)
    _type_choice(term, "\x1b", lambda: len(rig.stops("keep_going")) == 1, budget=5.0)
    assert term.wait_for(
        lambda: len(rig.stops("keep_going")) == 1, time.monotonic() + 5.0
    ), rig.audit()
    assert rig.server.dropped == [], "keep going sent something to the harness"
    release.set()

    _await(term, "REST-19", FINISH_WITHIN)
    rendered = term.buffer[mark:]
    for index in range(20):
        assert f"REST-{index:02d}" in rendered, rendered
    positions = [rendered.index(f"REST-{index:02d}") for index in range(20)]
    assert positions == sorted(positions), "events were reordered"
    assert _exit_code(term) == 0
    assert STOPPING_TEXT not in rendered

    keep = rig.stops("keep_going")
    assert keep[0]["outcome"] == "resumed"
    assert keep[0]["origin"] == "stop_prompt"
    assert keep[0]["reason"] == "key"
    assert rig.stops("cancel") == []
    assert rig.stops("steer") == []
    assert len(rig.server.requests) == 1


# -- (e) timeout -------------------------------------------------------------------

#: The timeout case's only difference from every other case here.
_SHORT_TIMEOUT = "import nvsh.panel; nvsh.panel.STOP_PROMPT_TIMEOUT = 1.5; "


def test_an_unanswered_prompt_times_out_into_keep_going(rig_factory):
    release = threading.Event()
    rig = rig_factory(
        "openai-compat",
        scripts=[_Script(before=["FIRST-TURN-TEXT"], release=release, after=["REST-OF-IT"])],
    )
    term = rig.terminal
    _ask(rig, patch=_SHORT_TIMEOUT)
    _await(term, "FIRST-TURN-TEXT")

    mark = _press(term, STOP_AND_CORRECT_LABEL)
    assert term.wait_for(
        lambda: len(rig.stops("keep_going")) == 1, time.monotonic() + 10.0
    ), rig.audit()
    assert rig.server.dropped == [], "the timeout sent something to the harness"
    release.set()

    _await(term, "REST-OF-IT", FINISH_WITHIN)
    assert _exit_code(term) == 0
    assert STOPPING_TEXT not in term.buffer[mark:]
    keep = rig.stops("keep_going")
    assert keep[0]["reason"] == "timeout", keep
    assert keep[0]["origin"] == "stop_prompt"
    assert rig.stops("cancel") == []
    assert len(rig.server.requests) == 1


# -- criterion 3 / spec assumption c35: a paused client loses nothing --------------

#: The flood: 200 chunks of ~2.2KB, about 450KB on the wire -- more than
#: twice a unix socket's send buffer (``/proc/sys/net/core/wmem_default``,
#: 212992 bytes on this kernel), which is the point: the daemon must
#: back-pressure a client that is not reading, never drop its events.
#: Chunk *count* is kept low on purpose: each one is a separate event
#: through the daemon's one-at-a-time ack, which is what costs time here,
#: not the bytes (4000 small chunks rendered at ~20/s and blew the budget).
FLOOD_CHUNKS = 200
CHUNK_FILL = 2200
PAUSE_SECONDS = 5.0


def _socket_buffer_bytes() -> int:
    try:
        return int(Path("/proc/sys/net/core/wmem_default").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):  # pragma: no cover - non-Linux
        return 212992


def test_a_client_paused_at_the_prompt_loses_no_events_and_is_not_client_gone(rig_factory):
    release = threading.Event()
    flood = [f"[chunk-{index:04d}]" + "." * CHUNK_FILL for index in range(FLOOD_CHUNKS)]
    rig = rig_factory(
        "openai-compat",
        scripts=[_Script(before=["FIRST-TURN-TEXT"], release=release, after=flood)],
    )
    term = rig.terminal
    _ask(rig)
    _await(term, "FIRST-TURN-TEXT")

    mark = _press(term, STOP_AND_CORRECT_LABEL)
    # The flood starts while the client is paused at the prompt and reads
    # nothing from the daemon socket at all.
    release.set()
    paused_until = time.monotonic() + PAUSE_SECONDS
    while time.monotonic() < paused_until:
        time.sleep(0.05)  # deliberate: the pause *is* the thing under test
    in_flight = rig.server.sent_bytes[-1]
    buffer_bytes = _socket_buffer_bytes()
    assert in_flight > buffer_bytes, (
        f"only {in_flight} bytes were in flight while the client was paused; "
        f"the socket buffer is {buffer_bytes} bytes, so nothing was proven"
    )
    print(f"\npaused {PAUSE_SECONDS}s: {in_flight} bytes in flight, buffer {buffer_bytes}")

    _type_choice(term, "\x1b", lambda: len(rig.stops("keep_going")) == 1, budget=5.0)
    assert _exit_code(term, budget=90.0) == 0

    # The panel wraps at the terminal's width, so a marker can be split
    # across a line break. Whitespace is dropped before matching; every
    # marker is whitespace-free, so nothing else can hide a lost chunk.
    rendered = re.sub(r"\s+", "", term.buffer[mark:])
    markers = [f"[chunk-{index:04d}]" for index in range(FLOOD_CHUNKS)]
    missing = [marker for marker in markers if marker not in rendered]
    assert missing == [], f"{len(missing)} chunks lost, first {missing[:5]}"
    positions = [rendered.index(marker) for marker in markers]
    assert positions == sorted(positions), "events were reordered"
    log = rig.daemon_log()
    assert log, "no daemon log: the turn never went through a daemon"
    assert "client-gone" not in log, log
    assert rig.stops("cancel") == []
