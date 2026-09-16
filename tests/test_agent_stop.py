"""End-to-end stop: the success signal of reliable-agent-stop (task t20).

For every adapter family and both request paths (the warm daemon session
and a one-shot ``/ask --agent <name>``), a real interactive ``bash`` runs
the real ``nvsh slash '/ask ...'`` client and panel on a pty, against a fake
harness that ignores its protocol-level cancel and holds a ``sleep 600``
grandchild. The test types Ctrl+C into the pty like an operator would and
measures, with :func:`time.monotonic`:

* the ``stopping… press again to kill`` line appears within **1s** of the
  first press (c22/h19);
* after the second press the harness process tree is gone and bash's prompt
  is back within **3s** (c21/h18).

Nothing here sleeps for a fixed time to "let things settle": every wait is a
poll against a deadline, and only the two thresholds above are asserted as
timings (plan risk r5 -- these run under ``pytest -n auto`` on loaded boxes).

openai-compat speaks HTTP, so it has no harness process tree: its case
asserts the stopping line and the prompt coming back, and that the stalled
request's connection was dropped.
"""

from __future__ import annotations

import fcntl
import json
import os
import pty
import select
import shutil
import signal
import struct
import subprocess  # nosec B404 - fixed argv, test only
import sys
import tempfile
import termios
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from nvsh import client_transport
from nvsh.panel import STOPPING_TEXT

FAKES_DIR = Path(__file__).parent / "fakes"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or not hasattr(os, "fork"), reason="needs bash and a pty"
)

#: bash's prompt in the pty: distinctive, and never part of the typed command.
PROMPT = "NVSH-E2E-PROMPT$ "

#: The acceptance thresholds (c22/h19 and c21/h18).
STOPPING_WITHIN = 1.0
STOPPED_WITHIN = 3.0

#: Generous bounds for everything that is *not* a measured threshold.
SETUP_TIMEOUT = 30.0


# -- processes -----------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie still answers kill(0); it is dead for our purposes
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def _kill_quietly(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _read_pid_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# -- the terminal ----------------------------------------------------------------

#: Run as ``python -c``: take stdin (the pty slave) as the controlling tty, exec argv.
_CTTY_EXEC = (
    "import fcntl, os, sys, termios; "
    "fcntl.ioctl(0, termios.TIOCSCTTY, 0); "
    "os.execvp(sys.argv[1], sys.argv[1:])"
)


class Terminal:
    """An interactive ``bash`` on a pty, read into one buffer as it talks."""

    def __init__(self, env: dict[str, str], cwd: Path) -> None:
        fd, slave = pty.openpty()
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 220, 0, 0))
        # A new session, then the pty slave (already on stdin) becomes its
        # controlling terminal, so Ctrl+C is the tty's SIGINT to bash's
        # foreground job exactly as at a real prompt. (No pty.fork(): forking
        # a multi-threaded pytest worker is deprecated.)
        self.proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
            [sys.executable, "-c", _CTTY_EXEC, "bash", "--norc", "--noprofile", "-i"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        os.close(slave)
        self.pid = self.proc.pid
        self.fd = fd
        self.buffer = ""
        self._raw = b""

    def pump(self, timeout: float) -> None:
        ready, _, _ = select.select([self.fd], [], [], max(0.0, timeout))
        if not ready:
            return
        try:
            chunk = os.read(self.fd, 65536)
        except OSError:
            return
        self._raw += chunk
        self.buffer = self._raw.decode("utf-8", errors="replace")

    def wait_for(self, predicate, deadline: float) -> bool:
        while True:
            if predicate():
                return True
            left = deadline - time.monotonic()
            if left <= 0:
                return predicate()
            self.pump(min(0.02, left))

    def type(self, data: str) -> None:
        os.write(self.fd, data.encode("utf-8"))

    def prompts(self) -> int:
        return self.buffer.count(PROMPT)

    def close(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except OSError:
            _kill_quietly(self.pid)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass


# -- openai-compat's stalling endpoint ------------------------------------------


class _StallHandler(BaseHTTPRequestHandler):
    """Accepts the chat request, opens the event stream, then says nothing."""

    server_version = "NvshStopE2E/1.0"

    def log_message(self, *_args):  # noqa: D401 - silence test server logging
        pass

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.flush()
        self.server.connected.set()  # type: ignore[attr-defined]
        # Hold the stream open until the client hangs up (or the test ends).
        while not self.server.closing.is_set():  # type: ignore[attr-defined]
            try:
                self.wfile.write(b": \n")  # an SSE comment: no event for the client
                self.wfile.flush()
            except OSError:
                self.server.dropped.set()  # type: ignore[attr-defined]
                return
            self.server.closing.wait(0.1)  # type: ignore[attr-defined]


# -- families ------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """One adapter family: which adapter, which fake stands in for its binary."""

    name: str
    adapter: str
    #: binary name on PATH -> fake script in tests/fakes
    binaries: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    #: extra ``[agents.<adapter>]`` settings; ``{base_url}`` is filled in
    settings: dict[str, str] = field(default_factory=dict)
    agy_events: dict | None = None
    #: The adapter has no protocol cancel, so its cancel() is already a signal
    #: to the process tree (``SubprocessAgent.cancel`` -> ``kill_tree``, and
    #: cold agy's terminate->kill): the first press may end the turn by itself.
    cancel_kills: bool = False

    @property
    def http(self) -> bool:
        return not self.binaries


FAMILIES = [
    Family("pi", "pi", {"pi": "pi"}, {"NVSH_FAKE_STALL_TURN": "1"}),
    Family("codex", "codex", {"codex": "codex-app-server"}, {"NVSH_FAKE_STALL_TURN": "1"}),
    Family("acp", "qwen", {"qwen": "acp"}, {"NVSH_FAKE_STALL_TURN": "1"}),
    Family(
        "agy",
        "agy",
        {"agy": "agy"},
        agy_events={"stdout": [], "exit_code": 0, "sleep_before": 3600},
        cancel_kills=True,
    ),
    Family("claude", "claude", {"claude": "claude"}, cancel_kills=True),
    Family("qwen-p", "qwen-p", {"qwen": "qwen"}, cancel_kills=True),
    Family(
        "openai-compat",
        "openai-compat",
        settings={"base_url": "{base_url}"},
        cancel_kills=True,  # cancel() closes the HTTP stream
    ),
]

PATHS = ["daemon", "one-shot"]


def _config_toml(family: Family, base_url: str) -> str:
    lines = ["[aliases]", f'default = "{family.adapter}"', ""]
    if family.settings:
        lines.append(f'[agents."{family.adapter}"]')
        for key, value in family.settings.items():
            lines.append(f'{key} = "{value.format(base_url=base_url)}"')
        lines.append("")
    return "\n".join(lines)


@dataclass
class Rig:
    terminal: Terminal
    env: dict[str, str]
    pid_file: Path
    server: ThreadingHTTPServer | None


@pytest.fixture
def rig_factory(tmp_path):
    """Build the pty, the fake binaries, the config and (for HTTP) the server."""
    made: list[Rig] = []
    # A short runtime dir: the daemon's unix socket path must fit in 108 bytes.
    runtime = Path(tempfile.mkdtemp(prefix="nvsh-t20-"))

    def build(family: Family) -> Rig:
        bindir = tmp_path / "bin"
        bindir.mkdir()
        for binary, fake in family.binaries.items():
            (bindir / binary).symlink_to(FAKES_DIR / fake)
        home = tmp_path / "home"
        home.mkdir()
        config_home = tmp_path / "config"
        (config_home / "nvsh").mkdir(parents=True)
        pid_file = tmp_path / "pids.json"

        server = None
        base_url = ""
        if family.http:
            server = ThreadingHTTPServer(("127.0.0.1", 0), _StallHandler)
            server.daemon_threads = True
            server.connected = threading.Event()  # type: ignore[attr-defined]
            server.dropped = threading.Event()  # type: ignore[attr-defined]
            server.closing = threading.Event()  # type: ignore[attr-defined]
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base_url = f"http://127.0.0.1:{server.server_port}"
        (config_home / "nvsh" / "config.toml").write_text(
            _config_toml(family, base_url), encoding="utf-8"
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
                "XDG_STATE_HOME": str(tmp_path / "state"),
                "XDG_RUNTIME_DIR": str(runtime),
                "PS1": PROMPT,
                "PROMPT_COMMAND": "",
                "HISTFILE": "/dev/null",
                "TERM": "xterm",
                "NO_COLOR": "1",
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
                "NVSH_FAKE_IGNORE_CANCEL": "1",
                "NVSH_FAKE_GRANDCHILD": "1",
                "NVSH_FAKE_PID_FILE": str(pid_file),
                "NVSH_TEST_ACP_UI_TIMEOUT": "120",
                "NVSH_FAKE_CODEX_APPROVAL_TIMEOUT": "120",
                **family.env,
            }
        )
        if family.agy_events is not None:
            events = tmp_path / "agy-events.json"
            events.write_text(json.dumps(family.agy_events), encoding="utf-8")
            env["NVSH_FAKE_EVENTS"] = str(events)
        rig = Rig(Terminal(env, tmp_path), env, pid_file, server)
        made.append(rig)
        return rig

    yield build

    for rig in made:
        rig.terminal.close()
        try:
            client_transport.stop(env=rig.env)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        pids = _read_pid_file(rig.pid_file) or {}
        for pid in pids.values():
            _kill_quietly(int(pid))
        if rig.server is not None:
            rig.server.closing.set()  # type: ignore[attr-defined]
            rig.server.shutdown()
            rig.server.server_close()
    _kill_daemon(runtime)
    shutil.rmtree(runtime, ignore_errors=True)


def _kill_daemon(runtime: Path) -> None:
    """Belt and braces: no daemon started under ``runtime`` outlives the test."""
    marker = str(runtime)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environ = (entry / "environ").read_bytes()
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"nvsh.daemon" in cmdline and marker.encode() in environ:
            _kill_quietly(int(entry.name))


# -- the test --------------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("family", FAMILIES, ids=[family.name for family in FAMILIES])
def test_two_presses_stop_every_family_on_both_paths(family: Family, path: str, rig_factory):
    rig = rig_factory(family)
    term = rig.terminal
    setup_deadline = time.monotonic() + SETUP_TIMEOUT
    assert term.wait_for(lambda: term.prompts() >= 1, setup_deadline), term.buffer

    agent_flag = "" if path == "daemon" else f"--agent {family.adapter} "
    term.type(f"{sys.executable} -m nvsh slash '/ask {agent_flag}why did it fail?'\n")

    # The turn is genuinely in flight: the harness (or the HTTP stream) is up,
    # and the panel has been streaming long enough to say it is waiting.
    if family.http:
        assert rig.server is not None
        connected = rig.server.connected  # type: ignore[attr-defined]
        assert term.wait_for(connected.is_set, setup_deadline), term.buffer
    else:
        assert term.wait_for(
            lambda: _read_pid_file(rig.pid_file) is not None, setup_deadline
        ), term.buffer
    mark = len(term.buffer)
    assert term.wait_for(
        lambda: "waiting for the agent" in term.buffer[mark:], setup_deadline
    ), term.buffer
    pids = [] if family.http else list((_read_pid_file(rig.pid_file) or {}).values())
    assert all(_pid_alive(pid) for pid in pids), f"harness tree not running: {pids}"

    # First press: the stopping line within 1s.
    mark = len(term.buffer)
    first = time.monotonic()
    term.type("\x03")
    assert term.wait_for(
        lambda: STOPPING_TEXT in term.buffer[mark:], first + STOPPING_WITHIN
    ), f"{family.name}/{path}: no stopping line within {STOPPING_WITHIN}s:\n{term.buffer}"
    stopping_after = time.monotonic() - first
    assert stopping_after <= STOPPING_WITHIN

    if not family.cancel_kills:
        # The harness ignores the polite stop: the turn is still open.
        assert term.prompts() == 1, term.buffer
        assert all(_pid_alive(pid) for pid in pids), "the harness honoured a cancel it ignores"

    # Second press: the tree is gone and the prompt is back within 3s.
    second = time.monotonic()
    term.type("\x03")
    deadline = second + STOPPED_WITHIN
    assert term.wait_for(
        lambda: term.prompts() >= 2, deadline
    ), f"{family.name}/{path}: prompt not back within {STOPPED_WITHIN}s:\n{term.buffer}"
    assert term.wait_for(
        lambda: not any(_pid_alive(pid) for pid in pids), deadline
    ), f"{family.name}/{path}: harness tree survived: {[p for p in pids if _pid_alive(p)]}"
    stopped_after = time.monotonic() - second
    assert stopped_after <= STOPPED_WITHIN
    assert "nvsh: interrupted" in term.buffer
    if path == "daemon":  # the warm session really served it: no one-shot fallback
        assert "falling back" not in term.buffer and "one-shot" not in term.buffer, term.buffer
    else:
        assert f"one-shot {family.adapter}" in term.buffer, term.buffer
    if rig.server is not None:
        # Not a threshold: the HTTP stream's socket closing is only observed on
        # the server's next write, so give it a generous bound.
        dropped = rig.server.dropped  # type: ignore[attr-defined]
        assert term.wait_for(dropped.is_set, time.monotonic() + 10), "stalled request kept open"
