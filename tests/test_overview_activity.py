"""``overview``'s live "Agent activity" section (task t4).

Backed by :func:`nvsh.client_transport.status`, which goes through
``control()`` -- a control message never autostarts a daemon (see
``nvsh/client_transport.py``'s docstring) -- so these tests assert the
no-daemon path is fast and side-effect-free, and that a daemon reporting an
active turn is reflected verbatim. A minimal fake unix-socket server stands
in for the real daemon: it speaks exactly the one-line-JSON-in,
JSON-lines-out wire protocol :func:`nvsh.client_transport.control` expects,
without pulling in the whole :class:`nvsh.daemon.Daemon`.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from nvsh import daemon as daemon_mod
from nvsh.cli import main
from nvsh.cli._commands import overview as overview_mod

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="the daemon needs unix domain sockets"
)


def _env(tmp_path: Path) -> dict[str, str]:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    return {"XDG_RUNTIME_DIR": str(run), "HOME": str(tmp_path)}


class _FakeDaemon:
    """Answers exactly one ``status`` control message with a canned state."""

    def __init__(self, path: Path, state: dict) -> None:
        self._path = path
        self._state = state
        path.parent.mkdir(parents=True, exist_ok=True)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(1)
        self._server.settimeout(5.0)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "_FakeDaemon":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.close()
        self._path.unlink(missing_ok=True)

    def _serve(self) -> None:
        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        with conn:
            stream = conn.makefile("rwb")
            stream.readline()  # the one request line; contents unused
            stream.write(
                (json.dumps({"kind": "status", "text": json.dumps(self._state)}) + "\n").encode()
            )
            stream.write((json.dumps({"kind": "done"}) + "\n").encode())
            stream.flush()


# --- no daemon --------------------------------------------------------------


def test_agent_activity_no_daemon_is_fast_and_spawns_nothing(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert not daemon_mod.socket_path(env).exists()

    started = time.monotonic()
    activity = overview_mod.agent_activity(env=env)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert activity == {"daemon": False}
    # A control message never autostarts a daemon: the socket still isn't
    # there afterward, proving nothing was spawned as a side effect.
    assert not daemon_mod.socket_path(env).exists()


def test_overview_json_no_daemon_reports_daemon_false(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    started = time.monotonic()
    rc = main(["overview", "--json"])
    elapsed = time.monotonic() - started

    assert rc == 0
    assert elapsed < 1.0
    payload = json.loads(capsys.readouterr().out)
    assert payload["agent_activity"] == {"daemon": False}


# --- a fake daemon with an active turn ---------------------------------------


def test_agent_activity_reports_active_turn(tmp_path: Path) -> None:
    env = _env(tmp_path)
    state = {
        "running": True,
        "pid": 4242,
        "socket": str(daemon_mod.socket_path(env)),
        "target": {"backend": "claude", "model": "opus"},
        "active_turn": {"shell": "shell-7", "started": time.time(), "elapsed": 3.5},
        "queued": [{"shell": "shell-8", "waiting": 1.2}],
    }
    with _FakeDaemon(daemon_mod.socket_path(env), state):
        activity = overview_mod.agent_activity(env=env)

    assert activity["daemon"] is True
    assert activity["active_turn"]["shell"] == "shell-7"
    assert activity["active_turn"]["target"] == {"backend": "claude", "model": "opus"}
    assert activity["active_turn"]["elapsed"] == 3.5
    assert activity["queued"] == [{"shell": "shell-8", "waiting": 1.2}]


def test_overview_json_with_active_turn_includes_shell_target_elapsed_queued(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    state = {
        "running": True,
        "pid": 4242,
        "socket": str(daemon_mod.socket_path(env)),
        "target": {"backend": "codex"},
        "active_turn": {"shell": "shell-1", "started": time.time(), "elapsed": 0.4},
        "queued": [],
    }
    with _FakeDaemon(daemon_mod.socket_path(env), state):
        rc = main(["overview", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    activity = payload["agent_activity"]
    assert activity["daemon"] is True
    assert activity["active_turn"]["shell"] == "shell-1"
    assert activity["active_turn"]["target"] == {"backend": "codex"}
    assert activity["active_turn"]["elapsed"] == 0.4
    assert activity["queued"] == []


# --- agent_sections()/cli_sections() stay static -----------------------------


def test_agent_sections_and_cli_sections_are_unaffected(tmp_path: Path) -> None:
    # These are reused verbatim elsewhere (the `cli overview` verb, the
    # `teken cli doctor` rubric), so the live daemon section must never leak
    # into them -- they must be identical whether or not a daemon is up.
    env = _env(tmp_path)
    before_agent = overview_mod.agent_sections()
    before_cli = overview_mod.cli_sections()

    state = {
        "running": True,
        "pid": 1,
        "socket": str(daemon_mod.socket_path(env)),
        "target": {"backend": "claude"},
        "active_turn": {"shell": "s", "started": time.time(), "elapsed": 1.0},
        "queued": [],
    }
    with _FakeDaemon(daemon_mod.socket_path(env), state):
        overview_mod.agent_activity(env=env)  # exercised, result unused
        after_agent = overview_mod.agent_sections()
        after_cli = overview_mod.cli_sections()

    assert after_agent == before_agent
    assert after_cli == before_cli
