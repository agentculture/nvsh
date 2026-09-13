"""Air-gapped run: with every non-loopback socket refused, the failure client
still reaches a localhost model endpoint and streams text (task t13, c38).

The guard monkeypatches ``socket.create_connection`` and ``socket.socket`` so
anything but 127.0.0.1 / ::1 / localhost / AF_UNIX raises, the way a machine
with no route off the box behaves.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nvsh import client as client_mod
from nvsh import panel as panel_mod

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

_SSE = [
    {"choices": [{"delta": {"content": "unified memory "}}]},
    {"choices": [{"delta": {"content": "is shared on GB10"}}]},
]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        self.server.last_body = json.loads(self.rfile.read(length)) if length else {}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in _SSE:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture()
def local_model():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.last_body = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture()
def no_network(monkeypatch):
    """Refuse every connection that is not loopback or an AF_UNIX socket."""
    real_create_connection = socket.create_connection
    real_socket = socket.socket

    def guarded_create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in _LOOPBACK:
            raise OSError(f"network is unreachable (air-gapped): {host}")
        return real_create_connection(address, *args, **kwargs)

    class GuardedSocket(real_socket):
        def connect(self, address):
            if self.family is socket.AF_INET or self.family is socket.AF_INET6:
                host = address[0] if isinstance(address, tuple) else address
                if str(host) not in _LOOPBACK:
                    raise OSError(f"network is unreachable (air-gapped): {host}")
            return super().connect(address)

    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "socket", GuardedSocket)


def test_air_gapped_failure_still_reaches_a_localhost_endpoint(
    tmp_path, monkeypatch, local_model, no_network
):
    config_dir = tmp_path / "config" / "nvsh"
    config_dir.mkdir(parents=True)
    base_url = f"http://127.0.0.1:{local_model.server_port}"
    (config_dir / "config.toml").write_text(
        "[agent]\n"
        'provider = "openai-compat"\n\n'
        "[agents.openai-compat]\n"
        f'base_url = "{base_url}"\n'
        'api_key_env = "NVSH_AIRGAP_KEY"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("NVSH_NO_DAEMON", "1")
    monkeypatch.setattr(client_mod, "_platform_block", lambda: "platform: dgx-spark")

    args = types.SimpleNamespace(
        exit=2, pipestatus="2", line="ls /nope", cwd=str(tmp_path), log="", json=False
    )
    p = panel_mod.Panel(out=io.StringIO(), in_=io.StringIO(), env={}, isatty=False)
    rc = client_mod.handle_failure(args, panel=p)

    out = p.out.getvalue()
    assert rc == 0
    assert "unified memory is shared on GB10" in out
    assert local_model.last_body is not None
    sent = json.dumps(local_model.last_body)
    assert "ls /nope" in sent


def test_a_non_loopback_endpoint_is_refused_when_air_gapped(tmp_path, monkeypatch, no_network):
    with pytest.raises(OSError):
        socket.create_connection(("example.com", 80), timeout=1)
