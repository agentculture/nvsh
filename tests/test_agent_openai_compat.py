"""Tests for nvsh.agent.openai_compat.OpenAICompatAgent (task t10).

Spins up a local http.server.ThreadingHTTPServer serving a canned SSE
stream (and a 401 variant), and asserts:
- streaming TEXT_DELTA events are yielded in order, then DONE on '[DONE]';
- the Authorization header is derived from an env-var name in config, never
  a literal key in config or source;
- HTTP errors (401) yield an ERROR event that distinguishes the status;
- capabilities are reported honestly.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nvsh.agent import AgentContext, AgentRequest, Capabilities, EventKind, RequestKind
from nvsh.agent.openai_compat import OpenAICompatAgent

_SSE_CHUNKS = [
    {"choices": [{"delta": {"content": "hello "}}]},
    {"choices": [{"delta": {"content": "world"}}]},
]


class _SSEHandler(BaseHTTPRequestHandler):
    server_version = "FakeOpenAICompat/1.0"

    def log_message(self, *_args):  # noqa: D401 - silence test server logging
        pass

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        auth = self.headers.get("Authorization", "")

        if "/unauthorized" in self.path:
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "unauthorized"}).encode("utf-8"))
            return

        self.server.last_auth_header = auth  # type: ignore[attr-defined]
        self.server.last_body = body  # type: ignore[attr-defined]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in _SSE_CHUNKS:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture()
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    server.last_auth_header = None
    server.last_body = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, command="ls /nope", exit_code=2)


def _context() -> AgentContext:
    return AgentContext(platform="dgx-spark", output="ls: /nope: No such file", cwd="/tmp")


def test_streams_text_deltas_then_done(fake_server, monkeypatch):
    monkeypatch.setenv("NVSH_TEST_API_KEY", "sekret-value")
    base_url = f"http://127.0.0.1:{fake_server.server_port}"
    agent = OpenAICompatAgent({"base_url": base_url, "api_key_env": "NVSH_TEST_API_KEY"})
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    kinds = [e.kind for e in events]
    assert kinds == [EventKind.TEXT_DELTA, EventKind.TEXT_DELTA, EventKind.DONE]
    assert events[0].text == "hello "
    assert events[1].text == "world"
    assert fake_server.last_auth_header == "Bearer sekret-value"


def test_no_auth_header_when_env_var_unset(fake_server, monkeypatch):
    monkeypatch.delenv("NVSH_TEST_API_KEY_UNSET", raising=False)
    base_url = f"http://127.0.0.1:{fake_server.server_port}"
    agent = OpenAICompatAgent({"base_url": base_url, "api_key_env": "NVSH_TEST_API_KEY_UNSET"})
    agent.start()
    try:
        list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert fake_server.last_auth_header == ""


def test_401_yields_error_event_distinguishing_status(fake_server, monkeypatch):
    monkeypatch.setenv("NVSH_TEST_API_KEY", "sekret-value")
    base_url = f"http://127.0.0.1:{fake_server.server_port}/unauthorized"
    agent = OpenAICompatAgent({"base_url": base_url, "api_key_env": "NVSH_TEST_API_KEY"})
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert [e.kind for e in events] == [EventKind.ERROR]
    assert "401" in events[0].error


def test_connection_error_yields_error_event(monkeypatch):
    monkeypatch.setenv("NVSH_TEST_API_KEY", "sekret-value")
    agent = OpenAICompatAgent(
        {"base_url": "http://127.0.0.1:1", "api_key_env": "NVSH_TEST_API_KEY"}
    )
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert events
    assert events[-1].kind == EventKind.ERROR


def test_capabilities_are_honest():
    agent = OpenAICompatAgent({"base_url": "http://127.0.0.1:1", "api_key_env": "X"})
    caps = agent.capabilities()
    assert caps == Capabilities(
        streaming=True,
        tool_calling=False,
        cancellation=True,
        persistent_session=False,
        local_model=True,
    )


def test_config_never_carries_a_literal_key():
    import inspect

    import nvsh.agent.openai_compat as mod
    from nvsh.config import default_toml

    source = inspect.getsource(mod)
    # api_key_env is a name, never a literal secret value.
    assert "api_key_env" in source
    assert "sk-" not in source
    assert "sekret-value" not in source
    assert "sekret-value" not in default_toml()


def test_connect_timeout_is_five_seconds():
    import inspect

    import nvsh.agent.openai_compat as mod

    source = inspect.getsource(mod)
    assert "timeout=5" in source
