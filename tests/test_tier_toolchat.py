"""Tests for nvsh.tiers.toolchat: stdlib-urllib OpenAI tool-call chat client.

Covers spec targets c25, h18: parsed tool_calls (streamed and non-streamed),
raw-fallback parsing, localhost-only enforcement, and stop-unblock semantics.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nvsh.tiers.toolchat import (
    ToolCall,
    ToolChat,
    ToolChatError,
    parse_raw_tool_calls,
    require_localhost,
)

# -- shared handler state --

# Each handler checks this class var to know what response to produce.
# Set by the test before calling complete().
_RESPONSE_TYPE: str = "tool_calls"


# -- generic handler --


class _ServerHandler(BaseHTTPRequestHandler):
    """A single handler class that produces different responses based on _RESPONSE_TYPE."""

    _request_body: str | None = None  # set during do_POST for test inspection

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        _ServerHandler._request_body = self.rfile.read(length).decode("utf-8") if length else ""

        rt = _RESPONSE_TYPE

        if rt == "tool_calls":
            self._reply_non_streamed("Running stats", [("gpu_stats", {"device": "0"})])
        elif rt == "stream_fragments":
            self._reply_streamed_fragments()
        elif rt == "two_parallel":
            self._reply_two_parallel()
        elif rt == "text_only":
            self._reply_text_only()
        elif rt == "raw_shape_a":
            self._reply_raw_shape_a()
        elif rt == "raw_shape_b":
            self._reply_raw_shape_b()
        elif rt == "invalid_args":
            self._reply_invalid_args()
        elif rt == "record_body":
            self._reply_text_only()
        elif rt == "echo":
            self._reply_text_only()
        elif rt == "http_500":
            self._send(500, "internal error")
        elif rt == "stalled":
            self._stalled()

    def _reply_non_streamed(
        self, text: str, calls: list[tuple[str, dict]], extra_content: str = ""
    ) -> None:
        if extra_content:
            content = text + extra_content if text else extra_content
        else:
            content = text
        tool_calls_list = [
            {
                "id": f"call_{i}",
                "function": {"name": name, "arguments": json.dumps(args)},
                "type": "function_call",
            }
            for i, (name, args) in enumerate(calls)
        ]
        body = {"choices": [{"message": {"content": content, "tool_calls": tool_calls_list}}]}
        self._send(200, json.dumps(body))

    def _reply_streamed_fragments(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.flush()
        # Build SSE data lines programmatically.
        # arguments is a string; accumulated across chunks:
        #   chunk1: {"device": "  +  chunk2: 0"}  +  chunk3: ""  →  {"device": "0"}
        # name is accumulated too: chunk1: gpu_ + chunk2: stats → gpu_stats
        c1_args = '{"device": "'
        c2_args = '0"}'
        c3_args = ""

        def sse(content, tcs):
            delta = {}
            if content:
                delta["content"] = content
            if tcs:
                delta["tool_calls"] = tcs
            return "data: " + json.dumps({"choices": [{"delta": delta}]})

        chunks = [
            sse("Running", [{"index": 0, "function": {"name": "gpu_", "arguments": c1_args}}]),
            # Second chunk appends name suffix AND arguments fragment
            sse(None, [{"index": 0, "function": {"name": "stats", "arguments": c2_args}}]),
            sse(" done", [{"index": 0, "function": {"arguments": c3_args}}]),
            "data: [DONE]",
        ]
        for chunk in chunks:
            self.wfile.write((chunk + "\n").encode())
            self.wfile.flush()
            time.sleep(0.02)

    def _reply_two_parallel(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.flush()
        # Two parallel tool calls split across chunks:
        # Index 0: name gpu_ + stats → gpu_stats; args {"dev + ice": " + 0"} → {"device": "0"}
        # Index 1: name mem_ + info → mem_info; args {"pa + th": " + /"} → {"path": "/"}
        idx0_arg1 = '{"dev'
        idx0_arg2 = 'ice": "'
        idx0_arg3 = '0"}'

        idx1_arg1 = '{"pa'
        idx1_arg2 = 'th": "'
        idx1_arg3 = '/"}'

        def sse(content, tcs):
            delta = {}
            if content:
                delta["content"] = content
            if tcs:
                delta["tool_calls"] = tcs
            return "data: " + json.dumps({"choices": [{"delta": delta}]})

        chunks = [
            sse(
                "Calling",
                [
                    {"index": 0, "function": {"name": "gpu_", "arguments": idx0_arg1}},
                    {"index": 1, "function": {"name": "mem_", "arguments": idx1_arg1}},
                ],
            ),
            sse(
                None,
                [
                    # Second chunk: append name suffix + args fragment for idx 0
                    {"index": 0, "function": {"name": "stats", "arguments": idx0_arg2}},
                    # Second chunk: append name suffix + args fragment for idx 1
                    {"index": 1, "function": {"name": "info", "arguments": idx1_arg2}},
                ],
            ),
            sse(
                None,
                [
                    {"index": 0, "function": {"arguments": idx0_arg3}},
                    {"index": 1, "function": {"arguments": idx1_arg3}},
                ],
            ),
            sse(
                " done",
                [
                    {"index": 0, "function": {"arguments": ""}},
                    {"index": 1, "function": {"arguments": ""}},
                ],
            ),
            "data: [DONE]",
        ]
        for chunk in chunks:
            self.wfile.write((chunk + "\n").encode())
            self.wfile.flush()
            time.sleep(0.02)

    def _reply_text_only(self) -> None:
        self._reply_non_streamed("hello world", [])

    def _reply_raw_shape_a(self) -> None:
        _bt = "`"  # noqa: F841
        _n = _bt * 2 + "name" + _bt * 2
        _f = _bt * 2 + "function" + _bt * 2
        _a = _bt * 2 + "arguments" + _bt * 2
        _c = _bt * 4
        call_obj = json.dumps({"name": "gpu_stats", "arguments": {}})
        # Shape A: markers wrap the JSON object, ```` must immediately follow it
        extra_content = f"Let me check: {_n}{_f}{_a}{call_obj}{_c}"
        self._reply_non_streamed("", [], extra_content=extra_content)

    def _reply_raw_shape_b(self) -> None:
        calls = json.dumps([{"name": "disk_usage", "arguments": {"path": "/"}}])
        content = f"<|tool_call_start|>{calls}<|tool_call_end|>"
        self._reply_non_streamed("", [], extra_content=content)

    def _reply_invalid_args(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "function": {
                                    "name": "bad_func",
                                    "arguments": "{not json",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        self._send(200, json.dumps(body))

    def _stalled(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.flush()
        time.sleep(5)

    def _send(self, status: int, body: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, fmt, *args):  # noqa: ARG002
        pass


# -- helpers --


def _make_server() -> tuple[ThreadingHTTPServer, int]:
    """Bind a ThreadingHTTPServer on a random ephemeral port."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ServerHandler)
    port = srv.server_address[1]
    return srv, port


def _make_toolchat(port: int, *, stream: bool = True) -> ToolChat:
    return ToolChat(f"http://127.0.0.1:{port}", "test-model", stream=stream)


# -- fixture: server --


class _ServerInfo:
    """Holder for server object + port."""

    def __init__(self, srv: ThreadingHTTPServer, port: int) -> None:
        self.srv = srv
        self.port = port


@pytest.fixture()
def server():
    """One HTTP server on an ephemeral port, shut down after the test."""
    srv, port = _make_server()
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    info = _ServerInfo(srv, port)
    yield info
    srv.shutdown()


# ------------------------------------------------------------------
# test_non_streamed_tool_calls
# ------------------------------------------------------------------


def test_non_streamed_tool_calls(server) -> None:
    global _RESPONSE_TYPE
    _RESPONSE_TYPE = "tool_calls"
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "check gpu"}], [])
    assert reply.tool_calls == (ToolCall(name="gpu_stats", arguments={"device": "0"}),)
    assert reply.text == "Running stats"


# ------------------------------------------------------------------
# test_streamed_tool_calls_fragments_are_joined
# ------------------------------------------------------------------


def test_streamed_tool_calls_fragments_are_joined(server) -> None:
    global _RESPONSE_TYPE
    _RESPONSE_TYPE = "stream_fragments"
    srv = server
    chat = _make_toolchat(srv.port, stream=True)
    reply = chat.complete([{"role": "user", "content": "check gpu"}], [])
    assert reply.tool_calls == (ToolCall(name="gpu_stats", arguments={"device": "0"}),)
    assert reply.text == "Running done"


# ------------------------------------------------------------------
# test_streamed_two_parallel_tool_calls_by_index
# ------------------------------------------------------------------


def test_streamed_two_parallel_tool_calls_by_index(server) -> None:
    global _RESPONSE_TYPE
    _RESPONSE_TYPE = "two_parallel"
    srv = server
    chat = _make_toolchat(srv.port, stream=True)
    reply = chat.complete([{"role": "user", "content": "check"}], [])
    names = {tc.name for tc in reply.tool_calls}
    assert names == {"gpu_stats", "mem_info"}
    assert reply.text == "Calling done"


# ------------------------------------------------------------------
# test_text_only_reply
# ------------------------------------------------------------------


def test_text_only_reply(server) -> None:
    global _RESPONSE_TYPE
    _RESPONSE_TYPE = "text_only"
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "hi"}], [])
    assert reply.text == "hello world"
    assert reply.tool_calls == ()


# ------------------------------------------------------------------
# test_raw_fallback_shape_a
# ------------------------------------------------------------------


def test_raw_fallback_shape_a(server) -> None:
    global _RESPONSE_TYPE
    _RESPONSE_TYPE = "raw_shape_a"
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "check"}], [])
    assert reply.tool_calls == (ToolCall(name="gpu_stats", arguments={}),)


# ------------------------------------------------------------------
# test_raw_fallback_shape_b
# ------------------------------------------------------------------


def test_raw_fallback_shape_b(server) -> None:
    global _RESPONSE_TYPE
    _RESPONSE_TYPE = "raw_shape_b"
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "check"}], [])
    assert reply.tool_calls == (ToolCall(name="disk_usage", arguments={"path": "/"}),)


# ------------------------------------------------------------------
# test_invalid_arguments_json_is_skipped
# ------------------------------------------------------------------


def test_invalid_arguments_json_is_skipped(server) -> None:
    global _RESPONSE_TYPE
    _RESPONSE_TYPE = "invalid_args"
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "check"}], [])
    assert reply.tool_calls == ()


# ------------------------------------------------------------------
# test_request_body_has_model_messages_tools
# ------------------------------------------------------------------


def test_request_body_has_model_messages_tools(server) -> None:
    global _RESPONSE_TYPE
    _RESPONSE_TYPE = "record_body"
    _ServerHandler._request_body = None
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    tools = [{"type": "function", "function": {"name": "check", "parameters": {}}}]
    chat.complete(
        [{"role": "user", "content": "hello"}],
        tools,
    )
    body = json.loads(_ServerHandler._request_body)  # type: ignore[arg-type]
    assert body["model"] == "test-model"
    assert len(body["messages"]) >= 1
    assert body["messages"][0]["role"] == "user"
    assert body["tools"] == tools


# ------------------------------------------------------------------
# test_require_localhost_refuses_other_hosts
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "http://localhost.example.com/v1",
        "http://127.0.0.1.example.com/v1",
        "https://localhost/v1",
        "localhost:8080",
    ],
)
def test_require_localhost_refuses_other_hosts(url) -> None:
    with pytest.raises(ToolChatError):
        require_localhost(url)


# ------------------------------------------------------------------
# test_require_localhost_accepts_local_hosts
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/v1",
        "http://localhost:1/v1",
        "http://[::1]:8080/v1",
    ],
)
def test_require_localhost_accepts_local_hosts(url) -> None:
    require_localhost(url)  # must not raise


# ------------------------------------------------------------------
# test_changing_port_is_config_only
# ------------------------------------------------------------------


def test_changing_port_is_config_only() -> None:
    """Two separate servers on different ports; two ToolChat objects; both work."""
    srv_a, port_a = _make_server()
    srv_b, port_b = _make_server()
    t_a = threading.Thread(target=srv_a.serve_forever, daemon=True)
    t_b = threading.Thread(target=srv_b.serve_forever, daemon=True)
    t_a.start()
    t_b.start()

    try:
        global _RESPONSE_TYPE
        _RESPONSE_TYPE = "echo"
        chat_a = _make_toolchat(port_a, stream=False)
        chat_b = _make_toolchat(port_b, stream=False)
        reply_a = chat_a.complete([{"role": "user", "content": "a"}], [])
        reply_b = chat_b.complete([{"role": "user", "content": "b"}], [])
        assert reply_a.text == "hello world"
        assert reply_b.text == "hello world"
    finally:
        srv_a.shutdown()
        srv_b.shutdown()


# ------------------------------------------------------------------
# test_connection_refused_raises_toolchaterror
# ------------------------------------------------------------------


def test_connection_refused_raises_toolchaterror() -> None:
    """Point at a port with no server → ToolChatError."""
    chat = _make_toolchat(19999, stream=False)
    with pytest.raises(ToolChatError):
        chat.complete([{"role": "user", "content": "hello"}], [])


# ------------------------------------------------------------------
# test_http_500_raises_toolchaterror
# ------------------------------------------------------------------


def test_http_500_raises_toolchaterror(server) -> None:
    global _RESPONSE_TYPE
    _RESPONSE_TYPE = "http_500"
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    with pytest.raises(ToolChatError):
        chat.complete([{"role": "user", "content": "hello"}], [])


# ------------------------------------------------------------------
# test_parse_raw_never_raises
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "`{bad`",
        "`[1,2]`",
        "10000x",
    ],
)
def test_parse_raw_never_raises(text) -> None:
    result = parse_raw_tool_calls(text)
    assert isinstance(result, tuple)


# ------------------------------------------------------------------
# test_stop_unblocks_a_stalled_read
# ------------------------------------------------------------------


def test_stop_unblocks_a_stalled_read() -> None:
    """Handler sends headers then sleeps 5 s; stop() must unblock in <2 s."""
    srv, port = _make_server()
    t_srv = threading.Thread(target=srv.serve_forever, daemon=True)
    t_srv.start()

    try:
        global _RESPONSE_TYPE
        _RESPONSE_TYPE = "stalled"
        chat = _make_toolchat(port, stream=False)
        errors: list[ToolChatError] = []

        def run_complete() -> None:
            try:
                chat.complete([{"role": "user", "content": "stalled"}], [])
            except ToolChatError as exc:
                errors.append(exc)

        t = threading.Thread(target=run_complete, daemon=True)
        t.start()
        time.sleep(0.2)
        chat.stop()
        t.join(timeout=2)
        assert not t.is_alive(), "complete() did not return within 2 s"
        assert len(errors) == 1, "expected a ToolChatError from the stop"
    finally:
        srv.shutdown()
