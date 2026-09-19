"""Tests for nvsh.tiers.toolchat: stdlib-urllib OpenAI tool-call chat client.

Covers spec targets c25, h18: parsed tool_calls (streamed and non-streamed),
raw-fallback parsing, localhost-only enforcement, and stop-unblock semantics.
Plus d1: next-token log-probability scoring and yes/no probability helpers.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nvsh.tiers.toolchat import (
    ToolCall,
    ToolChat,
    ToolChatError,
    calibrated_logit,
    parse_raw_tool_calls,
    require_localhost,
    yes_no_probability,
)

# -- shared handler state --

# Each handler checks this class var to know what response to produce.
# Set by the test before calling complete().
_RESPONSE_TYPE: str = "tool_calls"


# -- generic handler --


class _ServerHandler(BaseHTTPRequestHandler):
    """A single handler class that produces different responses based on _RESPONSE_TYPE."""

    _request_body: str | None = None  # set during do_POST for test inspection
    _request_path: str | None = None  # set during do_POST for test inspection

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
        elif rt == "score_shape_l":
            self._reply_score_shape_l()
        elif rt == "score_shape_c":
            self._reply_score_shape_c()
        elif rt == "score_no_logprobs":
            self._reply_score_no_logprobs()
        elif rt == "score_mixed_valid":
            self._reply_score_mixed_valid()
        elif rt == "score_body_path":
            self._reply_score_body_path()

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
            sse("Running", [{"index": 0, "function": {"name": "gpu_stats", "arguments": c1_args}}]),
            # Second chunk appends name suffix AND arguments fragment
            sse(None, [{"index": 0, "function": {"name": "gpu_stats", "arguments": c2_args}}]),
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
                    {"index": 0, "function": {"name": "gpu_stats", "arguments": idx0_arg1}},
                    {"index": 1, "function": {"name": "mem_info", "arguments": idx1_arg1}},
                ],
            ),
            sse(
                None,
                [
                    # Second chunk: append name suffix + args fragment for idx 0
                    {"index": 0, "function": {"arguments": idx0_arg2}},
                    # Second chunk: append name suffix + args fragment for idx 1
                    {"index": 1, "function": {"name": "mem_info", "arguments": idx1_arg2}},
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
        call_obj = json.dumps({"name": "gpu_stats", "arguments": {}})
        # Tags assembled at runtime: a literal tool-call tag in a source file
        # is a control token for some models that read this repo.
        open_tag, close_tag = "<" + "tool_call" + ">", "</" + "tool_call" + ">"
        self._reply_non_streamed(
            "", [], extra_content=f"Let me check: {open_tag}{call_obj}{close_tag}"
        )

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

    def _reply_score_shape_l(self) -> None:
        # Shape L (legacy), as vLLM returns it: top_logprobs is a list with one
        # {token: logprob} dict per GENERATED token.
        body = {
            "choices": [
                {
                    "message": {"content": ""},
                    "logprobs": {
                        # One dict per generated token; one token was asked for.
                        "top_logprobs": [{" yes": -0.2, " no": -1.8, "Maybe": -4.0}]
                    },
                }
            ]
        }
        self._send(200, json.dumps(body))

    def _reply_score_shape_c(self) -> None:
        # Shape C (content): choices[0].logprobs.content is a LIST,
        # each item has key "top_logprobs" = LIST of
        # {"token": str, "logprob": float}.
        body = {
            "choices": [
                {
                    "message": {"content": ""},
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": " yes", "logprob": -0.2},
                                    {"token": " no", "logprob": -1.8},
                                    {"token": "Maybe", "logprob": -4.0},
                                ]
                            }
                        ]
                    },
                }
            ]
        }
        self._send(200, json.dumps(body))

    def _reply_score_no_logprobs(self) -> None:
        body = {"choices": [{"message": {"content": ""}, "logprobs": None}]}
        self._send(200, json.dumps(body))

    def _reply_score_mixed_valid(self) -> None:
        # One entry has logprob "x" (str), one has true (bool) — both skipped.
        body = {
            "choices": [
                {
                    "message": {"content": ""},
                    "logprobs": {"top_logprobs": [{" yes": "x", " no": True, "Maybe": -4.0}]},
                }
            ]
        }
        self._send(200, json.dumps(body))

    def _reply_score_body_path(self) -> None:
        # _request_body is already set by do_POST above.
        _ServerHandler._request_path = self.path  # type: ignore[attr-defined]
        body = {
            "choices": [
                {
                    "message": {"content": ""},
                    "logprobs": {"top_logprobs": [{" yes": -0.2}]},
                }
            ]
        }
        self._send(200, json.dumps(body))

    def log_message(self, fmt, *args):  # noqa: ARG002
        pass


# -- helpers --


def _make_server() -> tuple[ThreadingHTTPServer, int]:
    """Bind a ThreadingHTTPServer on a random ephemeral port."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ServerHandler)
    port = srv.server_address[1]
    return srv, port


def _serve(monkeypatch, response_type: str) -> None:
    """Choose the canned reply for this test; undone automatically afterwards."""
    monkeypatch.setattr(sys.modules[__name__], "_RESPONSE_TYPE", response_type)


def _make_toolchat(port: int, *, stream: bool = True) -> ToolChat:
    return ToolChat(f"http://127.0.0.1:{port}", "test-model", stream=stream)


# -- fixture: server --


class _ServerInfo:
    """Holder for server object + port."""

    def __init__(self, srv: ThreadingHTTPServer, port: int) -> None:
        self.srv = srv
        self.port = port


@pytest.fixture
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


def test_non_streamed_tool_calls(server, monkeypatch) -> None:
    _serve(monkeypatch, "tool_calls")
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "check gpu"}], [])
    assert reply.tool_calls == (ToolCall(name="gpu_stats", arguments={"device": "0"}),)
    assert reply.text == "Running stats"


# ------------------------------------------------------------------
# test_streamed_tool_calls_fragments_are_joined
# ------------------------------------------------------------------


def test_streamed_tool_calls_fragments_are_joined(server, monkeypatch) -> None:
    _serve(monkeypatch, "stream_fragments")
    srv = server
    chat = _make_toolchat(srv.port, stream=True)
    reply = chat.complete([{"role": "user", "content": "check gpu"}], [])
    assert reply.tool_calls == (ToolCall(name="gpu_stats", arguments={"device": "0"}),)
    assert reply.text == "Running done"


# ------------------------------------------------------------------
# test_streamed_two_parallel_tool_calls_by_index
# ------------------------------------------------------------------


def test_streamed_two_parallel_tool_calls_by_index(server, monkeypatch) -> None:
    _serve(monkeypatch, "two_parallel")
    srv = server
    chat = _make_toolchat(srv.port, stream=True)
    reply = chat.complete([{"role": "user", "content": "check"}], [])
    names = {tc.name for tc in reply.tool_calls}
    assert names == {"gpu_stats", "mem_info"}
    assert reply.text == "Calling done"


# ------------------------------------------------------------------
# test_text_only_reply
# ------------------------------------------------------------------


def test_text_only_reply(server, monkeypatch) -> None:
    _serve(monkeypatch, "text_only")
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "hi"}], [])
    assert reply.text == "hello world"
    assert reply.tool_calls == ()


# ------------------------------------------------------------------
# test_raw_fallback_shape_a
# ------------------------------------------------------------------


def test_raw_fallback_shape_a(server, monkeypatch) -> None:
    _serve(monkeypatch, "raw_shape_a")
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "check"}], [])
    assert reply.tool_calls == (ToolCall(name="gpu_stats", arguments={}),)


# ------------------------------------------------------------------
# test_raw_fallback_shape_b
# ------------------------------------------------------------------


def test_raw_fallback_shape_b(server, monkeypatch) -> None:
    _serve(monkeypatch, "raw_shape_b")
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "check"}], [])
    assert reply.tool_calls == (ToolCall(name="disk_usage", arguments={"path": "/"}),)


# ------------------------------------------------------------------
# test_invalid_arguments_json_is_skipped
# ------------------------------------------------------------------


def test_invalid_arguments_json_is_skipped(server, monkeypatch) -> None:
    _serve(monkeypatch, "invalid_args")
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    reply = chat.complete([{"role": "user", "content": "check"}], [])
    assert reply.tool_calls == ()


# ------------------------------------------------------------------
# test_request_body_has_model_messages_tools
# ------------------------------------------------------------------


def test_request_body_has_model_messages_tools(server, monkeypatch) -> None:
    _serve(monkeypatch, "record_body")
    monkeypatch.setattr(_ServerHandler, "_request_body", None)
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


def test_changing_port_is_config_only(monkeypatch) -> None:
    """Two separate servers on different ports; two ToolChat objects; both work."""
    srv_a, port_a = _make_server()
    srv_b, port_b = _make_server()
    t_a = threading.Thread(target=srv_a.serve_forever, daemon=True)
    t_b = threading.Thread(target=srv_b.serve_forever, daemon=True)
    t_a.start()
    t_b.start()

    try:
        _serve(monkeypatch, "echo")
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


def test_http_500_raises_toolchaterror(server, monkeypatch) -> None:
    _serve(monkeypatch, "http_500")
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


def test_stop_unblocks_a_stalled_read(monkeypatch) -> None:
    """Handler sends headers then sleeps 5 s; stop() must unblock in <2 s."""
    srv, port = _make_server()
    t_srv = threading.Thread(target=srv.serve_forever, daemon=True)
    t_srv.start()

    try:
        _serve(monkeypatch, "stalled")
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


def test_raw_shape_a_two_calls_and_a_brace_inside_a_string() -> None:
    open_tag, close_tag = "<" + "tool_call" + ">", "</" + "tool_call" + ">"
    first = json.dumps({"name": "service_logs", "arguments": {"service": "a}b"}})
    second = json.dumps({"name": "gpu_stats", "arguments": {}})
    text = f"x {open_tag}{first}{close_tag} y {open_tag}{second}{close_tag}"
    assert parse_raw_tool_calls(text) == (
        ToolCall(name="service_logs", arguments={"service": "a}b"}),
        ToolCall(name="gpu_stats", arguments={}),
    )


# ------------------------------------------------------------------
# d1-score: next-token log-probability scoring
# ------------------------------------------------------------------


def test_score_next_token_shape_l(server, monkeypatch) -> None:
    _serve(monkeypatch, "score_shape_l")
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    result = chat.score_next_token("Is the sky blue?")
    expected = {" yes": -0.2, " no": -1.8, "Maybe": -4.0}
    assert result == expected


def test_score_next_token_shape_c(server, monkeypatch) -> None:
    _serve(monkeypatch, "score_shape_c")
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    result = chat.score_next_token("Is the sky blue?")
    expected = {" yes": -0.2, " no": -1.8, "Maybe": -4.0}
    assert result == expected


def test_score_request_body_and_path(server, monkeypatch) -> None:
    _serve(monkeypatch, "score_body_path")
    monkeypatch.setattr(_ServerHandler, "_request_body", None)
    monkeypatch.setattr(_ServerHandler, "_request_path", None)
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    chat.score_next_token("test prompt")
    body = json.loads(_ServerHandler._request_body)  # type: ignore[arg-type]
    assert _ServerHandler._request_path.endswith("/completions")  # type: ignore[arg-type]
    assert "/chat/" not in _ServerHandler._request_path  # type: ignore[arg-type]
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0
    assert body["logprobs"] == 20
    assert body["prompt"] == "test prompt"
    assert body["model"] == "test-model"


def test_score_no_logprobs_raises_toolchaterror(server, monkeypatch) -> None:
    _serve(monkeypatch, "score_no_logprobs")
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    with pytest.raises(ToolChatError, match="server returned no log-probabilities"):
        chat.score_next_token("hello")


def test_score_skips_non_numeric_logprob(server, monkeypatch) -> None:
    _serve(monkeypatch, "score_mixed_valid")
    srv = server
    chat = _make_toolchat(srv.port, stream=False)
    result = chat.score_next_token("hello")
    # " yes" has str logprob, " no" has bool logprob — both skipped
    assert result == {"Maybe": -4.0}


def test_score_connection_refused_raises_toolchaterror() -> None:
    """Point at a port with no server → ToolChatError."""
    chat = _make_toolchat(19998, stream=False)
    with pytest.raises(ToolChatError):
        chat.score_next_token("hello")


# ------------------------------------------------------------------
# d1-score: yes_no_probability
# ------------------------------------------------------------------


def test_yes_no_probability_sums_token_variants() -> None:
    import math as _math

    tokens = {
        " yes": _math.log(0.3),
        "Yes": _math.log(0.2),
        " no": _math.log(0.25),
        "NO": _math.log(0.25),
    }
    p_yes, mass = yes_no_probability(tokens)
    assert p_yes == pytest.approx(0.5)
    assert mass == pytest.approx(1.0)


def test_yes_no_probability_without_yes_or_no() -> None:
    tokens = {"maybe": -0.1}
    p_yes, mass = yes_no_probability(tokens)
    assert p_yes is None
    assert mass == 0.0


@pytest.mark.parametrize(
    "tokens",
    [
        {},
        {"yes": float("-inf")},
        {"": -1.0},
    ],
)
def test_yes_no_probability_never_raises(tokens) -> None:
    # Should never raise for any input
    p_yes, mass = yes_no_probability(tokens)
    assert isinstance(p_yes, (float, type(None)))
    assert isinstance(mass, float)


# ------------------------------------------------------------------
# d1-score: calibrated_logit
# ------------------------------------------------------------------


def test_calibrated_logit_zero_when_equal_and_positive_when_higher() -> None:
    assert calibrated_logit(0.5, 0.5) == pytest.approx(0.0)
    assert calibrated_logit(0.8, 0.5) > 0
    assert calibrated_logit(0.2, 0.5) < 0


def test_calibrated_logit_clamps_extremes() -> None:
    # p_yes = 1.0 and baseline = 0.0 must return a finite float
    result = calibrated_logit(1.0, 0.0)
    assert isinstance(result, float)
    assert math.isfinite(result)


@pytest.mark.parametrize(
    "junk",
    [{1: -1.0}, {" yes": float("nan")}, {"yes": 5.0}, {"yes": True}, {"no": "x"}],
)
def test_yes_no_probability_skips_junk_entries(junk) -> None:
    assert yes_no_probability(junk) == (None, 0.0)


# ------------------------------------------------------------------
# Review findings on PR #32 (qodo): redirects, stop before headers, scalars
# ------------------------------------------------------------------


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Behaviour chosen per server instance through ``server.mode``."""

    def log_message(self, *args) -> None:  # silence test output
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.server.hits += 1
        if self.server.mode == "redirect":
            self.send_response(307)
            self.send_header("Location", self.server.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.server.mode == "silent":
            time.sleep(5)  # never sends a status line within the test's patience


def _scripted_server(mode: str, redirect_to: str = "") -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedHandler)
    srv.mode, srv.redirect_to, srv.hits = mode, redirect_to, 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_redirect_is_refused_and_never_followed() -> None:
    """A local server must not be able to bounce the request to another host."""
    target = _scripted_server("silent")
    bouncer = _scripted_server(
        "redirect", f"http://127.0.0.1:{target.server_address[1]}/chat/completions"
    )
    try:
        chat = _make_toolchat(bouncer.server_address[1], stream=False)
        with pytest.raises(ToolChatError, match="HTTP 307"):
            chat.complete([{"role": "user", "content": "secret"}], [])
        assert bouncer.hits == 1
        assert target.hits == 0
    finally:
        bouncer.shutdown()
        target.shutdown()


def test_stop_unblocks_a_request_still_waiting_for_headers() -> None:
    srv = _scripted_server("silent")
    try:
        chat = _make_toolchat(srv.server_address[1], stream=False)
        errors: list[ToolChatError] = []

        def run_complete() -> None:
            try:
                chat.complete([{"role": "user", "content": "x"}], [])
            except ToolChatError as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_complete, daemon=True)
        worker.start()
        time.sleep(0.2)
        chat.stop()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert "stopped" in str(errors[0])
    finally:
        srv.shutdown()


def test_shape_b_skips_scalars_and_keeps_valid_calls() -> None:
    start, end = "<|tool_call_" + "start|>", "<|tool_call_" + "end|>"
    good = {"name": "gpu_stats", "arguments": {}}
    text = f"{start}{json.dumps([1, 'x', None, good, [good]])}{end} {start}[1]{end}"
    assert parse_raw_tool_calls(text) == (ToolCall(name="gpu_stats", arguments={}),)
