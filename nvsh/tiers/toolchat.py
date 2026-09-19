"""Tool-call chat client for an OpenAI-compatible server on THIS machine.

stdlib only. The transport is ``http.client`` rather than ``urllib`` for two
reasons that are both part of the contract:

* ``http.client`` never follows a redirect, so a local server cannot bounce a
  request (with its messages and tools) to another host. Any status other
  than 200 is an error.
* the connection's socket exists from ``connect()`` on, so :meth:`ToolChat.stop`
  can unblock a request that is still waiting for response *headers*, not only
  one that is reading a body.

Usage::

    chat = ToolChat("http://127.0.0.1:8080/v1", "my-model")
    reply = chat.complete(messages=[...], tools=[...])
    for call in reply.tool_calls:
        ...  # call.name, call.arguments -- untrusted: pass them through decide()
"""

from __future__ import annotations

import http.client
import json
import math
import re
import socket
import threading
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlsplit

_NO_LOGPROBS = "server returned no log-probabilities"
_HOST_ACCEPT: frozenset[str] = frozenset(("127.0.0.1", "localhost", "::1"))
_DEFAULT_PORT = 80


class ToolChatError(Exception):
    """Raised for every client-side failure; nothing else escapes this module."""


@dataclass(frozen=True)
class ToolCall:
    """A single tool call extracted from a chat completion. Untrusted."""

    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ChatReply:
    """The reply from the model: text content plus tool calls."""

    text: str
    tool_calls: tuple[ToolCall, ...]


# -- localhost enforcement --


def require_localhost(base_url: str) -> None:
    """Raise ``ToolChatError`` unless the URL is ``http://`` to this machine.

    The host is parsed and compared exactly; ``localhost.example.com`` starts
    with ``localhost`` and is refused.
    """
    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname
    except ValueError as exc:
        raise ToolChatError(f"invalid base URL: {exc}") from exc
    if parsed.scheme != "http":
        raise ToolChatError(f"only http:// scheme accepted (got {parsed.scheme!r})")
    if host is None or host not in _HOST_ACCEPT:
        raise ToolChatError(f"host must be 127.0.0.1, ::1, or localhost (got {host!r})")


# -- raw tool-call parsing --

# Shape A: a JSON object between tool-call tags. The body is taken between
# the TAGS (non-greedy across the closing tag, not across a brace), so nested
# braces and a "}" inside a string value survive; json.loads does the rest.
_RE_SHAPE_A = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Shape B: a JSON array between start/end markers.
_RE_SHAPE_B = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)


def parse_raw_tool_calls(text: str) -> tuple[ToolCall, ...]:
    """Extract tool calls a model printed as text instead of returning them.

    Two shapes, anywhere in the text, possibly several: a JSON object between
    tool-call tags, and a JSON array between start/end markers. Anything that
    does not parse, or is not an object with a string ``name`` and a dict
    ``arguments``, is skipped. Never raises; returns ``()`` when nothing fits.
    """
    if not isinstance(text, str):
        return ()
    calls: list[ToolCall] = []
    for candidate in _raw_candidates(text):
        call = _make_call(candidate)
        if call is not None:
            calls.append(call)
    return tuple(calls)


def _raw_candidates(text: str) -> Iterable[object]:
    for match in _RE_SHAPE_A.finditer(text):
        yield _loads_or_none(match.group(1))
    for match in _RE_SHAPE_B.finditer(text):
        decoded = _loads_or_none(match.group(1))
        if isinstance(decoded, list):
            yield from decoded


def _loads_or_none(raw: str) -> object:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _make_call(candidate: object) -> ToolCall | None:
    """A ToolCall from a decoded object, or None when it is not call-shaped."""
    if not isinstance(candidate, dict):
        return None
    name = candidate.get("name")
    arguments = candidate.get("arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments)


# -- structured tool-call parsing --


def _call_from_function(function: object) -> ToolCall | None:
    """A ToolCall from an OpenAI ``function`` object (arguments is a JSON string)."""
    if not isinstance(function, dict):
        return None
    arguments = function.get("arguments")
    decoded = _loads_or_none(arguments) if isinstance(arguments, str) else None
    return _make_call({"name": function.get("name"), "arguments": decoded})


def _calls_from_message(message: dict[str, Any]) -> list[ToolCall]:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    calls = [_call_from_function(_function_of(item)) for item in raw_calls]
    return [call for call in calls if call is not None]


def _function_of(item: object) -> object:
    return item.get("function") if isinstance(item, dict) else None


def _first_choice(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ToolChatError("server reply is not a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ToolChatError("server reply has no choices")
    return choices[0]


def _reply(text: str, calls: list[ToolCall]) -> ChatReply:
    """Fall back to tool calls printed in the text when none came structured."""
    found = tuple(calls) if calls else parse_raw_tool_calls(text)
    return ChatReply(text=text, tool_calls=found)


class _StreamAccumulator:
    """Joins SSE deltas: text is appended, tool-call fragments are joined by index."""

    def __init__(self) -> None:
        self.text = ""
        self._names: dict[int, str] = {}
        self._arguments: dict[int, str] = {}

    def add(self, delta: object) -> None:
        if not isinstance(delta, dict):
            return
        content = delta.get("content")
        if isinstance(content, str):
            self.text += content
        fragments = delta.get("tool_calls")
        if isinstance(fragments, list):
            for fragment in fragments:
                self._add_fragment(fragment)

    def _add_fragment(self, fragment: object) -> None:
        if not isinstance(fragment, dict) or not isinstance(fragment.get("index"), int):
            return
        index = fragment["index"]
        function = fragment.get("function")
        if not isinstance(function, dict):
            return
        name = function.get("name")
        if isinstance(name, str) and name and index not in self._names:
            # First non-empty name wins: some servers resend the name on every
            # chunk, which must not double it.
            self._names[index] = name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            self._arguments[index] = self._arguments.get(index, "") + arguments

    def calls(self) -> list[ToolCall]:
        built = (
            _call_from_function({"name": name, "arguments": self._arguments.get(index, "")})
            for index, name in sorted(self._names.items())
        )
        return [call for call in built if call is not None]


def _sse_payloads(response: http.client.HTTPResponse) -> Iterable[object]:
    """Decoded ``data:`` payloads of an SSE body, up to ``[DONE]`` or end of stream."""
    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        yield _loads_or_none(data)


def _delta_of(payload: object) -> object:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("delta")


# -- log-probability scoring --


def _valid_logprob(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


def _shape_l(logprobs: dict[str, Any]) -> dict[str, float]:
    """Legacy shape: ``top_logprobs`` is a list whose first item maps token -> logprob."""
    top = logprobs.get("top_logprobs")
    first = top[0] if isinstance(top, list) and top else None
    if not isinstance(first, dict):
        return {}
    return {
        token: float(value)
        for token, value in first.items()
        if isinstance(token, str) and _valid_logprob(value)
    }


def _shape_c(logprobs: dict[str, Any]) -> dict[str, float]:
    """Content shape: ``content[0].top_logprobs`` is a list of {token, logprob}."""
    content = logprobs.get("content")
    first = content[0] if isinstance(content, list) and content else None
    entries = first.get("top_logprobs") if isinstance(first, dict) else None
    if not isinstance(entries, list):
        return {}
    return {
        entry["token"]: float(entry["logprob"])
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("token"), str)
        and _valid_logprob(entry.get("logprob"))
    }


def yes_no_probability(top_logprobs: dict[str, float]) -> tuple[float | None, float]:
    """``(p_yes, mass)`` from a next-token distribution. Never raises.

    Token variants (" yes", "Yes", "YES") are summed. ``p_yes`` is normalised
    over yes+no only; ``mass`` is how much of the distribution those two words
    hold. ``(None, 0.0)`` when neither word appears.
    """
    if not isinstance(top_logprobs, dict):
        return (None, 0.0)
    masses = {"yes": 0.0, "no": 0.0}
    for token, logprob in top_logprobs.items():
        if not isinstance(token, str) or not _valid_logprob(logprob) or logprob > 0:
            continue  # a log-probability is never positive; skip junk, never raise
        word = token.strip().casefold()
        if word in masses:
            masses[word] += math.exp(logprob)
    mass = masses["yes"] + masses["no"]
    if mass > 0:
        return (masses["yes"] / mass, mass)
    return (None, 0.0)


def calibrated_logit(p_yes: float, baseline_p_yes: float) -> float:
    """How much more the model says yes for this request than for an empty one."""
    return _logit(p_yes) - _logit(baseline_p_yes)


def _logit(probability: float) -> float:
    clamped = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(clamped / (1 - clamped))


# -- client --


class ToolChat:
    """Minimal OpenAI-compatible chat client with tool calling, localhost only."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 30.0,
        stream: bool = True,
    ) -> None:
        require_localhost(base_url)
        parsed = urlsplit(base_url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or _DEFAULT_PORT
        self._prefix = parsed.path.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._stream = stream
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._cancelled = False

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        """One chat completion. Raises nothing except ``ToolChatError``."""
        body = {"model": self._model, "messages": messages, "tools": tools, "stream": self._stream}
        return self._request("/chat/completions", body, self._parse_completion)

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        """Token -> natural-log probability for the first generated token.

        One prefill and one token: nothing is generated beyond it. Raises
        nothing except ``ToolChatError``.
        """
        body = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": top,
            "stream": False,
        }
        return self._request("/completions", body, self._parse_score)

    def stop(self) -> None:
        """Make a blocked request fail promptly. Safe from another thread.

        Works before the response headers arrive as well as mid-body: the
        connection's socket is shut down, which unblocks any pending read.
        """
        with self._lock:
            self._cancelled = True
            sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already closed

    # -- internals --

    def _request(self, path: str, body: dict[str, Any], parse: Any) -> Any:
        connection = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        with self._lock:
            self._cancelled = False
        try:
            return parse(self._post(connection, path, body))
        except ToolChatError:
            raise
        except Exception as exc:  # transport, decoding and shape errors alike
            reason = "stopped" if self._cancelled else f"{type(exc).__name__}: {exc}"
            raise ToolChatError(f"request to {path} failed: {reason}") from exc
        finally:
            with self._lock:
                self._sock = None
            connection.close()

    def _post(
        self, connection: http.client.HTTPConnection, path: str, body: dict[str, Any]
    ) -> http.client.HTTPResponse:
        connection.connect()
        with self._lock:
            # Kept separately from the connection: http.client drops its own
            # reference once a "Connection: close" response starts, while the
            # body is still being read from the same socket.
            self._sock = connection.sock
            if self._cancelled:
                raise ToolChatError("request stopped before it was sent")
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        connection.request("POST", self._prefix + path, body=payload, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            # Includes every 3xx: a redirect is never followed.
            raise ToolChatError(f"server answered HTTP {response.status} for {path}")
        return response

    def _parse_completion(self, response: http.client.HTTPResponse) -> ChatReply:
        if self._stream:
            accumulator = _StreamAccumulator()
            for payload in _sse_payloads(response):
                accumulator.add(_delta_of(payload))
            return _reply(accumulator.text, accumulator.calls())
        message = _first_choice(json.loads(response.read())).get("message")
        if not isinstance(message, dict):
            raise ToolChatError("server reply has no message")
        content = message.get("content")
        return _reply(content if isinstance(content, str) else "", _calls_from_message(message))

    @staticmethod
    def _parse_score(response: http.client.HTTPResponse) -> dict[str, float]:
        logprobs = _first_choice(json.loads(response.read())).get("logprobs")
        if not isinstance(logprobs, dict):
            raise ToolChatError(_NO_LOGPROBS)
        scores = _shape_l(logprobs) or _shape_c(logprobs)
        if not scores:
            raise ToolChatError(_NO_LOGPROBS)
        return scores
