"""Tool-call chat client over stdlib HTTP: urllib only, OpenAI-compatible.

A minimal client for an OpenAI-compatible chat server on THIS machine that
supports tool calling.  No third-party dependencies — only ``http.server``
and ``urllib.request`` from the Python standard library.

Usage::

    from nvsh.tiers.toolchat import ToolChat, ChatReply

    chat = ToolChat("http://127.0.0.1:8000", "my-model")
    reply: ChatReply = chat.complete(
        messages=[{"role": "user", "content": "check gpu"}],
        tools=[{"type": "function", "function": {"name": "gpu_stats"}}],
    )
    for tc in reply.tool_calls:
        ...  # tc.name, tc.arguments
"""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

# -- exceptions --


class ToolChatError(Exception):
    """Raised for all client-side errors during HTTP communication."""


# -- data classes --


@dataclass(frozen=True)
class ToolCall:
    """A single tool call extracted from a chat completion."""

    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ChatReply:
    """The full reply from the model: text content + tool calls."""

    text: str
    tool_calls: tuple[ToolCall, ...]


# -- localhost enforcement --


_HOST_ACCEPT: frozenset[str] = frozenset(("127.0.0.1", "localhost", "::1"))


def require_localhost(base_url: str) -> None:
    """Raise ``ToolChatError`` unless the URL is ``http://`` to localhost.

    Parses the host with ``urllib.parse.urlsplit`` and checks against a
    whitelist.  A host like ``localhost.example.com`` is refused — only
    an *exact* match passes.
    """
    parsed = urlsplit(base_url)
    if parsed.scheme != "http":
        raise ToolChatError(f"only http:// scheme accepted (got {parsed.scheme!r})")
    host = parsed.hostname
    if host is None or host not in _HOST_ACCEPT:
        raise ToolChatError(f"host must be 127.0.0.1, ::1, or localhost (got {host!r})")


# -- raw tool-call parsing --

# Shape A: ``name`` ``function`` ``arguments`` <json object> ````
# The markers and JSON can be adjacent; backticks between them are shared.
# We match any number of backticks around the keywords and a closing run.
_RE_SHAPE_A = re.compile(
    r"`+name`+"  # ``name`` or more
    r"`+function`+"  # ``function`` or more
    r"`+arguments`+"  # ``arguments`` or more
    r"({.*?})"  # JSON object (non-greedy to match innermost {})
    r"`+"  # closing ```` or more
)

# Shape B: <|tool_call_start|> [JSON array] <|tool_call_end|>
_RE_SHAPE_B = re.compile(
    r"<\|tool_call_start\|>(\[.*?\])<\|tool_call_end\|>",
    re.DOTALL,
)


def parse_raw_tool_calls(text: str) -> tuple[ToolCall, ...]:
    """Extract ToolCalls from raw model output text.

    Supports two shapes anywhere in the text:

    A. ``name`` ``function`` ``arguments`` {"name": "f", "arguments": {}} ````

    B. <|tool_call_start|>[{"name": "f", "arguments": {}}]<|tool_call_end|>

    Each valid JSON object found with a string ``"name"`` and dict
    ``"arguments"`` becomes a ``ToolCall``.  Invalid fragments are skipped
    silently.  Never raises; returns ``()`` when nothing is found.
    """
    results: list[ToolCall] = []
    for _obj in _find_json_objects(text, _RE_SHAPE_A):
        call = _make_call(_obj)
        if call is not None:
            results.append(call)
    for _arr in _find_json_arrays(text, _RE_SHAPE_B):
        call = _make_call(_arr)
        if call is not None:
            results.append(call)
    return tuple(results)


def _find_json_objects(text: str, pattern: re.Pattern) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    for m in pattern.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                objs.append(obj)
        except json.JSONDecodeError:
            pass
    return objs


def _find_json_arrays(text: str, pattern: re.Pattern) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    for m in pattern.finditer(text):
        try:
            arr = json.loads(m.group(1))
            if isinstance(arr, list):
                objs.extend(arr)
        except json.JSONDecodeError:
            pass
    return objs


def _make_call(obj: dict[str, Any]) -> ToolCall | None:
    name = obj.get("name")
    args = obj.get("arguments")
    if isinstance(name, str) and isinstance(args, dict):
        return ToolCall(name=name, arguments=args)
    return None


# -- HTTP client --


class ToolChat:
    """OpenAI-compatible chat client with tool-call support.

    Uses only ``urllib.request`` for HTTP (no third-party deps).  The
    constructor calls :func:`require_localhost` to enforce that ``base_url``
    points at a localhost endpoint.

    ``complete()`` returns a :class:`ChatReply` whose ``text`` and
    ``tool_calls`` are always populated (never ``None``).  Errors from the
    HTTP layer or from invalid JSON are wrapped in :class:`ToolChatError`.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 30.0,
        stream: bool = True,
    ) -> None:
        require_localhost(base_url)
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._stream = stream
        self._cancelled = False
        self._response = None

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        """Send a chat request and return the parsed reply.

        POSTs to ``{base_url}/chat/completions`` with ``{"stream": <bool>}``.
        On success returns a :class:`ChatReply`; on error raises
        :class:`ToolChatError`.
        """
        self._cancelled = False
        url = f"{self._base_url}/chat/completions"
        payload = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "tools": tools,
                "stream": self._stream,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            resp = urllib.request.urlopen(req, timeout=self._timeout)  # nosec B310
        except urllib.error.HTTPError as exc:
            raise ToolChatError(f"HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ToolChatError(f"connection failed: {exc.reason}") from exc
        except OSError as exc:
            raise ToolChatError(f"connection failed: {exc}") from exc
        except TimeoutError:
            raise ToolChatError("request timed out") from None
        except ValueError as exc:
            raise ToolChatError(f"HTTP error: {exc}") from exc

        self._response = resp
        try:
            if self._stream:
                return self._parse_stream(resp)
            return self._parse_non_stream(resp)
        except ToolChatError:
            raise
        except Exception as exc:
            raise ToolChatError(f"parse error: {exc}") from exc
        finally:
            self._close_response()

    def stop(self) -> None:
        """Signal that a blocked ``complete()`` should return promptly.

        Safe to call from another thread.  Shuts down the underlying socket
        so a blocked read in another thread is interrupted immediately.
        """
        self._cancelled = True
        self._close_response()

    def _close_response(self) -> None:
        """Close the response and optionally shut down its socket.

        If ``self._cancelled`` is True (``stop()`` was called), shuts down
        the socket so a blocked read in another thread returns promptly.
        Otherwise just closes the response cleanly without touching the
        socket — the server expects a clean HTTP response close.
        """
        if self._response is None:
            return
        if self._cancelled:
            try:
                sock = getattr(getattr(self._response, "fp", None), "raw", None)
                sock = getattr(sock, "_sock", None) if sock is not None else None
                if isinstance(sock, socket.socket):
                    sock.shutdown(socket.SHUT_RDWR)
            except (OSError, AttributeError):
                pass
        try:
            self._response.close()
        except (OSError, AttributeError):
            pass
        self._response = None

    def _parse_non_stream(self, resp: Any) -> ChatReply:
        """Parse a non-streamed (HTTP-level) ChatCompletion response."""
        raw = resp.read().decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise ToolChatError("response is not valid JSON")

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise ToolChatError("no choices in response")

        message = choices[0].get("message") or {}
        content = message.get("content")
        text = content if isinstance(content, str) and content else ""

        tool_calls_raw = message.get("tool_calls")
        tool_calls = self._build_tool_calls_from_raw(tool_calls_raw)

        # Fallback: if no tool calls from the API fields, try raw parsing.
        if not tool_calls:
            tool_calls = parse_raw_tool_calls(text)

        return ChatReply(text=text, tool_calls=tool_calls)

    @staticmethod
    def _build_tool_calls_from_raw(
        raw_calls: list[dict] | None,
    ) -> tuple[ToolCall, ...]:
        """Build ToolCalls from the API's ``tool_calls`` field."""
        if not raw_calls or not isinstance(raw_calls, list):
            return ()
        results: list[ToolCall] = []
        for tc in raw_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            args_str = fn.get("arguments")
            if not isinstance(name, str):
                continue
            if not isinstance(args_str, str):
                continue
            try:
                args = json.loads(args_str)
                if isinstance(args, dict):
                    results.append(ToolCall(name=name, arguments=args))
            except json.JSONDecodeError:
                # Invalid arguments JSON → skip this call.
                pass
        return tuple(results)

    def _parse_stream(self, resp: Any) -> ChatReply:
        """Parse a streamed (SSE) ChatCompletion response."""
        text = ""
        # index -> {"name": str, "arguments": str}
        tc_accum: dict[int, dict[str, str]] = {}

        for raw_line in resp:
            if self._cancelled:
                raise ToolChatError("complete() was stopped")
            try:
                line = raw_line.decode("utf-8", errors="replace").strip()
            except (OSError, ValueError, AttributeError):
                break

            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            if not data:
                continue

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices")
            if not choices or not isinstance(choices, list):
                continue

            delta = choices[0].get("delta")
            if not isinstance(delta, dict):
                continue

            # Accumulate text.
            chunk_text = delta.get("content")
            if isinstance(chunk_text, str) and chunk_text:
                text += chunk_text

            # Accumulate tool_call fragments.
            chunk_tcs = delta.get("tool_calls")
            if isinstance(chunk_tcs, list):
                for frag in chunk_tcs:
                    if not isinstance(frag, dict):
                        continue
                    idx = frag.get("index")
                    if not isinstance(idx, int):
                        continue
                    acc = tc_accum.setdefault(idx, {"name": "", "arguments": ""})
                    fn = frag.get("function")
                    if isinstance(fn, dict):
                        fn_name = fn.get("name")
                        if isinstance(fn_name, str) and fn_name:
                            acc["name"] += fn_name  # accumulate, not replace
                        fn_args = fn.get("arguments")
                        if isinstance(fn_args, str):
                            acc["arguments"] += fn_args  # accumulate, not replace

        # Finalise accumulated tool calls.
        tool_calls: list[ToolCall] = []
        for _idx, acc in tc_accum.items():
            name = acc.get("name", "")
            args_str = acc.get("arguments", "")
            if not name:
                continue
            try:
                args = json.loads(args_str)
                if isinstance(args, dict):
                    tool_calls.append(ToolCall(name=name, arguments=args))
            except json.JSONDecodeError:
                # Invalid accumulated arguments → skip.
                pass
        tool_calls_tuple = tuple(tool_calls)

        # Fallback: if no tool calls from streaming, try raw parsing.
        if not tool_calls_tuple:
            tool_calls_tuple = parse_raw_tool_calls(text)

        return ChatReply(text=text, tool_calls=tool_calls_tuple)
