"""OpenAICompatAgent: stdlib urllib client for an OpenAI-compatible endpoint.

The fallback adapter when no headless CLI harness (pi, qwen, claude, codex)
is on PATH -- see ``nvsh.agent.registry.choose``. Talks plain
``POST {base_url}/chat/completions`` with ``"stream": true`` using only
``urllib.request`` (no third-party HTTP client; ``dependencies = []`` holds).

The bearer token, if any, is resolved at call time by
:func:`nvsh.config.resolve_bearer` -- from the environment variable *named*
by ``api_key_env``, from the file *named* by ``api_key_file``, or from the
default key file ``$XDG_CONFIG_HOME/nvsh/api_key`` -- never from a literal
key in config. With no source at all the ``Authorization`` header is simply
omitted rather than sent empty or fabricated. A refused source (a key file
other users can read, or one that is not there) is reported as a STATUS
event naming the file and its mode, never its content.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Iterator
from urllib.parse import urlsplit

from ..config import resolve_bearer
from ._subprocess import build_prompt, build_system_prompt
from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind, NvshAgent


class OpenAICompatAgent(NvshAgent):
    """Streams ``/chat/completions`` SSE ``data:`` lines as ``TEXT_DELTA`` events."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = dict(config or {})
        self._base_url = str(self._config.get("base_url", "")).rstrip("/")
        self._model = self._config.get("model", "default")
        self._cancelled = False
        self._closed = False
        self._response = None
        #: The last completed exchange, kept only so a steered message has
        #: the prior turn as its context (deviation d16). One turn, not a
        #: transcript: this adapter is stateless by design and nvsh's own
        #: context block carries the machine facts.
        self._last_prompt = ""
        self._last_reply = ""
        #: Operator text handed to :meth:`steer` and not yet sent.
        self._pending_steer: list[str] = []

    def start(self) -> None:
        self._cancelled = False
        self._closed = False

    def _headers(self) -> tuple[dict[str, str], str | None]:
        """Return ``(headers, diagnostic)`` -- the key lives only in the header."""
        headers = {"Content-Type": "application/json"}
        outcome = resolve_bearer(self._config)
        if outcome.bearer:
            headers["Authorization"] = f"Bearer {outcome.bearer}"
        return headers, outcome.diagnostic

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        # A cancel ends one turn, not the adapter: the daemon calls start()
        # once per warm session, so the flag has to be cleared per run or
        # every turn after the first stop streams nothing and ends in DONE.
        # Cleared *here*, not in the generator: a generator body only runs
        # at the first next(), and a cancel that lands between this call
        # and that first step belongs to this turn and must survive.
        self._cancelled = False
        return self._turn(request, context)

    def _turn(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        url = f"{self._base_url}/chat/completions"
        scheme = urlsplit(url).scheme
        if scheme not in ("http", "https"):
            yield AgentEvent(kind=EventKind.ERROR, error=f"unsupported URL scheme: {scheme!r}")
            return

        headers, key_diagnostic = self._headers()
        if key_diagnostic:
            yield AgentEvent(kind=EventKind.STATUS, text=key_diagnostic)

        prompt = build_prompt(request, context)
        messages = self._messages(prompt)
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "system", "content": build_system_prompt(context)}]
                + messages,
                "stream": True,
            }
        ).encode("utf-8")
        self._last_prompt = prompt
        self._last_reply = ""
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        failure = self._open_stream(req)
        if failure is not None:
            yield failure
            return

        try:
            yield from self._stream_events()
        finally:
            self._close_response()

    def _open_stream(self, req: urllib.request.Request) -> AgentEvent | None:
        """Open the SSE stream, or return the ERROR event that says why not."""
        try:
            # url's scheme is validated by run() (http/https only); base_url
            # is config-supplied, never user/network-controlled here.
            self._response = urllib.request.urlopen(req, timeout=5)  # nosec B310
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return AgentEvent(kind=EventKind.ERROR, error=f"HTTP {exc.code}: {body or exc.reason}")
        except urllib.error.URLError as exc:
            return AgentEvent(kind=EventKind.ERROR, error=f"connection failed: {exc.reason}")
        except OSError as exc:
            return AgentEvent(kind=EventKind.ERROR, error=f"connection failed: {exc}")
        return None

    @staticmethod
    def _sse_data(raw_line: bytes) -> str | None:
        """The payload of one ``data:`` line, or ``None`` for a line to skip."""
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            return None
        return line[len("data:") :].strip()

    @staticmethod
    def _content_delta(data: str) -> str | None:
        """The assistant text in one SSE chunk, or ``None`` if it carries none."""
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None
        choices = obj.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        return delta.get("content") or None

    def _stream_events(self) -> Iterator[AgentEvent]:
        """Map the open SSE stream to events, ending in DONE either way."""
        iterator = iter(self._response)
        while True:
            try:
                raw_line = next(iterator)
            except StopIteration:
                break
            except (OSError, AttributeError, ValueError):
                # A stalled stream's socket was just shut down from another
                # thread (force_stop(), which is what unblocks this very
                # read) -- that races http.client's own EOF handling here,
                # which can surface as an OSError or (rarely) an
                # AttributeError from its internal teardown rather than a
                # clean StopIteration. Either way the read is over now,
                # exactly like reaching EOF.
                break
            if self._cancelled:
                return
            data = self._sse_data(raw_line)
            if data is None:
                continue
            if data == "[DONE]":
                yield AgentEvent(kind=EventKind.DONE)
                return
            text = self._content_delta(data)
            if text:
                self._last_reply += text
                yield AgentEvent(kind=EventKind.TEXT_DELTA, text=text)
        # Stream closed without an explicit [DONE] -- treat as done anyway.
        if not self._cancelled:
            yield AgentEvent(kind=EventKind.DONE)

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        """This request's messages, with any steered text after the prior turn.

        There is no mid-turn channel here (one HTTP request per turn), so
        the operator's correction arrives as the *next* request. What this
        adapter adds is the thing only it still has: the previous exchange,
        sent ahead of it so the model can tell what it is being corrected
        about (deviation d16). Without it the correction would read as a
        fresh, contextless instruction.
        """
        steered = bool(self._pending_steer)
        self._pending_steer = []
        if not steered or not self._last_prompt:
            return [{"role": "user", "content": prompt}]
        return [
            {"role": "user", "content": self._last_prompt},
            {"role": "assistant", "content": self._last_reply},
            {"role": "user", "content": prompt},
        ]

    def steer(self, text: str) -> bool:
        """No mid-turn channel: remember to carry the prior turn forward.

        Always ``False`` -- the caller is told plainly that the text did not
        reach the running turn, so it sends it as the next request rather
        than pretending the agent has already read it. All this records is
        that the next request *is* a correction, which is what makes the
        previous exchange worth sending with it.
        """
        if text:
            self._pending_steer.append(text)
        return False

    def cancel(self) -> None:
        self._cancelled = True
        self._close_response()

    def _close_response(self) -> None:
        if self._response is not None:
            self._shutdown_socket()
            try:
                self._response.close()
            except (OSError, AttributeError):
                # AttributeError: the socket shutdown just above can wake a
                # concurrent blocked read in the turn's own thread, and its
                # EOF handling (``http.client``'s own ``_close_conn``) races
                # this thread to null out the response's internal buffer --
                # both sides are just trying to release the same resource.
                pass
            self._response = None

    def _shutdown_socket(self) -> None:
        """Shut down the response's socket before closing it.

        ``run()``'s ``_stream_events`` blocks on a plain read of the
        response (``for raw_line in self._response``) on whatever thread is
        driving the turn. A stalled backend that stops sending bytes and
        never closes its end leaves that read blocked for however long the
        request's own socket timeout is (5s) -- ``close()`` alone does not
        interrupt an in-progress blocking read from another thread. A POSIX
        ``shutdown(SHUT_RDWR)`` on the underlying socket does: it is the
        standard way to unblock a peer thread's blocking recv() on demand,
        which is what lets ``force_stop()`` return well under its 1s budget
        instead of waiting out the stall. Reaching for the private
        ``fp.raw._sock`` is the only way to get at that socket through
        ``http.client``/``urllib`` -- there is no public accessor -- and
        every step here is best-effort: a stop must never raise.
        """
        sock = getattr(getattr(self._response, "fp", None), "raw", None)
        sock = getattr(sock, "_sock", None) if sock is not None else None
        if isinstance(sock, socket.socket):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._close_response()
        self._closed = True

    def force_stop(self) -> None:
        """Cancel and release for certain: closes the response either way.

        ``cancel()``'s socket shutdown is what actually unblocks a stalled
        stream's blocked read; ``close()`` (idempotent) guarantees the
        response is gone even when ``cancel()`` was never called first.
        """
        try:
            self.cancel()
        finally:
            self.close()

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_calling=False,
            cancellation=True,
            persistent_session=False,
            local_model=True,
        )
