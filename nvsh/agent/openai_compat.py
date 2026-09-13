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
import urllib.error
import urllib.request
from typing import Iterator
from urllib.parse import urlsplit

from ..config import resolve_bearer
from ._subprocess import build_prompt
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
                "messages": messages,
                "stream": True,
            }
        ).encode("utf-8")
        self._last_prompt = prompt
        self._last_reply = ""
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            # url's scheme is validated above (http/https only); base_url is
            # config-supplied, never user/network-controlled at this call site.
            self._response = urllib.request.urlopen(req, timeout=5)  # nosec B310
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            yield AgentEvent(kind=EventKind.ERROR, error=f"HTTP {exc.code}: {body or exc.reason}")
            return
        except urllib.error.URLError as exc:
            yield AgentEvent(kind=EventKind.ERROR, error=f"connection failed: {exc.reason}")
            return
        except OSError as exc:
            yield AgentEvent(kind=EventKind.ERROR, error=f"connection failed: {exc}")
            return

        try:
            for raw_line in self._response:
                if self._cancelled:
                    return
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    yield AgentEvent(kind=EventKind.DONE)
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    self._last_reply += text
                    yield AgentEvent(kind=EventKind.TEXT_DELTA, text=text)
            # Stream closed without an explicit [DONE] -- treat as done anyway.
            if not self._cancelled:
                yield AgentEvent(kind=EventKind.DONE)
        finally:
            self._close_response()

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
            try:
                self._response.close()
            except OSError:
                pass
            self._response = None

    def close(self) -> None:
        if self._closed:
            return
        self._close_response()
        self._closed = True

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_calling=False,
            cancellation=True,
            persistent_session=False,
            local_model=True,
        )
