"""OpenAICompatAgent: stdlib urllib client for an OpenAI-compatible endpoint.

The fallback adapter when no headless CLI harness (pi, qwen, claude, codex)
is on PATH -- see ``nvsh.agent.registry.choose``. Talks plain
``POST {base_url}/chat/completions`` with ``"stream": true`` using only
``urllib.request`` (no third-party HTTP client; ``dependencies = []`` holds).

The bearer token, if any, is read at call time from the environment variable
*named* by ``api_key_env`` in config -- never a literal key. A missing env
var simply omits the ``Authorization`` header rather than sending an empty
or fabricated one.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterator
from urllib.parse import urlsplit

from ._subprocess import build_prompt
from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind, NvshAgent


class OpenAICompatAgent(NvshAgent):
    """Streams ``/chat/completions`` SSE ``data:`` lines as ``TEXT_DELTA`` events."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = dict(config or {})
        self._base_url = str(self._config.get("base_url", "")).rstrip("/")
        self._api_key_env = self._config.get("api_key_env")
        self._model = self._config.get("model", "default")
        self._cancelled = False
        self._closed = False
        self._response = None

    def start(self) -> None:
        self._cancelled = False
        self._closed = False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key_env:
            key = os.environ.get(self._api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        url = f"{self._base_url}/chat/completions"
        scheme = urlsplit(url).scheme
        if scheme not in ("http", "https"):
            yield AgentEvent(kind=EventKind.ERROR, error=f"unsupported URL scheme: {scheme!r}")
            return

        prompt = build_prompt(request, context)
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=self._headers(), method="POST")

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
                    yield AgentEvent(kind=EventKind.TEXT_DELTA, text=text)
            # Stream closed without an explicit [DONE] -- treat as done anyway.
            if not self._cancelled:
                yield AgentEvent(kind=EventKind.DONE)
        finally:
            self._close_response()

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
