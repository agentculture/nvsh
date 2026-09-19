"""The child process that owns the Needle3 engine (task t9).

Run as ``python -m nvsh.tiers.needle_worker`` by
:class:`~nvsh.tiers.needle.NeedleTier`. It reads one length-prefixed JSON
request frame at a time from stdin and writes one response frame per
request, and it is the *only* place in nvsh that imports ``cactus-needle``.

Three things this module is careful about:

* **Offline first.** ``NEEDLE_TELEMETRY=0``, ``DO_NOT_TRACK=1`` and
  ``HF_HUB_OFFLINE=1`` are set in ``os.environ`` *before* ``needle`` is
  imported (spec targets c12/h11). cactus-needle ships telemetry on by
  default and posts on every ``complete()``; it also reaches Hugging Face
  from import-time code paths. The import itself happens inside a function,
  never at module scope, so merely importing this module costs nothing.
* **Selection only.** The engine is given callables whose bodies do
  nothing, built from :data:`nvsh.ops.table.OPERATIONS`, and only
  ``complete()`` is ever called -- never ``run()``, which executes those
  callables. ``reset()`` clears the session between requests.
* **A clean protocol stream.** The real stdout fd is duplicated for frames
  and fd 1 is then pointed at stderr, so anything the native engine prints
  lands in the child's stderr (which the parent discards) instead of
  corrupting a frame.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from typing import Any, Callable, Iterable, MutableMapping

from ..ops import table as ops_table
from ..ops._model import ArgSpec, Operation
from .needle import HEADER_BYTES, MAX_FRAME_BYTES, pack_frame

#: Set before the engine is imported. ``NEEDLE_TELEMETRY``/``DO_NOT_TRACK``
#: switch cactus-needle's hosted telemetry off; ``HF_HUB_OFFLINE`` stops
#: huggingface_hub from making a network call (including the
#: ``force_download=True`` "download counter" fetch, spike s15).
OFFLINE_ENV = {
    "NEEDLE_TELEMETRY": "0",
    "DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
}


# -- environment --


def harden_env(environ: MutableMapping[str, str] | None = None) -> None:
    """Force the offline/telemetry-off variables into *environ*.

    Overwrites whatever was inherited: an operator (or a harness) with
    ``NEEDLE_TELEMETRY=1`` in their shell must not be able to turn
    telemetry back on for nvsh's child.
    """
    target = os.environ if environ is None else environ
    for key, value in OFFLINE_ENV.items():
        target[key] = value


# -- tool schemas, generated from the operation table --


def _arg_schema(spec: ArgSpec) -> dict[str, Any]:
    if spec.kind == "choice":
        return {"type": "string", "enum": list(spec.choices)}
    return {"type": "string"}


def tool_schema(operation: Operation) -> dict[str, Any]:
    """The JSON tool schema for one operation. Nothing is named by hand."""
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {spec.name: _arg_schema(spec) for spec in operation.args},
    }
    if operation.args:
        parameters["required"] = [spec.name for spec in operation.args]
    return {
        "name": operation.name,
        "description": operation.description,
        "parameters": parameters,
    }


def tool_schemas(operations: Iterable[Operation] | None = None) -> list[dict[str, Any]]:
    """One schema per registered operation, in table order."""
    table = ops_table.OPERATIONS if operations is None else operations
    return [tool_schema(operation) for operation in table]


def _inert_tool(operation: Operation) -> Callable[..., None]:
    """A callable with the operation's name and schema and an empty body.

    cactus-needle wants callables so it can execute them; nvsh never lets it
    (``complete()`` selects, ``run()`` executes and is never called), so the
    body does nothing at all. The schema is attached as ``_needle_tool``,
    which the library prefers over introspecting the signature.
    """

    def selected_only(**_arguments: object) -> None:
        """Deliberately empty: a tier selects, it never executes."""

    selected_only.__name__ = operation.name
    selected_only.__doc__ = operation.description
    selected_only._needle_tool = tool_schema(operation)  # type: ignore[attr-defined]
    return selected_only


def tool_functions(operations: Iterable[Operation] | None = None) -> list[Callable[..., None]]:
    """The do-nothing callables handed to ``needle.Needle(tools=...)``."""
    table = ops_table.OPERATIONS if operations is None else operations
    return [_inert_tool(operation) for operation in table]


# -- the engine --


def _import_needle() -> Any:  # pragma: no cover - needs cactus-needle installed
    """Import ``needle``. Never called at module scope."""
    import needle  # noqa: PLC0415 - deliberately lazy: see the module docstring

    return needle


def build_engine(
    weights_path: str | None,
    *,
    importer: Callable[[], Any] = _import_needle,
    operations: Iterable[Operation] | None = None,
) -> Any:
    """Build the engine, with the offline environment applied *first*."""
    harden_env()
    module = importer()
    kwargs: dict[str, Any] = {"tools": tool_functions(operations)}
    if weights_path:
        kwargs["weights"] = str(weights_path)
    return module.Needle(**kwargs)


def extract_selection(envelope: object) -> tuple[list, object]:
    """Reduce one ``complete()`` envelope to ``(calls, confidence)``.

    The whole of nvsh's knowledge of cactus-needle's return shape lives
    here. Measured against cactus-needle 3.0.2: ``complete()`` returns a
    dict with ``type`` (``"call"`` when it selected something),
    ``function_calls`` (a list of ``{"name", "arguments"}``) and
    ``confidence`` (a float, or ``None`` for fine-tuned weights). Anything
    else reduces to no calls; the parent's ``decide()`` is what judges the
    content.
    """
    if not isinstance(envelope, dict):
        return ([], None)
    calls = envelope.get("function_calls")
    return (calls if isinstance(calls, list) else [], envelope.get("confidence"))


class EngineSession:
    """Holds the engine across requests, resetting it between them."""

    def __init__(self, builder: Callable[[str | None], Any] | None = None) -> None:
        self._builder = builder if builder is not None else build_engine
        self._engine: Any = None

    def select(self, weights_path: str | None, text: str) -> tuple[list, object]:
        """One selection. Loads the engine on first use, resets it after that."""
        if self._engine is None:
            self._engine = self._builder(weights_path)
        else:
            self._engine.reset()
        return extract_selection(self._engine.complete(text))


# -- protocol --


_HEADER = struct.Struct(">I")


def _protocol_stream() -> Any:
    """Duplicate fd 1 for frames, then point fd 1 at stderr.

    Whatever the native engine writes to "stdout" after this lands in the
    child's stderr (discarded by the parent) and can never corrupt a frame.
    """
    frames = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return frames


def _read_exact(stream: Any, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: Any) -> dict | None:
    """One request frame, or ``None`` at end of input or on a bad frame."""
    header = _read_exact(stream, HEADER_BYTES)
    if header is None:
        return None
    size = _HEADER.unpack(header)[0]
    if size > MAX_FRAME_BYTES:
        return None
    payload = _read_exact(stream, size)
    if payload is None:
        return None
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def handle(session: EngineSession, request: dict) -> dict:
    """Answer one request. Any engine failure becomes an error response."""
    request_id = request.get("id")
    if request.get("op") != "select":
        return {"id": request_id, "ok": False, "error": f"unknown op {request.get('op')!r}"}
    try:
        calls, confidence = session.select(request.get("weights"), str(request.get("text") or ""))
    except Exception as exc:  # the engine is native code: nothing may escape
        return {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"id": request_id, "ok": True, "calls": calls, "confidence": confidence}


def _send(frames: Any, response: dict) -> None:
    try:
        payload = pack_frame(response)
    except (TypeError, ValueError) as exc:
        payload = pack_frame(
            {"id": response.get("id"), "ok": False, "error": f"unsendable reply: {exc}"}
        )
    frames.write(payload)


def main(argv: list[str] | None = None) -> int:
    """Serve request frames until stdin closes or a shutdown is asked for."""
    frames = _protocol_stream()
    session = EngineSession()
    while True:
        request = read_frame(sys.stdin.buffer)
        if request is None or request.get("op") == "shutdown":
            return 0
        _send(frames, handle(session, request))


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
