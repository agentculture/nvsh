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
* **Staged, never downloaded.** ``$HOME`` and ``NEEDLE3_LIB_PATH`` are
  pointed at the files :mod:`nvsh.tiers.needle_home` staged from the pinned
  wheel and weights, again *before* the import, and the verified base
  weights are symlinked where the library looks for them. Changing ``HOME``
  is safe because it only ever affects this child, which exists for nothing
  else. If the library's layout is not what nvsh staged for, the request
  fails with a clean error -- it never falls back to a download.
* **A clean protocol stream.** The real stdout fd is duplicated for frames
  and fd 1 is then pointed at stderr, so anything the native engine prints
  lands in the child's stderr (which the parent discards) instead of
  corrupting a frame.
"""

from __future__ import annotations

import importlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping

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


@dataclass(frozen=True)
class EngineSpec:
    """Where this child finds its engine: all of it staged and verified.

    ``tuned`` picks between the two ways cactus-needle can be given
    weights. Stock (``tuned=False``) loads the base model through its own
    cache -- which is why ``home`` exists -- and reports a real confidence.
    Tuned weights are passed as ``weights=``, which routes through the
    library's own ``FineTuneWorker`` subprocess and reports no confidence
    at all (``if self._tuned: response["confidence"] = None``).
    """

    lib: str | None = None
    weights: str | None = None
    home: str | None = None
    tuned: bool = False

    @classmethod
    def from_request(cls, request: Mapping[str, Any]) -> "EngineSpec":
        """Read a spec out of a request frame. Untrusted input: types are coerced."""
        return cls(
            lib=_as_text(request.get("lib")),
            weights=_as_text(request.get("weights")),
            home=_as_text(request.get("home")),
            tuned=bool(request.get("tuned")),
        )


def _as_text(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _import_needle() -> Any:  # pragma: no cover - needs cactus-needle installed
    """Import ``needle``. Never called at module scope."""
    import needle  # noqa: PLC0415 - deliberately lazy: see the module docstring

    return needle


def apply_engine_env(spec: EngineSpec, environ: MutableMapping[str, str] | None = None) -> None:
    """Point cactus-needle at the staged files -- before it is imported.

    ``$HOME`` is redirected because the library derives its weights cache
    from ``~``; changing it here only affects this child process, which
    exists for nothing else. ``NEEDLE3_LIB_PATH`` is the library's own
    override for the native engine, so it never looks in (or downloads to)
    its cache. The offline switches go on last and unconditionally.
    """
    target = os.environ if environ is None else environ
    if spec.home:
        target["HOME"] = spec.home
    if spec.lib:
        target["NEEDLE3_LIB_PATH"] = spec.lib
    harden_env(target)


def _library_fetch(module: Any) -> Any:
    """``needle.agent.fetch``, whether or not the package imported it already."""
    fetch = getattr(getattr(module, "agent", None), "fetch", None)
    if fetch is not None:
        return fetch
    return importlib.import_module(f"{module.__name__}.agent.fetch")


def link_base_weights(module: Any, weights_path: str) -> str:
    """Make the verified weights the base archive cactus-needle loads.

    The library has no override for the base weights path: it reads
    ``cache_dir(3)/base_weights(3)`` and downloads when that is missing.
    With ``$HOME`` already pointed at the staged home, that path is inside
    nvsh's cache, so a symlink to the pinned, sha256-verified file is
    enough -- and the download never happens.

    Raises ``RuntimeError`` when the library's layout is not what was
    measured. The caller turns that into an error reply; it never falls
    back to letting the library fetch (``HF_HUB_OFFLINE=1`` stays set).
    """
    try:
        fetch = _library_fetch(module)
        cache = Path(fetch.cache_dir(3))
        target = cache / fetch.base_weights(3)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"cactus-needle layout is not what nvsh staged for: {exc}") from exc

    verified = Path(weights_path).resolve()
    try:
        cache.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            if target.resolve() == verified:
                return str(target)
            target.unlink()
        target.symlink_to(verified)
    except OSError as exc:
        raise RuntimeError(f"could not stage the base weights at {target}: {exc}") from exc
    return str(target)


def build_engine(
    spec: EngineSpec,
    *,
    importer: Callable[[], Any] = _import_needle,
    operations: Iterable[Operation] | None = None,
    linker: Callable[[Any, str], str] = link_base_weights,
) -> Any:
    """Build the engine for *spec*, with the environment applied *first*."""
    apply_engine_env(spec)
    module = importer()
    kwargs: dict[str, Any] = {"tools": tool_functions(operations)}
    if spec.tuned:
        if not spec.weights:
            raise RuntimeError("tuned weights were asked for but no path was given")
        kwargs["weights"] = spec.weights
    elif spec.weights:
        linker(module, spec.weights)
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

    def __init__(self, builder: Callable[[EngineSpec], Any] | None = None) -> None:
        self._builder = builder if builder is not None else build_engine
        self._engine: Any = None

    def select(self, spec: EngineSpec, text: str) -> tuple[list, object]:
        """One selection. Loads the engine on first use, resets it after that."""
        if self._engine is None:
            self._engine = self._builder(spec)
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
        calls, confidence = session.select(
            EngineSpec.from_request(request), str(request.get("text") or "")
        )
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
