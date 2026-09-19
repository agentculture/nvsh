"""Tier 2's model runtime as a Docker container nvsh starts and stops.

This module is the **only** place in nvsh that launches a container, and the
launch line it builds comes from two sources and nothing else: the operator's
``[tiers.lfm]`` config and platform detection. No request text and no model
output reaches it -- :func:`render_launch` does not even take them as
arguments. A grep-style test in ``tests/test_tier_runtime_docker.py`` pins
that "docker run" appears nowhere else under ``nvsh/``.

Three things are tables rather than branches, so adding a device class or an
engine is data, not code:

* :data:`GPU_FLAGS` maps a :class:`~nvsh.platform._model.Platform` kind to the
  flag that exposes the GPU to a container. Measured (spec s17/s21, and
  ``docs/platforms.md``): the DGX Spark has no ``nvidia`` runtime and takes
  ``--gpus all``; Jetson's container toolkit runs in csv mode and *refuses*
  ``--gpus``, taking ``--runtime nvidia``. Anything else gets no flag and
  runs on the CPU. nvsh changes no Docker configuration on any of them.
* :data:`ENGINES` maps an engine name to its container port, its argument
  template, whether it needs the model directory mounted, and its health
  path. Switching engine is a config change only.
* the image is pinned by ``@sha256:`` digest -- an override in
  ``[tiers.lfm] image`` or an entry in ``pins.json``'s ``images`` list. A
  tag (``:latest`` or any other) is refused.

Name and port are derived from the OS user (``nvsh-tier2-<uid>`` on
``127.0.0.1:<base + uid % span>``) so two operators on one machine do not
collide, and the port is published on the loopback address only.

Nothing here is importable on the hot success path: the daemon reaches it
lazily, through :mod:`nvsh.tiers.manager`.
"""

from __future__ import annotations

import contextlib
import http.client
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit

from ..platform._model import Platform
from ..redact import redact
from .runtime import AttachedRuntime, Runtime, RuntimeUnavailable

#: The one executable this module ever names.
DOCKER = "docker"

#: Container naming and port allocation, per OS user. The span is as wide as
#: the unprivileged port range allows above ``PORT_BASE`` (18400..58399, all
#: <= 65535), so two ordinary uids far apart (uid 1000 and uid 2000, say) do
#: not land on the same host port; ``[tiers.lfm] port``
#: still overrides this for a site that needs a fixed number.
NAME_PREFIX = "nvsh-tier2-"
PORT_BASE = 18400
PORT_SPAN = 40000

#: The address the container publishes on, on the host. Nothing else.
LOOPBACK = "127.0.0.1"

#: What the engine binds to *inside* the container's own network namespace.
#: Not a host bind: the only thing that reaches the host is what
#: ``-p 127.0.0.1:<port>:<container port>`` publishes, and an engine bound
#: to the loopback of its own namespace would not be reachable at all.
_IN_CONTAINER_BIND = "0.0.0.0"  # nosec B104 - container namespace, published on 127.0.0.1 only

#: Where a mounted model directory appears inside the container, read-only.
MODEL_MOUNT = "/models"
#: Where a downloading engine keeps its model cache inside the container.
CACHE_MOUNT = "/cache"
#: The host side of that cache, under nvsh's own tier cache directory.
HF_CACHE_NAME = "hf"

DEFAULT_ENGINE = "llama-server"
DEFAULT_CTX = 4096
#: Up-front GPU memory share for engines that reserve one (vLLM, SGLang).
#: Small on purpose: Tier 2 serves a sub-1B model beside the operator's own work.
DEFAULT_GPU_FRACTION = 0.08
MIN_GPU_FRACTION = 0.02
MAX_GPU_FRACTION = 0.95
DEFAULT_STARTUP_SECONDS = 120.0
POLL_SECONDS = 1.0
PROBE_SECONDS = 2.0
LOG_TAIL_CHARS = 400

MANAGED = "managed"
ATTACH = "attach"

#: Exit status a shell reports for a command that is not on ``PATH``.
_NOT_FOUND = 127

_DOCKER_TIMEOUT = 20.0
_LAUNCH_TIMEOUT = 120.0

_RUNNING = "running"
_STOPPED = "stopped"

#: ``<repo>@sha256:<64 hex>`` and nothing else. A tag has no ``@sha256:`` and
#: is refused by the same check; the leading character must be alphanumeric,
#: so a ref can never arrive at docker looking like a flag.
_DIGEST_RE = re.compile(r"\A[A-Za-z0-9][^@\s]*@sha256:[0-9a-f]{64}\Z")

#: Docker's own wording for "the host port we asked to publish on is taken".
_PORT_ALLOCATED_RE = re.compile(r"port is already allocated", re.IGNORECASE)

#: Bounds for the numeric ``[tiers.lfm]`` settings the launch line reads.
_PORT_MIN = 1024
_PORT_MAX = 65535
_CTX_MIN = 256
_CTX_MAX = 1048576
_TIMEOUT_MAX = 3600.0

#: ``(argv, timeout) -> (returncode, combined output)``.
RunnerFn = Callable[[list[str], float], "tuple[int, str]"]
#: ``(base_url, timeout) -> is the endpoint answering?``
ProbeFn = Callable[[str, float], bool]

#: How each device class exposes its GPU to a container. Keyed by
#: ``Platform.kind``; a kind that is not here runs on the CPU.
GPU_FLAGS: Mapping[str, tuple[str, ...]] = {
    "dgx-spark": ("--gpus", "all"),
    "jetson": ("--runtime", "nvidia"),
}


@dataclass(frozen=True)
class EngineTemplate:
    """How one inference server is launched inside the container.

    ``args`` carries ``{model}``, ``{port}`` and ``{ctx}`` placeholders, and
    nothing else; ``port`` is what the server listens on *inside* the
    container; ``health_path`` is the root-relative path the readiness probe
    GETs.
    """

    port: int
    args: tuple[str, ...]
    needs_model_mount: bool
    health_path: str
    #: Extra arguments that turn on server-side tool-call parsing, with a
    #: ``{tool_parser}`` placeholder; added only when a parser is named.
    #: Without them vLLM answers a request that carries ``tools`` with 400.
    tool_parser_args: tuple[str, ...] = ()
    #: The parser used when ``[tiers.lfm] tool_call_parser`` is not set.
    default_tool_parser: str = ""
    #: True for an engine that downloads its model by id: its download cache
    #: is then kept on the host (``hf_cache_dir``) so a restart needs no network.
    downloads_model: bool = False


#: The engines ``[tiers.lfm] engine`` accepts, and how each is launched.
#: The endpoint never leaves this machine (published on loopback only), and
#: the engines serve plain HTTP there; TLS to 127.0.0.1 would protect nothing.
_SCHEME = "http://"  # NOSONAR python:S5332 - loopback only, see above

#: Placeholders the engine templates use, and the readiness path they share.
_MODEL = "{model}"
_PORT = "{port}"
_CTX = "{ctx}"
_HEALTH = "/health"

ENGINES: Mapping[str, EngineTemplate] = {
    "llama-server": EngineTemplate(
        port=8080,
        args=(
            "--model",
            _MODEL,
            "--host",
            _IN_CONTAINER_BIND,
            "--port",
            _PORT,
            "--ctx-size",
            _CTX,
        ),
        needs_model_mount=True,
        health_path=_HEALTH,
    ),
    "vllm": EngineTemplate(
        port=8000,
        args=(
            "--model",
            _MODEL,
            "--host",
            _IN_CONTAINER_BIND,
            "--port",
            _PORT,
            "--max-model-len",
            _CTX,
            "--gpu-memory-utilization",
            "{gpu_fraction}",
        ),
        needs_model_mount=False,
        health_path=_HEALTH,
        tool_parser_args=("--enable-auto-tool-choice", "--tool-call-parser", "{tool_parser}"),
        downloads_model=True,
        default_tool_parser="lfm2",  # verified in vllm/vllm-openai nightly, 2026-09-19
    ),
    "sglang": EngineTemplate(
        port=30000,
        args=(
            "--model-path",
            _MODEL,
            "--host",
            _IN_CONTAINER_BIND,
            "--port",
            _PORT,
            "--context-length",
            _CTX,
            "--mem-fraction-static",
            "{gpu_fraction}",
        ),
        needs_model_mount=False,
        health_path=_HEALTH,
        downloads_model=True,
        tool_parser_args=("--tool-call-parser", "{tool_parser}"),  # no default: unverified
    ),
}


# -- settings validation ---------------------------------------------------
#
# Every value below reaches ``docker`` as an argv element, so a setting that
# is present and malformed is refused outright rather than quietly replaced
# by a default: a silent fallback hides a typo, and a value that is merely
# *stringified* can smuggle a second volume through a ``:`` or a whole flag
# through a leading ``-``. Refusing is safe, because the caller turns a
# ``RuntimeUnavailable`` into one status line and the request escalates.


def _refuse(key: str, requirement: str, value: object) -> RuntimeUnavailable:
    """The one-line refusal for *key*. ``repr`` keeps it on a single line."""
    return RuntimeUnavailable(f"[tiers.lfm] {key} {requirement} (got {value!r})")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_word(value: str) -> bool:
    """No whitespace anywhere, and not a leading ``-`` that docker would read."""
    return value.split() == [value] and not value.startswith("-")


def check_port(value: object) -> int:
    """A published host port: an int in the unprivileged range."""
    if not _is_int(value) or not _PORT_MIN <= value <= _PORT_MAX:
        raise _refuse("port", f"must be an integer from {_PORT_MIN} to {_PORT_MAX}", value)
    return value


def check_ctx(value: object) -> int:
    """A context length the engine template substitutes into ``{ctx}``."""
    if not _is_int(value) or not _CTX_MIN <= value <= _CTX_MAX:
        raise _refuse("ctx", f"must be an integer from {_CTX_MIN} to {_CTX_MAX}", value)
    return value


def check_startup_timeout(value: object) -> float:
    """How long :meth:`DockerRuntime.ensure` waits for the engine to answer."""
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not numeric or not 0 < float(value) <= _TIMEOUT_MAX:  # type: ignore[arg-type]
        raise _refuse(
            "startup_timeout_seconds", f"must be a number of seconds in (0, {_TIMEOUT_MAX}]", value
        )
    return float(value)  # type: ignore[arg-type]


def check_host_dir(key: str, value: object) -> str:
    """A host directory safe to name in a ``-v`` argument (see check_model_dir)."""
    if not isinstance(value, str) or not value:
        raise _refuse(key, "must be an absolute path", value)
    if not os.path.isabs(value) or ":" in value or "," in value or not _is_word(value):
        raise _refuse(key, "must be an absolute path with no ':', ',' or whitespace", value)
    return value


def check_model_dir(value: object) -> str:
    """The host directory bind-mounted read-only at :data:`MODEL_MOUNT`.

    Absolute, and free of the ``:`` and ``,`` that separate a ``-v``
    argument's own fields -- otherwise a directory name alone could mount a
    second volume. Existence is not checked: rendering stays pure.
    """
    if not isinstance(value, str) or not value:
        raise _refuse("model_dir", "must be the absolute path of the model directory", value)
    if not os.path.isabs(value) or ":" in value or "," in value or not _is_word(value):
        raise _refuse("model_dir", "must be an absolute path with no ':', ',' or whitespace", value)
    return value


def check_model(value: object, *, mounted: bool) -> str:
    """The model the engine serves, as the engine's own argument.

    A mounted engine is handed ``<mount>/<name>``, so *name* must be a bare
    file name: a ``/`` or a ``..`` there would reach back out of the mount.
    An unmounted engine takes a model id, so one ``/`` is allowed
    (``owner/name``) and nothing else is.
    """
    if not isinstance(value, str) or not value:
        raise _refuse("model", "must name the model Tier 2 should serve", value)
    if not _is_word(value):
        raise _refuse("model", "must not start with '-' or contain whitespace", value)
    if mounted:
        return _check_mounted_model(value)
    if value.count("/") > 1 or "\\" in value:
        raise _refuse("model", "must be a model name or 'owner/name'", value)
    return value


def _check_mounted_model(value: str) -> str:
    if "/" in value or "\\" in value or value in (".", ".."):
        raise _refuse("model", "must be a bare file name inside [tiers.lfm] model_dir", value)
    return value


def _is_parser_name(text: str) -> bool:
    if not 0 < len(text) <= 40 or text.startswith("-"):
        return False
    return all(ch.islower() or ch.isdigit() or ch in "_-" for ch in text)


def check_tool_call_parser(value: object) -> str:
    """The name of the server's tool-call parser: a short lowercase word."""
    if not isinstance(value, str) or not _is_parser_name(value):
        raise _refuse("tool_call_parser", "must be a short lowercase parser name", value)
    return value


def check_gpu_memory_fraction(value: object) -> float:
    """The share of GPU memory an engine may reserve up front.

    vLLM and SGLang claim most of the GPU by default, which on a shared or
    unified-memory machine is the opposite of a small resident tier.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse("gpu_memory_fraction", "must be a number from 0.02 to 0.95", value)
    if not MIN_GPU_FRACTION <= float(value) <= MAX_GPU_FRACTION:
        raise _refuse("gpu_memory_fraction", "must be a number from 0.02 to 0.95", value)
    return float(value)


#: Checks for the settings that are optional but, when present, must be
#: well-formed. Keyed by setting name so adding one is a table entry.
SETTING_CHECKS: Mapping[str, Callable[[object], object]] = {
    "port": check_port,
    "ctx": check_ctx,
    "startup_timeout_seconds": check_startup_timeout,
    "model_dir": check_model_dir,
    "gpu_memory_fraction": check_gpu_memory_fraction,
    "tool_call_parser": check_tool_call_parser,
    "hf_cache_dir": lambda value: check_host_dir("hf_cache_dir", value),
}


def check_settings(settings: Mapping[str, object]) -> None:
    """Refuse every present-but-malformed launch setting. Pure; raises one line."""
    for key, check in SETTING_CHECKS.items():
        if settings.get(key) is not None:
            check(settings[key])


# -- per-user identity ----------------------------------------------------


def container_name(uid: int) -> str:
    """The container name this OS user's nvsh manages, and only that one."""
    return f"{NAME_PREFIX}{uid}"


def host_port(settings: Mapping[str, object], uid: int) -> int:
    """The loopback port this user's container publishes on.

    Derived from the uid so two operators on one machine do not collide;
    ``[tiers.lfm] port`` overrides it when a site needs a fixed number -- and
    a ``port`` that is present but unusable is refused, never ignored.
    """
    configured = settings.get("port")
    if configured is None:
        return PORT_BASE + (uid % PORT_SPAN)
    return check_port(configured)


# -- detection and config lookups -----------------------------------------


def gpu_flags(platform: Platform, setting: object) -> list[str]:
    """The flag that exposes the GPU on *platform*, or ``[]`` for a CPU run.

    ``setting`` is ``[tiers.lfm] gpu``: ``"off"`` forces the CPU path on any
    machine, anything else (including the default ``"auto"``) consults
    :data:`GPU_FLAGS`.
    """
    if str(setting or "auto").lower() == "off":
        return []
    return list(GPU_FLAGS.get(getattr(platform, "kind", ""), ()))


def engine_template(name: str) -> EngineTemplate:
    """The template for *name*, or ``RuntimeUnavailable`` naming what is accepted."""
    template = ENGINES.get(name)
    if template is None:
        accepted = ", ".join(sorted(ENGINES))
        raise RuntimeUnavailable(f"unknown Tier 2 engine {name!r}; accepted engines: {accepted}")
    return template


def machine_arch() -> str:
    """This machine's architecture, as ``pins.json`` spells it (``aarch64``, ...)."""
    return os.uname().machine


def pinned_images() -> list[dict]:
    """The ``images`` list from ``pins.json``. Never raises; ``[]`` when unreadable."""
    from .fetch import load_pins  # lazy: keeps this module cheap to import

    try:
        pins = load_pins()
    except (OSError, ValueError):
        return []
    images = pins.get("images") if isinstance(pins, dict) else None
    if not isinstance(images, list):
        return []
    return [entry for entry in images if isinstance(entry, dict)]


def _checked_ref(ref: str, origin: str) -> str:
    if _DIGEST_RE.match(ref):
        return ref
    raise RuntimeUnavailable(
        f"{origin} must name the image by @sha256: digest, not a tag (got {ref!r})"
    )


def resolve_image(settings: Mapping[str, object], engine: str) -> str:
    """The pinned image ref for *engine* on this machine.

    ``[tiers.lfm] image`` wins; otherwise ``pins.json``'s ``images`` list is
    searched for this engine on this architecture. Either way the ref must
    carry an ``@sha256:`` digest.
    """
    override = settings.get("image")
    if override is not None:
        return _checked_ref(str(override), "[tiers.lfm] image")
    arch = machine_arch()
    for entry in pinned_images():
        if entry.get("engine") == engine and entry.get("arch") == arch:
            return _checked_ref(str(entry.get("ref") or ""), "the pinned Tier 2 image")
    raise RuntimeUnavailable(
        f"no pinned image for engine {engine} on arch {arch}; set [tiers.lfm] image"
    )


# -- the launch line ------------------------------------------------------


def _model_dir(settings: Mapping[str, object]) -> str:
    return check_model_dir(settings.get("model_dir"))


def _model_ref(template: EngineTemplate, settings: Mapping[str, object]) -> str:
    model = check_model(settings.get("model"), mounted=template.needs_model_mount)
    if template.needs_model_mount:
        return f"{MODEL_MOUNT}/{model}"
    return model


def _ctx(settings: Mapping[str, object]) -> int:
    configured = settings.get("ctx")
    if configured is None:
        return DEFAULT_CTX
    return check_ctx(configured)


def _gpu_fraction(settings: Mapping[str, object]) -> float:
    configured = settings.get("gpu_memory_fraction")
    if configured is None:
        return DEFAULT_GPU_FRACTION
    return check_gpu_memory_fraction(configured)


def _mount_args(template: EngineTemplate, settings: Mapping[str, object], uid: int) -> list[str]:
    if template.needs_model_mount:
        return ["-v", f"{_model_dir(settings)}:{MODEL_MOUNT}:ro"]
    cache = settings.get("hf_cache_dir")
    if not template.downloads_model or cache is None:
        return []
    # The engine runs as the operator, not root, so what it downloads into the
    # host cache stays the operator's to delete (``nvsh uninstall`` does).
    host = check_host_dir("hf_cache_dir", cache)
    return ["--user", f"{uid}:{uid}", "-e", f"HF_HOME={CACHE_MOUNT}", "-v", f"{host}:{CACHE_MOUNT}"]


def _engine_args(template: EngineTemplate, settings: Mapping[str, object]) -> list[str]:
    values = {
        "model": _model_ref(template, settings),
        "port": str(template.port),
        "ctx": str(_ctx(settings)),
        "gpu_fraction": str(_gpu_fraction(settings)),
    }
    parser = settings.get("tool_call_parser") or template.default_tool_parser
    extra = template.tool_parser_args if parser else ()
    values["tool_parser"] = str(parser)
    return [arg.format(**values) for arg in (*template.args, *extra)]


def render_launch(settings: Mapping[str, object], platform: Platform, *, uid: int) -> list[str]:
    """The ``docker run`` argv for this config on this machine. Pure.

    Executes nothing and reads nothing but *settings*, *platform* and *uid*:
    no request text, no model output, no environment. The container is
    started detached and *without* ``--rm``, so :meth:`DockerRuntime.stop`'s
    ``docker rm`` is what actually reclaims it.

    Every setting it reads is validated first: a malformed one raises
    ``RuntimeUnavailable`` with one line naming the key, and no argv is
    produced at all.
    """
    check_settings(settings)
    engine = str(settings.get("engine") or DEFAULT_ENGINE)
    template = engine_template(engine)
    image = resolve_image(settings, engine)
    published = f"{LOOPBACK}:{host_port(settings, uid)}:{template.port}"
    argv = [DOCKER, "run", "-d", "--name", container_name(uid), "-p", published]
    argv += gpu_flags(platform, settings.get("gpu"))
    argv += _mount_args(template, settings, uid)
    argv.append(image)
    argv += _engine_args(template, settings)
    return argv


# -- what uninstall needs to say ------------------------------------------

#: Stand-ins used only to render a launch line for :func:`image_refs`, where
#: the model itself is irrelevant: we are asking which *image* was pulled.
_PROBE_MODEL = "model"
_PROBE_DIR = "/var/empty"


def image_refs(settings: Mapping[str, object], platform: Platform) -> list[str]:
    """Image refs a managed Tier 2 on *platform* could have pulled. Pure.

    ``[]`` when nothing resolves -- an unknown engine, no pin, a ref without
    a digest -- because in every one of those cases nvsh never pulled
    anything either. Also ``[]`` in ``[tiers.lfm] mode = "attach"``: nvsh
    never started a container, so it never pulled an image either, and
    reporting one would send the operator to remove an image nvsh does not
    own.
    """
    if str(settings.get("mode") or MANAGED) == ATTACH:
        return []
    probe = dict(settings)
    probe.setdefault("model", _PROBE_MODEL)
    probe.setdefault("model_dir", _PROBE_DIR)
    try:
        argv = render_launch(probe, platform, uid=0)
    except RuntimeUnavailable:
        return []
    return [part for part in argv if _DIGEST_RE.match(part)]


def leftover_note(refs: Sequence[str]) -> str:
    """What ``nvsh uninstall`` prints about images it deliberately leaves behind."""
    if not refs:
        return ""
    lines = ["nvsh does not delete container images; remove them yourself with:"]
    lines += [f"  {DOCKER} image rm {ref}" for ref in refs]
    return "\n".join(lines)


def stop_container(
    uid: int, runner: RunnerFn | None = None, *, timeout: float = _DOCKER_TIMEOUT
) -> str:
    """Stop and remove this OS user's Tier 2 container: ``nvsh uninstall``'s
    safety net, run *after* the daemon is stopped (the daemon's own
    ``close()`` already stops an attached container while it is alive).

    Touches :func:`container_name`'s name and nothing else -- exactly a
    ``docker stop`` then a ``docker rm``, never any other container, never
    ``docker rmi``, never ``docker system prune``. ``rm`` is attempted even
    when ``stop`` itself raises (a client-side timeout, say), so a container
    that failed to stop cleanly is still reclaimed rather than left behind.
    Never raises: a missing docker binary or an unreachable daemon folds into
    the returned one-line status instead of failing the caller, so a
    Docker-less uninstall still exits clean.
    """
    runner = runner if runner is not None else _default_runner
    name = container_name(uid)
    stop_code, stop_exc = _try_stop(runner, name, timeout)
    try:
        rm_code, rm_out = runner([DOCKER, "rm", name], timeout)
    except Exception as exc:  # noqa: BLE001 - docker missing/broken must not fail uninstall
        if stop_exc is not None:
            return f"docker not available: {exc}"
        return f"docker rm not available: {exc}"
    if rm_code == 0:
        if stop_exc is not None:
            return f"docker stop failed ({stop_exc}); removed {name}"
        return f"stopped and removed {name}"
    if stop_code is not None and stop_code != 0 and rm_code != 0:
        return f"no container named {name}"
    return f"docker rm {name} failed: {_tail(rm_out)}"


def _try_stop(runner: RunnerFn, name: str, timeout: float) -> tuple[int | None, Exception | None]:
    """Run ``docker stop`` for *name*, never raising.

    Returns ``(returncode, None)`` on an ordinary run, or ``(None, exc)``
    when the runner itself raised (a client-side timeout, say) -- the
    caller still goes on to attempt ``docker rm`` either way, so a
    container that failed to stop cleanly is still reclaimed.
    """
    try:
        stop_code, _stop_out = runner([DOCKER, "stop", name], timeout)
        return stop_code, None
    except Exception as exc:  # noqa: BLE001 - stop failing must not skip rm
        return None, exc


# -- defaults for the injected seams --------------------------------------


def _default_runner(argv: list[str], timeout: float) -> tuple[int, str]:  # pragma: no cover
    """Run *argv* and return ``(returncode, stdout+stderr)``. Real docker only."""
    import subprocess  # nosec B404 - fixed list argv, never shell=True

    completed = subprocess.run(  # nosec B603 - argv is built by render_launch, no shell
        argv, check=False, capture_output=True, text=True, timeout=timeout
    )
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def http_probe(base_url: str, timeout: float, *, path: str = _HEALTH) -> bool:
    """GET *path* on *base_url*'s host and port; ``True`` on HTTP 200.

    Never raises: a refused connection, a reset or a timeout is simply
    "not ready yet".
    """
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(
        parsed.hostname or LOOPBACK, parsed.port or 80, timeout=timeout
    )
    try:
        connection.request("GET", path)
        return connection.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _tail(text: str) -> str:
    """The last :data:`LOG_TAIL_CHARS` of *text*, redacted, on one line."""
    trimmed = " ".join(text.split())[-LOG_TAIL_CHARS:]
    return redact(trimmed.encode("utf-8", "surrogateescape")).decode("utf-8", "surrogateescape")


# -- the runtime ----------------------------------------------------------


class DockerRuntime:
    """A Tier 2 endpoint served by a container nvsh owns.

    Constructing one starts nothing and runs nothing; :meth:`ensure` is the
    only method that executes anything, and it only ever executes what
    :func:`render_launch` returned plus fixed ``docker`` sub-commands against
    :func:`container_name`'s name.
    """

    def __init__(
        self,
        settings: Mapping[str, object],
        platform: Platform,
        *,
        runner: RunnerFn | None = None,
        floor_check: Callable[[], object] | None = None,
        uid: Callable[[], int] = os.getuid,
        probe: ProbeFn | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = dict(settings)
        self._platform = platform
        self._runner = runner if runner is not None else _default_runner
        self._floor_check = floor_check
        self._probe = probe if probe is not None else self._default_probe
        self._clock = clock
        self._sleep = sleep
        self._uid = int(uid())
        self._name = container_name(self._uid)
        self._url = f"{_SCHEME}{LOOPBACK}:{host_port(self._settings, self._uid)}/v1"
        self._started = False

    # -- the Runtime protocol --------------------------------------------

    def ensure(self) -> str:
        """Return a ready base URL, starting the container if it is not up."""
        check_settings(self._settings)
        self._check_floor()
        self._require_docker()
        if self._state() != _RUNNING:
            argv = render_launch(self._settings, self._platform, uid=self._uid)
            self._clear()
            self._launch(argv)
        self._await_ready()
        self._started = True
        return self._url

    def stop(self) -> None:
        """Stop and remove our container. Never raises, whatever docker does."""
        for verb in ("stop", "rm"):
            with contextlib.suppress(Exception):
                self._runner([DOCKER, verb, self._name], _DOCKER_TIMEOUT)
        self._started = False

    def status(self) -> str:
        """One line for ``overview``/``doctor``. Runs nothing at all."""
        engine = str(self._settings.get("engine") or DEFAULT_ENGINE)
        state = "started" if self._started else "not started"
        return f"managed {engine} container {self._name} on {self._url} ({state})"

    # -- preconditions ----------------------------------------------------

    def _check_floor(self) -> None:
        if self._floor_check is None:
            return
        result = self._floor_check()
        if not getattr(result, "ok", True):
            detail = str(getattr(result, "status", "") or "")
            raise RuntimeUnavailable(detail or "not enough free memory to start Tier 2")

    def _require_docker(self) -> None:
        """Refuse with one line when docker is missing or its daemon is down."""
        info = [DOCKER, "info", "--format", "{{.ServerVersion}}"]
        try:
            code, _output = self._runner(info, _DOCKER_TIMEOUT)
        except OSError as exc:
            detail = f"Tier 2 needs Docker and it is not usable here: {exc}"
            raise RuntimeUnavailable(detail) from exc
        if code == _NOT_FOUND:
            raise RuntimeUnavailable("Tier 2 needs Docker: 'docker' is not on PATH")
        if code != 0:
            raise RuntimeUnavailable("Tier 2 needs Docker: its daemon is not reachable")

    # -- the container ----------------------------------------------------

    def _state(self) -> str:
        """``running``, ``stopped``, or ``""`` when no container of ours exists."""
        code, output = self._run([DOCKER, "inspect", "-f", "{{.State.Running}}", self._name])
        if code != 0:
            return ""
        return _RUNNING if output.strip() == "true" else _STOPPED

    def _clear(self) -> None:
        """Remove a stale container of our name so the launch can reuse it."""
        self._run([DOCKER, "rm", "-f", self._name])

    def _launch(self, argv: list[str]) -> None:
        code, output = self._run(argv, timeout=_LAUNCH_TIMEOUT)
        if code == 0:
            return
        tail = _tail(output)
        if _PORT_ALLOCATED_RE.search(tail):
            raise RuntimeUnavailable(
                f"could not start the Tier 2 container: {tail} "
                "-- set [tiers.lfm] port to an unused port"
            )
        raise RuntimeUnavailable(f"could not start the Tier 2 container: {tail}")

    def _await_ready(self) -> None:
        deadline = self._clock() + self._startup_seconds()
        while True:
            if self._probe(self._url, PROBE_SECONDS):
                return
            if self._clock() >= deadline:
                break
            self._sleep(POLL_SECONDS)
        # Read the logs before stopping: ``docker rm`` takes them with it.
        logs = self._logs()
        self.stop()
        raise RuntimeUnavailable(f"the Tier 2 runtime did not answer in time: {logs}")

    def _logs(self) -> str:
        _code, output = self._run([DOCKER, "logs", "--tail", "20", self._name])
        return _tail(output)

    def _startup_seconds(self) -> float:
        configured = self._settings.get("startup_timeout_seconds")
        if configured is None:
            return DEFAULT_STARTUP_SECONDS
        return check_startup_timeout(configured)

    def _default_probe(self, base_url: str, timeout: float) -> bool:
        engine = str(self._settings.get("engine") or DEFAULT_ENGINE)
        return http_probe(base_url, timeout, path=engine_template(engine).health_path)

    def _run(self, argv: list[str], *, timeout: float = _DOCKER_TIMEOUT) -> tuple[int, str]:
        """Run one docker sub-command, reporting a failure rather than raising."""
        try:
            return self._runner(argv, timeout)
        except Exception as exc:  # noqa: BLE001 - a broken runner is just "it failed"
            return (1, str(exc))


# -- building one from config ---------------------------------------------


def _attached(settings: Mapping[str, object]) -> Runtime:
    """An :class:`AttachedRuntime` for ``[tiers.lfm] base_url``.

    The URL's *host* is not checked here: ``AttachedRuntime.ensure`` already
    refuses anything but this machine, and having one place do it keeps the
    rule from drifting into two.
    """
    base_url = settings.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise RuntimeUnavailable('[tiers.lfm] mode = "attach" needs [tiers.lfm] base_url')
    return AttachedRuntime(base_url)


def build_runtime(settings: Mapping[str, object], platform: Platform, **kwargs: object) -> Runtime:
    """The runtime ``[tiers.lfm] mode`` asks for.

    ``attach`` returns an :class:`~nvsh.tiers.runtime.AttachedRuntime` and
    runs no docker command, ever -- the injected seams in *kwargs* are not
    even passed on, because there is nothing to inject them into.
    """
    mode = str(settings.get("mode") or MANAGED)
    if mode == ATTACH:
        return _attached(settings)
    if mode != MANAGED:
        raise RuntimeUnavailable(
            f"unknown [tiers.lfm] mode {mode!r}; accepted modes: {ATTACH}, {MANAGED}"
        )
    return DockerRuntime(_with_cache_dir(settings), platform, **kwargs)  # type: ignore[arg-type]


def _with_cache_dir(settings: Mapping[str, object]) -> Mapping[str, object]:
    """*settings* with ``hf_cache_dir`` defaulted to nvsh's own tier cache.

    Done here, not in :func:`render_launch`, so rendering stays a pure
    function of its arguments. The directory is created so Docker does not
    create it as root.
    """
    if settings.get("hf_cache_dir") is not None:
        return settings
    from .fetch import default_cache_dir  # lazy: keep this module light to import

    cache = default_cache_dir() / HF_CACHE_NAME
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        return settings  # no cache: the engine downloads inside the container
    return {**settings, "hf_cache_dir": str(cache)}
