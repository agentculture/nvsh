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

#: Container naming and port allocation, per OS user.
NAME_PREFIX = "nvsh-tier2-"
PORT_BASE = 18400
PORT_SPAN = 1000

#: The address the container publishes on, on the host. Nothing else.
LOOPBACK = "127.0.0.1"

#: What the engine binds to *inside* the container's own network namespace.
#: Not a host bind: the only thing that reaches the host is what
#: ``-p 127.0.0.1:<port>:<container port>`` publishes, and an engine bound
#: to the loopback of its own namespace would not be reachable at all.
_IN_CONTAINER_BIND = "0.0.0.0"  # nosec B104 - container namespace, published on 127.0.0.1 only

#: Where a mounted model directory appears inside the container, read-only.
MODEL_MOUNT = "/models"

DEFAULT_ENGINE = "llama-server"
DEFAULT_CTX = 4096
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
#: is refused by the same check.
_DIGEST_RE = re.compile(r"\A[^@\s]+@sha256:[0-9a-f]{64}\Z")

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


#: The engines ``[tiers.lfm] engine`` accepts, and how each is launched.
ENGINES: Mapping[str, EngineTemplate] = {
    "llama-server": EngineTemplate(
        port=8080,
        args=(
            "--model",
            "{model}",
            "--host",
            _IN_CONTAINER_BIND,
            "--port",
            "{port}",
            "--ctx-size",
            "{ctx}",
        ),
        needs_model_mount=True,
        health_path="/health",
    ),
    "vllm": EngineTemplate(
        port=8000,
        args=(
            "--model",
            "{model}",
            "--host",
            _IN_CONTAINER_BIND,
            "--port",
            "{port}",
            "--max-model-len",
            "{ctx}",
        ),
        needs_model_mount=False,
        health_path="/health",
    ),
    "sglang": EngineTemplate(
        port=30000,
        args=(
            "--model-path",
            "{model}",
            "--host",
            _IN_CONTAINER_BIND,
            "--port",
            "{port}",
            "--context-length",
            "{ctx}",
        ),
        needs_model_mount=False,
        health_path="/health",
    ),
}


# -- per-user identity ----------------------------------------------------


def container_name(uid: int) -> str:
    """The container name this OS user's nvsh manages, and only that one."""
    return f"{NAME_PREFIX}{uid}"


def host_port(settings: Mapping[str, object], uid: int) -> int:
    """The loopback port this user's container publishes on.

    Derived from the uid so two operators on one machine do not collide;
    ``[tiers.lfm] port`` overrides it when a site needs a fixed number.
    """
    configured = settings.get("port")
    if isinstance(configured, int) and not isinstance(configured, bool):
        return configured
    return PORT_BASE + (uid % PORT_SPAN)


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
    directory = settings.get("model_dir")
    if not isinstance(directory, str) or not directory:
        raise RuntimeUnavailable(
            "this engine serves a local file; set [tiers.lfm] model_dir to its directory"
        )
    return directory


def _model_ref(template: EngineTemplate, settings: Mapping[str, object]) -> str:
    model = settings.get("model")
    if not isinstance(model, str) or not model:
        raise RuntimeUnavailable("set [tiers.lfm] model to the model Tier 2 should serve")
    if template.needs_model_mount:
        return f"{MODEL_MOUNT}/{model}"
    return model


def _ctx(settings: Mapping[str, object]) -> int:
    configured = settings.get("ctx")
    if isinstance(configured, int) and not isinstance(configured, bool):
        return configured
    return DEFAULT_CTX


def _mount_args(template: EngineTemplate, settings: Mapping[str, object]) -> list[str]:
    if not template.needs_model_mount:
        return []
    return ["-v", f"{_model_dir(settings)}:{MODEL_MOUNT}:ro"]


def _engine_args(template: EngineTemplate, settings: Mapping[str, object]) -> list[str]:
    values = {
        "model": _model_ref(template, settings),
        "port": str(template.port),
        "ctx": str(_ctx(settings)),
    }
    return [arg.format(**values) for arg in template.args]


def render_launch(settings: Mapping[str, object], platform: Platform, *, uid: int) -> list[str]:
    """The ``docker run`` argv for this config on this machine. Pure.

    Executes nothing and reads nothing but *settings*, *platform* and *uid*:
    no request text, no model output, no environment. The container is
    started detached and *without* ``--rm``, so :meth:`DockerRuntime.stop`'s
    ``docker rm`` is what actually reclaims it.
    """
    engine = str(settings.get("engine") or DEFAULT_ENGINE)
    template = engine_template(engine)
    image = resolve_image(settings, engine)
    published = f"{LOOPBACK}:{host_port(settings, uid)}:{template.port}"
    argv = [DOCKER, "run", "-d", "--name", container_name(uid), "-p", published]
    argv += gpu_flags(platform, settings.get("gpu"))
    argv += _mount_args(template, settings)
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
    anything either.
    """
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


# -- defaults for the injected seams --------------------------------------


def _default_runner(argv: list[str], timeout: float) -> tuple[int, str]:  # pragma: no cover
    """Run *argv* and return ``(returncode, stdout+stderr)``. Real docker only."""
    import subprocess  # nosec B404 - fixed list argv, never shell=True

    completed = subprocess.run(  # nosec B603 - argv is built by render_launch, no shell
        argv, check=False, capture_output=True, text=True, timeout=timeout
    )
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def http_probe(base_url: str, timeout: float, *, path: str = "/health") -> bool:
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
        self._url = f"http://{LOOPBACK}:{host_port(self._settings, self._uid)}/v1"
        self._started = False

    # -- the Runtime protocol --------------------------------------------

    def ensure(self) -> str:
        """Return a ready base URL, starting the container if it is not up."""
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
        if code != 0:
            raise RuntimeUnavailable(f"could not start the Tier 2 container: {_tail(output)}")

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
        if isinstance(configured, (int, float)) and not isinstance(configured, bool):
            return float(configured)
        return DEFAULT_STARTUP_SECONDS

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
    return DockerRuntime(settings, platform, **kwargs)  # type: ignore[arg-type]
