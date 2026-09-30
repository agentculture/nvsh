"""Render an operation + args into an argv list for the detected platform.

``render()`` is a pure function: it never runs a subprocess, never probes a
CLI with ``--help``, and never builds a shell string. Which verbs a device
CLI supports is a static table (``DEVICE_CLI_VERBS``) keyed by the CLI's
binary name, populated from what was actually measured on real hardware
(see the plan's t2 task and ``docs/platforms.md``) -- not discovered at
request time.

Callers must validate ``args`` against ``nvsh.ops.table.validate()`` before
calling ``render()``; this module assumes the operation name and args are
already grounded and well-typed.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from nvsh.platform._model import DeviceCli, Platform

# ---------------------------------------------------------------------------
# Static per-CLI verb table
# ---------------------------------------------------------------------------
#
# Measured 2026-09-29 (plan device-cli-alignment-spark-thor-orin): all three
# device CLIs accept `<verb> --json` for the same ten read-only machine-state
# operations below, at or above the version in DEVICE_CLI_MIN_VERSIONS:
#
# - dgx-spark-cli 0.8.0 (`spark`; branch build, unreleased at the time of
#   measuring) adds `power`; 0.7.0 (measured 2026-09-19) had the other nine.
# - jetson-thor-cli 0.5.0 (`thor`) has all ten, `power` included.
# - jetson-orin-cli 0.6.0 (`orin`; branch build, unreleased) has all ten;
#   0.5.0 had no machine verbs at all.
#
# `swap_status` is `swap status --json`, not bare `swap --json`: `swap` has
# other (non-read-only) subcommands on every CLI, and `status` is the one
# that is read-only. No device CLI verb backs a mutating operation: those
# keep their system fallbacks below.
#
# A CLI older than its floor (or whose `--version` could not be parsed) is
# treated exactly as if it were absent: every operation uses its system
# fallback. The floors are data, next to the verbs they guard.

_SPARK_VERBS: dict[str, list[str]] = {
    "machine_status": ["status"],
    "memory_stats": ["memory"],
    "gpu_stats": ["gpu"],
    "disk_stats": ["disk"],
    "thermal_stats": ["thermal"],
    "container_list": ["containers"],
    "network_info": ["network"],
    "process_list": ["processes"],
    "swap_status": ["swap", "status"],
    "power_get": ["power"],
}

_THOR_VERBS: dict[str, list[str]] = dict(_SPARK_VERBS)

_ORIN_VERBS: dict[str, list[str]] = dict(_SPARK_VERBS)

DEVICE_CLI_VERBS: dict[str, dict[str, list[str]]] = {
    "spark": _SPARK_VERBS,
    "thor": _THOR_VERBS,
    "orin": _ORIN_VERBS,
}

#: The oldest release of each device CLI whose verbs match its table above.
DEVICE_CLI_MIN_VERSIONS: dict[str, tuple[int, ...]] = {
    "spark": (0, 8, 0),
    "thor": (0, 5, 0),
    "orin": (0, 6, 0),
}

# Which device CLI binaries are candidates for a given Platform.kind, in
# lookup order. On "jetson" the board (Platform.board(), from the device
# tree) narrows this to that board's own CLI; when the board is unknown both
# are tried, thor first.
_CLI_CANDIDATES: dict[str, tuple[str, ...]] = {
    "dgx-spark": ("spark",),
    "jetson": ("thor", "orin"),
}

#: Platform.board() -> the only device CLI that board uses.
_BOARD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "thor": ("thor",),
    "orin": ("orin",),
}

_LEADING_VERSION = re.compile(r"v?(\d+(?:\.\d+)*)")


def _version_tuple(text: str | None) -> tuple[int, ...] | None:
    """``"0.8.0"`` -> ``(0, 8, 0)``; the leading numeric release only, so a
    branch build's suffix (``0.8.0.dev1``) counts as that release. ``None``
    when *text* is ``None`` or does not start with a number."""
    if text is None:
        return None
    match = _LEADING_VERSION.match(text.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def usable_device_cli(platform: Platform, cli: str) -> DeviceCli | None:
    """The detected device CLI *cli* when it is at or above its floor.

    Fails closed: absent, no floor on record, or a version that is missing,
    unparseable or older than :data:`DEVICE_CLI_MIN_VERSIONS` -> ``None``.
    """
    floor = DEVICE_CLI_MIN_VERSIONS.get(cli)
    found = platform.device_cli(cli)
    if floor is None or found is None:
        return None
    version = _version_tuple(found.version)
    if version is None or version < floor:
        return None
    return found


def cli_meets_floor(platform: Platform, cli: str) -> bool:
    """True when *cli* is detected on *platform* at or above its floor."""
    return usable_device_cli(platform, cli) is not None


def _candidates(platform: Platform) -> tuple[str, ...]:
    candidates = _CLI_CANDIDATES.get(platform.kind, ())
    if platform.kind != "jetson":
        return candidates
    board = platform.board()
    return _BOARD_CANDIDATES.get(board, candidates) if board else candidates


# nvpmodel's numeric mode IDs are per-board and were not measured for every
# mode on every board. `max_performance` -> `0` is the one mapping that is
# certain (nvpmodel's mode 0 is always MAXN/max-performance on every Jetson
# board nvsh has been run on -- see docs/platforms.md's nvpmodel_power_mode
# rows for thor/orin, both reporting MAXN at mode 0). `balanced` and
# `low_power` map to different mode IDs on different boards (Thor's and
# Orin's nvpmodel tables are not the same), so rendering those would be a
# guess dressed up as a command -- render() returns None for them instead.
_NVPMODEL_MAX_PERFORMANCE_MODE_ID = "0"


_Fallback = Callable[[dict[str, str], Platform], "list[str] | None"]


def _static(argv: list[str]) -> _Fallback:
    """An argument-free fallback: the same argv on every platform."""
    return lambda _args, _platform: list(argv)


def _none(_args: dict[str, str], _platform: Platform) -> list[str] | None:
    return None


def _fallback_service_status(args: dict[str, str], _platform: Platform) -> list[str]:
    return ["systemctl", "status", "--no-pager", args["service"]]


def _fallback_service_logs(args: dict[str, str], _platform: Platform) -> list[str]:
    return ["journalctl", "-u", args["service"], "-n", "50", "--no-pager"]


def _fallback_service_restart(args: dict[str, str], _platform: Platform) -> list[str]:
    return ["sudo", "systemctl", "restart", args["service"]]


def _fallback_container_restart(args: dict[str, str], _platform: Platform) -> list[str]:
    return ["docker", "restart", args["container"]]


def _fallback_power_get(_args: dict[str, str], platform: Platform) -> list[str] | None:
    if platform.kind == "jetson":
        return ["nvpmodel", "-q"]
    # dgx-spark/rtx/generic: no nvpmodel, no other widely-present way to
    # query a power mode without the device CLI.
    return None


def _fallback_power_set(args: dict[str, str], platform: Platform) -> list[str] | None:
    mode = args.get("mode")
    if platform.kind == "jetson" and mode == "max_performance":
        return ["sudo", "nvpmodel", "-m", _NVPMODEL_MAX_PERFORMANCE_MODE_ID]
    # balanced/low_power: mode id is per-board and unverified -- see the
    # comment on _NVPMODEL_MAX_PERFORMANCE_MODE_ID above. Also covers
    # dgx-spark/rtx/generic, which have no nvpmodel at all.
    return None


#: Operations that never go to a device CLI: the same argv everywhere.
_FIXED_ARGV: dict[str, list[str]] = {
    # nvsh's own doctor verb: never platform- or device-CLI-dependent.
    "nvsh_doctor": ["nvsh", "doctor", "--json"],
}

#: The system (non-device-CLI) argv for each operation. A fallback that
#: returns ``None`` means no single, non-shell, exiting argv exists for that
#: operation on that platform; each such entry says why.
_SYSTEM_FALLBACKS: dict[str, _Fallback] = {
    "memory_stats": _static(["free", "-m"]),
    "disk_stats": _static(["df", "-h"]),
    "container_list": _static(["docker", "ps"]),
    "network_info": _static(["ip", "-brief", "addr"]),
    "process_list": _static(["ps", "-eo", "pid,rss,comm", "--sort=-rss"]),
    "swap_status": _static(["swapon", "--show"]),
    # nvidia-smi with no flags is a single call that exits; it is present
    # on dgx-spark, rtx and (per the thor/orin fixtures in
    # tests/fixtures/platform) both Jetson boards nvsh targets.
    "gpu_stats": _static(["nvidia-smi"]),
    "service_status": _fallback_service_status,
    "service_logs": _fallback_service_logs,
    "service_restart": _fallback_service_restart,
    "container_restart": _fallback_container_restart,
    # CPU/GPU/board temperatures live in several separate files under
    # /sys/class/thermal/thermal_zone*/temp; reading and labelling all of
    # them in one shot needs a loop, which needs a shell -- and this module
    # never builds a shell string. There is no single sensors-style binary
    # guaranteed present across dgx-spark/jetson/rtx either (lm-sensors is
    # not installed by default on any of them).
    "thermal_stats": _none,
    # "the current state of this machine" is exactly the composite view a
    # device CLI's `status` verb assembles; no single system command
    # produces the same summary, so there is nothing honest to fall back to.
    "machine_status": _none,
    "power_get": _fallback_power_get,
    "power_set": _fallback_power_set,
}


def _safe_argument(value: object) -> bool:
    """True when *value* can sit in an argv slot without being read as an option."""
    return (
        isinstance(value, str)
        and value != ""
        and not value.startswith("-")
        and all(ch.isprintable() for ch in value)
    )


def render(operation_name: str, args: dict[str, str], platform: Platform) -> list[str] | None:
    """Render *operation_name* with *args* into an argv list for *platform*.

    Returns a list of strings (never a shell string), or ``None`` when no
    single, non-shell, exiting command exists for this operation on this
    platform. ``args`` must already have passed ``nvsh.ops.table.validate()``.
    """
    if not all(_safe_argument(value) for value in args.values()):
        # Defence in depth behind grounding: a value that starts with "-"
        # would be read as an OPTION by systemctl/docker/journalctl even
        # though it is a single argv element.
        return None

    if operation_name in _FIXED_ARGV:
        return list(_FIXED_ARGV[operation_name])

    for cli in _candidates(platform):
        found = usable_device_cli(platform, cli)
        verb = DEVICE_CLI_VERBS[cli].get(operation_name) if found else None
        if found is not None and verb is not None:
            # The resolved path, not the bare name: a CLI installed through
            # nvsh's own extra lives in nvsh's env bin, which is not on PATH.
            return [found.path, *verb, "--json"]

    return _SYSTEM_FALLBACKS.get(operation_name, _none)(args, platform)
