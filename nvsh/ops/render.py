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

from nvsh.platform._model import Platform

# ---------------------------------------------------------------------------
# Static per-CLI verb table
# ---------------------------------------------------------------------------
#
# Measured against dgx-spark-cli 0.7.0 (`spark`) and jetson-thor-cli 0.5.0
# (`thor`) on 2026-09-19: both accept `<verb> --json` for every read-only
# machine-state operation below. `power` exists only on thor (thor is a
# Jetson board with `nvpmodel`; the Spark's `spark` CLI has no power verb).
# `swap_status` is `swap status --json`, not bare `swap --json`: `swap` has
# other (non-read-only) subcommands on both CLIs, and `status` is the one
# that is read-only.
#
# jetson-orin-cli 0.5.0 (`orin`) has NO machine verbs yet -- it is listed
# with an empty verb map so every operation falls through to the system
# fallback even when `orin` is on PATH, and so a future orin-cli release
# that adds verbs has one table to extend, not a special-cased absence.

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
}

_THOR_VERBS: dict[str, list[str]] = {
    **_SPARK_VERBS,
    "power_get": ["power"],
}

_ORIN_VERBS: dict[str, list[str]] = {}

DEVICE_CLI_VERBS: dict[str, dict[str, list[str]]] = {
    "spark": _SPARK_VERBS,
    "thor": _THOR_VERBS,
    "orin": _ORIN_VERBS,
}

# Which device CLI binaries are candidates for a given Platform.kind, in
# lookup order. `Platform.kind` can't distinguish thor from orin (both
# report "jetson" -- see nvsh/platform/_detect.py's _classify()), so both
# are tried and whichever CLI is actually on PATH (platform.get("<x>_cli"))
# wins.
_CLI_CANDIDATES: dict[str, tuple[str, ...]] = {
    "dgx-spark": ("spark",),
    "jetson": ("thor", "orin"),
}

# nvpmodel's numeric mode IDs are per-board and were not measured for every
# mode on every board. `max_performance` -> `0` is the one mapping that is
# certain (nvpmodel's mode 0 is always MAXN/max-performance on every Jetson
# board nvsh has been run on -- see docs/platforms.md's nvpmodel_power_mode
# rows for thor/orin, both reporting MAXN at mode 0). `balanced` and
# `low_power` map to different mode IDs on different boards (Thor's and
# Orin's nvpmodel tables are not the same), so rendering those would be a
# guess dressed up as a command -- render() returns None for them instead.
_NVPMODEL_MAX_PERFORMANCE_MODE_ID = "0"


# Argument-free operations that always fall back to the same static argv,
# regardless of args or platform.
_STATIC_FALLBACK_ARGV: dict[str, list[str]] = {
    "memory_stats": ["free", "-m"],
    "disk_stats": ["df", "-h"],
    "container_list": ["docker", "ps"],
    "network_info": ["ip", "-brief", "addr"],
    "process_list": ["ps", "-eo", "pid,rss,comm", "--sort=-rss"],
    "swap_status": ["swapon", "--show"],
    # nvidia-smi with no flags is a single call that exits; it is present
    # on dgx-spark, rtx and (per the thor/orin fixtures in
    # tests/fixtures/platform) both Jetson boards nvsh targets.
    "gpu_stats": ["nvidia-smi"],
}


def _fallback_service_status(args: dict[str, str]) -> list[str]:
    return ["systemctl", "status", "--no-pager", args["service"]]


def _fallback_service_logs(args: dict[str, str]) -> list[str]:
    return ["journalctl", "-u", args["service"], "-n", "50", "--no-pager"]


def _fallback_service_restart(args: dict[str, str]) -> list[str]:
    return ["sudo", "systemctl", "restart", args["service"]]


def _fallback_container_restart(args: dict[str, str]) -> list[str]:
    return ["docker", "restart", args["container"]]


# Operations whose fallback argv is static in shape but needs one arg value
# substituted in.
_ARG_FALLBACKS = {
    "service_status": _fallback_service_status,
    "service_logs": _fallback_service_logs,
    "service_restart": _fallback_service_restart,
    "container_restart": _fallback_container_restart,
}


def _fallback_power_get(platform: Platform) -> list[str] | None:
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


def _system_fallback(
    operation_name: str, args: dict[str, str], platform: Platform
) -> list[str] | None:
    """The system (non-device-CLI) argv for *operation_name*, or ``None``.

    ``None`` means: no single, non-shell, exiting argv exists for this
    operation on this platform. Each ``None`` case below says why.
    """
    if operation_name in _STATIC_FALLBACK_ARGV:
        return list(_STATIC_FALLBACK_ARGV[operation_name])

    if operation_name in _ARG_FALLBACKS:
        return _ARG_FALLBACKS[operation_name](args)

    if operation_name == "thermal_stats":
        # CPU/GPU/board temperatures live in several separate files under
        # /sys/class/thermal/thermal_zone*/temp; reading and labelling all
        # of them in one shot needs a loop, which needs a shell -- and this
        # module never builds a shell string. There is no single sensors-
        # style binary guaranteed present across dgx-spark/jetson/rtx
        # either (lm-sensors is not installed by default on any of them).
        # So there is no single exiting argv for this operation anywhere
        # nvsh runs without a device CLI: return None rather than invent
        # one.
        return None

    if operation_name == "machine_status":
        # "the current state of this machine" is exactly the composite
        # view a device CLI's `status` verb assembles; no single system
        # command produces the same summary, so there is nothing honest to
        # fall back to.
        return None

    if operation_name == "power_get":
        return _fallback_power_get(platform)

    if operation_name == "power_set":
        return _fallback_power_set(args, platform)

    return None


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

    if operation_name == "nvsh_doctor":
        # nvsh's own doctor verb: always available, never platform- or
        # device-CLI-dependent.
        return ["nvsh", "doctor", "--json"]

    for cli in _CLI_CANDIDATES.get(platform.kind, ()):
        cli_value = platform.get(f"{cli}_cli")
        if cli_value is not None and cli_value.present:
            verb = DEVICE_CLI_VERBS.get(cli, {}).get(operation_name)
            if verb is not None:
                return [cli, *verb, "--json"]

    return _system_fallback(operation_name, args, platform)
