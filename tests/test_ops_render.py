"""Tests for nvsh.ops.render.render(): operation + platform -> argv list.

Table-tests every registered operation for each platform kind, both with
the relevant device CLI present (spark/thor/orin) and absent (system
fallback), asserting the exact argv list render() returns.
"""

from __future__ import annotations

import pytest

from nvsh.ops.render import DEVICE_CLI_VERBS, render
from nvsh.ops.table import names as operation_names
from nvsh.platform._model import PATH, Platform, Value


def _present(name: str) -> Value:
    return Value(
        name=name, text=f"/usr/local/bin/{name}", source=f"{name} (PATH)", method=PATH, present=True
    )


def _absent(name: str) -> Value:
    return Value(name=name, text=None, source=f"{name} (PATH)", method=PATH, present=False)


def _platform(
    kind: str, *, spark: bool = False, thor: bool = False, orin: bool = False
) -> Platform:
    values = (
        (_present("spark_cli") if spark else _absent("spark_cli")),
        (_present("thor_cli") if thor else _absent("thor_cli")),
        (_present("orin_cli") if orin else _absent("orin_cli")),
    )
    return Platform(kind=kind, values=values)


SPARK_PRESENT = _platform("dgx-spark", spark=True)
SPARK_ABSENT = _platform("dgx-spark")
THOR_PRESENT = _platform("jetson", thor=True)
ORIN_PRESENT = _platform("jetson", orin=True)
JETSON_ABSENT = _platform("jetson")
RTX = _platform("rtx")
GENERIC = _platform("generic")

_SERVICE_ARGS = {"service": "nvsh-daemon"}
_CONTAINER_ARGS = {"container": "mycontainer"}


# ---------------------------------------------------------------------------
# Every operation nvsh.ops registers must be handled by this table so a new
# operation can't silently ship without a rendering decision.
# ---------------------------------------------------------------------------


def test_every_operation_is_covered_by_this_test_module():
    covered = {
        "machine_status",
        "memory_stats",
        "gpu_stats",
        "disk_stats",
        "thermal_stats",
        "container_list",
        "network_info",
        "process_list",
        "power_get",
        "swap_status",
        "nvsh_doctor",
        "service_status",
        "service_logs",
        "power_set",
        "service_restart",
        "container_restart",
    }
    assert covered == set(operation_names())


# ---------------------------------------------------------------------------
# render() always returns a list of strings or None, never a shell string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,args,platform",
    [
        ("memory_stats", {}, SPARK_PRESENT),
        ("memory_stats", {}, SPARK_ABSENT),
        ("service_status", _SERVICE_ARGS, GENERIC),
    ],
)
def test_render_result_shape(op, args, platform):
    result = render(op, args, platform)
    assert result is None or (
        isinstance(result, list) and all(isinstance(item, str) for item in result)
    )
    if result is not None:
        rejoined = " ".join(result)
        assert ";" not in rejoined and "|" not in rejoined and "&&" not in rejoined


# ---------------------------------------------------------------------------
# Device CLI present: spark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,expected",
    [
        ("machine_status", ["spark", "status", "--json"]),
        ("memory_stats", ["spark", "memory", "--json"]),
        ("gpu_stats", ["spark", "gpu", "--json"]),
        ("disk_stats", ["spark", "disk", "--json"]),
        ("thermal_stats", ["spark", "thermal", "--json"]),
        ("container_list", ["spark", "containers", "--json"]),
        ("network_info", ["spark", "network", "--json"]),
        ("process_list", ["spark", "processes", "--json"]),
        ("swap_status", ["spark", "swap", "status", "--json"]),
        # spark has no power verb -- falls through to the system fallback,
        # which is also None on dgx-spark (no nvpmodel there).
        ("power_get", None),
    ],
)
def test_spark_device_cli_present(op, expected):
    assert render(op, {}, SPARK_PRESENT) == expected


# ---------------------------------------------------------------------------
# Device CLI present: thor (has power, unlike spark)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,expected",
    [
        ("machine_status", ["thor", "status", "--json"]),
        ("memory_stats", ["thor", "memory", "--json"]),
        ("gpu_stats", ["thor", "gpu", "--json"]),
        ("disk_stats", ["thor", "disk", "--json"]),
        ("thermal_stats", ["thor", "thermal", "--json"]),
        ("container_list", ["thor", "containers", "--json"]),
        ("network_info", ["thor", "network", "--json"]),
        ("process_list", ["thor", "processes", "--json"]),
        ("swap_status", ["thor", "swap", "status", "--json"]),
        ("power_get", ["thor", "power", "--json"]),
    ],
)
def test_thor_device_cli_present(op, expected):
    assert render(op, {}, THOR_PRESENT) == expected


# ---------------------------------------------------------------------------
# Device CLI present: orin -- no verbs registered yet, so every machine
# operation falls straight through to the system fallback even though
# orin_cli is present on PATH.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,expected",
    [
        ("machine_status", None),
        ("memory_stats", ["free", "-m"]),
        ("gpu_stats", ["nvidia-smi"]),
        ("disk_stats", ["df", "-h"]),
        ("thermal_stats", None),
        ("container_list", ["docker", "ps"]),
        ("network_info", ["ip", "-brief", "addr"]),
        ("process_list", ["ps", "-eo", "pid,rss,comm", "--sort=-rss"]),
        ("swap_status", ["swapon", "--show"]),
        ("power_get", ["nvpmodel", "-q"]),
    ],
)
def test_orin_device_cli_present_but_has_no_verbs(op, expected):
    assert render(op, {}, ORIN_PRESENT) == expected
    assert DEVICE_CLI_VERBS["orin"] == {}


# ---------------------------------------------------------------------------
# Device CLI absent: dgx-spark kind, spark_cli not on PATH
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,expected",
    [
        ("machine_status", None),
        ("memory_stats", ["free", "-m"]),
        ("gpu_stats", ["nvidia-smi"]),
        ("disk_stats", ["df", "-h"]),
        ("thermal_stats", None),
        ("container_list", ["docker", "ps"]),
        ("network_info", ["ip", "-brief", "addr"]),
        ("process_list", ["ps", "-eo", "pid,rss,comm", "--sort=-rss"]),
        ("swap_status", ["swapon", "--show"]),
        # not jetson -> no nvpmodel fallback either
        ("power_get", None),
    ],
)
def test_spark_kind_device_cli_absent(op, expected):
    assert render(op, {}, SPARK_ABSENT) == expected


# ---------------------------------------------------------------------------
# Device CLI absent: jetson kind, neither thor_cli nor orin_cli on PATH
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,expected",
    [
        ("machine_status", None),
        ("memory_stats", ["free", "-m"]),
        ("gpu_stats", ["nvidia-smi"]),
        ("disk_stats", ["df", "-h"]),
        ("thermal_stats", None),
        ("container_list", ["docker", "ps"]),
        ("network_info", ["ip", "-brief", "addr"]),
        ("process_list", ["ps", "-eo", "pid,rss,comm", "--sort=-rss"]),
        ("swap_status", ["swapon", "--show"]),
        ("power_get", ["nvpmodel", "-q"]),
    ],
)
def test_jetson_kind_device_cli_absent(op, expected):
    assert render(op, {}, JETSON_ABSENT) == expected


# ---------------------------------------------------------------------------
# rtx / generic kinds never have a device CLI candidate at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", [RTX, GENERIC])
@pytest.mark.parametrize(
    "op,expected",
    [
        ("machine_status", None),
        ("memory_stats", ["free", "-m"]),
        ("gpu_stats", ["nvidia-smi"]),
        ("power_get", None),
    ],
)
def test_rtx_and_generic_kinds_use_system_fallback(op, expected, platform):
    assert render(op, {}, platform) == expected


# ---------------------------------------------------------------------------
# nvsh_doctor is always the same, on every platform, device CLI or not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform",
    [SPARK_PRESENT, SPARK_ABSENT, THOR_PRESENT, ORIN_PRESENT, JETSON_ABSENT, RTX, GENERIC],
)
def test_nvsh_doctor_always_renders_the_same(platform):
    assert render("nvsh_doctor", {}, platform) == ["nvsh", "doctor", "--json"]


# ---------------------------------------------------------------------------
# Operations that take an argument: service_status/service_logs/
# service_restart/container_restart never come from a device CLI (no
# device CLI has service verbs), and each argument value occupies exactly
# one argv element.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform",
    [SPARK_PRESENT, SPARK_ABSENT, THOR_PRESENT, ORIN_PRESENT, JETSON_ABSENT, RTX, GENERIC],
)
def test_service_status_always_system_command(platform):
    assert render("service_status", _SERVICE_ARGS, platform) == [
        "systemctl",
        "status",
        "--no-pager",
        "nvsh-daemon",
    ]


@pytest.mark.parametrize(
    "platform",
    [SPARK_PRESENT, SPARK_ABSENT, THOR_PRESENT, ORIN_PRESENT, JETSON_ABSENT, RTX, GENERIC],
)
def test_service_logs_always_system_command(platform):
    assert render("service_logs", _SERVICE_ARGS, platform) == [
        "journalctl",
        "-u",
        "nvsh-daemon",
        "-n",
        "50",
        "--no-pager",
    ]


@pytest.mark.parametrize(
    "platform",
    [SPARK_PRESENT, SPARK_ABSENT, THOR_PRESENT, ORIN_PRESENT, JETSON_ABSENT, RTX, GENERIC],
)
def test_service_restart_always_system_command(platform):
    assert render("service_restart", _SERVICE_ARGS, platform) == [
        "sudo",
        "systemctl",
        "restart",
        "nvsh-daemon",
    ]


@pytest.mark.parametrize(
    "platform",
    [SPARK_PRESENT, SPARK_ABSENT, THOR_PRESENT, ORIN_PRESENT, JETSON_ABSENT, RTX, GENERIC],
)
def test_container_restart_always_system_command(platform):
    assert render("container_restart", _CONTAINER_ARGS, platform) == [
        "docker",
        "restart",
        "mycontainer",
    ]


def test_service_argument_is_one_argv_element_even_with_spaces():
    # A service/container name that happens to contain a space must not be
    # split across argv elements or need shell quoting -- it's one element.
    result = render("service_status", {"service": "my service with spaces"}, GENERIC)
    assert result == ["systemctl", "status", "--no-pager", "my service with spaces"]
    assert len(result) == 4


# ---------------------------------------------------------------------------
# power_set: mutating, never comes from a device CLI (none has a mutating
# verb). max_performance is the one nvpmodel mode id (0) that's certain on
# every board measured; balanced/low_power render None because the mode id
# differs per board and hasn't been verified for either.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform,mode,expected",
    [
        (THOR_PRESENT, "max_performance", ["sudo", "nvpmodel", "-m", "0"]),
        (JETSON_ABSENT, "max_performance", ["sudo", "nvpmodel", "-m", "0"]),
        (ORIN_PRESENT, "max_performance", ["sudo", "nvpmodel", "-m", "0"]),
        (THOR_PRESENT, "balanced", None),
        (THOR_PRESENT, "low_power", None),
        (SPARK_PRESENT, "max_performance", None),
        (SPARK_ABSENT, "max_performance", None),
        (RTX, "max_performance", None),
        (GENERIC, "max_performance", None),
    ],
)
def test_power_set(platform, mode, expected):
    assert render("power_set", {"mode": mode}, platform) == expected


# ---------------------------------------------------------------------------
# thermal_stats: no single, non-shell, exiting argv exists on any of these
# platforms (device CLI absent), on purpose -- see the comment in render.py.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", [SPARK_ABSENT, JETSON_ABSENT, RTX, GENERIC, ORIN_PRESENT])
def test_thermal_stats_is_none_without_a_device_cli_that_has_the_verb(platform):
    assert render("thermal_stats", {}, platform) is None


# ---------------------------------------------------------------------------
# machine_status: same story -- no generic composite-status command exists.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", [SPARK_ABSENT, JETSON_ABSENT, RTX, GENERIC, ORIN_PRESENT])
def test_machine_status_is_none_without_a_device_cli_that_has_the_verb(platform):
    assert render("machine_status", {}, platform) is None
