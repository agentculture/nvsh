"""Tests for nvsh.ops.render.render(): operation + platform -> argv list.

Table-tests every registered operation for each platform kind, both with
the relevant device CLI present (spark/thor/orin) and absent (system
fallback), asserting the exact argv list render() returns.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

from nvsh.ops.render import (
    DEVICE_CLI_MIN_VERSIONS,
    DEVICE_CLI_VERBS,
    cli_meets_floor,
    render,
    usable_device_cli,
)
from nvsh.ops.table import get as get_operation
from nvsh.ops.table import names as operation_names
from nvsh.platform import ON_PATH, Platform, Value
from nvsh.platform._model import FILE, PATH, SUBPROCESS, device_cli_source

REPO_ROOT = Path(__file__).resolve().parent.parent

# The module, not the re-exported function nvsh.ops.render also names.
render_mod = importlib.import_module("nvsh.ops.render")

#: Marks a CLI as present on disk but with no parseable ``--version``.
NO_VERSION = "no-version"


def _cli_path(cli: str) -> str:
    # Deliberately not on any PATH a test host has: render() must use the
    # resolved path detection found, never the bare name (h17: a CLI the
    # nvsh extra installed lives in nvsh's own env bin, off PATH).
    return f"/opt/nvsh-env/bin/{cli}"


def _cli_values(cli: str, version: str | None) -> tuple[Value, Value]:
    """``<cli>_cli`` / ``<cli>_cli_version`` as detection would build them.

    *version* ``None`` = the CLI is absent; :data:`NO_VERSION` = present but
    ``--version`` printed nothing parseable; anything else = that version.
    """
    name, version_name, version_source = f"{cli}_cli", f"{cli}_cli_version", f"{cli} --version"
    if version is None:
        return (
            Value(name, None, device_cli_source(cli, None), PATH, False),
            Value(version_name, None, version_source, SUBPROCESS, False),
        )
    cli_value = Value(name, _cli_path(cli), device_cli_source(cli, ON_PATH), PATH, True)
    if version == NO_VERSION:
        return cli_value, Value(version_name, None, version_source, SUBPROCESS, False)
    return cli_value, Value(version_name, version, version_source, SUBPROCESS, True)


def _platform(
    kind: str,
    *,
    board: str | None = None,
    spark: str | None = None,
    thor: str | None = None,
    orin: str | None = None,
) -> Platform:
    """A Platform built from the real nvsh.platform types.

    Each CLI keyword is that CLI's ``--version`` (``None`` = absent).
    """
    values: list[Value] = []
    for cli, version in (("spark", spark), ("thor", thor), ("orin", orin)):
        values.extend(_cli_values(cli, version))
    values.append(
        Value(
            "jetson_board",
            board,
            "/proc/device-tree/model",
            FILE,
            board is not None,
        )
    )
    return Platform(kind=kind, values=tuple(values))


# Floors are restated here on purpose, not read from render.py: a floor
# change must be a deliberate test change too.
SPARK_FLOOR, THOR_FLOOR, ORIN_FLOOR = "0.8.0", "0.5.0", "0.6.0"

SPARK_PRESENT = _platform("dgx-spark", spark=SPARK_FLOOR)
SPARK_ABSENT = _platform("dgx-spark")
THOR_PRESENT = _platform("jetson", board="thor", thor=THOR_FLOOR)
ORIN_PRESENT = _platform("jetson", board="orin", orin=ORIN_FLOOR)
ORIN_BELOW_FLOOR = _platform("jetson", board="orin", orin="0.5.0")
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
        assert ";" not in rejoined
        assert "|" not in rejoined
        assert "&&" not in rejoined


# ---------------------------------------------------------------------------
# The verb tables and floors, pinned exactly: any key gained or lost in a
# _*_VERBS table (or a floor moved) fails here until the expected rows below
# change with it.
# ---------------------------------------------------------------------------

#: The ten read-only machine operations every device CLI maps, and the verb
#: each renders to. Restated literally (not read from render.py).
EXPECTED_VERBS: dict[str, list[str]] = {
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

READ_ONLY_DEVICE_OPS = tuple(EXPECTED_VERBS)


def test_verb_tables_are_exactly_the_expected_rows():
    assert render_mod._SPARK_VERBS == EXPECTED_VERBS
    assert render_mod._THOR_VERBS == EXPECTED_VERBS
    assert render_mod._ORIN_VERBS == EXPECTED_VERBS
    assert DEVICE_CLI_VERBS == {
        "spark": EXPECTED_VERBS,
        "thor": EXPECTED_VERBS,
        "orin": EXPECTED_VERBS,
    }


def test_every_device_verb_is_a_read_only_operation():
    """No device CLI verb backs a mutating operation (power_set, service_*,
    container_restart keep their system fallbacks)."""
    for verbs in DEVICE_CLI_VERBS.values():
        for op in verbs:
            assert get_operation(op).read_only, op


def test_floors_are_exactly_the_expected_versions():
    assert DEVICE_CLI_MIN_VERSIONS == {"spark": (0, 8, 0), "thor": (0, 5, 0), "orin": (0, 6, 0)}
    assert set(DEVICE_CLI_MIN_VERSIONS) == set(DEVICE_CLI_VERBS)


# ---------------------------------------------------------------------------
# cli_meets_floor / usable_device_cli: integer-tuple comparison, fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version,expected",
    [
        ("0.8.0", True),
        ("0.8.1", True),
        ("0.10.0", True),  # integer tuples, not string order ("0.10" < "0.8")
        ("1.0", True),
        ("v0.8.0", True),
        # a suffix on the floor release (a branch build) counts as that release
        ("0.8.0.dev1", True),
        ("0.7.9", False),
        ("0.7", False),
        ("0.8", False),  # (0, 8) < (0, 8, 0)
        ("garbage", False),
        (NO_VERSION, False),  # present, --version unparseable: fail closed
        (None, False),  # absent
    ],
)
def test_cli_meets_floor_spark(version, expected):
    platform = _platform("dgx-spark", spark=version)
    assert cli_meets_floor(platform, "spark") is expected
    usable = usable_device_cli(platform, "spark")
    assert (usable is not None) is expected
    if usable is not None:
        assert usable.path == _cli_path("spark")


def test_cli_without_a_floor_never_meets_it():
    assert cli_meets_floor(SPARK_PRESENT, "jetson-cli") is False


# ---------------------------------------------------------------------------
# The table: kind x board x present / absent / below-floor, per op.
# ---------------------------------------------------------------------------

#: System fallback per platform kind (what render() returns with no usable
#: device CLI). dgx-spark has no nvpmodel, so power_get has no fallback there.
_COMMON_FALLBACK: dict[str, list[str] | None] = {
    "machine_status": None,
    "memory_stats": ["free", "-m"],
    "gpu_stats": ["nvidia-smi"],
    "disk_stats": ["df", "-h"],
    "thermal_stats": None,
    "container_list": ["docker", "ps"],
    "network_info": ["ip", "-brief", "addr"],
    "process_list": ["ps", "-eo", "pid,rss,comm", "--sort=-rss"],
    "swap_status": ["swapon", "--show"],
}
FALLBACK = {
    "dgx-spark": {**_COMMON_FALLBACK, "power_get": None},
    "jetson": {**_COMMON_FALLBACK, "power_get": ["nvpmodel", "-q"]},
}

#: (label, kind, board, cli, floor, one version below the floor)
BOARDS = (
    ("spark", "dgx-spark", None, "spark", SPARK_FLOOR, "0.7.0"),
    ("thor", "jetson", "thor", "thor", THOR_FLOOR, "0.4.9"),
    ("orin", "jetson", "orin", "orin", ORIN_FLOOR, "0.5.0"),
)
STATES = ("at-floor", "above-floor", "absent", "below-floor", "no-version")


def _state_version(state: str, floor: str, below: str) -> str | None:
    return {
        "at-floor": floor,
        "above-floor": "9.0.0",
        "absent": None,
        "below-floor": below,
        "no-version": NO_VERSION,
    }[state]


def _table_rows():
    for label, kind, board, cli, floor, below in BOARDS:
        for state in STATES:
            for op in READ_ONLY_DEVICE_OPS:
                usable = state in ("at-floor", "above-floor")
                expected = (
                    [_cli_path(cli), *EXPECTED_VERBS[op], "--json"]
                    if usable
                    else FALLBACK[kind][op]
                )
                yield pytest.param(
                    kind,
                    board,
                    cli,
                    _state_version(state, floor, below),
                    op,
                    expected,
                    id=f"{label}-{state}-{op}",
                )


@pytest.mark.parametrize("kind,board,cli,version,op,expected", list(_table_rows()))
def test_render_table(kind, board, cli, version, op, expected):
    platform = _platform(kind, board=board, **{cli: version})
    assert render(op, {}, platform) == expected


def test_render_table_covers_every_board_and_state():
    """10 read-only ops x 3 boards x 5 states: the table is complete."""
    assert len(list(_table_rows())) == 10 * 3 * len(STATES)


@pytest.mark.parametrize(
    "platform",
    [SPARK_PRESENT, THOR_PRESENT, ORIN_PRESENT],
    ids=["spark", "thor", "orin"],
)
def test_ten_of_ten_read_only_ops_use_the_device_cli_at_its_floor(platform):
    rendered = [render(op, {}, platform) for op in READ_ONLY_DEVICE_OPS]
    assert all(argv is not None and argv[-1] == "--json" for argv in rendered)
    assert all(argv[0].startswith("/opt/nvsh-env/bin/") for argv in rendered)  # type: ignore


def test_orin_below_floor_uses_every_system_fallback():
    for op in READ_ONLY_DEVICE_OPS:
        assert render(op, {}, ORIN_BELOW_FLOOR) == FALLBACK["jetson"][op]


# ---------------------------------------------------------------------------
# Board selection on kind "jetson"
# ---------------------------------------------------------------------------


def test_thor_board_ignores_an_orin_cli():
    platform = _platform("jetson", board="thor", orin=ORIN_FLOOR)
    assert render("machine_status", {}, platform) is None
    assert render("power_get", {}, platform) == ["nvpmodel", "-q"]


def test_orin_board_ignores_a_thor_cli():
    platform = _platform("jetson", board="orin", thor=THOR_FLOOR)
    assert render("machine_status", {}, platform) is None


def test_orin_board_prefers_orin_even_when_thor_is_also_present():
    platform = _platform("jetson", board="orin", thor=THOR_FLOOR, orin=ORIN_FLOOR)
    assert render("gpu_stats", {}, platform) == [_cli_path("orin"), "gpu", "--json"]


def test_unknown_board_tries_thor_then_orin():
    both = _platform("jetson", thor=THOR_FLOOR, orin=ORIN_FLOOR)
    assert render("gpu_stats", {}, both) == [_cli_path("thor"), "gpu", "--json"]
    orin_only = _platform("jetson", orin=ORIN_FLOOR)
    assert render("gpu_stats", {}, orin_only) == [_cli_path("orin"), "gpu", "--json"]


def test_unknown_board_skips_a_thor_below_its_floor():
    platform = _platform("jetson", thor="0.4.0", orin=ORIN_FLOOR)
    assert render("gpu_stats", {}, platform) == [_cli_path("orin"), "gpu", "--json"]


def test_board_is_ignored_off_jetson():
    platform = _platform("dgx-spark", board="thor", spark=SPARK_FLOOR, thor=THOR_FLOOR)
    assert render("gpu_stats", {}, platform) == [_cli_path("spark"), "gpu", "--json"]


# ---------------------------------------------------------------------------
# Criterion 4: no operation name appears outside nvsh/ops/table.py and the
# data tables in nvsh/ops/render.py -- never in a comparison or branch.
# ---------------------------------------------------------------------------


_OP_LITERAL = re.compile("[\"'](" + "|".join(map(re.escape, operation_names())) + ")[\"']")


def test_no_operation_name_outside_the_table_and_render_tables():
    pattern = _OP_LITERAL
    allowed = {REPO_ROOT / "nvsh" / "ops" / "table.py", REPO_ROOT / "nvsh" / "ops" / "render.py"}
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for path in sorted((REPO_ROOT / "nvsh").rglob("*.py"))
        if path not in allowed
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []


def test_render_names_operations_only_as_table_keys():
    """In render.py an operation name may only be a key of a module-level
    dict literal (a verb, fallback or floor table) -- never compared to."""
    tree = ast.parse((REPO_ROOT / "nvsh" / "ops" / "render.py").read_text(encoding="utf-8"))
    ops = set(operation_names())
    keys: set[int] = set()
    for node in tree.body:
        value = getattr(node, "value", None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Dict):
            keys.update(id(key) for key in value.keys if key is not None)
    stray = [
        f"line {node.lineno}: {node.value}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value in ops and id(node) not in keys
    ]
    assert stray == []


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


@pytest.mark.parametrize("platform", [SPARK_ABSENT, JETSON_ABSENT, RTX, GENERIC, ORIN_BELOW_FLOOR])
def test_thermal_stats_is_none_without_a_device_cli_that_has_the_verb(platform):
    assert render("thermal_stats", {}, platform) is None


# ---------------------------------------------------------------------------
# machine_status: same story -- no generic composite-status command exists.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", [SPARK_ABSENT, JETSON_ABSENT, RTX, GENERIC, ORIN_BELOW_FLOOR])
def test_machine_status_is_none_without_a_device_cli_that_has_the_verb(platform):
    assert render("machine_status", {}, platform) is None


@pytest.mark.parametrize("value", ["-f", "--now", "-f --rm", "", "a\nb", "a\x00b"])
@pytest.mark.parametrize(
    ("operation", "arg"),
    [
        ("service_restart", "service"),
        ("service_status", "service"),
        ("service_logs", "service"),
        ("container_restart", "container"),
    ],
)
def test_option_like_or_unprintable_argument_renders_nothing(operation, arg, value):
    """One argv element can still be read as an option; refuse it outright."""
    assert (
        render(operation, {arg: value}, _platform("jetson", board="thor", thor=THOR_FLOOR)) is None
    )


def test_shell_metacharacters_stay_one_argv_element():
    argv = render("service_restart", {"service": "x; rm -rf /"}, _platform("generic"))
    assert argv == ["sudo", "systemctl", "restart", "x; rm -rf /"]
