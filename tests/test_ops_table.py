"""Operation table model, validation, and the 16-entry registry.

Tests for ``nvsh.ops``: the frozen dataclasses, the 16-operation table,
and every validation path (unknown, missing, unexpected, wrong-type,
blank, bad-choice, and never-raise).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from nvsh.ops._model import ValidationError
from nvsh.ops.table import OPERATIONS, get, names, validate

# ---------------------------------------------------------------------------
# test_table_has_the_sixteen_operations_in_order
# ---------------------------------------------------------------------------


def test_table_has_the_sixteen_operations_in_order() -> None:
    """All 16 names in the exact order the plan specifies."""
    expected = (
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
    )
    assert len(OPERATIONS) == 16
    assert tuple(op.name for op in OPERATIONS) == expected


# ---------------------------------------------------------------------------
# test_read_only_flags
# ---------------------------------------------------------------------------


def test_read_only_flags() -> None:
    """Exactly three operations are mutating; the rest are read-only."""
    mutating = ("power_set", "service_restart", "container_restart")
    expected_mutating = set(mutating)
    found_mutating: set[str] = set()
    for op in OPERATIONS:
        if not op.read_only:
            found_mutating.add(op.name)
    assert found_mutating == expected_mutating


# ---------------------------------------------------------------------------
# test_validate_accepts_valid_calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,args",
    [
        ("machine_status", {}),
        ("memory_stats", {}),
        ("gpu_stats", {}),
        ("disk_stats", {}),
        ("thermal_stats", {}),
        ("container_list", {}),
        ("network_info", {}),
        ("process_list", {}),
        ("power_get", {}),
        ("swap_status", {}),
        ("nvsh_doctor", {}),
        ("service_status", {"service": "nvsh"}),
        ("service_logs", {"service": "nvsh"}),
        ("power_set", {"mode": "max_performance"}),
        ("power_set", {"mode": "balanced"}),
        ("power_set", {"mode": "low_power"}),
        ("service_restart", {"service": "nvsh"}),
        ("container_restart", {"container": "nvsh-agent"}),
    ],
)
def test_validate_accepts_valid_calls(name: str, args: dict[str, object]) -> None:
    """Every known operation with correct arguments returns None."""
    result = validate(name, args)
    assert result is None, f"expected valid, got {result}"


# ---------------------------------------------------------------------------
# test_validate_unknown_operation
# ---------------------------------------------------------------------------


def test_validate_unknown_operation() -> None:
    """A name not in the table returns ValidationError(code='unknown_operation')."""
    err = validate("fake_thing", {})
    assert isinstance(err, ValidationError)
    assert err.code == "unknown_operation"
    assert "fake_thing" in err.message


# ---------------------------------------------------------------------------
# test_validate_missing_and_unexpected_argument
# ---------------------------------------------------------------------------


def test_validate_missing_argument() -> None:
    """Required argument omitted → missing_argument."""
    err = validate("service_status", {})
    assert isinstance(err, ValidationError)
    assert err.code == "missing_argument"
    assert "service" in err.message


def test_validate_unexpected_argument() -> None:
    """A key not declared on the operation → unexpected_argument."""
    err = validate("machine_status", {"bogus": "x"})
    assert isinstance(err, ValidationError)
    assert err.code == "unexpected_argument"
    assert "bogus" in err.message


# ---------------------------------------------------------------------------
# test_validate_wrong_type_and_blank
# ---------------------------------------------------------------------------


def test_validate_wrong_type() -> None:
    """Non-string value for a str argument → wrong_type."""
    err = validate("service_status", {"service": 42})
    assert isinstance(err, ValidationError)
    assert err.code == "wrong_type"
    assert "service" in err.message


def test_validate_blank_string() -> None:
    """Empty or whitespace-only string → wrong_type."""
    err = validate("service_status", {"service": ""})
    assert isinstance(err, ValidationError)
    assert err.code == "wrong_type"
    err2 = validate("service_status", {"service": "   "})
    assert isinstance(err2, ValidationError)
    assert err2.code == "wrong_type"


def test_validate_args_not_dict() -> None:
    """Non-dict args value → wrong_type."""
    err = validate("machine_status", "not a dict")
    assert isinstance(err, ValidationError)
    assert err.code == "wrong_type"
    err2 = validate("machine_status", [1, 2])
    assert isinstance(err2, ValidationError)
    assert err2.code == "wrong_type"


# ---------------------------------------------------------------------------
# test_validate_bad_choice
# ---------------------------------------------------------------------------


def test_validate_bad_choice() -> None:
    """A Literal choice argument with an invalid value → bad_choice."""
    err = validate("power_set", {"mode": "turbo"})
    assert isinstance(err, ValidationError)
    assert err.code == "bad_choice"
    assert "mode" in err.message
    assert "turbo" in err.message


# ---------------------------------------------------------------------------
# test_validate_never_raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        None,
        5,
        [],
        {"a": object()},
        "a" * 10000,
    ],
)
def test_validate_never_raises(name: object) -> None:
    """validate() never raises, no matter what is thrown at it."""
    result = validate(name, {})
    assert isinstance(result, (ValidationError, type(None)))


@pytest.mark.parametrize(
    "args",
    [
        None,
        5,
        [],
        {"a": object()},
        "a" * 10000,
    ],
)
def test_validate_never_raises_args(args: object) -> None:
    """validate(name, weird_args) never raises either."""
    result = validate("machine_status", args)
    assert isinstance(result, (ValidationError, type(None)))


# ---------------------------------------------------------------------------
# test_ops_imports_only_stdlib
# ---------------------------------------------------------------------------


def test_ops_imports_only_stdlib() -> None:
    """nvsh.ops pulls in no third-party packages (needle, numpy, requests, ...)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; import nvsh.ops; print(json.dumps(sorted(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, f"import nvsh.ops failed: {proc.stderr}"

    modules = json.loads(proc.stdout)
    suspects = (
        "needle",
        "numpy",
        "requests",
        "yaml",
        "ruamel",
    )
    offenders = [m for m in modules if any(m.startswith(p) for p in suspects)]
    assert not offenders, f"nvsh.ops pulled in third-party modules: {offenders}"


# ---------------------------------------------------------------------------
# get() and names() sanity
# ---------------------------------------------------------------------------


def test_get_returns_none_for_unknown() -> None:
    assert get("nonexistent") is None


def test_get_returns_operation() -> None:
    op = get("machine_status")
    assert op is not None
    assert op.name == "machine_status"


def test_names_returns_all_sixteen() -> None:
    ns = names()
    assert len(ns) == 16
    assert ns == tuple(op.name for op in OPERATIONS)
