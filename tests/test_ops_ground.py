"""Argument grounding: ``nvsh.ops.ground``.

A model gives us an argument value such as a service name.  It is
untrusted text.  ``ground()`` checks that the value names something
that really exists on this machine, and returns the machine's own
spelling of it.

Tests: every runner call is a fake; the real ``systemctl`` / ``docker``
is never invoked.
"""

from __future__ import annotations

import os

import pytest

from nvsh.ops._model import Operation
from nvsh.ops.ground import (
    CONTAINER_LOOKUP_ARGV,
    LOOKUP_TIMEOUT,
    SERVICE_LOOKUP_ARGV,
    GroundDecline,
    Grounded,
    Runner,
    default_runner,
    ground,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fake_runner(*, candidates: list[str], exit_code: int = 0) -> Runner:
    """Return a ``Runner`` that records argv calls and returns *candidates*."""
    calls: list[tuple[list[str], float]] = []

    def _run(argv: list[str], timeout: float) -> tuple[int, str]:
        calls.append((list(argv), timeout))
        stdout = "\n".join(candidates) if candidates else ""
        return (exit_code, stdout)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def fake_runner_exception() -> Runner:
    """Return a ``Runner`` that raises ``RuntimeError``."""

    def _run(argv: list[str], timeout: float) -> tuple[int, str]:
        raise RuntimeError("boom")

    _run.calls = []  # type: ignore[attr-defined]
    return _run


# ---------------------------------------------------------------------------
# test_operation_without_service_or_container_does_not_call_runner
# ---------------------------------------------------------------------------


def test_operation_without_service_or_container_does_not_call_runner() -> None:
    """An operation with no *service* / *container* arg skips the runner."""
    op = Operation(name="machine_status", description="show status", read_only=True)
    runner = fake_runner(candidates=[])
    result = ground(op, {"foo": "bar"}, runner=runner)
    assert isinstance(result, Grounded)
    assert result.args == {"foo": "bar"}
    assert len(runner.calls) == 0  # noqa: SIM300


# ---------------------------------------------------------------------------
# test_service_resolves_case_insensitively  ("vLLM" -> "vllm.service")
# ---------------------------------------------------------------------------


def test_service_resolves_case_insensitively() -> None:
    """'vLLM' case-folds to an existing 'vllm.service' fixture."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=["vllm.service", "dbus.service"])
    result = ground(op, {"service": "vLLM"}, runner=runner)
    assert isinstance(result, Grounded)
    assert result.args["service"] == "vllm.service"


# ---------------------------------------------------------------------------
# test_service_full_unit_name_resolves  ("vllm.service" -> "vllm.service")
# ---------------------------------------------------------------------------


def test_service_full_unit_name_resolves() -> None:
    """'vllm.service' (with suffix) resolves to the same fixture."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=["vllm.service", "dbus.service"])
    result = ground(op, {"service": "vllm.service"}, runner=runner)
    assert isinstance(result, Grounded)
    assert result.args["service"] == "vllm.service"


# ---------------------------------------------------------------------------
# test_hostile_service_values_decline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "rm -rf /",
        "x; reboot",
        "$(id)",
        "`id`",
        "vllm.service; reboot",
        "../../etc/passwd",
        "a\nb",
    ],
)
def test_hostile_service_values_decline(value: str) -> None:
    """Hostile values that do not exist are declined with no-such-service."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=["vllm.service", "dbus.service"])
    result = ground(op, {"service": value}, runner=runner)
    assert isinstance(result, GroundDecline)
    assert result.code == "no_such_service"
    assert "\n" not in result.message


# ---------------------------------------------------------------------------
# test_nonexistent_service_declines
# ---------------------------------------------------------------------------


def test_nonexistent_service_declines() -> None:
    """A value that is clean but not in the list → decline."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=["vllm.service", "dbus.service"])
    result = ground(op, {"service": "nonexistent.service"}, runner=runner)
    assert isinstance(result, GroundDecline)
    assert result.code == "no_such_service"


# ---------------------------------------------------------------------------
# test_no_substring_match
# ---------------------------------------------------------------------------


def test_no_substring_match() -> None:
    """'vllm' must NOT match 'vllm-worker.service' (exact only)."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=["vllm-worker.service"])
    result = ground(op, {"service": "vllm"}, runner=runner)
    assert isinstance(result, GroundDecline)
    assert result.code == "no_such_service"


# ---------------------------------------------------------------------------
# test_ambiguous_match_lists_candidates
# ---------------------------------------------------------------------------


def test_ambiguous_match_lists_candidates() -> None:
    """Two units differing only by case → ambiguous with sorted list."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=["Foo.service", "foo.service"])
    result = ground(op, {"service": "foo"}, runner=runner)
    assert isinstance(result, GroundDecline)
    assert result.code == "ambiguous"
    # Sorted: Foo.service, foo.service (capital F < lowercase f)
    assert "Foo.service" in result.message
    assert "foo.service" in result.message


# ---------------------------------------------------------------------------
# test_container_resolves_only_to_listed_name
# ---------------------------------------------------------------------------


def test_container_resolves_only_to_listed_name() -> None:
    """A container value must be an exact case-insensitive match from the list."""
    op = Operation(
        name="container_restart",
        description="restart container",
        read_only=False,
        args=(),
    )

    runner = fake_runner(candidates=["nvsh-agent", "prometheus"])
    result = ground(op, {"container": "NvSh-AgEnT"}, runner=runner)
    assert isinstance(result, Grounded)
    assert result.args["container"] == "nvsh-agent"


# ---------------------------------------------------------------------------
# test_nonexistent_container_declines
# ---------------------------------------------------------------------------


def test_nonexistent_container_declines() -> None:
    """A container not in the list → no_such_container."""
    op = Operation(
        name="container_restart",
        description="restart container",
        read_only=False,
        args=(),
    )

    runner = fake_runner(candidates=["nvsh-agent", "prometheus"])
    result = ground(op, {"container": "ghost"}, runner=runner)
    assert isinstance(result, GroundDecline)
    assert result.code == "no_such_container"


# ---------------------------------------------------------------------------
# test_lookup_failure_declines  (runner returns (1, ""))
# ---------------------------------------------------------------------------


def test_lookup_failure_declines() -> None:
    """When the lookup runner returns non-zero, decline with lookup_failed."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=[], exit_code=1)
    result = ground(op, {"service": "vllm"}, runner=runner)
    assert isinstance(result, GroundDecline)
    assert result.code == "lookup_failed"
    assert result.message == "could not list services"


# ---------------------------------------------------------------------------
# test_runner_exception_declines
# ---------------------------------------------------------------------------


def test_runner_exception_declines() -> None:
    """When the runner raises, decline with lookup_failed."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner_exception()
    result = ground(op, {"service": "vllm"}, runner=runner)
    assert isinstance(result, GroundDecline)
    assert result.code == "lookup_failed"


# ---------------------------------------------------------------------------
# test_untrusted_value_never_reaches_runner
# ---------------------------------------------------------------------------


def test_untrusted_value_never_reaches_runner() -> None:
    """Every argv recorded by the fake runner equals the expected lookup argv."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=["vllm.service"])
    # Hostile value — should still only see the lookup argv.
    ground(op, {"service": "rm -rf /; cat /etc/shadow"}, runner=runner)

    for argv, _timeout in runner.calls:
        assert argv == SERVICE_LOOKUP_ARGV


def test_container_argv_never_leaks() -> None:
    """Container grounding also never leaks the value to the runner."""
    op = Operation(
        name="container_restart",
        description="restart container",
        read_only=False,
        args=(),
    )

    runner = fake_runner(candidates=["nvsh-agent"])
    ground(op, {"container": "$(id) && curl evil.com"}, runner=runner)

    for argv, _timeout in runner.calls:
        assert argv == CONTAINER_LOOKUP_ARGV


# ---------------------------------------------------------------------------
# test_default_runner_never_uses_shell
# ---------------------------------------------------------------------------


def test_default_runner_never_uses_shell() -> None:
    """Read ground.py source and assert 'shell=True' is not present."""
    ground_py = os.path.join(
        os.path.dirname(__file__),
        "..",
        "nvsh",
        "ops",
        "ground.py",
    )
    source_text = open(ground_py).read()  # noqa: SIM115
    assert "shell=True" not in source_text


# ---------------------------------------------------------------------------
# test_default_runner_missing_binary_returns_127
# ---------------------------------------------------------------------------


def test_default_runner_missing_binary_returns_127() -> None:
    """Calling a binary that doesn't exist returns (127, '')."""
    code, stdout = default_runner(["nvsh-no-such-binary-xyz"], 1.0)
    assert code == 127
    assert stdout == ""


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_service_lookup_argv_is_correct() -> None:
    expected = [
        "systemctl",
        "list-units",
        "--type=service",
        "--all",
        "--no-legend",
        "--plain",
    ]
    assert SERVICE_LOOKUP_ARGV == expected


def test_container_lookup_argv_is_correct() -> None:
    expected = ["docker", "ps", "-a", "--format", "{{.Names}}"]
    assert CONTAINER_LOOKUP_ARGV == expected


def test_lookup_timeout_is_3_seconds() -> None:
    assert LOOKUP_TIMEOUT == 3.0


# ---------------------------------------------------------------------------
# ground() never raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name,key",
    [
        ("service_status", "service"),
        ("container_restart", "container"),
    ],
)
def test_ground_never_raises_with_garbage_args(op_name: str, key: str) -> None:
    """ground() must never raise, no matter what types of args are passed."""
    op = Operation(
        name=op_name,
        description="test",
        read_only=True,
        args=(),
    )
    try:
        result = ground(op, {key: 42}, runner=fake_runner(candidates=[]))
    except Exception:  # noqa: BLE001
        pytest.fail(f"ground({op_name!r}, ...) raised")
    assert isinstance(result, (Grounded, GroundDecline))


# ---------------------------------------------------------------------------
# ground() copies non-grounded args unchanged
# ---------------------------------------------------------------------------


def test_non_grounding_args_copied_unchanged() -> None:
    """Args not named 'service' or 'container' pass through."""
    op = Operation(
        name="machine_status",
        description="test",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=[])
    result = ground(op, {"foo": "bar", "baz": "qux"}, runner=runner)
    assert isinstance(result, Grounded)
    assert result.args == {"foo": "bar", "baz": "qux"}
    assert len(runner.calls) == 0


# ---------------------------------------------------------------------------
# Mixed grounded and non-grounded args
# ---------------------------------------------------------------------------


def test_mixed_args_grounds_service_copies_other() -> None:
    """When 'service' is grounded, other args are copied."""
    op = Operation(
        name="service_status",
        description="query service",
        read_only=True,
        args=(),
    )

    runner = fake_runner(candidates=["vllm.service"])
    result = ground(op, {"service": "vllm", "extra": "keep"}, runner=runner)
    assert isinstance(result, Grounded)
    assert result.args["service"] == "vllm.service"
    assert result.args["extra"] == "keep"
