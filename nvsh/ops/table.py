"""The 16-operation registry and its validate() gateway.

``OPERATIONS`` is a frozen tuple of :class:`.Operation` objects, ordered
exactly as the plan specifies.  ``get()``, ``names()``, and ``validate()``
are the public API.
"""

from __future__ import annotations

from nvsh.ops._model import ArgSpec, Operation, ValidationError

# ---------------------------------------------------------------------------
# The 16 operations (read-only first, then mutating, in plan order)
# ---------------------------------------------------------------------------

OPERATIONS: tuple[Operation, ...] = (
    Operation(
        name="machine_status",
        description="Show the current state of this machine.",
        read_only=True,
    ),
    Operation(
        name="memory_stats",
        description="Show the current memory usage statistics.",
        read_only=True,
    ),
    Operation(
        name="gpu_stats",
        description="Show GPU memory and utilisation statistics.",
        read_only=True,
    ),
    Operation(
        name="disk_stats",
        description="Show disk usage across all mounted filesystems.",
        read_only=True,
    ),
    Operation(
        name="thermal_stats",
        description="Show temperatures of this machine: how hot the CPU, GPU and board are.",
        read_only=True,
    ),
    Operation(
        name="container_list",
        description="List all containers currently running or known.",
        read_only=True,
    ),
    Operation(
        name="network_info",
        description="Show network interfaces and their current status.",
        read_only=True,
    ),
    Operation(
        name="process_list",
        description="List the current processes on this machine.",
        read_only=True,
    ),
    Operation(
        name="power_get",
        description="Read the current power mode configuration.",
        read_only=True,
    ),
    Operation(
        name="swap_status",
        description="Show the current swap usage and configuration.",
        read_only=True,
    ),
    Operation(
        name="nvsh_doctor",
        description="Run a health-check diagnostic pass on the nvsh environment.",
        read_only=True,
    ),
    Operation(
        name="service_status",
        description="Query the status of a named service.",
        read_only=True,
        args=(ArgSpec(name="service", kind="str"),),
    ),
    Operation(
        name="service_logs",
        description="Show the recent logs for a named service.",
        read_only=True,
        args=(ArgSpec(name="service", kind="str"),),
    ),
    Operation(
        name="power_set",
        description="Set the power mode for this machine.",
        read_only=False,
        args=(
            ArgSpec(
                name="mode",
                kind="choice",
                choices=("max_performance", "balanced", "low_power"),
            ),
        ),
    ),
    Operation(
        name="service_restart",
        description="Restart a named service.",
        read_only=False,
        args=(ArgSpec(name="service", kind="str"),),
    ),
    Operation(
        name="container_restart",
        description="Restart a named container.",
        read_only=False,
        args=(ArgSpec(name="container", kind="str"),),
    ),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get(name: str) -> Operation | None:
    """Return the operation with *name*, or ``None`` if not found."""
    for op in OPERATIONS:
        if op.name == name:
            return op
    return None


def names() -> tuple[str, ...]:
    """Return every registered operation name, in table order."""
    return tuple(op.name for op in OPERATIONS)


def validate(name: object, args: object) -> ValidationError | None:
    """Validate *name* and *args* against the operation table.

    Returns ``None`` when the call is valid, a :class:`ValidationError`
    describing the first problem found otherwise.  **Never raises**, no
    matter what types are passed.
    """
    # 1. name must be a str
    if not isinstance(name, str):
        return ValidationError(
            code="unknown_operation",
            message=f"{name!r} is not a valid operation name",
        )

    # 2. name must exist in the table
    op = get(name)
    if op is None:
        return ValidationError(
            code="unknown_operation",
            message=f"unknown operation {name!r}",
        )

    # 3. args must be a dict
    if not isinstance(args, dict):
        return ValidationError(
            code="wrong_type",
            message=f"arguments to {name!r} must be a dict, not {type(args).__name__}",
        )

    arg_map: dict[str, object] = args

    # 4. Every declared argument is required and must not be unexpected
    declared_names = {a.name for a in op.args}

    for arg_spec in op.args:
        if arg_spec.name not in arg_map:
            return ValidationError(
                code="missing_argument",
                message=f"{name!r} requires argument {arg_spec.name!r}",
            )

    for key in arg_map:
        if key not in declared_names:
            return ValidationError(
                code="unexpected_argument",
                message=f"{name!r} does not accept argument {key!r}",
            )

    # 5. Validate each argument value
    for arg_spec in op.args:
        value = arg_map[arg_spec.name]

        # Must be a string
        if not isinstance(value, str):
            return ValidationError(
                code="wrong_type",
                message=(
                    f"{name!r} argument {arg_spec.name!r} "
                    f"must be a string, not {type(value).__name__}"
                ),
            )

        # Empty or whitespace-only is wrong_type
        if not value.strip():
            return ValidationError(
                code="wrong_type",
                message=f"{name!r} argument {arg_spec.name!r} must not be empty",
            )

        # Literal choices
        if arg_spec.kind == "choice" and value not in arg_spec.choices:
            return ValidationError(
                code="bad_choice",
                message=(
                    f"{name!r} argument {arg_spec.name!r} must be one of "
                    f"{arg_spec.choices}, not {value!r}"
                ),
            )

    return None  # everything is fine
