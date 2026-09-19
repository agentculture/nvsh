"""Argument grounding: check that untrusted argument values name real units.

A model gives us an argument value such as a service name.  It is
untrusted text.  ``ground()`` checks that the value names something
that really exists on this machine, and returns the machine's own
spelling of it.

Public API
----------
* ``Runner`` — ``Callable[[list[str], float], tuple[int, str]]``
* ``SERVICE_LOOKUP_ARGV`` — systemctl argv for listing services
* ``CONTAINER_LOOKUP_ARGV`` — docker argv for listing containers
* ``LOOKUP_TIMEOUT`` — seconds (3.0)
* ``Grounded`` — result: args with each value replaced by the machine's
  own spelling.
* ``GroundDecline`` — result: decline code + one-line message.
* ``default_runner`` — subprocess.run wrapper, never uses shell.
* ``ground`` — the entry-point that decides which lookup to run.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

from nvsh.ops._model import Operation

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# (argv, timeout_seconds) -> (exit_code, stdout)
Runner = Callable[[list[str], float], tuple[int, str]]

# ---------------------------------------------------------------------------
# Lookup constants
# ---------------------------------------------------------------------------

SERVICE_LOOKUP_ARGV: list[str] = [
    "systemctl",
    "list-units",
    "--type=service",
    "--all",
    "--no-legend",
    "--plain",
]

CONTAINER_LOOKUP_ARGV: list[str] = [
    "docker",
    "ps",
    "-a",
    "--format",
    "{{.Names}}",
]

LOOKUP_TIMEOUT: float = 3.0

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grounded:
    """The argument value was grounded to a real unit.

    ``args`` contains the original arguments, with every grounded
    value replaced by the machine's own spelling (if grounded).
    """

    args: dict[str, str]


@dataclass(frozen=True)
class GroundDecline:
    """The argument could not be grounded.

    ``code`` is one of: ``"no_such_service"``, ``"no_such_container"``,
    ``"ambiguous"``, ``"lookup_failed"``.
    ``message`` is a one-line human-readable description (no newlines).
    """

    code: str
    message: str


# ---------------------------------------------------------------------------
# default_runner — the real subprocess wrapper
# ---------------------------------------------------------------------------


def default_runner(argv: list[str], timeout: float) -> tuple[int, str]:
    """Call subprocess.run on *argv* with *timeout* seconds.

    Never passes shell mode — *argv* is a fixed list defined in
    this module, so subprocess cannot be tricked into parsing shell
    syntax.

    On ``FileNotFoundError``, ``subprocess.TimeoutExpired`` or
    ``OSError`` returns ``(127, "")``.
    """
    # nosec B603: argv is a fixed list defined in this module (not user input).
    try:
        proc = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            stdin=subprocess.DEVNULL,  # a lookup must never wait on a terminal
            text=True,
            timeout=timeout,
            check=False,
        )
        return (proc.returncode, proc.stdout)
    except (subprocess.TimeoutExpired, OSError):  # FileNotFoundError is an OSError
        return (127, "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scrub(value: str) -> str:
    """Replace every non-printable-ASCII character with ``?``, truncate to 40."""
    cleaned = "".join(c if c.isascii() and c.isprintable() else "?" for c in value)
    return cleaned[:40]


_UNIT_SUFFIX = ".service"


def _parse_services(output: str) -> list[str]:
    """Unit names from ``systemctl list-units --plain`` output."""
    tokens = (line.split()[0] for line in output.splitlines() if line.split())
    return [token for token in tokens if token.endswith(_UNIT_SUFFIX)]


def _parse_containers(output: str) -> list[str]:
    """Container names from ``docker ps --format {{.Names}}`` output."""
    return [line.strip() for line in output.splitlines() if line.strip()]


@dataclass(frozen=True)
class _Kind:
    """How one groundable argument is looked up, matched and reported."""

    lookup_argv: tuple[str, ...]
    parse: Callable[[str], list[str]]
    wanted: Callable[[str], tuple[str, ...]]
    noun: str
    plural: str


def _service_names(value: str) -> tuple[str, ...]:
    """A service matches as given or with the unit suffix added ('vllm' -> 'vllm.service')."""
    return (value, value + _UNIT_SUFFIX)


_KINDS: dict[str, _Kind] = {
    "service": _Kind(
        tuple(SERVICE_LOOKUP_ARGV), _parse_services, _service_names, "service", "services"
    ),
    "container": _Kind(
        tuple(CONTAINER_LOOKUP_ARGV),
        _parse_containers,
        lambda value: (value,),
        "container",
        "containers",
    ),
}


def _match(value: str, candidates: list[str], kind: _Kind) -> str | GroundDecline:
    """The machine's own spelling of *value*, or a decline.

    Exact, case-insensitive comparison only -- no substring, prefix or fuzzy
    match. EVERY candidate is examined before a match is returned, so two
    names that differ only by case are reported as ambiguous instead of the
    first one listed winning (which, for ``container_restart``, would let
    ``docker ps`` ordering pick the target).
    """
    wanted = {name.casefold() for name in kind.wanted(value)}
    matches = sorted({candidate for candidate in candidates if candidate.casefold() in wanted})
    if len(matches) == 1:
        return matches[0]
    if matches:
        return GroundDecline(
            code="ambiguous", message=f"{_scrub(value)} matches: {', '.join(matches)}"
        )
    return GroundDecline(
        code=f"no_such_{kind.noun}", message=f"no such {kind.noun}: {_scrub(value)}"
    )


def _lookup(kind: _Kind, runner: Runner) -> list[str] | GroundDecline:
    failed = GroundDecline(code="lookup_failed", message=f"could not list {kind.plural}")
    try:
        exit_code, output = runner(list(kind.lookup_argv), LOOKUP_TIMEOUT)
    except Exception:  # noqa: BLE001
        # The runner is injectable; whatever it raises is a failed lookup.
        return failed
    if exit_code != 0 or not isinstance(output, str):
        return failed
    return kind.parse(output)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def ground(
    operation: Operation,
    args: dict[str, str],
    runner: Runner = default_runner,
) -> Grounded | GroundDecline:
    """Ground the ``service`` / ``container`` arguments *operation* declares.

    Every other argument is copied unchanged, and an operation that declares
    neither never calls the runner. The untrusted value is only ever COMPARED
    with the lookup output: it is never passed to the runner and never put in
    a shell string. Never raises.
    """
    declared = {spec.name for spec in operation.args}
    result: dict[str, str] = dict(args)
    for key, kind in _KINDS.items():
        if key not in declared or key not in args:
            continue
        value = args[key]
        if not isinstance(value, str):
            return GroundDecline(
                code="lookup_failed", message=f"{key} must be a string to be grounded"
            )
        candidates = _lookup(kind, runner)
        if isinstance(candidates, GroundDecline):
            return candidates
        matched = _match(value, candidates, kind)
        if isinstance(matched, GroundDecline):
            return matched
        result[key] = matched
    return Grounded(args=result)
