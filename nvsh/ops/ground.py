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
            text=True,
            timeout=timeout,
            check=False,
        )
        return (proc.returncode, proc.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return (127, "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scrub(value: str) -> str:
    """Replace every non-printable-ASCII character with ``?``, truncate to 40."""
    cleaned = "".join(c if c.isascii() and c.isprintable() else "?" for c in value)
    return cleaned[:40]


def _parse_services(output: str) -> list[str]:
    """Extract service names from systemctl list-units output."""
    candidates: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        token = stripped.split()[0]
        if token.endswith(".service"):
            candidates.append(token)
    return candidates


def _parse_containers(output: str) -> list[str]:
    """Extract container names from docker ps --format output."""
    candidates: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped:
            candidates.append(stripped)
    return candidates


def _match_service(value: str, candidates: list[str]) -> Grounded | GroundDecline:
    """Try to ground a service value against *candidates*.

    Collects ALL matching candidates (both direct and augmented forms),
    then checks for ambiguity before deciding.
    """
    value_folded = value.casefold()
    all_matches: list[str] = []

    # 1. Exact match (case-insensitive) of value against a candidate
    for candidate in candidates:
        if candidate.casefold() == value_folded:
            all_matches.append(candidate)

    # 2. For services: value + ".service" against a candidate
    if not value.endswith(".service"):
        augmented_folded = (value + ".service").casefold()
        for candidate in candidates:
            if candidate.casefold() == augmented_folded:
                all_matches.append(candidate)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_matches: list[str] = []
    for m in all_matches:
        if m not in seen:
            seen.add(m)
            unique_matches.append(m)

    # 3. Check for ambiguity
    if len(unique_matches) > 1:
        matches_sorted = sorted(unique_matches)
        return GroundDecline(
            code="ambiguous",
            message=f"{value} matches: {', '.join(matches_sorted)}",
        )

    # 4. Single match
    if len(unique_matches) == 1:
        return Grounded(args={"service": unique_matches[0]})

    # 5. No match
    shown = _scrub(value)
    return GroundDecline(
        code="no_such_service",
        message=f"no such service: {shown}",
    )


def _match_container(value: str, candidates: list[str]) -> Grounded | GroundDecline:
    """Try to ground a container value against *candidates*."""
    value_folded = value.casefold()

    # Exact match (case-insensitive)
    for candidate in candidates:
        if candidate.casefold() == value_folded:
            return Grounded(args={"container": candidate})

    # Check for ambiguity
    matches: list[str] = []
    for candidate in candidates:
        if candidate.casefold() == value_folded:
            matches.append(candidate)

    if len(matches) > 1:
        matches_sorted = sorted(matches)
        return GroundDecline(
            code="ambiguous",
            message=f"{value} matches: {', '.join(matches_sorted)}",
        )

    # No match
    shown = _scrub(value)
    return GroundDecline(
        code="no_such_container",
        message=f"no such container: {shown}",
    )


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def ground(
    operation: Operation,
    args: dict[str, str],
    runner: Runner = default_runner,
) -> Grounded | GroundDecline:
    """Ground the *service* / *container* arguments on *operation*.

    Rules
    -----
    1. Only arguments named ``"service"`` and ``"container"`` are
       grounded.  Every other argument is copied unchanged.
    2. A ``"service"`` argument triggers a systemctl lookup;
       a ``"container"`` argument triggers a docker lookup.
    3. Matching is case-insensitive, exact only (no substring / fuzzy).
    4. ``ground()`` never raises — the runner's exception is caught.
    5. ``ground()`` never builds a shell string and never passes the
       untrusted value to the runner.
    """
    # Build the grounded result dict
    result: dict[str, str] = dict(args)
    grounded_keys: list[str] = []

    for key in ("service", "container"):
        if key not in args:
            continue

        value = args[key]

        # Non-string value — can't ground it; decline.
        if not isinstance(value, str):
            return GroundDecline(
                code="lookup_failed",
                message=f"grounding {key!r} requires a string value, not {type(value).__name__}",
            )
        grounded_keys.append(key)

        # Select the right lookup argv
        if key == "service":
            lookup_argv = SERVICE_LOOKUP_ARGV
        else:
            lookup_argv = CONTAINER_LOOKUP_ARGV

        # Run the lookup
        try:
            exit_code, output = runner(lookup_argv, LOOKUP_TIMEOUT)
        except Exception:  # noqa: BLE001
            # Runner itself failed → decline
            return GroundDecline(
                code="lookup_failed",
                message=f"lookup failed for {key}",
            )

        # Non-zero exit → failure
        if exit_code != 0:
            if key == "service":
                return GroundDecline(
                    code="lookup_failed",
                    message="could not list services",
                )
            else:
                return GroundDecline(
                    code="lookup_failed",
                    message="could not list containers",
                )

        # Parse candidates
        if key == "service":
            candidates = _parse_services(output)
        else:
            candidates = _parse_containers(output)

        # Match
        if key == "service":
            match_result = _match_service(value, candidates)
        else:
            match_result = _match_container(value, candidates)

        if isinstance(match_result, GroundDecline):
            return match_result
        elif isinstance(match_result, Grounded):
            result.update(match_result.args)

    return Grounded(args=result)
