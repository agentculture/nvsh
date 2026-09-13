"""Trigger rules: decide whether a failed command should auto-call the agent.

Pure functions only, stdlib-only, no I/O. Nothing here reads $?, PIPESTATUS,
the clock or the filesystem — the bash hook (t8) gathers those and builds a
:class:`TriggerEvent`; this module only classifies it.

Rule source: the "Behavior rules" trigger list in this repo's CLAUDE.md.
130 (Ctrl-C), 141 (SIGPIPE), grep/diff exiting 1, ``false``, ``test``/``[ ]``,
and commands running inside a script that handles its own errors are not
errors. Pipeline status honours whatever ``pipefail``-aware exit code the
caller already computed — this module never re-derives it from PIPESTATUS.
Interactive or long-running programs never auto-trigger mid-run. Automatic
calls are rate-limited. Pattern triggers (a tool that prints an error but
exits 0) are opt-in via ``TriggerEvent.patterns``.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

#: Default rate-limit window, in seconds, between automatic agent calls.
DEFAULT_RATE_WINDOW = 30.0

#: First-word program classes that are interactive or long-running and must
#: never auto-trigger mid-run. Exported so t8's bash hook can reuse the same
#: list for its cheap pre-filter.
INTERACTIVE_PROGRAMS: frozenset[str] = frozenset(
    {
        "vim",
        "htop",
        "jtop",
        "less",
        "ssh",
        "docker",
        "kubectl",
        "tmux",
        "screen",
        "su",
    }
)

#: ``sudo`` alone is not interactive (most `sudo <cmd>` invocations are
#: one-shot), but these specific second words open an interactive session.
_INTERACTIVE_SUDO_SUBCOMMANDS: frozenset[str] = frozenset({"-i", "su"})

#: grep-family tools whose exit 1 means "no match", not an error.
_GREP_FAMILY: frozenset[str] = frozenset({"grep", "egrep", "fgrep", "zgrep", "rgrep"})


@dataclass(frozen=True)
class RateState:
    """Pure rate-limiter state. The caller persists and replays this."""

    last_auto_call: float | None = None


@dataclass(frozen=True)
class TriggerEvent:
    """Everything :func:`decide` needs, supplied by the caller.

    The caller (the bash hook / entrypoint) is responsible for measuring
    ``exit_code`` with ``pipefail`` already honoured as bash reports it, and
    for supplying ``now`` (its own clock read) and the previous
    ``rate_state``. This module never reads the clock or re-derives an exit
    status from ``pipestatus``.
    """

    command: str
    exit_code: int
    pipestatus: tuple[int, ...] = ()
    pipefail: bool = False
    in_script: bool = False
    output: str = ""
    patterns: tuple[str, ...] = ()
    now: float = 0.0
    rate_window: float = DEFAULT_RATE_WINDOW
    rate_state: RateState = field(default_factory=RateState)


@dataclass(frozen=True)
class Decision:
    """The outcome of :func:`decide`.

    ``action`` is one of ``"skip"`` (do nothing) or ``"ask"`` (auto-trigger
    the agent panel). ``rate_state`` is the updated rate-limiter state the
    caller should persist for the next call.
    """

    action: str
    reason: str
    rule_id: str
    rate_state: RateState


def decide(event: TriggerEvent) -> Decision:
    """Classify one command outcome as ``"skip"`` or ``"ask"``."""

    if event.exit_code == 0:
        if event.patterns and _pattern_matches(event.output, event.patterns):
            return _rate_limited_ask(
                event,
                rule_id="pattern_trigger",
                reason="command exited 0 but its output matched an opt-in error pattern",
            )
        return _skip(event, reason="command succeeded (exit 0)", rule_id="exit_zero")

    if event.exit_code == 130:
        return _skip(
            event, reason="exit 130 is SIGINT (Ctrl-C), not an error", rule_id="sigint_130"
        )

    if event.exit_code == 141:
        return _skip(event, reason="exit 141 is SIGPIPE, not an error", rule_id="sigpipe_141")

    if event.in_script:
        return _skip(
            event,
            reason="command ran inside a script that handles its own errors",
            rule_id="in_script",
        )

    program = _program_class(event.command)
    if program is not None:
        return _skip(
            event,
            reason=f"'{program}' is an interactive/pass-through program",
            rule_id="interactive_program",
        )

    last_word = _pipeline_last_word(event.command)
    last_segment = _last_segment(event.command)

    if last_word in _GREP_FAMILY and event.exit_code == 1:
        return _skip(
            event, reason="grep-family exit 1 means no match, not an error", rule_id="grep_no_match"
        )

    if last_word == "diff" and event.exit_code == 1:
        return _skip(
            event, reason="diff exit 1 means inputs differ, not an error", rule_id="diff_differ"
        )

    if last_word == "false":
        return _skip(event, reason="'false' always exits non-zero by design", rule_id="false_cmd")

    if last_word == "test" or last_segment.startswith("["):
        return _skip(
            event,
            reason="test/[ ] exit 1 is a boolean result, not an error",
            rule_id="test_bracket",
        )

    return _rate_limited_ask(
        event, rule_id="real_error", reason=f"command exited {event.exit_code}"
    )


def _skip(event: TriggerEvent, *, reason: str, rule_id: str) -> Decision:
    return Decision(action="skip", reason=reason, rule_id=rule_id, rate_state=event.rate_state)


def _rate_limited_ask(event: TriggerEvent, *, rule_id: str, reason: str) -> Decision:
    allowed, new_state = _check_rate_limit(event.rate_state, event.now, event.rate_window)
    if not allowed:
        return Decision(
            action="skip",
            reason="auto-trigger rate-limited: another call is within the configured window",
            rule_id="rate_limited",
            rate_state=new_state,
        )
    return Decision(action="ask", reason=reason, rule_id=rule_id, rate_state=new_state)


def _check_rate_limit(state: RateState, now: float, window: float) -> tuple[bool, RateState]:
    """Pure rate check: takes ``now``/``window`` from the caller, no clock reads.

    Returns ``(allowed, new_state)``. A refused call leaves the state
    unchanged (the window does not reset), which is what "records it" means
    here: the refusal shows up in the returned Decision's reason/rule_id.
    """

    if state.last_auto_call is not None and (now - state.last_auto_call) < window:
        return False, state
    return True, RateState(last_auto_call=now)


def _program_class(command: str) -> str | None:
    first = _first_word(command)
    if first in INTERACTIVE_PROGRAMS:
        return first
    if first == "sudo":
        tokens = command.split()
        if len(tokens) >= 2 and tokens[1] in _INTERACTIVE_SUDO_SUBCOMMANDS:
            return f"sudo {tokens[1]}"
    return None


def _first_word(command: str) -> str:
    tokens = _tokenize(command)
    return tokens[0] if tokens else ""


def _last_segment(command: str) -> str:
    segments = command.split("|")
    return segments[-1].strip() if segments else command.strip()


def _pipeline_last_word(command: str) -> str:
    tokens = _tokenize(_last_segment(command))
    return tokens[0] if tokens else ""


def _tokenize(text: str) -> list[str]:
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        # Unbalanced quotes etc. - fall back to a naive split rather than
        # raising; this module never lets classification blow up on odd input.
        return text.split()


def _pattern_matches(output: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in output for pattern in patterns if pattern)
