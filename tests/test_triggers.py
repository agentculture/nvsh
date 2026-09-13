"""Table-driven tests for nvsh.triggers.decide().

One row per rule in CLAUDE.md's "Behavior rules" trigger list, plus the
pass-through programs called out in the plan (ssh, docker exec, kubectl exec,
tmux, screen, su, sudo -i, sudo su). Every non-error row must yield
``action == "skip"`` with a non-empty reason.
"""

from __future__ import annotations

import pytest

from nvsh.triggers import INTERACTIVE_PROGRAMS, Decision, RateState, TriggerEvent, decide

# ---------------------------------------------------------------------------
# The main table: (description, command, exit_code, pipestatus, pipefail,
# in_script, expected_action, expected_rule_id)
# ---------------------------------------------------------------------------

TABLE = [
    # -- success is never an error --
    ("success exit 0", "ls /tmp", 0, (0,), False, False, "skip", "exit_zero"),
    # -- 130 / 141: signals, not errors --
    ("ctrl-c 130", "sleep 10", 130, (130,), False, False, "skip", "sigint_130"),
    ("sigpipe 141", "yes | head -1", 141, (141, 0), False, False, "skip", "sigpipe_141"),
    # -- grep/diff "no match" exit codes --
    ("grep no match", "grep foo file.txt", 1, (1,), False, False, "skip", "grep_no_match"),
    ("egrep no match", "egrep foo file.txt", 1, (1,), False, False, "skip", "grep_no_match"),
    ("diff differs", "diff a.txt b.txt", 1, (1,), False, False, "skip", "diff_differ"),
    # -- false always "fails" by design --
    ("bare false", "false", 1, (1,), False, False, "skip", "false_cmd"),
    # -- test / [ ] booleans --
    ("test builtin", "test -f /nope", 1, (1,), False, False, "skip", "test_bracket"),
    ("bracket test", "[ -f /nope ]", 1, (1,), False, False, "skip", "test_bracket"),
    # -- commands that handle their own errors inside a script --
    (
        "in-script real error",
        "python bad_script.py",
        1,
        (1,),
        False,
        True,
        "skip",
        "in_script",
    ),
    # -- interactive / long-running programs never auto-trigger mid-run --
    ("vim", "vim /etc/hosts", 1, (1,), False, False, "skip", "interactive_program"),
    ("htop", "htop", 1, (1,), False, False, "skip", "interactive_program"),
    ("jtop", "jtop", 1, (1,), False, False, "skip", "interactive_program"),
    ("less", "less /var/log/syslog", 1, (1,), False, False, "skip", "interactive_program"),
    ("ssh", "ssh orin", 255, (255,), False, False, "skip", "interactive_program"),
    (
        "docker exec",
        "docker exec -it foo bash",
        1,
        (1,),
        False,
        False,
        "skip",
        "interactive_program",
    ),
    (
        "kubectl exec",
        "kubectl exec -it pod -- bash",
        1,
        (1,),
        False,
        False,
        "skip",
        "interactive_program",
    ),
    ("tmux", "tmux attach", 1, (1,), False, False, "skip", "interactive_program"),
    ("screen", "screen -r", 1, (1,), False, False, "skip", "interactive_program"),
    ("su", "su - root", 1, (1,), False, False, "skip", "interactive_program"),
    ("sudo -i", "sudo -i", 1, (1,), False, False, "skip", "interactive_program"),
    ("sudo su", "sudo su", 1, (1,), False, False, "skip", "interactive_program"),
    # -- pipeline status: honour pipefail as given, do not re-derive --
    (
        "pipe without pipefail reports last stage success",
        "false | true",
        0,
        (1, 0),
        False,
        False,
        "skip",
        "exit_zero",
    ),
    (
        "pipe with pipefail reports pipeline failure",
        "false | true",
        1,
        (1, 0),
        True,
        False,
        "ask",
        "real_error",
    ),
    # -- a real, unqualified error --
    ("ls of missing path", "ls /nope", 2, (2,), False, False, "ask", "real_error"),
]


@pytest.mark.parametrize(
    "desc,command,exit_code,pipestatus,pipefail,in_script,expected_action,expected_rule_id",
    TABLE,
    ids=[row[0] for row in TABLE],
)
def test_trigger_table(
    desc, command, exit_code, pipestatus, pipefail, in_script, expected_action, expected_rule_id
):
    event = TriggerEvent(
        command=command,
        exit_code=exit_code,
        pipestatus=pipestatus,
        pipefail=pipefail,
        in_script=in_script,
        now=1000.0,
    )

    decision = decide(event)

    assert isinstance(decision, Decision)
    assert decision.action == expected_action
    assert decision.rule_id == expected_rule_id
    assert decision.reason  # every decision explains itself


def test_every_non_error_row_has_a_reason():
    for desc, command, exit_code, pipestatus, pipefail, in_script, action, rule_id in TABLE:
        if action != "skip":
            continue
        event = TriggerEvent(
            command=command,
            exit_code=exit_code,
            pipestatus=pipestatus,
            pipefail=pipefail,
            in_script=in_script,
            now=1.0,
        )
        decision = decide(event)
        assert decision.action == "skip", desc
        assert decision.reason, desc


def test_pure_no_module_level_io_dependencies():
    """decide() must not import time/os/subprocess at call time (stdlib only, pure)."""
    import inspect

    import nvsh.triggers as triggers_module

    source = inspect.getsource(triggers_module)
    for forbidden in ("import time", "import subprocess", "import os\n"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Rate limiting: pure, driven entirely by (now, window, rate_state) supplied
# by the caller.
# ---------------------------------------------------------------------------


def test_rate_limiter_allows_first_auto_call():
    event = TriggerEvent(command="ls /nope", exit_code=2, now=100.0, rate_window=30.0)
    decision = decide(event)
    assert decision.action == "ask"
    assert decision.rule_id == "real_error"
    assert decision.rate_state.last_auto_call == 100.0


def test_rate_limiter_refuses_second_call_inside_window_and_records_it():
    first = decide(TriggerEvent(command="ls /nope", exit_code=2, now=100.0, rate_window=30.0))
    assert first.action == "ask"

    second = decide(
        TriggerEvent(
            command="ls /also-nope",
            exit_code=2,
            now=110.0,
            rate_window=30.0,
            rate_state=first.rate_state,
        )
    )

    assert second.action == "skip"
    assert second.rule_id == "rate_limited"
    assert second.reason
    # the refusal is recorded: state still reflects the last successful call
    assert second.rate_state.last_auto_call == first.rate_state.last_auto_call


def test_rate_limiter_allows_call_after_window_elapses():
    first = decide(TriggerEvent(command="ls /nope", exit_code=2, now=100.0, rate_window=30.0))
    third = decide(
        TriggerEvent(
            command="ls /nope",
            exit_code=2,
            now=131.0,
            rate_window=30.0,
            rate_state=first.rate_state,
        )
    )
    assert third.action == "ask"
    assert third.rate_state.last_auto_call == 131.0


def test_rate_state_default_allows_first_call():
    state = RateState()
    assert state.last_auto_call is None


def test_skip_decisions_do_not_consume_the_rate_limit():
    first_skip = decide(TriggerEvent(command="false", exit_code=1, now=100.0))
    assert first_skip.action == "skip"
    # a subsequent real error at the same instant is still allowed to ask
    second = decide(
        TriggerEvent(
            command="ls /nope",
            exit_code=2,
            now=100.0,
            rate_state=first_skip.rate_state,
        )
    )
    assert second.action == "ask"


# ---------------------------------------------------------------------------
# Pattern triggers: opt-in only, for exit-0 commands that print an error.
# ---------------------------------------------------------------------------


def test_pattern_trigger_is_opt_in_and_ignored_by_default():
    event = TriggerEvent(
        command="some-tool --run",
        exit_code=0,
        output="ERROR: something went wrong",
        now=1.0,
    )
    decision = decide(event)
    assert decision.action == "skip"
    assert decision.rule_id == "exit_zero"


def test_pattern_trigger_fires_when_opted_in_and_matched():
    event = TriggerEvent(
        command="some-tool --run",
        exit_code=0,
        output="ERROR: something went wrong",
        patterns=("ERROR:",),
        now=1.0,
    )
    decision = decide(event)
    assert decision.action == "ask"
    assert decision.rule_id == "pattern_trigger"


def test_pattern_trigger_skips_when_opted_in_but_not_matched():
    event = TriggerEvent(
        command="some-tool --run",
        exit_code=0,
        output="all good",
        patterns=("ERROR:",),
        now=1.0,
    )
    decision = decide(event)
    assert decision.action == "skip"
    assert decision.rule_id == "exit_zero"


# ---------------------------------------------------------------------------
# Program-class frozenset export (t8 reuses this for the hook).
# ---------------------------------------------------------------------------


def test_interactive_programs_is_a_frozenset_of_first_words():
    assert isinstance(INTERACTIVE_PROGRAMS, frozenset)
    for expected in (
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
    ):
        assert expected in INTERACTIVE_PROGRAMS


def test_decide_imports_only_stdlib():
    import ast
    import pathlib

    module_path = pathlib.Path(__file__).parent.parent / "nvsh" / "triggers.py"
    tree = ast.parse(module_path.read_text())
    allowed_stdlib = {"dataclasses", "shlex", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed_stdlib, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                assert node.module.split(".")[0] in allowed_stdlib, node.module
