"""``nvsh learn`` — the learnability affordance.

Prints a structured self-teaching prompt. Must satisfy the agent-first rubric:
>=200 chars and mention purpose, command map, exit codes, --json, and explain.
"""

from __future__ import annotations

import argparse

from nvsh import __version__
from nvsh.cli._output import emit_result

_TEXT = """\
nvsh — an agent-first shell for NVIDIA Jetson, DGX Spark and RTX Spark.

Purpose
-------
Runs your commands like a normal shell; when a command fails, it hands the
error and device context to an agent (shell -> agent) to diagnose and propose
a fix, which the human confirms before anything runs. Early scaffold: the
agent-first verbs below exist today; the shell itself is planned (issues #1, #2).

Commands
--------
  nvsh whoami             Identity from culture.yaml.
  nvsh learn              This self-teaching prompt.
  nvsh explain <path>...  Markdown docs for any noun/verb path.
  nvsh overview           Descriptive snapshot of the agent.
  nvsh doctor             Check the agent-identity invariants.
  nvsh cli overview       Describe the CLI surface itself.
  nvsh approve check ...  Check whether a command is already approved.

Machine-readable output
-----------------------
Every command supports --json. Errors in JSON mode emit
{"code", "message", "remediation"} to stderr. Stdout and stderr never mix.

Exit-code policy
----------------
  0 success
  1 user-input error (bad flag, bad path, missing arg)
  2 environment / setup error
  3+ reserved

More detail
-----------
  nvsh explain nvsh
"""


def _as_json_payload() -> dict[str, object]:
    return {
        "tool": "nvsh",
        "version": __version__,
        "purpose": (
            "Agent-first shell for NVIDIA Jetson, DGX Spark and RTX Spark (planned): will run "
            "commands normally and hand failures to an agent to diagnose and propose a fix. "
            "Currently an agent-first CLI scaffold; only the commands listed here exist."
        ),
        "commands": [
            {"path": ["whoami"], "summary": "Identity probe from culture.yaml."},
            {"path": ["learn"], "summary": "Self-teaching prompt."},
            {"path": ["explain"], "summary": "Markdown docs by path."},
            {"path": ["overview"], "summary": "Descriptive snapshot of the agent."},
            {"path": ["doctor"], "summary": "Check the agent-identity invariants."},
            {"path": ["cli", "overview"], "summary": "Describe the CLI surface."},
            {"path": ["approve", "check"], "summary": "Check whether a command is approved."},
        ],
        "exit_codes": {
            "0": "success",
            "1": "user-input error",
            "2": "environment/setup error",
        },
        "json_support": True,
        "explain_pointer": "nvsh explain <path>",
    }


def cmd_learn(args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        emit_result(_as_json_payload(), json_mode=True)
    else:
        emit_result(_TEXT, json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "learn",
        help="Print a structured self-teaching prompt for agent consumers.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_learn)
