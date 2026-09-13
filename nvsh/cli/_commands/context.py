"""``nvsh context --show`` — print exactly what would be sent to the agent.

The device-context rule in CLAUDE.md ("report what was detected and how;
don't guess", "support ``--show-context``") only means anything if the
operator can see the real bytes. This verb prints the prompt text
:func:`nvsh.agent.pi.build_prompt` builds from the recorded failure's
request and context — platform block with every value's source, the
redacted output slice, the cwd — with nothing added and nothing summarised.

``--json`` wraps the same prompt alongside its parts, for an agent reading
the verb instead of a human.
"""

from __future__ import annotations

import argparse

from nvsh.cli._errors import EXIT_ENV_ERROR, CliError


def cmd_context_show(args: argparse.Namespace) -> int:
    from nvsh.client import context_show

    try:
        return context_show(json_mode=bool(getattr(args, "json", False)))
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"could not assemble the context: {exc}",
            remediation="check that $XDG_STATE_HOME/nvsh is readable, then retry",
        ) from exc


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "context",
        help="Show exactly the context that would be sent to the agent "
        "(see 'nvsh explain context').",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Print the assembled context for the last failure (default action).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_context_show, json=False)
