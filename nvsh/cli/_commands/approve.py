"""``nvsh approve`` — check/manage the glob-pattern approval store.

Backs the "propose, don't run" contract: an agent-proposed command is never
executed without operator approval. This noun group is the shared decision
point other components (the bash-hook extension, the daemon client) call
into via the CLI so the approval logic lives in exactly one place —
:mod:`nvsh.approvals`.

* ``nvsh approve check <cmd>``       — {decision, pattern}
* ``nvsh approve add <pattern>``     — persist (or, with ``--session``, hold
                                        in memory only for this process)
* ``nvsh approve list``              — {user: [...], session: [...]}
* ``nvsh approve remove <pattern>``  — drop from both lists
"""

from __future__ import annotations

import argparse

from nvsh.approvals import ApprovalError, Approvals
from nvsh.cli._errors import EXIT_USER_ERROR, CliError
from nvsh.cli._output import emit_result


def cmd_approve_check(args: argparse.Namespace) -> int:
    approvals = Approvals.load()
    scope, pattern = approvals.matches(args.cmd)
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result({"decision": scope, "pattern": pattern}, json_mode=True)
    else:
        text = f"decision: {scope}" + (f"\npattern: {pattern}" if pattern else "")
        emit_result(text, json_mode=False)
    return 0


def cmd_approve_add(args: argparse.Namespace) -> int:
    approvals = Approvals.load()
    scope = "session" if getattr(args, "session", False) else "user"
    try:
        approvals.add(args.pattern, scope=scope)
    except ApprovalError as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=str(exc),
            remediation="choose a narrower pattern; 'sudo *', 'rm *' and bare '*' are refused",
        ) from exc
    if scope == "user":
        approvals.save()
    json_mode = bool(getattr(args, "json", False))
    result = {"added": args.pattern, "scope": scope}
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(f"approved ({scope}): {args.pattern}", json_mode=False)
    return 0


def cmd_approve_list(args: argparse.Namespace) -> int:
    approvals = Approvals.load()
    json_mode = bool(getattr(args, "json", False))
    payload = {"user": list(approvals.user_patterns), "session": list(approvals.session_patterns)}
    if json_mode:
        emit_result(payload, json_mode=True)
    else:
        lines = ["user:"]
        lines.extend(f"  {p}" for p in payload["user"])
        lines.append("session:")
        lines.extend(f"  {p}" for p in payload["session"])
        emit_result("\n".join(lines), json_mode=False)
    return 0


def cmd_approve_remove(args: argparse.Namespace) -> int:
    approvals = Approvals.load()
    approvals.remove(args.pattern)
    approvals.save()
    json_mode = bool(getattr(args, "json", False))
    result = {"removed": args.pattern}
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(f"removed: {args.pattern}", json_mode=False)
    return 0


def _no_verb(args: argparse.Namespace) -> int:
    return cmd_approve_list(args)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "approve",
        help="Check or manage the approved-command pattern store (see 'nvsh explain approve').",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=_no_verb, json=False)
    noun_sub = p.add_subparsers(dest="approve_command", parser_class=type(p))

    check = noun_sub.add_parser("check", help="Decide whether a command is already approved.")
    check.add_argument("cmd", help="The full command line to check.")
    check.add_argument("--json", action="store_true", help="Emit structured JSON.")
    check.set_defaults(func=cmd_approve_check)

    add = noun_sub.add_parser("add", help="Approve a glob pattern.")
    add.add_argument("pattern", help="fnmatch glob matched against the full command line.")
    add.add_argument("--session", action="store_true", help="Hold in memory only for this process.")
    add.add_argument("--json", action="store_true", help="Emit structured JSON.")
    add.set_defaults(func=cmd_approve_add)

    lst = noun_sub.add_parser("list", help="List approved patterns.")
    lst.add_argument("--json", action="store_true", help="Emit structured JSON.")
    lst.set_defaults(func=cmd_approve_list)

    rm = noun_sub.add_parser("remove", help="Remove a pattern from both lists.")
    rm.add_argument("pattern", help="The pattern to remove.")
    rm.add_argument("--json", action="store_true", help="Emit structured JSON.")
    rm.set_defaults(func=cmd_approve_remove)
