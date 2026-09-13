"""``nvsh approve`` — check/manage the glob-pattern approval store.

Backs the "propose, don't run" contract: an agent-proposed command is never
executed without operator approval. This noun group is the shared decision
point other components (the bash-hook extension, the daemon client) call
into via the CLI so the approval logic lives in exactly one place —
:mod:`nvsh.approvals`.

* ``nvsh approve check <cmd>``       — {decision, pattern}
* ``nvsh approve add <pattern>``     — persist to ``approved.toml`` (or, with
                                        ``--session``, to the login session's
                                        runtime-dir store, which a later
                                        process still reads and logout wipes);
                                        with ``--scope`` the argument is a
                                        command line and nvsh derives one
                                        pattern per pipeline stage (d24)
* ``nvsh approve list``              — {user: [...], session: [...]}
* ``nvsh approve remove <pattern>``  — drop from both lists
* ``nvsh approve audit``             — append one decision to the audit log
                                        (used by the pi approval extension,
                                        ``nvsh/agent/pi_ext/approval.ts``, a
                                        pure forwarder that never decides
                                        policy itself)
"""

from __future__ import annotations

import argparse

from nvsh.agent.audit import AuditLog
from nvsh.approvals import (
    SCOPES,
    ApprovalError,
    Approvals,
    base_scope,
    command_refusal_reason,
    patterns_for,
)
from nvsh.cli._errors import EXIT_USER_ERROR, CliError
from nvsh.cli._output import emit_result


def cmd_approve_check(args: argparse.Namespace) -> int:
    approvals = Approvals.load()
    scope, pattern = approvals.matches(args.cmd)
    # d24: a command line is decided per stage, so "ask" has a *place* --
    # name the stage that is holding the line back rather than the whole
    # pipeline.
    stage = approvals.unapproved_stage(args.cmd) if scope == "ask" else None
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result({"decision": scope, "pattern": pattern, "stage": stage}, json_mode=True)
    else:
        text = f"decision: {scope}"
        if pattern:
            text += f"\npattern: {pattern}"
        if stage:
            text += f"\nstage: {stage}"
        emit_result(text, json_mode=False)
    return 0


def cmd_approve_add(args: argparse.Namespace) -> int:
    """Approve a pattern, or -- with ``--scope`` -- a whole command line.

    Two shapes, one writer (deviation d24). Without ``--scope`` the
    positional is a glob and is stored as given, exactly as before. With
    ``--scope`` it is a *command line*: nvsh splits it into stages and
    derives one pattern per stage through :func:`nvsh.approvals.patterns_for`
    -- the same helper the panel's scope line and its store write use. That
    is what lets the pi approval extension offer the four scope choices
    without re-implementing (or drifting from) the widening rules.
    """
    approvals = Approvals.load()
    requested = getattr(args, "scope", None)
    if requested:
        reason = command_refusal_reason(args.pattern)
        if reason is not None:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f"refusing to approve {args.pattern!r}: {reason}",
                remediation="run it once instead; no scope pre-approves this command",
            )
        scope = base_scope(requested)
        patterns = patterns_for(args.pattern, requested)
        if not patterns:
            raise CliError(
                code=EXIT_USER_ERROR,
                message="nothing to approve: the command line is empty",
                remediation="pass the command line you want approved",
            )
    else:
        scope = "session" if getattr(args, "session", False) else "user"
        patterns = [args.pattern]
    for pattern in patterns:
        try:
            approvals.add(pattern, scope=scope)
        except ApprovalError as exc:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=str(exc),
                remediation="choose a narrower pattern; 'sudo *', 'rm *' and bare '*' are refused",
            ) from exc
    if scope == "user":
        approvals.save()
    json_mode = bool(getattr(args, "json", False))
    result = {"added": args.pattern, "scope": scope, "patterns": patterns}
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(f"approved ({scope}): {' '.join(patterns)}", json_mode=False)
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
    found = approvals.remove(args.pattern)
    approvals.save()
    json_mode = bool(getattr(args, "json", False))
    # `removed` stays the long-standing key; `found` says whether the pattern
    # was actually there, so a typo no longer reads as a successful removal.
    result = {"removed": args.pattern, "found": found}
    if json_mode:
        emit_result(result, json_mode=True)
    elif found:
        emit_result(f"removed: {args.pattern}", json_mode=False)
    else:
        emit_result(f"no such approval: {args.pattern}", json_mode=False)
    return 0


def cmd_approve_audit(args: argparse.Namespace) -> int:
    """Append one ``{tool, command}`` decision to the audit log.

    Called by ``nvsh/agent/pi_ext/approval.ts`` after every branch (allow,
    once, session, user, deny/block) so every tool call the extension
    forwards a decision on is recorded, even one it never asked the
    operator about (an existing user/session match).
    """
    entry = AuditLog().record(
        "tool_call",
        proposal={"tool": args.tool, "command": args.command},
        decision=args.decision,
    )
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result({"recorded": True, "ts": entry["ts"]}, json_mode=True)
    else:
        emit_result(f"recorded: {args.tool} {args.command!r} -> {args.decision}", json_mode=False)
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
    add.add_argument(
        "--session",
        action="store_true",
        help="Approve for this login session only (gone at logout).",
    )
    add.add_argument(
        "--scope",
        choices=SCOPES,
        help=(
            "Treat the argument as a command line and derive one pattern per "
            "stage for this scope ('-specific' keeps the first argument)."
        ),
    )
    add.add_argument("--json", action="store_true", help="Emit structured JSON.")
    add.set_defaults(func=cmd_approve_add)

    lst = noun_sub.add_parser("list", help="List approved patterns.")
    lst.add_argument("--json", action="store_true", help="Emit structured JSON.")
    lst.set_defaults(func=cmd_approve_list)

    rm = noun_sub.add_parser("remove", help="Remove a pattern from both lists.")
    rm.add_argument("pattern", help="The pattern to remove.")
    rm.add_argument("--json", action="store_true", help="Emit structured JSON.")
    rm.set_defaults(func=cmd_approve_remove)

    audit = noun_sub.add_parser("audit", help="Append one tool-call decision to the audit log.")
    audit.add_argument("--tool", required=True, help="The tool name, e.g. 'bash'.")
    audit.add_argument("--command", required=True, help="The full command line.")
    audit.add_argument(
        "--decision",
        required=True,
        choices=(
            "user",
            "user-specific",
            "session",
            "session-specific",
            "ask",
            "once",
            "deny",
            "block",
        ),
        help="The decision reached for this command.",
    )
    audit.add_argument("--json", action="store_true", help="Emit structured JSON.")
    audit.set_defaults(func=cmd_approve_audit)
