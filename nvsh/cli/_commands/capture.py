"""``nvsh capture`` — print the captured-output slice for the current shell.

Backs ``--show-context`` for the session log described in
``docs/architecture.md``'s output-capture section: the bash hook runs each
interactive session under ``script(1)`` (or, inside tmux, ``tmux pipe-pane``)
so a failure's real output is recoverable without any re-run. This verb
surfaces :func:`nvsh.capture.last_slice` for a human or an agent to inspect
directly, never printing the underlying log path.
"""

from __future__ import annotations

import argparse
import os

from nvsh.capture import capture_source, last_slice, session_log_path
from nvsh.cli._output import emit_result


def cmd_capture_show(args: argparse.Namespace) -> int:
    env = dict(os.environ)
    pid = getattr(args, "pid", None)
    if pid is None:
        pid = env.get("NVSH_SHELL_PID")
    if pid is None:
        pid = os.getppid()
    pid = int(pid)

    source = capture_source(env)
    log = session_log_path(env, pid)
    result = last_slice(log, source=source)

    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        payload = {
            "status": result.status,
            "source": result.source,
            "bytes_total": result.bytes_total,
            "redaction_rules": result.redaction_rules,
            "text": result.text,
        }
        emit_result(payload, json_mode=True)
    else:
        header = f"status: {result.status}\nsource: {result.source}"
        body = result.text if result.text else "(no captured output)"
        emit_result(f"{header}\n\n{body}", json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "capture",
        help="Show the last captured command output (see 'nvsh explain capture').",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Print the last captured-output slice for the current shell (default action).",
    )
    p.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Shell PID to read the session log for (default: $NVSH_SHELL_PID or the parent PID).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_capture_show, json=False)
