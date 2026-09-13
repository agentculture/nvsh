"""``nvsh slash`` / ``nvsh complete`` — the CLI side of :mod:`nvsh.slash`.

Thin verbs: all the routing, registry and platform-filtering logic lives in
:mod:`nvsh.slash`; these handlers only translate argparse's ``Namespace``
into that module's calls and format the result per the CLI output contract
(``register(sub)``, ``CliError``, ``emit_result``, ``--json`` on every verb).

``nvsh/shell/readline.bash`` is the reason both verbs exist: the bash layer
never contains the slash-command list, it only calls ``nvsh complete --json``
for the palette and per-command arguments, and ``nvsh slash <line>`` to
dispatch (see ``docs/shell-integration.md``).
"""

from __future__ import annotations

import argparse
import os

from nvsh import slash as slash_mod
from nvsh.cli._output import emit_result


def _platform_kind(args: argparse.Namespace) -> str:
    override = getattr(args, "platform", None)
    if override:
        return str(override)
    from nvsh.platform import detect

    return detect().kind


def cmd_slash(args: argparse.Namespace) -> int:
    kind = _platform_kind(args)
    draft = os.environ.get("NVSH_DRAFT") or None
    exit_code = slash_mod.dispatch(args.line, platform_kind=kind, draft=draft)
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        text = (args.line or "").strip()
        if text.startswith("/"):
            text = text[1:]
        name = text.split(" ", 1)[0] if text else ""
        emit_result({"command": name, "exit_code": exit_code}, json_mode=True)
    return exit_code


def cmd_complete(args: argparse.Namespace) -> int:
    kind = _platform_kind(args)
    words = list(getattr(args, "words", None) or [])
    items = slash_mod.complete(words, kind)
    payload = {"items": [item.to_dict() for item in items]}
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(payload, json_mode=True)
    else:
        emit_result("\n".join(item.value for item in items), json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("slash", help="Dispatch one '/verb ...' line (see 'nvsh explain slash').")
    p.add_argument("line", help="The full slash-command line, '/' included.")
    p.add_argument("--json", action="store_true", help="Emit {command, exit_code} JSON.")
    p.add_argument(
        "--platform",
        default=None,
        help="Override the detected platform kind (dgx-spark/jetson/rtx/generic; for tests).",
    )
    p.set_defaults(func=cmd_slash, json=False)

    c = sub.add_parser("complete", help="Tab-completion candidates (see 'nvsh explain complete').")
    c.add_argument("--json", action="store_true", help="Emit {items: [{value, description}]}.")
    c.add_argument(
        "--platform",
        default=None,
        help="Override the detected platform kind (dgx-spark/jetson/rtx/generic; for tests).",
    )
    c.add_argument(
        "words",
        nargs="*",
        help="Words after '--': the command being completed, then the partial word.",
    )
    c.set_defaults(func=cmd_complete, json=False)
