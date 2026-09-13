"""``nvsh agent`` — list/choose/install NvshAgent harness backends.

Backs the harness chooser (see :mod:`nvsh.agent.registry`):

* ``nvsh agent list``       — {adapters: [{name, installed, binary, description, configured}]}
* ``nvsh agent use <name>`` — validates *name*, writes ``[agent] provider`` via :mod:`nvsh.config`
  (and, for ``openai-compat``, says where to put the gateway key when no
  bearer resolves — deviation d10)
* ``nvsh agent install pi`` — prints the npm install command; runs it only with ``--yes``
                               or an interactive 'y' (never on its own)
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess  # nosec B404 - fixed argv below, no shell=True
import sys

from nvsh import config as nvsh_config
from nvsh.agent import registry
from nvsh.cli._errors import EXIT_USER_ERROR, CliError
from nvsh.cli._output import emit_result

#: Help text every ``--json`` flag in this verb group shares.
_JSON_HELP = "Emit structured JSON."

#: Where a gateway key goes when nothing exports one. The placeholder
#: spelling, never a resolved path (d10).
KEY_HINT = (
    f"no bearer resolved: put the gateway's key in "
    f"{nvsh_config.DEFAULT_KEY_FILE_DISPLAY} (mode 0600), or set api_key_file / "
    "api_key_env under [agents.openai-compat] in config.toml"
)


def cmd_agent_list(args: argparse.Namespace) -> int:
    cfg = nvsh_config.load()
    rows = registry.available_adapters()
    for row in rows:
        row["configured"] = row["name"] == cfg.agent_provider
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result({"adapters": rows}, json_mode=True)
    else:
        lines = []
        for row in rows:
            status = "installed" if row["installed"] else "not installed"
            marker = " (configured)" if row["configured"] else ""
            lines.append(f"{row['name']}: {status}{marker} — {row['description']}")
        emit_result("\n".join(lines), json_mode=False)
    return 0


def cmd_agent_use(args: argparse.Namespace) -> int:
    name = args.name
    if name not in registry.ADAPTERS:
        valid = ", ".join(sorted(registry.ADAPTERS))
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"unknown agent '{name}'",
            remediation=f"choose one of: {valid}",
        )
    cfg = nvsh_config.set_provider(name)
    json_mode = bool(getattr(args, "json", False))
    result: dict = {"provider": cfg.agent_provider}
    note = None
    if name == "openai-compat":
        outcome = nvsh_config.resolve_bearer(cfg.agents.get(name, {}))
        result["bearer_source"] = outcome.source
        note = outcome.diagnostic or (KEY_HINT if outcome.source is None else None)
        result["note"] = note
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        text = f"provider set to: {cfg.agent_provider}"
        if note:
            text += f"\n{note}"
        emit_result(text, json_mode=False)
    return 0


def cmd_agent_install(args: argparse.Namespace) -> int:
    if args.name != "pi":
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"unknown install target '{args.name}'",
            remediation="the only install target today is 'pi'",
        )

    command = registry.PI_INSTALL_CMD
    has_npm = shutil.which("npm") is not None
    ran = False

    if not has_npm:
        pass  # print the command but don't offer to run it -- there's nothing to run it with
    elif getattr(args, "yes", False):
        subprocess.run(shlex.split(command), check=False)  # nosec B603 - fixed argv, no shell
        ran = True
    elif sys.stdin.isatty():
        answer = input(f"Run '{command}'? [y/N] ")
        if answer.strip().lower() == "y":
            subprocess.run(shlex.split(command), check=False)  # nosec B603
            ran = True

    result = {"command": command, "ran": ran, "npm_available": has_npm}
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        text = f"command: {command}\nran: {ran}"
        if not has_npm:
            text += "\nnote: npm not found on PATH; install node first"
        emit_result(text, json_mode=False)
    return 0


def _no_verb(args: argparse.Namespace) -> int:
    return cmd_agent_list(args)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "agent",
        help="List, choose, or install NvshAgent harness backends (see 'nvsh explain agent').",
    )
    p.add_argument("--json", action="store_true", help=_JSON_HELP)
    p.set_defaults(func=_no_verb, json=False)
    noun_sub = p.add_subparsers(dest="agent_command", parser_class=type(p))

    lst = noun_sub.add_parser("list", help="List known harness backends and their PATH status.")
    lst.add_argument("--json", action="store_true", help=_JSON_HELP)
    lst.set_defaults(func=cmd_agent_list)

    use = noun_sub.add_parser("use", help="Set the configured harness backend.")
    use.add_argument("name", help="One of: pi, qwen, claude, codex, openai-compat.")
    use.add_argument("--json", action="store_true", help=_JSON_HELP)
    use.set_defaults(func=cmd_agent_use)

    install = noun_sub.add_parser("install", help="Print (and optionally run) an install command.")
    install.add_argument("name", help="Install target (only 'pi' today).")
    install.add_argument(
        "--yes", action="store_true", help="Run the install command without prompting."
    )
    install.add_argument("--json", action="store_true", help=_JSON_HELP)
    install.set_defaults(func=cmd_agent_install)
