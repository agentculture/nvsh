"""``nvsh agent`` — list/choose/install NvshAgent harness backends.

Backs the harness chooser (see :mod:`nvsh.agent.registry`):

* ``nvsh agent list``       — {adapters: [{name, installed, binary, path, hosted,
  capabilities, default, configured, description}]}, sorted so the row whose
  name matches the resolved ``[aliases].default`` (falling back to
  ``[agent] provider`` when no alias table is set — see
  :meth:`nvsh.config.Config.resolve_target`) comes first.
* ``nvsh agent use <name>`` — validates *name*, writes ``[agent] provider`` AND
  ``[aliases].default`` via :mod:`nvsh.config` (:func:`nvsh.config.set_provider`
  does both) (and, for ``openai-compat``, says where to put the gateway key when
  no bearer resolves — deviation d10)
* ``nvsh agent install pi`` — prints the npm install command; runs it only with ``--yes``
                               or an interactive 'y' (never on its own)

``build_adapter_rows`` is the shared row-builder :mod:`nvsh.slash`'s
``/agent`` handler imports, so the CLI and slash surfaces never drift apart.
"""

from __future__ import annotations

import argparse
import dataclasses
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


def resolved_default_backend(cfg: nvsh_config.Config) -> str:
    """The backend name ``'default'`` resolves to right now.

    Goes through :meth:`nvsh.config.Config.resolve_target`, which falls back
    to the legacy ``[agent] provider`` when ``[aliases]`` has no ``default``
    entry (or no ``[aliases]`` table at all) — so a config with only
    ``[agent] provider`` still resolves a default here.
    """
    try:
        backend, _model, _effort, _alias = cfg.resolve_target(nvsh_config.DEFAULT_ALIAS)
    except nvsh_config.ConfigError:
        return cfg.agent_provider
    return backend


def _adapter_capabilities(name: str, cfg: nvsh_config.Config) -> dict | None:
    """Cheaply construct adapter *name* and report its capabilities, or ``None``.

    Every adapter's ``__init__`` is cheap and never spawns a subprocess (that
    happens lazily on first turn), so building one just to read
    ``.capabilities()`` is safe for a listing verb. A factory that validates
    its own settings and raises is caught here rather than failing the whole
    ``agent list`` call -- the row just reports ``capabilities: null``.
    """
    spec = registry.ADAPTERS[name]
    try:
        agent = spec.factory(cfg)
        return dataclasses.asdict(agent.capabilities())
    except Exception:  # noqa: BLE001 - defensive: factories may validate/raise
        return None


def build_adapter_rows(cfg: nvsh_config.Config) -> list[dict]:
    """Enriched 'nvsh agent list' rows, the resolved default backend sorted first.

    Each row from :func:`nvsh.agent.registry.available_adapters` (name,
    installed, binary, description) gains ``path``, ``hosted``,
    ``capabilities`` and two boolean markers: ``default`` (this is what
    ``'default'``/a bare ``--agent`` currently resolves to) and
    ``configured`` (kept for backward compatibility: this is the legacy
    ``[agent] provider``, which coincides with ``default`` unless
    ``[aliases].default`` overrides it).
    """
    default_backend = resolved_default_backend(cfg)
    rows = registry.available_adapters()
    for row in rows:
        spec = registry.ADAPTERS[row["name"]]
        row["path"] = spec.path
        row["hosted"] = spec.hosted
        row["capabilities"] = _adapter_capabilities(row["name"], cfg)
        row["default"] = row["name"] == default_backend
        row["configured"] = row["name"] == cfg.agent_provider
    # list.sort is stable: rows keep registry order except the default's hop
    # to the front, so the ordering stays deterministic and easy to test.
    rows.sort(key=lambda row: 0 if row["default"] else 1)
    return rows


def _agent_list_text(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        status = "installed" if row["installed"] else "not installed"
        tags = []
        if row["default"]:
            tags.append("default")
        if row["hosted"]:
            tags.append("hosted")
        marker = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"{row['name']} [{row['path']}]: {status}{marker} — {row['description']}")
    return "\n".join(lines)


def cmd_agent_list(args: argparse.Namespace) -> int:
    cfg = nvsh_config.load()
    rows = build_adapter_rows(cfg)
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result({"adapters": rows}, json_mode=True)
    else:
        emit_result(_agent_list_text(rows), json_mode=False)
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
