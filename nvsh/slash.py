"""The slash-command registry: the single source of truth for ``/verb`` lines.

``nvsh/shell/readline.bash`` holds no command list (see
``docs/shell-integration.md``); everything it knows about the palette and
per-command arguments comes from two CLI calls this module backs:
``nvsh complete --json`` (:func:`complete`) and ``nvsh slash <line>``
(:func:`dispatch`). Both are also reachable directly for tests and for
``Ctrl+G`` (``nvsh slash "/ask"`` with ``NVSH_DRAFT`` set).

Each :class:`SlashCommand` bundles everything one verb needs: its name and
aliases, a human description, an optional argument schema (for ``--help``
and documentation), a completion provider for its own arguments, the
handler that actually runs it, a ``safety`` classification, and an optional
``platforms`` set that hides the command everywhere else (``/power`` and
``/clocks`` are Jetson-only stubs, proving the filtering works — see
``nvsh.platform.detect().kind``).

Deferred (spec claim c22): ``/remember-good``, ``/diff-good`` and
``/restore-good`` are not registered here.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Callable, Mapping

from .agent.base import RequestKind
from .panel import Panel

#: Safety classifications a :class:`SlashCommand` may carry.
SAFETY_READ_ONLY = "read_only"
SAFETY_AGENT = "agent"
SAFETY_MUTATES_CONFIG = "mutates_config"

_JETSON_ONLY = frozenset({"jetson"})


@dataclass(frozen=True)
class Item:
    """One completion candidate: what ``nvsh complete --json`` emits."""

    value: str
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"value": self.value, "description": self.description}


@dataclass(frozen=True)
class ArgSpec:
    """One documented argument of a slash command (for ``/help`` and docs)."""

    name: str
    description: str = ""


@dataclass(frozen=True)
class SlashInvocation:
    """Everything a handler needs: the parsed line plus the calling shell's state."""

    name: str
    args: list[str]
    rest: str
    line: str
    draft: str | None
    env: Mapping[str, str]
    panel: Panel | None
    platform_kind: str

    def panel_or(self) -> Panel:
        return self.panel if self.panel is not None else Panel(env=self.env)


CompletionFn = Callable[[list[str]], list[Item]]
HandlerFn = Callable[[SlashInvocation], int]


@dataclass(frozen=True)
class SlashCommand:
    """One registered ``/verb``."""

    name: str
    handler: HandlerFn
    aliases: tuple[str, ...] = ()
    description: str = ""
    arg_schema: tuple[ArgSpec, ...] = ()
    completion: CompletionFn | None = None
    safety: str = SAFETY_READ_ONLY
    platforms: frozenset[str] | None = None

    def visible_on(self, platform_kind: str) -> bool:
        """Is this command part of the palette on *platform_kind*?"""
        return self.platforms is None or platform_kind in self.platforms


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


def _handle_ask(inv: SlashInvocation) -> int:
    from .client import ask

    return ask(inv.rest, draft=inv.draft, panel=inv.panel, env=inv.env, kind=RequestKind.SLASH)


def _handle_fix(inv: SlashInvocation) -> int:
    from .client import fix

    return fix(panel=inv.panel, env=inv.env)


def _handle_explain(inv: SlashInvocation) -> int:
    from .client import explain

    return explain(panel=inv.panel, env=inv.env)


def _handle_retry(inv: SlashInvocation) -> int:
    from .client import retry

    return retry(panel=inv.panel, env=inv.env)


def _handle_context(inv: SlashInvocation) -> int:
    from .client import context_show

    json_mode = "--json" in inv.args
    return context_show(json_mode=json_mode, env=inv.env)


def _agent_lines(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        status = "installed" if row["installed"] else "not installed"
        marker = " (configured)" if row.get("configured") else ""
        lines.append(f"{row['name']}: {status}{marker} — {row['description']}")
    return "\n".join(lines)


def _handle_agent(inv: SlashInvocation) -> int:
    from . import config as nvsh_config
    from .agent import registry

    panel = inv.panel_or()
    sub = inv.args[0] if inv.args else "list"

    if sub == "list":
        cfg = nvsh_config.load()
        rows = registry.available_adapters()
        for row in rows:
            row["configured"] = row["name"] == cfg.agent_provider
        panel.line(_agent_lines(rows))
        return 0

    if sub == "use":
        if len(inv.args) < 2:
            panel.line("nvsh: /agent use <name>")
            return 1
        name = inv.args[1]
        if name not in registry.ADAPTERS:
            valid = ", ".join(sorted(registry.ADAPTERS))
            panel.line(f"nvsh: unknown agent '{name}' (choose one of: {valid})")
            return 1
        cfg = nvsh_config.set_provider(name)
        panel.line(f"provider set to: {cfg.agent_provider}")
        return 0

    panel.line("nvsh: /agent [list|use <name>]")
    return 1


def _handle_help(inv: SlashInvocation) -> int:
    panel = inv.panel_or()
    lines = ["nvsh slash commands:"]
    for cmd in visible_commands(inv.platform_kind):
        aliases = f" (aliases: {', '.join('/' + a for a in cmd.aliases)})" if cmd.aliases else ""
        lines.append(f"  /{cmd.name}{aliases} — {cmd.description}")
    panel.line("\n".join(lines))
    return 0


def _handle_undo(inv: SlashInvocation) -> int:
    from . import client_transport
    from .client import _read_json, _shell_pid, _write_private_json, last_failure_path

    panel = inv.panel_or()
    shell_id = _shell_pid(dict(inv.env))
    events = client_transport.control("undo", shell_id=shell_id, env=inv.env)
    if events:
        panel.line("nvsh: undid the last agent turn")
        return 0

    # No daemon (never started, or crashed): the only local state nvsh keeps
    # about a pending proposal lives in last-failure.json. Clear it, but
    # never run anything on the machine -- /undo only ever rewinds nvsh's
    # own view of the conversation (see spec's /undo decision).
    path = last_failure_path(inv.env)
    data = _read_json(path)
    if data and "pending_proposal" in data:
        data.pop("pending_proposal", None)
        _write_private_json(path, data)
    panel.line("nvsh: no daemon running; cleared the pending proposal")
    return 0


def _handle_doctor(inv: SlashInvocation) -> int:
    import argparse

    from .cli._commands.doctor import cmd_doctor

    args = argparse.Namespace(
        json="--json" in inv.args,
        prompt_command=inv.env.get("NVSH_PROMPT_COMMAND") or None,
        bind_p=inv.env.get("NVSH_BIND_P") or None,
        keymap=inv.env.get("NVSH_KEYMAP") or None,
    )
    return cmd_doctor(args)


def _handle_approve(inv: SlashInvocation) -> int:
    from .approvals import ApprovalError, Approvals

    panel = inv.panel_or()
    args = inv.args
    sub = args[0] if args else "list"
    approvals = Approvals.load()

    if sub == "list":
        lines = ["user:"]
        lines.extend(f"  {p}" for p in approvals.user_patterns)
        lines.append("session:")
        lines.extend(f"  {p}" for p in approvals.session_patterns)
        panel.line("\n".join(lines))
        return 0

    if sub == "add":
        if len(args) < 2:
            panel.line("nvsh: /approve add <pattern> [--session]")
            return 1
        pattern = args[1]
        session = "--session" in args[2:]
        try:
            approvals.add(pattern, scope="session" if session else "user")
        except ApprovalError as exc:
            panel.line(f"nvsh: {exc}")
            return 1
        if not session:
            approvals.save()
        panel.line(f"approved ({'session' if session else 'user'}): {pattern}")
        return 0

    if sub == "remove":
        if len(args) < 2:
            panel.line("nvsh: /approve remove <pattern>")
            return 1
        approvals.remove(args[1])
        approvals.save()
        panel.line(f"removed: {args[1]}")
        return 0

    panel.line("nvsh: /approve [list|add <pattern> [--session]|remove <pattern>]")
    return 1


def _stub_handler(name: str) -> HandlerFn:
    def handler(inv: SlashInvocation) -> int:
        inv.panel_or().line(f"nvsh: /{name} not implemented in this milestone")
        return 0

    return handler


# ---------------------------------------------------------------------------
# completion providers
# ---------------------------------------------------------------------------


def _complete_doctor(_args: list[str]) -> list[Item]:
    return [
        Item("--json", "emit structured JSON"),
        Item("--strict", "fail on warnings, not only errors"),
    ]


def _complete_agent(args: list[str]) -> list[Item]:
    from .agent import registry

    items = [
        Item("list", "list known harness backends and their PATH status"),
        Item("use", "set the configured harness backend"),
    ]
    if not args or args[0] != "list":
        items.extend(Item(name, "") for name in sorted(registry.ADAPTERS))
    return items


def _complete_approve(_args: list[str]) -> list[Item]:
    return [
        Item("list", "list approved patterns"),
        Item("add", "approve a glob pattern"),
        Item("remove", "remove a pattern"),
        Item("--session", "hold the pattern in memory only for this process"),
    ]


def _complete_context(_args: list[str]) -> list[Item]:
    return [Item("--show", "print the assembled context"), Item("--json", "emit structured JSON")]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand(
        name="ask",
        handler=_handle_ask,
        description="ask the agent a free-form question with the machine's context",
        arg_schema=(ArgSpec("text", "the question"),),
        safety=SAFETY_AGENT,
    ),
    SlashCommand(
        name="fix",
        handler=_handle_fix,
        description="ask for the smallest fix for the last recorded failure",
        safety=SAFETY_AGENT,
    ),
    SlashCommand(
        name="explain",
        handler=_handle_explain,
        description="ask what the last recorded failure means on this machine",
        safety=SAFETY_AGENT,
    ),
    SlashCommand(
        name="retry",
        handler=_handle_retry,
        description="re-run the last failed command, after confirmation",
        safety=SAFETY_AGENT,
    ),
    SlashCommand(
        name="context",
        handler=_handle_context,
        description="show exactly the context that would be sent to the agent",
        arg_schema=(ArgSpec("--show", "default"), ArgSpec("--json", "structured output")),
        completion=_complete_context,
    ),
    SlashCommand(
        name="agent",
        handler=_handle_agent,
        description="list or choose the NvshAgent harness backend",
        arg_schema=(ArgSpec("list"), ArgSpec("use <name>")),
        completion=_complete_agent,
        safety=SAFETY_MUTATES_CONFIG,
    ),
    SlashCommand(
        name="help",
        handler=_handle_help,
        description="list the slash-command palette",
    ),
    SlashCommand(
        name="undo",
        handler=_handle_undo,
        description="drop the last agent turn and its proposal (never touches the machine)",
    ),
    SlashCommand(
        name="approve",
        handler=_handle_approve,
        description="check or manage the approved-command pattern store",
        arg_schema=(
            ArgSpec("list"),
            ArgSpec("add <pattern> [--session]"),
            ArgSpec("remove <pattern>"),
        ),
        completion=_complete_approve,
        safety=SAFETY_MUTATES_CONFIG,
    ),
    SlashCommand(
        name="doctor",
        handler=_handle_doctor,
        description="run the in-shell checks (hook, bindings, backend, platform)",
        arg_schema=(ArgSpec("--json"), ArgSpec("--strict")),
        completion=_complete_doctor,
    ),
    SlashCommand(
        name="power",
        handler=_stub_handler("power"),
        description="Jetson power-mode controls (stub)",
        platforms=_JETSON_ONLY,
    ),
    SlashCommand(
        name="clocks",
        handler=_stub_handler("clocks"),
        description="Jetson clocks controls (stub)",
        platforms=_JETSON_ONLY,
    ),
)

#: name/alias (without the leading '/') -> command.
REGISTRY: dict[str, SlashCommand] = {}
for _cmd in _COMMANDS:
    REGISTRY[_cmd.name] = _cmd
    for _alias in _cmd.aliases:
        REGISTRY[_alias] = _cmd
del _cmd
del _COMMANDS


def resolve(name: str) -> SlashCommand | None:
    """Look up *name* (no leading ``/``), honouring aliases."""
    return REGISTRY.get(name.lower())


def visible_commands(platform_kind: str) -> list[SlashCommand]:
    """Every distinct command visible on *platform_kind*, in registration order."""
    seen: set[str] = set()
    out: list[SlashCommand] = []
    for cmd in REGISTRY.values():
        if cmd.name in seen or not cmd.visible_on(platform_kind):
            continue
        seen.add(cmd.name)
        out.append(cmd)
    return out


def _detect_platform_kind() -> str:
    from .platform import detect

    return detect().kind


def complete(words: list[str], platform_kind: str | None = None) -> list[Item]:
    """Completion candidates for ``nvsh complete --json [-- <words>...]``.

    ``words`` empty -> the full slash-command palette, ``/``-prefixed
    (bash's initial-word completion never calls with ``--``). Non-empty ->
    ``words[0]`` names the command and the rest are its own words so far;
    unknown or hidden commands get no candidates.
    """
    kind = platform_kind if platform_kind is not None else _detect_platform_kind()
    if not words:
        return [Item(f"/{cmd.name}", cmd.description) for cmd in visible_commands(kind)]

    first = words[0].lstrip("/")
    cmd = resolve(first)
    if cmd is None or not cmd.visible_on(kind) or cmd.completion is None:
        return []
    return cmd.completion(list(words[1:]))


def _panel_for(panel: Panel | None, env: Mapping[str, str]) -> Panel:
    return panel if panel is not None else Panel(env=env)


def dispatch(
    line: str,
    env: Mapping[str, str] | None = None,
    platform_kind: str | None = None,
    *,
    draft: str | None = None,
    panel: Panel | None = None,
) -> int:
    """Route one ``/verb ...`` line (``nvsh slash <line>``).

    Parses with :mod:`shlex` so quoted arguments survive; an unknown or
    platform-hidden command (and anything that fails to parse) reports a
    user error and never guesses.
    """
    resolved = dict(os.environ if env is None else env)
    kind = platform_kind if platform_kind is not None else _detect_platform_kind()
    text = (line or "").strip()
    if text.startswith("/"):
        text = text[1:]
    if not text:
        _panel_for(panel, resolved).line("nvsh: empty slash command (try /help)")
        return 1

    try:
        parts = shlex.split(text)
    except ValueError:
        parts = text.split()
    if not parts:
        _panel_for(panel, resolved).line("nvsh: empty slash command (try /help)")
        return 1

    name, args = parts[0], parts[1:]
    cmd = resolve(name)
    if cmd is None or not cmd.visible_on(kind):
        _panel_for(panel, resolved).line(f"nvsh: unknown slash command '/{name}' (try /help)")
        return 1

    rest = text[len(name) :].strip()
    invocation = SlashInvocation(
        name=cmd.name,
        args=args,
        rest=rest,
        line=line,
        draft=draft,
        env=resolved,
        panel=panel,
        platform_kind=kind,
    )
    return cmd.handler(invocation)
