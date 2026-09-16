"""``nvsh overview`` — read-only descriptive snapshot of the agent.

Describes the agent to an agent reader: identity (from culture.yaml), the verb
surface, and the sibling-pattern artifacts this template carries. The shared
section/render helpers here are reused by the ``cli`` noun's ``overview`` (see
:mod:`nvsh.cli._commands.cli`).

Descriptive verbs never hard-fail on a missing target path — an optional
positional ``target`` is accepted and ignored (overview describes this agent,
not an external target), so ``overview <bogus-path>`` still exits 0.
"""

from __future__ import annotations

import argparse
from typing import Mapping

from nvsh import client_transport
from nvsh.cli._commands.whoami import report
from nvsh.cli._output import emit_result

#: How long ``overview`` waits to connect to a running daemon before giving
#: up and reporting "no daemon running". Short on purpose: `overview` is a
#: read-only, always-fast descriptive verb, and ``status`` never autostarts
#: a daemon, so this only bounds the connect wait against a stale socket
#: that exists but will never answer.
_ACTIVITY_CONNECT_TIMEOUT = 0.5

_ARTIFACTS = [
    "culture.yaml + AGENTS.colleague.md — mesh identity (suffix + backend)",
    ".claude/skills/ — the canonical guildmaster skill kit (cite-don't-import)",
    "docs/skill-sources.md — skill provenance ledger",
    "pyproject.toml + .github/workflows/ — buildable, deployable package baseline",
]

_VERBS = [
    "whoami — identity probe (nick, version, backend, model)",
    "learn — structured self-teaching prompt",
    "explain <path> — markdown docs for a topic",
    "overview — this descriptive snapshot",
    "doctor — check the agent-identity invariants",
]


def agent_sections() -> list[dict[str, object]]:
    """Sections describing the agent (used by the global verb)."""
    ident = report()
    return [
        {
            "title": "Identity",
            "items": [
                f"nick: {ident['nick']}",
                f"version: {ident['version']}",
                f"backend: {ident['backend']}",
                f"model: {ident['model']}",
            ],
        },
        {"title": "Verbs", "items": list(_VERBS)},
        {"title": "Sibling-pattern artifacts", "items": list(_ARTIFACTS)},
    ]


def cli_sections() -> list[dict[str, object]]:
    """Sections describing the CLI surface itself (used by `cli overview`)."""
    return [
        {
            "title": "Verbs",
            "items": list(_VERBS) + ["cli overview — describe the CLI surface (this command)"],
        },
        {
            "title": "Conventions",
            "items": [
                "every command supports --json",
                "results to stdout, errors/diagnostics to stderr (never mixed)",
                "exit codes: 0 success, 1 user error, 2 environment error, 3+ reserved",
            ],
        },
    ]


def agent_activity(
    env: Mapping[str, str] | None = None, timeout: float = _ACTIVITY_CONNECT_TIMEOUT
) -> dict[str, object]:
    """The live daemon's current turn, without starting one.

    Backed by :func:`nvsh.client_transport.status`, which goes through
    ``control()`` -- a control message never autostarts a daemon, so a
    missing socket returns ``{"running": False, ...}`` immediately and this
    never spawns a process. ``timeout`` only bounds the connect wait against
    a stale socket file that exists but has nothing listening on it.
    """
    state = client_transport.status(env=env, timeout=timeout)
    if not state.get("running"):
        return {"daemon": False}

    activity: dict[str, object] = {"daemon": True, "queued": state.get("queued") or []}
    active = state.get("active_turn")
    if isinstance(active, Mapping):
        activity["active_turn"] = {
            "shell": active.get("shell"),
            "target": state.get("target"),
            "elapsed": active.get("elapsed"),
        }
    return activity


def _activity_section(activity: Mapping[str, object]) -> dict[str, object]:
    """Render :func:`agent_activity`'s result as one overview section."""
    if not activity.get("daemon"):
        return {"title": "Agent activity", "items": ["no daemon running"]}

    items: list[str] = []
    active = activity.get("active_turn")
    if isinstance(active, Mapping):
        target = active.get("target")
        backend = target.get("backend") if isinstance(target, Mapping) else target
        elapsed = float(active.get("elapsed") or 0.0)
        items.append(
            f"active turn: shell {active.get('shell')} (target {backend}, "
            f"{elapsed:.1f}s elapsed)"
        )
    else:
        items.append("idle")

    queued = activity.get("queued") or []
    if queued:
        items.append(f"queued: {len(queued)} waiting")

    return {"title": "Agent activity", "items": items}


def render_text(subject: str, sections: list[dict[str, object]]) -> str:
    lines = [f"# {subject}", ""]
    for section in sections:
        lines.append(f"## {section['title']}")
        for item in section["items"]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip()


def emit_overview(
    subject: str,
    sections: list[dict[str, object]],
    *,
    json_mode: bool,
    extra: Mapping[str, object] | None = None,
) -> None:
    if json_mode:
        payload: dict[str, object] = {"subject": subject, "sections": sections}
        if extra:
            payload.update(extra)
        emit_result(payload, json_mode=True)
    else:
        emit_result(render_text(subject, sections), json_mode=False)


def cmd_overview(args: argparse.Namespace) -> int:
    # `target` is accepted for rubric compatibility (descriptive verbs must not
    # hard-fail on a missing path) but overview describes this agent itself.
    #
    # The live "Agent activity" section is rendered here, not folded into
    # `agent_sections()`: that function is reused verbatim elsewhere (and by
    # tests that need its static, daemon-independent output), so the live
    # section is appended only to what this command actually renders.
    activity = agent_activity()
    sections = agent_sections() + [_activity_section(activity)]
    emit_overview(
        "nvsh",
        sections,
        json_mode=bool(getattr(args, "json", False)),
        extra={"agent_activity": activity},
    )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "overview",
        help="Read-only descriptive snapshot of the agent (identity, verbs, artifacts).",
    )
    p.add_argument(
        "target",
        nargs="?",
        help="Ignored — overview always describes this agent itself. Accepted so a "
        "stray path argument never hard-fails.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_overview)
