"""Harness chooser: which NvshAgent backends nvsh knows about and which to use.

A dict-of-factories registry (shape borrowed from
``culture_core/cli/agents.py``'s ``_BACKEND_DAEMON_FACTORIES``): each entry
names a binary to look for on PATH and a zero-arg-from-config factory that
lazily imports the concrete adapter, so importing this module never pulls in
subprocess/urllib machinery it doesn't need.

``choose()`` is the function ``nvsh setup`` (task t21) calls to pick a
backend non-interactively: the configured provider if it is installed, else
``openai-compat`` (always available -- it needs no binary), with a reason
string explaining why.

``install_offer()`` only ever *returns* the npm install command; it never
runs it. The caller (the CLI's ``nvsh agent install pi``) decides whether to
actually execute it, and only after an explicit ``--yes`` or interactive
confirmation.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import Config
from .base import NvshAgent

WhichFn = Callable[[str], Optional[str]]
PromptFn = Callable[[str], str]

#: The only install target this module offers today. npm via nvm on the
#: Spark; the Jetsons have no node today (see docs/platforms.md).
PI_INSTALL_CMD = "npm install -g @earendil-works/pi-coding-agent"


@dataclass(frozen=True)
class AdapterSpec:
    """One registered backend: how to detect it and how to build it."""

    name: str
    binary: str | None  # None means "no binary needed" (openai-compat).
    factory: Callable[[Config], NvshAgent]
    description: str
    needs_node: bool = False


def _make_pi(config: Config) -> NvshAgent:
    try:
        from .pi import PiAgent
    except ImportError as exc:  # pragma: no cover - covered once t9 merges PiAgent
        raise RuntimeError("pi adapter not available yet: nvsh.agent.pi is not present") from exc
    settings = config.agents.get("pi", {})
    provider = settings.get("provider")
    model = settings.get("model")
    return PiAgent(
        provider=str(provider) if provider is not None else None,
        model=str(model) if model is not None else None,
    )


def _make_qwen(config: Config) -> NvshAgent:
    from .qwen import QwenAgent

    return QwenAgent(config.agents.get("qwen", {}))


def _make_claude(config: Config) -> NvshAgent:
    from .claude import ClaudeAgent

    return ClaudeAgent(config.agents.get("claude", {}))


def _make_codex(config: Config) -> NvshAgent:
    from .codex import CodexAgent

    return CodexAgent(config.agents.get("codex", {}))


def _make_openai_compat(config: Config) -> NvshAgent:
    from .openai_compat import OpenAICompatAgent

    return OpenAICompatAgent(config.agents.get("openai-compat", {}))


#: Registered in the order 'nvsh agent list' reports them.
ADAPTERS: dict[str, AdapterSpec] = {
    "pi": AdapterSpec(
        name="pi",
        binary="pi",
        factory=_make_pi,
        description="Pi coding agent (pi --mode rpc); nemotron/associate default.",
        needs_node=True,
    ),
    "qwen": AdapterSpec(
        name="qwen",
        binary="qwen",
        factory=_make_qwen,
        description="Qwen Code CLI (qwen -p).",
        needs_node=True,
    ),
    "claude": AdapterSpec(
        name="claude",
        binary="claude",
        factory=_make_claude,
        description="Claude Code CLI (claude -p --output-format stream-json).",
        needs_node=True,
    ),
    "codex": AdapterSpec(
        name="codex",
        binary="codex",
        factory=_make_codex,
        description="Codex CLI (codex exec --json).",
        needs_node=True,
    ),
    "openai-compat": AdapterSpec(
        name="openai-compat",
        binary=None,
        factory=_make_openai_compat,
        description="Stdlib urllib client against an OpenAI-compatible endpoint.",
        needs_node=False,
    ),
}


def installed(name: str, which: WhichFn = shutil.which) -> bool:
    """Is adapter ``name`` usable right now? ``openai-compat`` always is."""
    spec = ADAPTERS[name]
    if spec.binary is None:
        return True
    return which(spec.binary) is not None


def available_adapters(which: WhichFn = shutil.which) -> list[dict]:
    """Rows for 'nvsh agent list': name, installed, binary, description."""
    return [
        {
            "name": spec.name,
            "installed": installed(spec.name, which),
            "binary": spec.binary,
            "description": spec.description,
        }
        for spec in ADAPTERS.values()
    ]


def choose(config: Config, which: WhichFn = shutil.which) -> tuple[str, str]:
    """Pick a backend: the configured provider if installed, else openai-compat.

    Returns ``(name, reason)``. ``reason`` always explains the pick, so
    ``nvsh setup`` (t21) can print it verbatim -- e.g. "pi not on PATH and
    node missing; using openai-compat against http://host:8000/v1".
    """
    configured = config.agent_provider
    if configured in ADAPTERS and installed(configured, which):
        return configured, f"{configured} is configured and on PATH"

    if configured not in ADAPTERS:
        why = f"configured provider '{configured}' is unknown"
    else:
        spec = ADAPTERS[configured]
        node_present = which("node") is not None
        if spec.needs_node and not node_present:
            why = f"{configured} not on PATH and node missing"
        else:
            why = f"{configured} not on PATH"

    base_url = config.agents.get("openai-compat", {}).get("base_url")
    if base_url:
        reason = f"{why}; using openai-compat against {base_url}"
    else:
        reason = f"{why}; using openai-compat (no base_url configured, defaults apply)"
    return "openai-compat", reason


def install_offer(name: str, which: WhichFn = shutil.which, prompt: PromptFn = input) -> str | None:
    """Return the install command for ``name`` if the user approves it, else ``None``.

    Never executes anything itself -- the caller runs the returned command
    after its own approval step. Only ``pi`` has an install offer today, and
    only when ``npm`` is on PATH.
    """
    if name != "pi":
        return None
    if which("npm") is None:
        return None
    answer = prompt("Install pi via npm? [y/N] ")
    if answer.strip().lower() == "y":
        return PI_INSTALL_CMD
    return None


def no_harness_message() -> str:
    """Failure-panel text offered when no harness is installed at all.

    Takes no ``which``: the two options are the same whatever is on PATH --
    this text is only ever shown once nothing was found.
    """
    return (
        "No harness is installed. Options:\n"
        "  - install pi\n"
        "  - choose another harness (nvsh agent use <name>)"
    )
