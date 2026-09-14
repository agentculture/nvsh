"""Harness chooser: which NvshAgent backends nvsh knows about and which to use.

A dict-of-factories registry (shape borrowed from
``culture_core/cli/agents.py``'s ``_BACKEND_DAEMON_FACTORIES``): each entry
names a binary to look for on PATH and a zero-arg-from-config factory that
lazily imports the concrete adapter, so importing this module never pulls in
subprocess/urllib machinery it doesn't need.

``choose()`` is the function ``nvsh setup`` (task t21) calls to pick a
backend non-interactively: the configured provider if it is installed, else
``openai-compat`` (always available -- it needs no binary), with a reason
string explaining why. It also accepts a ``forced`` target (an alias name, a
literal ``'backend[/model[/effort]]'`` string, or a :class:`~.base.Target`)
for ``nvsh --agent <target> ...`` invocations (task t20): a forced target is
resolved through :meth:`Config.resolve_target` and either wins outright or
fails loudly with a :class:`~nvsh.cli._errors.CliError` naming the missing
binary -- it never silently falls back to ``openai-compat`` the way the
unforced path does.

``install_offer()`` only ever *returns* the npm install command; it never
runs it. The caller (the CLI's ``nvsh agent install pi``) decides whether to
actually execute it, and only after an explicit ``--yes`` or interactive
confirmation.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Callable, Optional

from ..cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from ..config import Config, ConfigError
from .base import NvshAgent, Target

WhichFn = Callable[[str], Optional[str]]
PromptFn = Callable[[str], str]

#: The only install target this module offers today. npm via nvm on the
#: Spark; the Jetsons have no node today (see docs/platforms.md).
PI_INSTALL_CMD = "npm install -g @earendil-works/pi-coding-agent"

#: Valid ``AdapterSpec.path`` values -- the wire/transport protocol the
#: adapter speaks to its backend, independent of ``binary``/``hosted``.
PATH_VALUES = {"rpc", "stream-json", "app-server", "acp", "http"}


@dataclass(frozen=True)
class AdapterSpec:
    """One registered backend: how to detect it, how to build it, what it is."""

    name: str
    binary: str | None  # None means "no binary needed" (openai-compat).
    factory: Callable[[Config], NvshAgent]
    description: str
    #: Transport protocol the adapter speaks: 'rpc' (pi), 'stream-json'
    #: (claude, agy), 'app-server' (codex), 'acp' (qwen, kiro) or 'http'
    #: (openai-compat).
    path: str
    #: Whether the backend talks to a hosted (non-local) model/service.
    hosted: bool
    needs_node: bool = False


def _str_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _list_or_none(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    return [str(item) for item in value]


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
        effort=_str_or_none(settings.get("effort")),
        extra_args=_list_or_none(settings.get("extra_args")),
        approval=str(settings.get("approval", "nvsh")),
    )


def _make_qwen(config: Config) -> NvshAgent:
    """ACP-mode qwen (task t8/c21): ``qwen --acp`` over :class:`~.acp.AcpAgent`.

    ``nvsh.agent.acp`` is imported lazily so this module never drags the
    ACP machinery in for a caller that only wants ``ADAPTERS`` metadata.
    ``acp.build`` owns the mode/approval logic (decision c53): ``plan`` and
    read-only unless ``[agents.qwen] approval = "harness"``.
    """
    from .acp import build

    return build("qwen", config.agents.get("qwen", {}))


def _make_qwen_print(config: Config) -> NvshAgent:
    """The print-mode ``qwen --output-format stream-json`` adapter (t13), kept
    reachable as the no-ACP fallback under the ``qwen-p`` name ('qwen' itself
    means ACP). It runs read-only (``--approval-mode plan``). An operator
    selects it explicitly -- ``[aliases] default = "qwen-p"`` or
    ``@qwen-p/<model>`` -- when the ACP path is unwanted. Settings come from
    ``[agents.qwen-p]``, falling back to ``[agents.qwen]`` so an operator who
    never split the two tables still gets sane defaults.
    """
    from .qwen import QwenAgent

    settings = config.agents.get("qwen-p") or config.agents.get("qwen", {})
    return QwenAgent(settings)


def _make_claude(config: Config) -> NvshAgent:
    from .claude import ClaudeAgent

    return ClaudeAgent(config.agents.get("claude", {}))


def _make_codex(config: Config) -> NvshAgent:
    from .codex import CodexAgent

    return CodexAgent(config.agents.get("codex", {}))


def _make_agy(config: Config) -> NvshAgent:
    """``agy -p --output-format stream-json`` (lazy import, see ``_make_qwen``)."""
    from .agy import AgyAgent

    settings = config.agents.get("agy", {})
    return AgyAgent(
        model=_str_or_none(settings.get("model")),
        effort=_str_or_none(settings.get("effort")),
        extra_args=_list_or_none(settings.get("extra_args")),
        approval=str(settings.get("approval", "nvsh")),
    )


def _make_kiro(config: Config) -> NvshAgent:
    """``kiro-cli acp`` over :class:`~.acp.AcpAgent` (lazy import, see ``_make_qwen``)."""
    from .acp import build

    return build("kiro", config.agents.get("kiro", {}))


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
        path="rpc",
        hosted=False,
        needs_node=True,
    ),
    "qwen": AdapterSpec(
        name="qwen",
        binary="qwen",
        factory=_make_qwen,
        description="Qwen Code CLI over ACP (qwen --acp).",
        path="acp",
        hosted=False,
        needs_node=True,
    ),
    "qwen-p": AdapterSpec(
        name="qwen-p",
        binary="qwen",
        factory=_make_qwen_print,
        description="Qwen Code CLI print mode (stream-json, read-only); no-ACP fallback.",
        path="stream-json",
        hosted=False,
        needs_node=True,
    ),
    "claude": AdapterSpec(
        name="claude",
        binary="claude",
        factory=_make_claude,
        description="Claude Code CLI (claude -p --output-format stream-json).",
        path="stream-json",
        hosted=True,
        needs_node=True,
    ),
    "codex": AdapterSpec(
        name="codex",
        binary="codex",
        factory=_make_codex,
        description="Codex CLI (codex exec --json).",
        path="app-server",
        hosted=True,
        needs_node=True,
    ),
    "agy": AdapterSpec(
        name="agy",
        binary="agy",
        factory=_make_agy,
        description="Agy CLI (stream-json over stdout).",
        path="stream-json",
        hosted=True,
        needs_node=False,
    ),
    "kiro": AdapterSpec(
        name="kiro",
        binary="kiro-cli",
        factory=_make_kiro,
        description="Kiro CLI over ACP (kiro-cli acp).",
        path="acp",
        hosted=True,
        needs_node=False,
    ),
    "openai-compat": AdapterSpec(
        name="openai-compat",
        binary=None,
        factory=_make_openai_compat,
        description="Stdlib urllib client against an OpenAI-compatible endpoint.",
        path="http",
        hosted=False,
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


def _forced_backend(config: Config, forced: str | Target) -> str:
    """Resolve a ``forced`` argument to a bare backend name.

    A ``Target`` names its backend directly; a string is resolved through
    :meth:`Config.resolve_target`, so both an alias (``'fast'``) and a
    literal ``'backend[/model[/effort]]'`` string (``'claude/opus'``) work.
    """
    if isinstance(forced, Target):
        return forced.backend
    try:
        backend, _model, _effort, _alias = config.resolve_target(forced)
    except ConfigError as exc:
        raise CliError(
            EXIT_USER_ERROR,
            f"cannot resolve forced target {forced!r}: {exc}",
            remediation="use a configured alias, or a 'backend[/model[/effort]]' string",
        ) from exc
    return backend


def choose(
    config: Config,
    which: WhichFn = shutil.which,
    forced: str | Target | None = None,
) -> tuple[str, str]:
    """Pick a backend: the configured provider if installed, else openai-compat.

    Returns ``(name, reason)``. ``reason`` always explains the pick, so
    ``nvsh setup`` (t21) can print it verbatim -- e.g. "pi not on PATH and
    node missing; using openai-compat against http://host:8000/v1".

    ``forced`` (task t20's ``nvsh --agent <target>``) overrides the
    configured provider entirely: an alias name, a literal
    ``'backend[/model[/effort]]'`` string, or a :class:`~.base.Target` is
    resolved via :meth:`Config.resolve_target` (or read straight off the
    ``Target``) and either wins outright -- when its binary is on PATH -- or
    fails loudly with a :class:`~nvsh.cli._errors.CliError` naming the
    missing binary. A forced target never silently falls back to
    ``openai-compat``; that fallback is only for the unforced path below.
    """
    if forced is not None:
        backend = _forced_backend(config, forced)
        if backend not in ADAPTERS:
            raise CliError(
                EXIT_USER_ERROR,
                f"unknown backend {backend!r} in forced target {forced!r}",
                remediation=f"choose one of: {', '.join(sorted(ADAPTERS))}",
            )
        spec = ADAPTERS[backend]
        if not installed(backend, which):
            raise CliError(
                EXIT_ENV_ERROR,
                f"{spec.binary} is not installed (forced backend {backend!r})",
                remediation=f"install {spec.binary}, or drop --agent to let nvsh choose",
            )
        return backend, f"forced via --agent {forced!r}"

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
