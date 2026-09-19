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
#: ``fixture`` is the demo adapter's: its "protocol" is a committed JSON
#: file in the package. ``inproc`` is a local response tier's (``needle``,
#: and ``lfm`` once task t19 registers it): it speaks to a child process
#: this same daemon starts, not to a separately installed CLI.
PATH_VALUES = {"rpc", "stream-json", "app-server", "acp", "http", "fixture", "inproc"}

#: Adapters :func:`probe` never offers. ``openai-compat`` is excluded
#: because it is always "installed" and would win every auto-pick; ``demo``
#: for the same reason and a stronger one -- it answers from a fixture, so
#: auto-picking it would silently replace the operator's harness with a
#: canned reply. ``needle`` (and ``lfm``, task t19) is excluded because it
#: is a single tier, not a full agent: it only ever answers
#: instruction-shaped requests, so auto-picking it as the default would mean
#: every ordinary failure silently gets a Tier-1-only answer instead of the
#: full agent. ``demo``/``needle`` stay selectable *per request*
#: (``--agent demo``, ``@needle``, an alias whose target is one of them),
#: but ``nvsh agent use``/``nvsh setup --agent`` refuse to persist either as
#: the default -- see :data:`NOT_PERSISTABLE_DEFAULT_REASONS` -- because
#: that would make every ordinary failure replay a canned fixture or a
#: Tier-1-only answer instead of calling a real backend.
PROBE_EXCLUDED = frozenset({"openai-compat", "demo", "needle"})

#: Shared by every place that refuses to persist ``demo`` as the default
#: backend (``nvsh agent use demo``, ``nvsh setup --agent demo``, and
#: doctor's ``default_target_not_demo`` check): ``demo`` is a scripted
#: fixture replay (see ``nvsh/agent/demo.py``'s module docstring), not a
#: real backend, so it must never become what a bare ``--agent`` or the
#: daemon's warm session resolves to. Per-request use (``--agent demo``,
#: ``@demo``, an alias pointing at demo) is unaffected -- only persisting
#: it as the default is refused.
DEMO_DEFAULT_MESSAGE = (
    "demo is a scripted fixture (no model, no network); it cannot be the persisted default agent"
)
DEMO_DEFAULT_HINT = (
    "choose a real backend with 'nvsh agent use <name>' (see 'nvsh agent list'); "
    "run the demo for one request with --agent demo or @demo instead"
)

#: Same refusal, for ``needle`` (see ``nvsh/agent/needle.py``'s module
#: docstring): a Tier 1 tool-selecting model answers instruction-shaped
#: requests only, and declines everything else rather than calling a real
#: harness -- exactly the wrong shape for the backend nvsh always falls
#: back to.
NEEDLE_DEFAULT_MESSAGE = (
    "needle is a Tier 1 local-response model (instruction-shaped requests only); "
    "it cannot be the persisted default agent"
)
NEEDLE_DEFAULT_HINT = (
    "choose a real backend with 'nvsh agent use <name>' (see 'nvsh agent list'); "
    "run needle for one request with --agent needle or @needle instead"
)

#: ``nvsh agent use``/``nvsh setup --agent``'s refusal table for adapters
#: that must never be the *persisted default* (each stays selectable per
#: request). One dict instead of a growing chain of ``if name == ...``:
#: ``lfm`` (task t19) joins this the same way -- add its ``(message, hint)``
#: pair here, no new branch.
NOT_PERSISTABLE_DEFAULT_REASONS: dict[str, tuple[str, str]] = {
    "demo": (DEMO_DEFAULT_MESSAGE, DEMO_DEFAULT_HINT),
    "needle": (NEEDLE_DEFAULT_MESSAGE, NEEDLE_DEFAULT_HINT),
}

#: The set form of :data:`NOT_PERSISTABLE_DEFAULT_REASONS`'s keys, for a
#: plain membership check where the reason text isn't needed.
NOT_PERSISTABLE_DEFAULT = frozenset(NOT_PERSISTABLE_DEFAULT_REASONS)


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
    #: Overrides :func:`installed`'s default rule (``which(binary)``, or
    #: ``True`` when ``binary`` is ``None``) with a zero-arg predicate. Used
    #: by adapters with no binary at all whose "installed" question is not
    #: "always yes" (``openai-compat``/``demo``'s case) but "is the Python
    #: flavor importable" -- ``needle`` (and ``lfm``, task t19).
    installed_check: Callable[[], bool] | None = None


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


def _make_demo(config: Config) -> NvshAgent:
    """The fixture-replaying demo adapter (lazy import, see ``_make_qwen``)."""
    from .demo import DemoAgent

    return DemoAgent(config.agents.get("demo", {}))


def _make_openai_compat(config: Config) -> NvshAgent:
    from .openai_compat import OpenAICompatAgent

    return OpenAICompatAgent(config.agents.get("openai-compat", {}))


def _make_needle(config: Config) -> NvshAgent:
    """The explicit Tier-1-only adapter (lazy import, see ``_make_qwen``).

    Settings come from ``[tiers]``, not ``[agents.needle]``: Tier 1 has no
    harness-style knobs of its own, only the ones the tier ladder already
    defines (``needle_min_confidence``, ``memory_floor_mb``, ...).
    """
    from .needle import NeedleAgent

    return NeedleAgent(config.tiers)


def _needle_flavor_installed() -> bool:
    """Whether the ``needle`` (``cactus-needle``) Python package is
    importable. ``find_spec`` only locates the module -- it is never
    imported, so this never runs the native engine's own module-level code
    (mirrors ``nvsh/doctor_checks.py``'s ``_tier_flavor_installed``).
    """
    import importlib.util

    try:
        return importlib.util.find_spec("needle") is not None
    except (ImportError, ValueError):
        return False


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
    "demo": AdapterSpec(
        name="demo",
        binary=None,
        factory=_make_demo,
        description="Scripted demo backend; replays a committed fixture (no model, no network).",
        path="fixture",
        hosted=False,
        needs_node=False,
    ),
    "needle": AdapterSpec(
        name="needle",
        binary=None,
        factory=_make_needle,
        description="Needle3 Tier 1, explicit-only (@needle); local, instruction-shaped requests.",
        path="inproc",
        hosted=False,
        needs_node=False,
        installed_check=_needle_flavor_installed,
    ),
}


def installed(name: str, which: WhichFn = shutil.which) -> bool:
    """Is adapter ``name`` usable right now?

    ``spec.installed_check`` wins when set (``needle``'s: is the flavor
    importable). Otherwise: ``openai-compat`` and ``demo`` always are --
    neither has a binary, so ``which`` is not consulted at all (never with
    ``None``, which would raise) -- and everything else is on PATH or not.
    """
    spec = ADAPTERS[name]
    if spec.installed_check is not None:
        return spec.installed_check()
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


def _tool_calling(name: str, config: Config) -> bool:
    """Derive ``tool_calling`` for adapter ``name`` the same way
    ``build_adapter_rows`` does (``nvsh/cli/_commands/agent.py``'s
    ``_adapter_capabilities``): construct the adapter and read
    ``.capabilities().tool_calling`` -- without importing the CLI module.
    Every adapter's ``__init__`` is cheap and never spawns a subprocess, so
    building one just to read this is safe for a probe. A factory that
    validates its own settings and raises is treated as ``tool_calling=False``
    rather than failing the whole probe.
    """
    try:
        agent = ADAPTERS[name].factory(config)
        return bool(agent.capabilities().tool_calling)
    except Exception:  # noqa: BLE001 - defensive: factories may validate/raise
        return False


def steer_capable(name: str, config: Config) -> bool:
    """Whether adapter ``name`` can take a mid-turn correction, built the
    same way as :func:`_tool_calling`: construct the adapter (cheap, no
    subprocess) and read ``.capabilities().steer`` -- ``start()`` is never
    called, so this starts no process. Used client-side (stop-choice-prompt
    c31) to pick the choice prompt's ``[t]`` label without asking the
    daemon: ``true`` only for ``pi`` and ``codex``. A factory that
    validates its own settings and raises is treated as ``steer=False``
    rather than failing the caller.
    """
    try:
        agent = ADAPTERS[name].factory(config)
        return bool(agent.capabilities().steer)
    except Exception:  # noqa: BLE001 - defensive: factories may validate/raise
        return False


def probe(which: WhichFn = shutil.which, config: Config | None = None) -> list[dict]:
    """Installed adapters (:data:`PROBE_EXCLUDED` left out), tool-calling first.

    Each row carries ``name``, ``hosted`` and ``tool_calling``. Rows are
    ordered tool-calling adapters first, then the rest -- within each group,
    ``ADAPTERS`` registration order is preserved (``list.sort`` is stable).
    ``tool_calling`` comes from :func:`_tool_calling`, the same underlying
    source (the constructed adapter's ``.capabilities()``) that
    ``build_adapter_rows`` reports for 'nvsh agent list'.

    ``config`` defaults to a bare :class:`~nvsh.config.Config` -- enough to
    derive ``tool_calling`` for every adapter's default settings; pass the
    live config to reflect an operator's ``[agents.<name>]`` overrides
    (e.g. ``[agents.qwen] approval = "harness"``).

    Adapters that share one ``binary`` (``qwen`` and its ``qwen-p``
    print-mode fallback) are de-duplicated down to a single row -- the
    first adapter registered for that binary, in ``ADAPTERS`` order -- so a
    single installed harness never surfaces two rows for setup to prompt
    between (Qodo #2, PR #12 review). The dropped adapter stays fully
    selectable via :func:`choose`'s ``forced`` argument, aliases, and
    ``nvsh agent use``; it is only excluded from this auto-pick table.
    """
    if config is None:
        config = Config()
    seen_binaries: set[str] = set()
    rows = []
    for name, spec in ADAPTERS.items():
        if name in PROBE_EXCLUDED or not installed(name, which):
            continue
        if spec.binary is not None:
            if spec.binary in seen_binaries:
                continue
            seen_binaries.add(spec.binary)
        rows.append(
            {
                "name": name,
                "hosted": spec.hosted,
                "tool_calling": _tool_calling(name, config),
            }
        )
    rows.sort(key=lambda row: 0 if row["tool_calling"] else 1)
    return rows


def _forced_backend(config: Config, forced: str | Target) -> str:
    """Resolve a ``forced`` argument to a bare backend name.

    A ``Target`` names its backend directly; a string is resolved through
    :meth:`Config.resolve_target`, so both an alias (``'fast'``) and a
    literal ``'backend[/model[/effort]]'`` string (``'claude/opus'``) work.
    """
    if isinstance(forced, Target):
        return forced.backend
    # A bare adapter name (``claude``, ``@codex``) is a valid target even
    # when no alias spells it: the shell's ``@target`` grammar already
    # accepts it (nvsh/client.py ``_bare_backend_target``), and ``nvsh setup
    # --agent claude`` is the documented on-ramp. Checked before
    # ``resolve_target`` so an alias of the same name still wins there.
    bare = forced[1:] if forced.startswith("@") else forced
    if bare in ADAPTERS and forced not in config.aliases:
        return bare
    try:
        backend, _model, _effort, _alias = config.resolve_target(forced)
    except ConfigError as exc:
        raise CliError(
            EXIT_USER_ERROR,
            f"cannot resolve forced target {forced!r}: {exc}",
            remediation="use a configured alias, or a 'backend[/model[/effort]]' string",
        ) from exc
    return backend


def _choose_forced(config: Config, which: WhichFn, forced: str | Target) -> tuple[str, str]:
    """The ``forced is not None`` branch of :func:`choose`, split out to keep
    ``choose`` under the cognitive-complexity limit (SonarCloud python:S3776,
    PR #12 review). Behavior and every message are unchanged: resolve
    ``forced`` to a backend name, then either return it (when its binary is
    on PATH) or raise the same :class:`~nvsh.cli._errors.CliError` as before.
    """
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


def _unavailable_reason(configured: str, which: WhichFn) -> str:
    """Explain why ``configured`` was not picked, for the unforced path of
    :func:`choose` (split out for python:S3776, PR #12 review). Same three
    messages as before, byte-for-byte.
    """
    if configured not in ADAPTERS:
        return f"configured provider '{configured}' is unknown"
    spec = ADAPTERS[configured]
    node_present = which("node") is not None
    if spec.needs_node and not node_present:
        return f"{configured} not on PATH and node missing"
    return f"{configured} not on PATH"


def choose(
    config: Config,
    which: WhichFn = shutil.which,
    forced: str | Target | None = None,
) -> tuple[str, str]:
    """Pick a backend: the configured provider if installed, else whatever
    :func:`probe` finds installed, else openai-compat.

    Returns ``(name, reason)``. ``reason`` always explains the pick, so
    ``nvsh setup`` (t21) can print it verbatim -- e.g. "pi not on PATH and
    node missing; installed: claude, codex; picked claude".

    ``forced`` (task t20's ``nvsh --agent <target>``) overrides the
    configured provider entirely: an alias name, a literal
    ``'backend[/model[/effort]]'`` string, or a :class:`~.base.Target` is
    resolved via :meth:`Config.resolve_target` (or read straight off the
    ``Target``) and either wins outright -- when its binary is on PATH -- or
    fails loudly with a :class:`~nvsh.cli._errors.CliError` naming the
    missing binary. A forced target never silently falls back to
    ``openai-compat``; that fallback is only for the unforced path below.
    See :func:`_choose_forced` for this branch's logic.

    When the configured provider is not installed (:func:`_unavailable_reason`
    explains why), :func:`probe` is consulted: if anything is installed, the
    first probe row (tool-calling adapters first) is picked and the full
    probe list is named in the reason. Only when ``probe`` finds nothing
    installed does the pick fall back to ``openai-compat``, exactly as
    before.
    """
    if forced is not None:
        return _choose_forced(config, which, forced)

    configured = config.agent_provider
    if configured in ADAPTERS and installed(configured, which):
        return configured, f"{configured} is configured and on PATH"

    why = _unavailable_reason(configured, which)

    probed = probe(which, config)
    if probed:
        picked = probed[0]["name"]
        installed_names = ", ".join(row["name"] for row in probed)
        reason = f"{why}; installed: {installed_names}; picked {picked}"
        return picked, reason

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
