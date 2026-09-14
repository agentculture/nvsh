"""Stdlib TOML config loader for nvsh (stable-contract).

Reads ``$XDG_CONFIG_HOME/nvsh/config.toml`` (defaulting
``$XDG_CONFIG_HOME`` to ``~/.config`` via :meth:`pathlib.Path.home`, never a
hard-coded path). A missing file yields the built-in defaults untouched.

This module never reads or stores a literal API key. Backends that need a
key configure ``api_key_env`` — the *name* of an environment variable the
caller reads at call time — never a value. A ``config.toml`` that sets a
literal ``api_key`` is rejected outright (:class:`ConfigError`), so a key
pasted into the file by mistake fails loudly instead of being silently
absorbed and later leaked. A key may also live in a *file* --
``api_key_file``, or the default ``$XDG_CONFIG_HOME/nvsh/api_key`` -- which
:func:`resolve_bearer` reads at call time, refusing any file another user
can read.

Unknown keys — at the top level or inside a known table — are rejected with
a :class:`ConfigError` that lists the valid keys, so a typo in the config
file is a loud failure instead of a silently ignored setting.
"""

from __future__ import annotations

import os
import re
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

#: Top-level tables this config format recognizes.
_VALID_TOP_KEYS = {"agent", "agents", "aliases", "sessions", "triggers"}

#: Keys recognized inside ``[agent]``.
_VALID_AGENT_KEYS = {"provider"}

#: Keys recognized inside any ``[agents.<name>]`` table. Union of every
#: backend's fields — a given backend only uses a subset (e.g. ``pi`` never
#: sets ``base_url``). ``api_key`` is deliberately NOT in this set: it is
#: rejected explicitly below with a dedicated message, not silently allowed
#: through as "unknown". ``effort`` and ``extra_args`` and ``approval`` are
#: per-harness knobs (task t5): ``effort`` and ``model`` are opaque strings
#: nvsh never validates against an enum (decision c24) -- a given backend
#: decides what values it accepts; ``approval`` is checked against a fixed
#: two-value set below because it controls a nvsh-vs-harness code path, not
#: a backend-specific string.
_VALID_AGENT_BACKEND_KEYS = {
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "api_key_file",
    "effort",
    "extra_args",
    "approval",
    "fixture",
}

#: Valid values for ``[agents.<name>].approval``.
_VALID_APPROVAL_VALUES = {"nvsh", "harness"}

#: Reserved alias name: ``[aliases].default`` is what a bare ``nvsh --agent
#: default`` (or no ``--agent`` at all) resolves to.
DEFAULT_ALIAS = "default"

#: Keys recognized inside ``[sessions]``.
_VALID_SESSIONS_KEYS = {"max"}

#: Keys recognized inside ``[triggers]``.
_VALID_TRIGGERS_KEYS = {"rate_window_seconds", "opt_in_patterns"}

_DEFAULT_AGENTS: dict[str, dict[str, object]] = {
    "pi": {"provider": "nemotron", "model": "associate"},
}


class ConfigError(ValueError):
    """Raised when ``config.toml`` is malformed or carries an unknown/refused key."""


def _parse_alias_target(spec: str) -> tuple[str, str | None, str | None]:
    """Parse a ``'backend[/model[/effort]]'`` string into its segments.

    At most three ``/``-separated segments, none of them empty. ``model``
    and ``effort`` are opaque strings (decision c24): never validated
    against an enum, just passed through.
    """
    parts = spec.split("/")
    if not (1 <= len(parts) <= 3) or any(part == "" for part in parts):
        raise ConfigError(
            f"invalid alias target {spec!r}: expected 'backend[/model[/effort]]' "
            "(1-3 non-empty '/'-separated segments)"
        )
    backend = parts[0]
    model = parts[1] if len(parts) > 1 else None
    effort = parts[2] if len(parts) > 2 else None
    return backend, model, effort


@dataclass
class Config:
    """Resolved nvsh configuration — defaults merged with ``config.toml``."""

    agent_provider: str = "pi"
    agents: dict[str, dict[str, object]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _DEFAULT_AGENTS.items()}
    )
    aliases: dict[str, str] = field(default_factory=dict)
    sessions_max: int = 1
    triggers: dict[str, object] = field(default_factory=dict)

    def resolve_target(self, name: str) -> tuple[str, str | None, str | None, bool]:
        """Resolve *name* to ``(backend, model, effort, alias)``.

        *name* is either an alias registered in ``[aliases]`` (including the
        reserved ``'default'``, which falls back to the legacy ``[agent]``
        provider when no ``[aliases].default`` is set), or a literal
        ``'backend[/model[/effort]]'`` target string (an optional leading
        ``@`` is stripped, so ``'@claude/sonnet/medium'`` resolves the same
        as ``'claude/sonnet/medium'``). ``alias`` tells the caller whether
        *name* was resolved by alias lookup (``True``) or parsed directly as
        a literal target string (``False``). When an alias omits the model
        segment, the model falls back to that backend's configured
        ``[agents.<name>].model``.
        """
        if name in self.aliases:
            backend, model, effort = _parse_alias_target(self.aliases[name])
            model = model or self.agents.get(backend, {}).get("model")
            return backend, model, effort, True

        if name == DEFAULT_ALIAS:
            backend = self.agent_provider
            model = self.agents.get(backend, {}).get("model")
            return backend, model, None, True

        stripped = name[1:] if name.startswith("@") else name
        if "/" in stripped:
            backend, model, effort = _parse_alias_target(stripped)
            model = model or self.agents.get(backend, {}).get("model")
            return backend, model, effort, False

        raise ConfigError(
            f"unknown alias {name!r}: not in [aliases] and not a "
            "'backend[/model[/effort]]' target"
        )


def default_toml() -> str:
    """Return the ``config.toml`` template text for ``nvsh config init``.

    Uses only localhost placeholders and never a literal secret — the
    ``openai-compat`` example points at ``api_key_env``, the name of an
    environment variable, not a key value.
    """
    return """\
# nvsh config — see docs/config.example.toml for the full annotated example.

[agent]
provider = "pi"

[agents.pi]
provider = "nemotron"
model = "associate"

[agents.openai-compat]
base_url = "http://localhost:8000/v1"
api_key_env = "NVSH_API_KEY"
# Or keep the key in a file nvsh reads at call time (mode 0600). Unset, nvsh
# still falls back to the default key file below.
# api_key_file = "$XDG_CONFIG_HOME/nvsh/api_key"

[agents.claude]
provider = "claude"
model = "sonnet"

# [aliases] maps a short name to a 'backend[/model[/effort]]' target.
# 'default' is reserved: it is what a bare --agent (or no --agent at all)
# resolves to. When [aliases].default is absent, 'default' falls back to
# [agent] provider above and that backend's configured model, no effort.
[aliases]
default = "pi"
reviewer = "claude/opus/high"
local = "pi"

[sessions]
max = 1

[triggers]
rate_window_seconds = 60
opt_in_patterns = []
"""


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    return _toml_str(str(value))


def _dump_toml(cfg: Config) -> str:
    """Serialize *cfg* back to ``config.toml`` text (round-trips through :func:`load`).

    Deliberately minimal (no third-party TOML writer -- stdlib ``tomllib``
    is read-only): covers exactly the shapes :func:`load` accepts. Never
    emits a literal ``api_key`` -- callers only ever set ``api_key_env``.
    """
    lines = ["[agent]", f"provider = {_toml_str(cfg.agent_provider)}", ""]

    for name, table in cfg.agents.items():
        lines.append(f"[agents.{name}]")
        for key, value in table.items():
            lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")

    lines.append("[aliases]")
    for name, spec in cfg.aliases.items():
        lines.append(f"{name} = {_toml_str(spec)}")
    lines.append("")

    lines.append("[sessions]")
    lines.append(f"max = {_toml_scalar(cfg.sessions_max)}")
    lines.append("")

    lines.append("[triggers]")
    for key, value in cfg.triggers.items():
        lines.append(f"{key} = {_toml_scalar(value)}")
    lines.append("")

    return "\n".join(lines)


def save(cfg: Config, path: Path | None = None) -> None:
    """Write *cfg* to ``config.toml`` (default ``$XDG_CONFIG_HOME/nvsh/config.toml``)."""
    target = path if path is not None else _default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_dump_toml(cfg), encoding="utf-8")


def set_provider(name: str, path: Path | None = None) -> Config:
    """Load the current config, set ``[agent] provider = name``, save, and return it.

    Preserves every other table already on disk (loads first, mutates,
    saves the full merged config back).
    """
    cfg = load(path)
    cfg.agent_provider = name
    cfg.aliases[DEFAULT_ALIAS] = name
    save(cfg, path)
    return cfg


def _config_dir(env: Mapping[str, str] | None = None) -> Path:
    resolved = os.environ if env is None else env
    xdg = resolved.get("XDG_CONFIG_HOME")
    if xdg:
        base = Path(xdg)
    else:
        home = resolved.get("HOME")
        base = (Path(home) if home else Path.home()) / ".config"
    return base / "nvsh"


def _default_path() -> Path:
    return _config_dir() / "config.toml"


def _reject_unknown(table: dict, valid: set[str], where: str) -> None:
    unknown = set(table) - valid
    if unknown:
        bad = ", ".join(sorted(unknown))
        allowed = ", ".join(sorted(valid))
        raise ConfigError(f"unknown key(s) in {where}: {bad} (valid keys: {allowed})")


def _table(raw: dict, name: str, what: str) -> dict:
    """``raw[name]`` as a table (``{}`` when absent), or a :class:`ConfigError`."""
    table = raw.get(name, {})
    if not isinstance(table, dict):
        raise ConfigError(what)
    return table


def _apply_agent(raw: dict, cfg: Config) -> None:
    agent_table = _table(raw, "agent", "[agent] must be a table")
    _reject_unknown(agent_table, _VALID_AGENT_KEYS, "[agent]")
    if "provider" in agent_table:
        cfg.agent_provider = agent_table["provider"]


def _apply_agents(raw: dict, cfg: Config) -> None:
    agents_table = _table(raw, "agents", "[agents] must be a table of tables")
    if not agents_table:
        return
    merged: dict[str, dict[str, object]] = {k: dict(v) for k, v in _DEFAULT_AGENTS.items()}
    for name, backend_table in agents_table.items():
        if not isinstance(backend_table, dict):
            raise ConfigError(f"[agents.{name}] must be a table")
        if "api_key" in backend_table:
            raise ConfigError(
                f"[agents.{name}] sets 'api_key' directly — nvsh never reads or "
                "stores literal API keys; use 'api_key_env' to name an "
                "environment variable instead"
            )
        _reject_unknown(backend_table, _VALID_AGENT_BACKEND_KEYS, f"[agents.{name}]")
        if "approval" in backend_table and backend_table["approval"] not in _VALID_APPROVAL_VALUES:
            allowed = ", ".join(sorted(_VALID_APPROVAL_VALUES))
            raise ConfigError(
                f"[agents.{name}] approval={backend_table['approval']!r} is invalid "
                f"(valid values: {allowed})"
            )
        if "extra_args" in backend_table:
            extra_args = backend_table["extra_args"]
            if not isinstance(extra_args, list) or not all(
                isinstance(item, str) for item in extra_args
            ):
                raise ConfigError(f"[agents.{name}] extra_args must be a list of strings")
        merged.setdefault(name, {})
        merged[name].update(backend_table)
    cfg.agents = merged


def _apply_aliases(raw: dict, cfg: Config) -> None:
    aliases_table = _table(raw, "aliases", "[aliases] must be a table")
    for name, spec in aliases_table.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(name)):
            raise ConfigError(
                f"[aliases] name {name!r} must be a bare key (letters, digits, '_' or '-')"
            )
        if not isinstance(spec, str):
            raise ConfigError(f"[aliases] {name!r} must be a string 'backend[/model[/effort]]'")
        _parse_alias_target(spec)  # validated eagerly; raises ConfigError on a bad shape
    cfg.aliases = dict(aliases_table)


def _apply_sessions(raw: dict, cfg: Config) -> None:
    sessions_table = _table(raw, "sessions", "[sessions] must be a table")
    _reject_unknown(sessions_table, _VALID_SESSIONS_KEYS, "[sessions]")
    if "max" in sessions_table:
        cfg.sessions_max = sessions_table["max"]


def _apply_triggers(raw: dict, cfg: Config) -> None:
    triggers_table = _table(raw, "triggers", "[triggers] must be a table")
    _reject_unknown(triggers_table, _VALID_TRIGGERS_KEYS, "[triggers]")
    cfg.triggers = dict(triggers_table)


def _read_toml(target: Path) -> dict:
    try:
        return tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"malformed TOML in {target}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {target}: {exc}") from exc


def load(path: Path | None = None) -> Config:
    """Load config from ``path`` (default ``$XDG_CONFIG_HOME/nvsh/config.toml``).

    A missing file yields :class:`Config` defaults untouched. Never reads or
    stores API keys — a literal ``api_key`` anywhere under ``[agents.*]``
    raises :class:`ConfigError`.
    """
    target = path if path is not None else _default_path()
    if not target.is_file():
        return Config()

    raw = _read_toml(target)
    _reject_unknown(raw, _VALID_TOP_KEYS, "top level")

    cfg = Config()
    _apply_agent(raw, cfg)
    _apply_agents(raw, cfg)
    _apply_aliases(raw, cfg)
    _apply_sessions(raw, cfg)
    _apply_triggers(raw, cfg)
    return cfg


# ---------------------------------------------------------------------------
# bearer resolution (deviation d10)
# ---------------------------------------------------------------------------

#: Name of the key file nvsh reads when nothing else is configured.
DEFAULT_KEY_FILENAME = "api_key"

#: Placeholder spelling of that file, for messages. Never a resolved path:
#: an operator may paste a doctor line anywhere, and the directory is
#: machine-identifying (see the d4 rule for endpoint URLs).
DEFAULT_KEY_FILE_DISPLAY = "$XDG_CONFIG_HOME/nvsh/" + DEFAULT_KEY_FILENAME

#: Generic source labels. They name *where* a bearer came from and never
#: carry the key, nor any prefix of it, nor the directory it lives in.
KEY_SOURCE_FILE = "api_key_file"
KEY_SOURCE_DEFAULT_FILE = "the default key file"
NO_BEARER_NOTE = "no bearer configured"

#: ``$VAR`` / ``${VAR}``. ``re.ASCII`` keeps ``\w`` to ``[A-Za-z0-9_]``, the
#: shell's own variable-name alphabet.
_VAR_REF_RE = re.compile(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)", re.ASCII)


@dataclass(frozen=True)
class BearerResolution:
    """The outcome of looking for an API bearer, with no key in any label.

    ``bearer`` is the key itself (or ``None``); ``source`` is a generic
    label for a message; ``diagnostic`` explains a *refused* source (a key
    file another user can read, or one that is not there) in a line safe to
    print -- it reports the file's name and mode, never its content.
    """

    bearer: str | None = None
    source: str | None = None
    diagnostic: str | None = None


def default_key_file(env: Mapping[str, str] | None = None) -> Path:
    """``$XDG_CONFIG_HOME/nvsh/api_key``, falling back to ``$HOME/.config/...``."""
    return _config_dir(env) / DEFAULT_KEY_FILENAME


def expand_path(raw: str, env: Mapping[str, str] | None = None) -> Path:
    """Expand ``~`` and ``$VAR``/``${VAR}`` in a configured path against *env*."""
    resolved = os.environ if env is None else env

    def _sub(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        return resolved.get(name, match.group(0))

    text = _VAR_REF_RE.sub(_sub, raw.strip())
    if text.startswith("~"):
        home = resolved.get("HOME") or str(Path.home())
        text = home + text[1:].lstrip("/") if text == "~" else home + text[1:]
    return Path(text)


def _read_key_file(path: Path) -> BearerResolution:
    """Read one key file, refusing it when anyone but the owner can read it."""
    name = path.name
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return BearerResolution(diagnostic=f"key file {name} is not readable")
    if mode & 0o077:
        return BearerResolution(
            diagnostic=(
                f"ignoring key file {name}: mode {mode:04o} lets other users read it "
                f"(chmod 600 it)"
            )
        )
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return BearerResolution(diagnostic=f"key file {name} is not readable")
    if not text:
        return BearerResolution(diagnostic=f"key file {name} is empty")
    return BearerResolution(bearer=text, source=KEY_SOURCE_FILE)


def resolve_bearer(
    settings: Mapping[str, object], env: Mapping[str, str] | None = None
) -> BearerResolution:
    """Find the bearer for one ``[agents.<name>]`` table, in a fixed order.

    1. ``api_key_env``, when that variable is set and non-empty in *env*.
    2. ``api_key_file``, when configured (a configured-but-unusable file is
       reported and stops the search -- nvsh does not silently reach for a
       different key than the one the operator pointed at).
    3. the default key file, when it exists.
    4. nothing: the caller sends no ``Authorization`` header at all.

    The key itself only ever travels in :attr:`BearerResolution.bearer`.
    """
    resolved = os.environ if env is None else env

    env_name = settings.get("api_key_env")
    if env_name:
        value = (resolved.get(str(env_name)) or "").strip()
        if value:
            return BearerResolution(bearer=value, source=f"${env_name}")

    configured = settings.get("api_key_file")
    if configured:
        return _read_key_file(expand_path(str(configured), resolved))

    fallback = default_key_file(resolved)
    if fallback.exists():
        outcome = _read_key_file(fallback)
        if outcome.source is not None:
            return BearerResolution(bearer=outcome.bearer, source=KEY_SOURCE_DEFAULT_FILE)
        return outcome

    return BearerResolution()
