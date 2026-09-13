"""Stdlib TOML config loader for nvsh (stable-contract).

Reads ``$XDG_CONFIG_HOME/nvsh/config.toml`` (defaulting
``$XDG_CONFIG_HOME`` to ``~/.config`` via :meth:`pathlib.Path.home`, never a
hard-coded path). A missing file yields the built-in defaults untouched.

This module never reads or stores a literal API key. Backends that need a
key configure ``api_key_env`` — the *name* of an environment variable the
caller reads at call time — never a value. A ``config.toml`` that sets a
literal ``api_key`` is rejected outright (:class:`ConfigError`), so a key
pasted into the file by mistake fails loudly instead of being silently
absorbed and later leaked.

Unknown keys — at the top level or inside a known table — are rejected with
a :class:`ConfigError` that lists the valid keys, so a typo in the config
file is a loud failure instead of a silently ignored setting.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

#: Top-level tables this config format recognizes.
_VALID_TOP_KEYS = {"agent", "agents", "sessions", "triggers"}

#: Keys recognized inside ``[agent]``.
_VALID_AGENT_KEYS = {"provider"}

#: Keys recognized inside any ``[agents.<name>]`` table. Union of every
#: backend's fields — a given backend only uses a subset (e.g. ``pi`` never
#: sets ``base_url``). ``api_key`` is deliberately NOT in this set: it is
#: rejected explicitly below with a dedicated message, not silently allowed
#: through as "unknown".
_VALID_AGENT_BACKEND_KEYS = {"provider", "model", "base_url", "api_key_env"}

#: Keys recognized inside ``[sessions]``.
_VALID_SESSIONS_KEYS = {"max"}

#: Keys recognized inside ``[triggers]``.
_VALID_TRIGGERS_KEYS = {"rate_window_seconds", "opt_in_patterns"}

_DEFAULT_AGENTS: dict[str, dict[str, object]] = {
    "pi": {"provider": "nemotron", "model": "associate"},
}


class ConfigError(ValueError):
    """Raised when ``config.toml`` is malformed or carries an unknown/refused key."""


@dataclass
class Config:
    """Resolved nvsh configuration — defaults merged with ``config.toml``."""

    agent_provider: str = "pi"
    agents: dict[str, dict[str, object]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _DEFAULT_AGENTS.items()}
    )
    sessions_max: int = 1
    triggers: dict[str, object] = field(default_factory=dict)


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
    save(cfg, path)
    return cfg


def _config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "nvsh"


def _default_path() -> Path:
    return _config_dir() / "config.toml"


def _reject_unknown(table: dict, valid: set[str], where: str) -> None:
    unknown = set(table) - valid
    if unknown:
        bad = ", ".join(sorted(unknown))
        allowed = ", ".join(sorted(valid))
        raise ConfigError(f"unknown key(s) in {where}: {bad} (valid keys: {allowed})")


def load(path: Path | None = None) -> Config:
    """Load config from ``path`` (default ``$XDG_CONFIG_HOME/nvsh/config.toml``).

    A missing file yields :class:`Config` defaults untouched. Never reads or
    stores API keys — a literal ``api_key`` anywhere under ``[agents.*]``
    raises :class:`ConfigError`.
    """
    target = path if path is not None else _default_path()
    if not target.is_file():
        return Config()

    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"malformed TOML in {target}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {target}: {exc}") from exc

    _reject_unknown(raw, _VALID_TOP_KEYS, "top level")

    cfg = Config()

    agent_table = raw.get("agent", {})
    if not isinstance(agent_table, dict):
        raise ConfigError("[agent] must be a table")
    _reject_unknown(agent_table, _VALID_AGENT_KEYS, "[agent]")
    if "provider" in agent_table:
        cfg.agent_provider = agent_table["provider"]

    agents_table = raw.get("agents", {})
    if not isinstance(agents_table, dict):
        raise ConfigError("[agents] must be a table of tables")
    if agents_table:
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
            merged.setdefault(name, {})
            merged[name].update(backend_table)
        cfg.agents = merged

    sessions_table = raw.get("sessions", {})
    if not isinstance(sessions_table, dict):
        raise ConfigError("[sessions] must be a table")
    _reject_unknown(sessions_table, _VALID_SESSIONS_KEYS, "[sessions]")
    if "max" in sessions_table:
        cfg.sessions_max = sessions_table["max"]

    triggers_table = raw.get("triggers", {})
    if not isinstance(triggers_table, dict):
        raise ConfigError("[triggers] must be a table")
    _reject_unknown(triggers_table, _VALID_TRIGGERS_KEYS, "[triggers]")
    cfg.triggers = dict(triggers_table)

    return cfg
