"""AuditLog: append-only JSONL record of every proposal/decision/outcome."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping

from .base import Target, target_to_dict


def default_audit_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolve ``$XDG_STATE_HOME/nvsh/audit.jsonl`` through an injectable env mapping.

    Falls back to ``$HOME/.local/state`` when ``XDG_STATE_HOME`` is unset, per the
    XDG base directory spec. ``env`` defaults to ``os.environ`` but callers (and
    tests) can inject any mapping to avoid touching real process state.
    """
    resolved_env = os.environ if env is None else env
    xdg_state_home = resolved_env.get("XDG_STATE_HOME")
    if xdg_state_home:
        base = Path(xdg_state_home)
    else:
        home = resolved_env.get("HOME") or os.path.expanduser("~")
        base = Path(home) / ".local" / "state"
    return base / "nvsh" / "audit.jsonl"


def _to_jsonable(value: object) -> object:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_jsonable(v) for key, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(v) for key, v in value.items()}
    return value


class AuditLog:
    """Appends JSON lines to a file, creating its parent dir 0700 and itself 0600.

    One line per :meth:`record` call: ``{ts, event, proposal, decision, outcome}``.
    """

    def __init__(self, path: str | Path | None = None, env: Mapping[str, str] | None = None):
        self.path = Path(path) if path is not None else default_audit_path(env)
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)

    def record(
        self,
        event: str,
        proposal: object = None,
        decision: object = None,
        outcome: object = None,
        target: Target | Mapping[str, object] | None = None,
    ) -> dict:
        """Append one JSON line and return the entry that was written.

        ``target`` is optional (t18): existing call sites (``nvsh/agent/
        loop.py``, ``nvsh/installers.py``) never pass it and keep recording
        ``"target": null``. A :class:`~nvsh.agent.base.Target` is encoded via
        :func:`~nvsh.agent.base.target_to_dict`; a caller that already has a
        plain dict (e.g. decoded off the wire) may pass that instead.
        """
        if isinstance(target, Target):
            target_data: object = target_to_dict(target)
        elif target is not None:
            target_data = dict(target)
        else:
            target_data = None
        entry = {
            "ts": time.time(),
            "event": event,
            "proposal": _to_jsonable(proposal),
            "decision": decision,
            "outcome": _to_jsonable(outcome),
            "target": target_data,
        }
        line = json.dumps(entry, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        os.chmod(self.path, 0o600)
        return entry

    def read_all(self) -> list[dict]:
        """Read back every recorded entry (test/debug convenience, not on the hot path)."""
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
