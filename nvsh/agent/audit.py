"""AuditLog: append-only JSONL record of every proposal/decision/outcome."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping

from .base import Target, target_to_dict

# The stop-lifecycle kinds record_stop accepts (reliable-agent-stop spec):
# Ctrl+C/Esc cancel, a second-press force kill, steering a running turn,
# replacing it, exiting a busy prompt, declining the agent outright, and
# nvsh doctor --apply clearing a hung turn. Any other kind is a programming
# error in the caller, not a new stop path, so record_stop rejects it.
STOP_KINDS = frozenset(
    {
        "cancel",
        "force_kill",
        "steer",
        "replace",
        "busy_exit",
        "declined",
        "doctor_apply",
        "keep_going",
    }
)

# stop-choice-prompt (t3): where a stop-prompt outcome originated, and why a
# keep_going was recorded (an explicit key press versus the 30s timeout).
STOP_ORIGINS = frozenset({"stop_prompt", "busy_prompt"})
STOP_REASONS = frozenset({"key", "timeout"})


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


def _encode_target(target: Target | Mapping[str, object] | None) -> object:
    """Encode ``target`` the way every audit entry shares: dataclass, dict or None."""
    if isinstance(target, Target):
        return target_to_dict(target)
    if target is not None:
        return dict(target)
    return None


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
        entry = {
            "ts": time.time(),
            "event": event,
            "proposal": _to_jsonable(proposal),
            "decision": decision,
            "outcome": _to_jsonable(outcome),
            "target": _encode_target(target),
        }
        self._write_entry(entry)
        return entry

    def record_stop(
        self,
        kind: str,
        shell: object,
        target: Target | Mapping[str, object] | None,
        elapsed: object,
        outcome: object,
        *,
        origin: str | None = None,
        reason: str | None = None,
        correction: str | None = None,
    ) -> dict:
        """Append one ``event='stop'`` line for the stop/cancel/steer/replace/

        busy-exit/declined/doctor-apply/keep_going lifecycle (t3,
        reliable-agent-stop and stop-choice-prompt). ``kind`` must be one of
        :data:`STOP_KINDS`; anything else is a programming error in the
        caller and raises ``ValueError`` rather than silently recording an
        unrecognised stop path. ``target`` follows :meth:`record`'s
        encoding: a :class:`~nvsh.agent.base.Target` dataclass, a plain
        dict, or ``None``.

        ``origin``, ``reason`` and ``correction`` are keyword-only and
        optional so every existing call site (``nvsh/client.py``,
        ``nvsh/cli/_commands/doctor.py``) keeps working unedited:

        - ``origin`` names where a stop-prompt outcome came from --
          ``"stop_prompt"`` or ``"busy_prompt"`` -- and is written only when
          given.
        - ``reason`` distinguishes an explicit ``keep_going`` key press from
          the 30s timeout -- ``"key"`` or ``"timeout"`` -- and is written
          only when given.
        - ``correction`` is the operator's typed correction text. It is
          never written to the log: only its length, as ``correction_chars``,
          is recorded, and only when ``correction`` is given. There is no
          parameter that accepts and stores the raw text under any other
          name; passing one (e.g. ``text=...``) is a ``TypeError`` because
          no such keyword exists.
        """
        if kind not in STOP_KINDS:
            raise ValueError(f"unknown stop kind: {kind!r} (expected one of {sorted(STOP_KINDS)})")
        if origin is not None and origin not in STOP_ORIGINS:
            raise ValueError(
                f"unknown stop origin: {origin!r} (expected one of {sorted(STOP_ORIGINS)})"
            )
        if reason is not None and reason not in STOP_REASONS:
            raise ValueError(
                f"unknown stop reason: {reason!r} (expected one of {sorted(STOP_REASONS)})"
            )
        entry = {
            "ts": time.time(),
            "event": "stop",
            "kind": kind,
            "shell": shell,
            "target": _encode_target(target),
            "elapsed": elapsed,
            "outcome": _to_jsonable(outcome),
        }
        if origin is not None:
            entry["origin"] = origin
        if reason is not None:
            entry["reason"] = reason
        if correction is not None:
            entry["correction_chars"] = len(correction)
        self._write_entry(entry)
        return entry

    def _write_entry(self, entry: dict) -> None:
        line = json.dumps(entry, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        os.chmod(self.path, 0o600)

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
