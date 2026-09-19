"""Tier measurement records: append-only JSONL with size-capped rotation.

Covers spec targets c6, c35, h30. Follows nvsh/agent/audit.py conventions:
parent directory 0o700, file 0o600, json.dumps(sort_keys=True), path under
XDG_STATE_HOME/nvsh.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from ..redact import redact

DEFAULT_CAP_BYTES = 8 * 1024 * 1024
ROTATED_FILES = 4


def default_records_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolve ``$XDG_STATE_HOME/nvsh/tiers.jsonl`` through an injectable env mapping.

    Falls back to ``$HOME/.local/state`` when ``XDG_STATE_HOME`` is unset, per the
    XDG base directory spec.
    """
    resolved_env = os.environ if env is None else env
    xdg_state_home = resolved_env.get("XDG_STATE_HOME")
    if xdg_state_home:
        base = Path(xdg_state_home)
    else:
        home = resolved_env.get("HOME") or os.path.expanduser("~")
        base = Path(home) / ".local" / "state"
    return base / "nvsh" / "tiers.jsonl"


@dataclass(frozen=True)
class TierRecord:
    """A single tier-measurement entry written to the JSONL log.

    Fields mirror the agent-first specification: tier name, the kind of
    trigger that caused the call, the operation performed, per-operation
    arguments, confidence score, round-trip latency, any decline reason,
    which tier was escalated to, the operator's final decision, and
    optionally the raw request text.
    """

    tier: str  # "needle", "lfm" or "agent"
    request_kind: str  # "failure", "slash" or "explicit"
    operation: str | None = None
    args: dict[str, str] = field(default_factory=dict)
    confidence: float | None = None
    latency_ms: float = 0.0
    decline_reason: str | None = None
    escalated_to: str | None = None
    operator_decision: str | None = None  # "approved", "declined" or None
    request_text: str | None = None


def _redact_str(value: str) -> str:
    """Redact a single string value through the redaction pipeline."""
    return redact(value.encode("utf-8")).decode("utf-8", "replace")


def _redact_entry(entry: dict) -> dict:
    """Return a copy of *entry* with every string value redacted."""
    result: dict = {}
    for k, v in entry.items():
        if isinstance(v, str):
            result[k] = _redact_str(v)
        elif isinstance(v, dict):
            result[k] = {rk: _redact_str(rv) if isinstance(rv, str) else rv for rk, rv in v.items()}
        else:
            result[k] = v
    return result


class TierRecords:
    """Appends JSON lines to a tier log, creating its parent dir 0700 and itself 0600.

    Supports size-based rotation: when the current file reaches ``cap_bytes //
    ROTATED_FILES`` the oldest file is evicted and remaining files shift up.
    Total on-disk size never exceeds ``cap_bytes``.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        cap_bytes: int = DEFAULT_CAP_BYTES,
        store_request_text: bool = False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path if path is not None else default_records_path()
        self._cap_bytes = cap_bytes
        self._store_request_text = store_request_text
        self._clock = clock
        try:
            self._ensure_dir()
        except OSError:
            # If we can't create the directory, write() will silently fail.
            pass

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)

    def write(self, record: TierRecord) -> None:
        """Append one JSON line for *record*. Never raises on OSError."""
        try:
            self._rotate_if_needed()
            entry = asdict(record)
            entry["ts"] = self._clock()

            if not self._store_request_text:
                entry.pop("request_text", None)
                entry["request_chars"] = len(record.request_text or "")
            else:
                # Redacted text kept in the entry
                entry["request_text"] = _redact_str(record.request_text or "")

            entry = _redact_entry(entry)
            line = json.dumps(entry, sort_keys=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            os.chmod(self.path, 0o600)
            self._enforce_cap()
        except OSError:
            # A failed measurement must never break the shell.
            pass

    def read_all(self) -> list[dict]:
        """Read back every recorded entry across rotated files, oldest first."""
        entries: list[dict] = []
        files = self._list_files()
        for fpath in files:
            if not fpath.exists():
                continue
            with open(fpath, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def _rotate_if_needed(self) -> None:
        """If the current file is >= cap // ROTATED_FILES, rotate out the oldest."""
        if not self.path.exists():
            return
        current_size = self.path.stat().st_size
        threshold = self._cap_bytes // ROTATED_FILES
        if current_size >= threshold:
            self._rotate()

    def _rotate(self) -> None:
        """Delete .3, rename .2 -> .3, .1 -> .2, current -> .1."""
        base = self.path
        for i in range(ROTATED_FILES - 1, 0, -1):
            src = Path(str(base) + f".{i - 1}") if i > 1 else base
            dst = Path(str(base) + f".{i}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
        # Clear the primary file (it will be re-created on next write)
        try:
            base.unlink(missing_ok=True)
        except OSError:
            pass
        # Ensure total on-disk size never exceeds the configured cap.
        self._enforce_cap()

    def _enforce_cap(self) -> None:
        """Prune oldest rotated files until total on-disk <= cap_bytes."""
        while self._total_size() > self._cap_bytes:
            self._prune_oldest()

    def _total_size(self) -> int:
        return sum(f.stat().st_size for f in self._list_files() if f.exists())

    def _prune_oldest(self) -> None:
        """Delete .3, shift .2 -> .3, .1 -> .2.  Base is untouched."""
        base = self.path
        for i in range(3, 1, -1):
            src = Path(str(base) + f".{i - 1}")
            dst = Path(str(base) + f".{i}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)

    def _list_files(self) -> list[Path]:
        """Return file paths oldest-first: .3 .. .1, then base (newest)."""
        files: list[Path] = []
        for i in range(ROTATED_FILES - 1, 0, -1):
            files.append(Path(str(self.path) + f".{i}"))
        files.append(self.path)
        return files
