"""Glob-pattern approval store for agent-proposed commands (stable-contract).

nvsh never runs an agent-suggested command without operator confirmation
("propose, don't run" — see CLAUDE.md). This module answers, for a given
candidate command line, whether it is already approved:

* ``"user"``    — matches a pattern the operator persisted (``approved.toml``).
* ``"session"`` — matches a pattern approved for this shell session only
                  (in-memory; never written to disk).
* ``"ask"``     — no match; the operator must be asked.

Patterns are ``fnmatch`` globs matched against the *full* command line
(after whitespace normalization), not just argv[0] — ``"docker ps*"``
matches ``docker ps -a`` but not ``docker exec ...``.

``add()`` refuses obviously dangerous patterns outright: a bare ``"*"``,
anything starting with ``sudo`` or ``rm`` (with or without a following
glob), so a single approval can never blanket-authorize destructive or
privilege-escalating commands.
"""

from __future__ import annotations

import fnmatch
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

#: Shipped as the starting ``user_patterns`` for a fresh install — narrow,
#: read-only-ish diagnostic commands an operator is unlikely to mind
#: auto-approving.
DEFAULT_PATTERNS: tuple[str, ...] = (
    "nvidia-smi *",
    "docker ps*",
    "journalctl *",
    "systemctl status *",
    "df *",
    "free *",
    "nvsh *",
)

_WHITESPACE_RE = re.compile(r"\s+")


class ApprovalError(ValueError):
    """Raised by :meth:`Approvals.add` when a pattern is refused."""


def _normalize(cmd: str) -> str:
    return _WHITESPACE_RE.sub(" ", cmd.strip())


def _refusal_reason(pattern: str) -> str | None:
    normalized = _normalize(pattern)
    if normalized == "*":
        return "a bare '*' would approve every command"
    first_word = normalized.split(" ", 1)[0]
    if first_word in ("sudo", "rm"):
        return f"patterns starting with '{first_word}' are never approved"
    return None


def _config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "nvsh"


def _default_path() -> Path:
    return _config_dir() / "approved.toml"


@dataclass
class Approvals:
    """Holds the persisted (``user_patterns``) and in-memory-only
    (``session_patterns``) approval lists for one nvsh process."""

    path: Path
    user_patterns: list[str] = field(default_factory=list)
    session_patterns: list[str] = field(default_factory=list)

    @classmethod
    def default(cls, path: Path | None = None) -> "Approvals":
        """A fresh store seeded with :data:`DEFAULT_PATTERNS`, not read from disk."""
        return cls(path=path or _default_path(), user_patterns=list(DEFAULT_PATTERNS))

    @classmethod
    def load(cls, path: Path | None = None) -> "Approvals":
        """Load ``user_patterns`` from ``path`` (default: XDG approved.toml).

        A missing file yields :meth:`default`. Session patterns are always
        empty on load — they never persist.
        """
        target = path or _default_path()
        if not target.is_file():
            return cls.default(path=target)
        try:
            raw = tomllib.loads(target.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ApprovalError(f"malformed TOML in {target}: {exc}") from exc
        patterns = raw.get("user_patterns", [])
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            raise ApprovalError(f"{target}: 'user_patterns' must be a list of strings")
        return cls(path=target, user_patterns=list(patterns))

    def save(self) -> None:
        """Write ``user_patterns`` to :attr:`path` as 0600 TOML. Never writes session patterns."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["user_patterns = ["]
        for pattern in self.user_patterns:
            escaped = pattern.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'  "{escaped}",')
        lines.append("]")
        text = "\n".join(lines) + "\n"

        # Create/truncate with 0600 from the start (no world/group-readable window).
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
        finally:
            pass
        os.chmod(self.path, 0o600)

    def add(self, pattern: str, scope: str = "user") -> None:
        """Add ``pattern`` to ``user_patterns`` or ``session_patterns``.

        Raises :class:`ApprovalError` (a :class:`ValueError`) for a pattern
        matching the refusal rules, or for an unknown ``scope``, without
        modifying either list.
        """
        if scope not in ("user", "session"):
            raise ApprovalError(f"unknown scope: {scope!r} (expected 'user' or 'session')")
        reason = _refusal_reason(pattern)
        if reason is not None:
            raise ApprovalError(f"refusing to approve {pattern!r}: {reason}")
        normalized_pattern = _normalize(pattern)
        target = self.user_patterns if scope == "user" else self.session_patterns
        if normalized_pattern not in target:
            target.append(normalized_pattern)

    def remove(self, pattern: str) -> None:
        """Remove ``pattern`` from both lists if present (idempotent)."""
        if pattern in self.user_patterns:
            self.user_patterns.remove(pattern)
        if pattern in self.session_patterns:
            self.session_patterns.remove(pattern)

    def matches(self, cmd: str) -> tuple[str, str | None]:
        """Return ``(scope, pattern)`` for the first match, checking user before session.

        ``scope`` is ``"user"``, ``"session"`` or ``"ask"``; ``pattern`` is the
        matching glob, or ``None`` when nothing matched.
        """
        normalized = _normalize(cmd)
        for pattern in self.user_patterns:
            if fnmatch.fnmatchcase(normalized, pattern):
                return "user", pattern
        for pattern in self.session_patterns:
            if fnmatch.fnmatchcase(normalized, pattern):
                return "session", pattern
        return "ask", None

    def decide(self, cmd: str) -> str:
        """Return ``"user"``, ``"session"`` or ``"ask"`` for ``cmd``."""
        scope, _ = self.matches(cmd)
        return scope
