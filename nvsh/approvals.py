"""Glob-pattern approval store for agent-proposed commands (stable-contract).

nvsh never runs an agent-suggested command without operator confirmation
("propose, don't run" — see CLAUDE.md). This module answers, for a given
candidate command line, whether it is already approved:

* ``"user"``    — matches a pattern the operator persisted (``approved.toml``).
* ``"session"`` — matches a pattern approved for this *login session* only
                  (``$XDG_RUNTIME_DIR/nvsh/session-approvals.toml``).
* ``"ask"``     — no match; the operator must be asked.

**Where session approvals live (deviation d15).** They used to be a plain
in-memory list, which made the scope a no-op in practice: every caller that
approves one — ``nvsh approve add <cmd> --session`` (the pi approval
extension shells out to it) and each ``nvsh hook`` invocation — is a
throwaway process, so the pattern died before the next command could match
it, and the operator was asked again immediately. They are now written to
``$XDG_RUNTIME_DIR/nvsh/session-approvals.toml`` (mode 0600, dir 0700),
which is exactly the lifetime the word "session" promises: the runtime dir
is per-user tmpfs that systemd creates at login and removes at logout, so
the approvals die with the login session and never survive a reboot. They
are *never* written to ``approved.toml`` — only ``user`` scope persists
across logins. When ``XDG_RUNTIME_DIR`` is unset the store falls back to
``/run/user/<uid>`` if it exists and otherwise to a per-uid directory under
the system temp dir, so nvsh keeps working on a bare ssh into a Jetson
without ever quietly promoting a session approval to a permanent one.

Patterns are ``fnmatch`` globs matched against the *full* command line
(after whitespace normalization), not just argv[0] — ``"docker ps*"``
matches ``docker ps -a`` but not ``docker exec ...``.

``add()`` refuses obviously dangerous patterns outright: a bare ``"*"``, and
anything whose executable token can *match* ``sudo`` or ``rm`` -- literally
(``"sudo reboot"``) or through a glob (``"sudo*"``, ``"rm*"``, ``"* -rf /"``)
-- so a single approval can never blanket-authorize destructive or
privilege-escalating commands. :meth:`Approvals.matches` enforces the same
policy again on the candidate command, so a pattern that reached the store
some other way (a hand-edited ``approved.toml``) still cannot auto-approve
``sudo rm -rf /``.
"""

from __future__ import annotations

import fnmatch
import os
import re
import tempfile
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

#: Executables no approval -- pattern or stored -- may ever authorize.
_NEVER_APPROVED_EXECUTABLES: tuple[str, ...] = ("sudo", "rm")


class ApprovalError(ValueError):
    """Raised by :meth:`Approvals.add` when a pattern is refused."""


def _normalize(cmd: str) -> str:
    return _WHITESPACE_RE.sub(" ", cmd.strip())


def refusal_reason(pattern: str) -> str | None:
    """Why ``pattern`` may never be approved, or ``None`` when it may.

    Public so a caller can ask *before* offering the operator a scope key
    (the panel's ``[s]``/``[u]``) instead of letting :meth:`Approvals.add`
    raise after the fact.
    """
    normalized = _normalize(pattern)
    if normalized == "*":
        return "a bare '*' would approve every command"
    first_word = normalized.split(" ", 1)[0]
    # The executable token is a glob too: comparing it literally let `sudo*`
    # and `rm*` (and a bare `*` in argv[0]) through, and `matches()` would
    # then happily auto-approve `sudo rm -rf /`. Refuse any first token that
    # *can match* a forbidden executable, not just one that equals it.
    for forbidden in _NEVER_APPROVED_EXECUTABLES:
        if first_word == forbidden or fnmatch.fnmatchcase(forbidden, first_word):
            return f"patterns starting with '{forbidden}' are never approved"
    return None


def command_refusal_reason(cmd: str) -> str | None:
    """Why ``cmd`` may never be *auto*-approved, whatever patterns are stored.

    The same policy as :func:`refusal_reason`, applied to the candidate
    command rather than to the pattern, so a pattern that reached the store
    some other way (a hand-edited ``approved.toml``, a file written by an
    older nvsh) still cannot authorize a privileged or destructive command.
    """
    normalized = _normalize(cmd)
    if not normalized:
        return None
    first_word = normalized.split(" ", 1)[0]
    if first_word in _NEVER_APPROVED_EXECUTABLES:
        return f"'{first_word}' commands are never auto-approved"
    return None


#: Backwards-compatible private alias (this module's own callers).
_refusal_reason = refusal_reason


def _config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "nvsh"


def _default_path() -> Path:
    return _config_dir() / "approved.toml"


def runtime_dir() -> Path:
    """``$XDG_RUNTIME_DIR/nvsh`` — the login session's own scratch directory.

    Falls back to ``/run/user/<uid>`` when the variable is unset but the
    directory exists (a bare ``ssh host`` often loses the variable), and to
    a per-uid directory under the system temp dir otherwise. Never falls
    back to a persistent location: a session approval must not outlive the
    session just because an environment variable went missing.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "nvsh"
    uid = os.getuid()
    candidate = Path(f"/run/user/{uid}")
    if candidate.is_dir():
        return candidate / "nvsh"
    return Path(tempfile.gettempdir()) / f"nvsh-{uid}"


def _default_session_path() -> Path:
    return runtime_dir() / "session-approvals.toml"


def _read_pattern_list(path: Path, key: str) -> list[str]:
    """Read ``key`` (a list of strings) out of the TOML at ``path``."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ApprovalError(f"malformed TOML in {path}: {exc}") from exc
    patterns = raw.get(key, [])
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
        raise ApprovalError(f"{path}: {key!r} must be a list of strings")
    return list(patterns)


def _write_pattern_list(path: Path, key: str, patterns: list[str]) -> None:
    """Write ``patterns`` as 0600 TOML under ``key``, creating the dir 0700."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:  # pragma: no cover - a shared dir we do not own
        pass
    lines = [f"{key} = ["]
    for pattern in patterns:
        escaped = pattern.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  "{escaped}",')
    lines.append("]")
    text = "\n".join(lines) + "\n"

    # Create/truncate with 0600 from the start (no world/group-readable window).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


@dataclass
class Approvals:
    """Holds the login-persistent (``user_patterns``) and session-scoped
    (``session_patterns``) approval lists.

    Both lists are backed by a file: ``path`` (``approved.toml``, under the
    XDG *config* dir) and ``session_path`` (``session-approvals.toml``,
    under the XDG *runtime* dir). Only the second one goes away at logout.
    """

    path: Path
    user_patterns: list[str] = field(default_factory=list)
    session_patterns: list[str] = field(default_factory=list)
    session_path: Path | None = None

    def _session_file(self) -> Path:
        return self.session_path if self.session_path is not None else _default_session_path()

    @classmethod
    def default(cls, path: Path | None = None, session_path: Path | None = None) -> "Approvals":
        """A fresh store seeded with :data:`DEFAULT_PATTERNS`, not read from disk."""
        return cls(
            path=path or _default_path(),
            user_patterns=list(DEFAULT_PATTERNS),
            session_path=session_path or _default_session_path(),
        )

    @classmethod
    def load(cls, path: Path | None = None, session_path: Path | None = None) -> "Approvals":
        """Load both lists from disk (defaults: the XDG config/runtime paths).

        A missing ``approved.toml`` yields the :data:`DEFAULT_PATTERNS`. A
        missing session file simply means no session approvals are in force
        — a fresh login, or a runtime dir the system wiped. A *malformed*
        session file is ignored rather than fatal: it must never be able to
        stop nvsh from asking the operator (deviation d15).
        """
        target = path or _default_path()
        session_target = session_path or _default_session_path()
        if not target.is_file():
            approvals = cls.default(path=target, session_path=session_target)
        else:
            approvals = cls(
                path=target,
                user_patterns=_read_pattern_list(target, "user_patterns"),
                session_path=session_target,
            )
        if session_target.is_file():
            try:
                approvals.session_patterns = _read_pattern_list(session_target, "session_patterns")
            except ApprovalError:
                approvals.session_patterns = []
        return approvals

    def save(self) -> None:
        """Write ``user_patterns`` to :attr:`path` as 0600 TOML, and the
        session list to its own runtime-dir file. ``approved.toml`` never
        receives a session pattern."""
        _write_pattern_list(self.path, "user_patterns", self.user_patterns)
        self.save_session()

    def save_session(self) -> None:
        """Write ``session_patterns`` to the runtime-dir file (best effort).

        Best effort on purpose: a read-only or missing runtime dir must
        degrade to "the operator gets asked again", never to a traceback at
        the prompt.
        """
        try:
            _write_pattern_list(self._session_file(), "session_patterns", self.session_patterns)
        except OSError:
            return

    def add(self, pattern: str, scope: str = "user") -> None:
        """Add ``pattern`` to ``user_patterns`` or ``session_patterns``.

        A ``session`` addition is written through to the runtime-dir file
        immediately, because every caller that makes one is a throwaway
        process (``nvsh approve add --session``, one ``nvsh hook`` run) and
        an in-memory-only session scope would never match anything
        (deviation d15). A ``user`` addition still needs an explicit
        :meth:`save`, as before.

        Raises :class:`ApprovalError` (a :class:`ValueError`) for a pattern
        matching the refusal rules, or for an unknown ``scope``, without
        modifying either list.
        """
        if scope not in ("user", "session"):
            raise ApprovalError(f"unknown scope: {scope!r} (expected 'user' or 'session')")
        reason = refusal_reason(pattern)
        if reason is not None:
            raise ApprovalError(f"refusing to approve {pattern!r}: {reason}")
        normalized_pattern = _normalize(pattern)
        target = self.user_patterns if scope == "user" else self.session_patterns
        if normalized_pattern not in target:
            target.append(normalized_pattern)
        if scope == "session":
            self.save_session()

    def remove(self, pattern: str) -> bool:
        """Remove ``pattern`` from both lists if present (idempotent).

        Normalizes the argument the way :meth:`add` normalizes what it
        stores, so ``remove("kubectl get  *")`` really does drop the stored
        ``"kubectl get *"``. Returns whether anything was removed, so the
        CLI can tell "removed" from "no such approval" instead of always
        reporting success.
        """
        normalized = _normalize(pattern)
        removed = False
        for target in (self.user_patterns, self.session_patterns):
            for candidate in (pattern, normalized):
                if candidate in target:
                    target.remove(candidate)
                    removed = True
        return removed

    def matches(self, cmd: str) -> tuple[str, str | None]:
        """Return ``(scope, pattern)`` for the first match, checking user before session.

        ``scope`` is ``"user"``, ``"session"`` or ``"ask"``; ``pattern`` is the
        matching glob, or ``None`` when nothing matched.
        """
        normalized = _normalize(cmd)
        if command_refusal_reason(normalized) is not None:
            # No stored pattern, however it got there, may pre-authorize a
            # privileged or destructive command: the operator is always asked.
            return "ask", None
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
