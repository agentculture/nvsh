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

Patterns are ``fnmatch`` globs matched against one *stage* of the command
line (after whitespace normalization), not just argv[0] — ``"docker ps*"``
matches ``docker ps -a`` but not ``docker exec ...``.

**Stages, and why every one of them has to match (deviation d24).** A
command line is split on ``|``, ``&&``, ``||``, ``&``, ``;`` and newlines,
honouring quotes, so ``ssh orin "ps | head"`` is one stage whose first
argument is ``orin``. :meth:`Approvals.matches` requires *every* stage to
match some approved pattern: a broad ``ls *`` can never pull
``ls | sudo tee /etc/x`` through, and :meth:`Approvals.unapproved_stage`
names the stage that is holding the line back. A line carrying a subshell
or a command substitution (``(...)``, ``$(...)``, backticks) is opaque —
what it really runs is not in its own text — so it is treated as one stage
that no pattern may ever match.

**Four scopes, two forms (deviation d24).** An operator on the Spark was
offered ``'ssh *'`` for ``ssh orin "ps -eo ... | head -n 20"`` and found it
far too broad. :func:`pattern_for` therefore knows two forms of each scope:
``session``/``user`` are the d15 keys (the exact line / ``<first word> *``),
and ``session-specific``/``user-specific`` keep the first argument as well
(``ssh orin *``). :func:`patterns_for` applies the chosen form to every
stage, and it is the one helper the panel's scope line, the client's store
write, the details view and — through ``nvsh approve add --scope`` — the
pi approval extension all share.

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
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from nvsh import runtimedir

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

    Deviation d24 widens it in two directions. It is applied to *every*
    stage of the line, not only to the first, so a broad ``ls *`` can never
    pull ``ls | sudo tee /etc/x`` through; and a line carrying a subshell or
    a command substitution is refused outright, because what it really runs
    is not visible in its own text.
    """
    if not _normalize(cmd):
        return None
    if is_opaque(cmd):
        return "a subshell or command substitution is never auto-approved"
    for stage in stages(cmd):
        first_word = stage.split(" ", 1)[0]
        if first_word in _NEVER_APPROVED_EXECUTABLES:
            return f"'{first_word}' commands are never auto-approved"
    return None


# ---------------------------------------------------------------------------
# stages and patterns (deviation d24)
# ---------------------------------------------------------------------------

#: The scope tokens :func:`pattern_for` and :meth:`Approvals.add` accept.
#: ``session``/``user`` are the d15 keys (``[s]``/``[u]``); the ``-specific``
#: pair are d24's uppercase keys (``[S]``/``[U]``), which keep the command's
#: *first argument* in the pattern.
SCOPES: tuple[str, ...] = ("session", "session-specific", "user", "user-specific")

#: Characters that make a line opaque: whatever is inside them is a command
#: nvsh cannot see, so the line is one stage that is never auto-approved.
#: Backslash-escaped parens (``find ... \( ... \)``) are literal and do not
#: count -- the scanner skips an escaped character without inspecting it.
_OPAQUE_CHARS = "()`"


def _scan(text: str):
    """Yield ``(index, char, in_quote)`` over ``text``, honouring shell quoting.

    One scanner for all three d24 walks (opaque detection, stage splitting,
    raw-word splitting) so they can never disagree about what is quoted.
    ``in_quote`` is the quote character currently open, or ``None``.
    """
    quote: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < n:
                yield i, ch, quote
                yield i + 1, text[i + 1], quote
                i += 2
                continue
            yield i, ch, quote
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            yield i, ch, None
            yield i + 1, text[i + 1], "\\"  # escaped: data, never a separator
            i += 2
            continue
        if ch in "'\"":
            quote = ch
            yield i, ch, None
            i += 1
            continue
        yield i, ch, None
        i += 1


def is_opaque(command: str) -> bool:
    """Does ``command`` contain a subshell, ``$(...)`` or a backtick?

    Such a line is treated conservatively as one stage that no approval may
    ever match: the commands it actually runs are inside the parentheses,
    where a per-stage pattern cannot see them.
    """
    return any(ch in _OPAQUE_CHARS for _i, ch, quoted in _scan(command) if quoted is None)


def split_stages(command: str) -> list[str]:
    """Split ``command`` into stages on ``|``, ``&&``, ``||``, ``&``, ``;``, newline.

    Quoting is honoured, so ``ssh orin "ps | head"`` is ONE stage. Each
    stage comes back whitespace-normalized, and empty stages (a trailing
    ``;``, ``|&``) are dropped. This is the raw splitter; :func:`stages` is
    what callers want, because it also handles the opaque case.
    """
    parts: list[str] = []
    current: list[str] = []
    skip = -1
    chars = list(_scan(command))
    for index, ch, quoted in chars:
        if index == skip:
            continue
        if quoted is not None:
            current.append(ch)
            continue
        if ch in (";", "\n"):
            parts.append("".join(current))
            current = []
            continue
        if ch in ("|", "&"):
            following = command[index + 1 : index + 2]
            if following == ch:
                skip = index + 1
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return [stage for stage in (_normalize(part) for part in parts) if stage]


def stages(command: str) -> list[str]:
    """The stages of ``command``: one per pipeline/list element (d24).

    An opaque line (:func:`is_opaque`) is deliberately *not* split -- it
    comes back as the single normalized line, which
    :func:`command_refusal_reason` then refuses outright.
    """
    if is_opaque(command):
        normalized = _normalize(command)
        return [normalized] if normalized else []
    return split_stages(command)


def _raw_words(text: str, limit: int) -> list[str]:
    """The first ``limit`` whitespace-separated words of ``text``, quotes kept.

    Kept *raw* (``'"my host"'`` stays quoted) because the result goes into
    an ``fnmatch`` pattern matched against the raw, whitespace-normalized
    command line -- a de-quoted word would never match it.
    """
    words: list[str] = []
    current: list[str] = []
    for _index, ch, quoted in _scan(text):
        if len(words) >= limit:
            return words
        if quoted is None and ch == " ":
            if current:
                words.append("".join(current))
                current = []
            continue
        current.append(ch)
    if current and len(words) < limit:
        words.append("".join(current))
    return words


def base_scope(scope: str) -> str:
    """``"session"`` or ``"user"`` -- which list ``scope`` writes into."""
    if scope not in SCOPES:
        raise ApprovalError(f"unknown scope: {scope!r} (expected one of {', '.join(SCOPES)})")
    return "session" if scope.startswith("session") else "user"


def pattern_for(command: str, scope: str) -> str:
    """The glob ``scope`` would store for the single stage ``command``.

    The one pure helper every caller shares -- the panel's scope line, the
    client's store write, the details view and (through
    ``nvsh approve add --scope``) the pi approval extension:

    * ``session``          -- the exact line, unwidened.
    * ``user``             -- ``"<first word> *"``: this *kind* of command.
    * ``session-specific`` /
      ``user-specific``    -- ``"<first word> <first argument> *"``, d24's
      answer to an operator on the Spark who was offered ``'ssh *'`` for
      ``ssh orin "..."`` and found it far too broad.

    When the line has no second word there is nothing to be specific about,
    so the ``-specific`` scopes fall back to their plain form.
    """
    base = base_scope(scope)
    normalized = _normalize(command)
    if not normalized:
        return normalized
    words = _raw_words(normalized, 2)
    first = words[0]
    if not scope.endswith("-specific"):
        return normalized if base == "session" else f"{first} *"
    if len(words) < 2:
        return normalized if base == "session" else f"{first} *"
    return f"{first} {words[1]} *"


def patterns_for(command: str, scope: str) -> list[str]:
    """One pattern per stage of ``command``, in order, deduplicated.

    ``patterns_for("ps -eo pid | head -n 20", "user")`` is
    ``["ps *", "head *"]``: approving a pipeline approves each of its
    stages, and :meth:`Approvals.matches` then requires every stage of a
    later candidate to match something.
    """
    out: list[str] = []
    for stage in stages(command):
        pattern = pattern_for(stage, scope)
        if pattern and pattern not in out:
            out.append(pattern)
    return out


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
    return runtimedir.fallback_dir(uid)


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
        target_scope = base_scope(scope)
        reason = refusal_reason(pattern)
        if reason is not None:
            raise ApprovalError(f"refusing to approve {pattern!r}: {reason}")
        normalized_pattern = _normalize(pattern)
        target = self.user_patterns if target_scope == "user" else self.session_patterns
        if normalized_pattern not in target:
            target.append(normalized_pattern)
        if target_scope == "session":
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

    def _match_stage(self, stage: str) -> tuple[str, str | None]:
        """``(scope, pattern)`` for one stage, user patterns before session ones."""
        for pattern in self.user_patterns:
            if fnmatch.fnmatchcase(stage, pattern):
                return "user", pattern
        for pattern in self.session_patterns:
            if fnmatch.fnmatchcase(stage, pattern):
                return "session", pattern
        return "ask", None

    def matches(self, cmd: str) -> tuple[str, str | None]:
        """Return ``(scope, pattern)`` for ``cmd``, requiring *every* stage to match.

        ``scope`` is ``"user"``, ``"session"`` or ``"ask"``; ``pattern`` is the
        matching glob (the matching globs joined by ``" | "`` when the line
        has more than one stage), or ``None`` when nothing matched.

        Deviation d24: a command line is matched per stage, not as one
        string. ``ls | sudo tee /etc/x`` is not covered by ``ls *`` -- the
        second stage has to be approved on its own, and it never can be.
        The reported scope is the *narrower* lifetime in play, so a line
        whose stages are half session-approved reads as ``session``.
        """
        if command_refusal_reason(cmd) is not None:
            # No stored pattern, however it got there, may pre-authorize a
            # privileged, destructive or opaque command: the operator is
            # always asked.
            return "ask", None
        stage_list = stages(cmd)
        if not stage_list:
            return "ask", None
        scopes: list[str] = []
        patterns: list[str] = []
        for stage in stage_list:
            scope, pattern = self._match_stage(stage)
            if scope == "ask" or pattern is None:
                return "ask", None
            scopes.append(scope)
            patterns.append(pattern)
        return ("session" if "session" in scopes else "user"), " | ".join(patterns)

    def unapproved_stage(self, cmd: str) -> str | None:
        """The first stage of ``cmd`` that is not approved, or ``None``.

        What ``nvsh approve check`` reports so an operator (or the pi
        extension) can see *which* part of a pipeline is holding the line
        back, instead of a bare ``ask`` for the whole thing.
        """
        stage_list = stages(cmd)
        if not stage_list:
            return None
        if is_opaque(cmd):
            return stage_list[0]
        for stage in stage_list:
            if stage.split(" ", 1)[0] in _NEVER_APPROVED_EXECUTABLES:
                return stage
        for stage in stage_list:
            if self._match_stage(stage)[0] == "ask":
                return stage
        return None

    def decide(self, cmd: str) -> str:
        """Return ``"user"``, ``"session"`` or ``"ask"`` for ``cmd``."""
        scope, _ = self.matches(cmd)
        return scope
