"""Pure functions over rc-file text: where ``nvsh setup`` inserts its block, and how
it removes it again.

Nothing here touches the filesystem except :class:`RcPath` and the two backup
helpers, and every one of those goes through :meth:`RcPath.validate` — the
single validation choke point for the operator-supplied ``--rc PATH`` (see
"Validating ``--rc``" below). Every other function is a pure ``text -> text``
transform so it can be tested without ever reading the real ``$HOME``.

The block is delimited by ``MARK_START_PREFIX`` (carrying a short content
hash of its own body) and ``MARK_END``. :func:`insert_block` and
:func:`remove_block` are exact inverses of each other when *block* is passed
back unchanged: ``remove_block(insert_block(text, block)) == (text, True,
False)``. That is what lets ``nvsh setup`` be idempotent (a second run
detects the identical block already present and writes nothing) and what
lets ``nvsh uninstall`` restore the rc byte-for-byte when the block was
never hand-edited.

Validating ``--rc``
-------------------

``--rc PATH`` is operator input that ends up in an ``open()``, so it is
validated once, in one place, before any filesystem call sees it (SonarCloud
``pythonsecurity:S2083``). :meth:`RcPath.validate` refuses:

* an empty path, or one containing a ``NUL`` byte;
* any ``..`` component (no traversal, lexically, before resolution);
* a path *inside* ``$HOME`` whose symlinks resolve *outside* ``$HOME``
  (the classic "point ~/.bashrc at someone else's file" trick);
* a path outside ``$HOME`` that the calling uid does not own, or whose
  directory is world-writable without the sticky bit (so ``--rc`` can still
  name a file the operator genuinely owns elsewhere, but never a drop box
  another user can swap under us);
* anything that exists but is not a regular file.

Everything that survives is handed back as an :class:`RcPath`, and the
``read``/``write``/``backup`` methods on that object are the only way the rc
file is touched. Those methods never hand a path to the filesystem at all:
they open the validated home directory and then open the file *relative to
that descriptor* by its bare name (:func:`_open_in_dir`), so no directory
component an operator typed ever reaches an ``open()`` and no parent can be
swapped in between the validation and the write.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

#: Opening marker line prefix; the content hash is appended after it.
MARK_START_PREFIX = "# >>> nvsh setup >>>"
#: Closing marker line, verbatim.
MARK_END = "# <<< nvsh setup <<<"

_CASE_DASH_RE = re.compile(r"^\s*case\s+\$-\s+in\b")
_SINGLE_LINE_DASH_RE = re.compile(r"\$-.*\*i\*")
_HASH_RE = re.compile(r"sha256:([0-9a-f]+)")


def _hash_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def build_block(body_lines: list[str]) -> str:
    """Render the marked block from *body_lines* (the lines between the markers).

    The opening marker carries a short ``sha256:`` digest of the body so
    :func:`remove_block` can tell whether the block was hand-edited since it
    was written.
    """
    body = "\n".join(body_lines) + "\n"
    digest = _hash_body(body)
    return f"{MARK_START_PREFIX} sha256:{digest}\n{body}{MARK_END}\n"


def has_block(text: str) -> bool:
    """Does *text* contain a (possibly edited) nvsh setup block?"""
    return MARK_START_PREFIX in text and MARK_END in text


def find_insert_point(text: str) -> int:
    """Return the line index right after the rc's interactive guard.

    Detected as either a ``case $- in ... esac`` block or a single-line test
    on ``$-`` for interactivity (``[[ $- != *i* ]] && return`` and similar).
    Falls back to the top of the file (index ``0``) when neither is found —
    this also covers an empty file.
    """
    lines = text.splitlines(keepends=True)

    for i, line in enumerate(lines):
        if _CASE_DASH_RE.match(line):
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "esac":
                    return j + 1
            return len(lines)

    for i, line in enumerate(lines):
        if _SINGLE_LINE_DASH_RE.search(line):
            return i + 1

    return 0


def insert_block(text: str, block: str) -> str:
    """Insert *block* (as built by :func:`build_block`) at :func:`find_insert_point`."""
    idx = find_insert_point(text)
    lines = text.splitlines(keepends=True)
    return "".join(lines[:idx]) + block + "".join(lines[idx:])


def remove_block(text: str) -> tuple[str, bool, bool]:
    """Strip the marked block out of *text*.

    Returns ``(new_text, removed, edited)``:

    * ``removed`` is ``False`` (and *text* is returned unchanged) when no
      block is present at all.
    * ``edited`` is ``True`` when the body's actual content hash no longer
      matches the hash recorded on the opening marker — i.e. someone hand-
      edited the block after ``nvsh setup`` wrote it.
    """
    lines = text.splitlines(keepends=True)
    start_i = end_i = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(MARK_START_PREFIX):
            start_i = i
        elif start_i is not None and line.strip() == MARK_END:
            end_i = i
            break

    if start_i is None or end_i is None:
        return text, False, False

    body_lines = lines[start_i + 1 : end_i]
    declared = _HASH_RE.search(lines[start_i])
    declared_hash = declared.group(1) if declared else None
    computed_hash = _hash_body("".join(body_lines))
    edited = declared_hash != computed_hash

    new_text = "".join(lines[:start_i]) + "".join(lines[end_i + 1 :])
    return new_text, True, edited


# --------------------------------------------------------------------------
# the validated rc path
# --------------------------------------------------------------------------


class RcPathError(ValueError):
    """An ``--rc`` path that nvsh refuses to read, write or back up."""


def _within(candidate: Path, root: Path) -> bool:
    """Is *candidate* *root* itself or below it? Purely lexical on two absolute paths."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _home_root(home: Path | str | None) -> Path:
    """The user's home, symlinks resolved. ``$HOME`` when set, else the passwd entry."""
    raw = Path(home) if home is not None else Path(os.path.expanduser("~"))
    return Path(os.path.realpath(raw))


def _require_caller_owns(target: Path) -> None:
    """Refuse a path outside ``$HOME`` unless the caller owns it in a safe directory."""
    probe = target if target.exists() else target.parent
    try:
        info = os.stat(probe)
        parent_info = os.stat(target.parent)
    except OSError as exc:
        raise RcPathError(f"cannot use {target} as an rc file: {exc}") from exc
    if info.st_uid != os.getuid():
        raise RcPathError(
            f"refusing to edit {target}: it is outside your home and you do not own it"
        )
    # A world-writable directory is a drop box: anyone could swap the file
    # under us between the check and the write. The sticky bit (as on /tmp)
    # takes that away again, since only the owner may then replace entries.
    world_writable = bool(parent_info.st_mode & stat.S_IWOTH)
    sticky = bool(parent_info.st_mode & stat.S_ISVTX)
    if world_writable and not sticky:
        raise RcPathError(f"refusing to edit {target}: its directory is world-writable")


#: Flags for the rc/backup writes below: create or truncate, write-only.
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def _open_in_dir(directory: Path, name: str, flags: int, mode: int = 0o666) -> int:
    """Open the file *name* **inside** *directory*, by name only.

    *directory* is opened first and the file is then opened relative to that
    descriptor, so the only thing the kernel resolves for the file itself is
    a bare name with no directory component in it: nothing an operator typed
    can steer the open somewhere else, and the parent cannot be swapped
    between :meth:`RcPath.validate` and the write. *name* is rejected
    outright if it is not a bare name (this is belt and braces --
    :meth:`RcPath.validate` has already reduced the path to one).
    """
    if not name or os.sep in name or name in (os.curdir, os.pardir):
        raise RcPathError(f"refusing to use {name!r} as an rc file name")
    dir_fd = os.open(os.fspath(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        return os.open(name, flags, mode, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def _read_in_dir(directory: Path, name: str) -> str:
    """Read ``<directory>/<name>`` through :func:`_open_in_dir`."""
    with os.fdopen(_open_in_dir(directory, name, os.O_RDONLY), "r", encoding="utf-8") as handle:
        return handle.read()


def _write_in_dir(directory: Path, name: str, text: str) -> None:
    """Write *text* to ``<directory>/<name>`` through :func:`_open_in_dir`."""
    with os.fdopen(_open_in_dir(directory, name, _WRITE_FLAGS), "w", encoding="utf-8") as handle:
        handle.write(text)


@dataclass(frozen=True)
class RcPath:
    """A ``--rc`` path that has passed :meth:`validate`, plus the I/O nvsh does on it.

    Construct it only through :meth:`validate`. ``path`` is the lexical
    (``..``-free, absolute) path so diagnostics quote what the operator
    typed; ``real`` is the fully symlink-resolved path the checks ran
    against; ``root`` is the validated home directory every read and write
    is performed *inside* (see :func:`_open_in_dir`).
    """

    path: Path
    real: Path
    root: Path

    # -- construction ------------------------------------------------------

    @classmethod
    def validate(cls, raw: "RcPath | Path | str", *, home: Path | str | None = None) -> "RcPath":
        """Validate *raw* and return the :class:`RcPath` every rc read/write goes through."""
        if isinstance(raw, RcPath):
            return raw

        text = os.fspath(raw)
        if not text.strip():
            raise RcPathError("rc path must not be empty")
        if "\x00" in text:
            raise RcPathError("rc path must not contain a NUL byte")

        candidate = Path(text).expanduser()
        if any(part == os.pardir for part in candidate.parts):
            raise RcPathError(f"rc path must not contain '..' components: {text}")

        lexical = Path(os.path.abspath(candidate))
        if any(part == os.pardir for part in lexical.parts):  # pragma: no cover - belt and braces
            raise RcPathError(f"rc path must not contain '..' components: {text}")
        real = Path(os.path.realpath(lexical))
        root = _home_root(home)

        # The rc file lives directly in the operator's home (c36: the block
        # goes into $HOME/.bashrc). Anything else is refused, and the path the
        # I/O uses is rebuilt from the trusted root plus the file's *name* so
        # no operator-supplied directory component ever reaches open().
        if not _within(lexical, root) or lexical.parent != root:
            raise RcPathError(
                f"refusing to edit {lexical}: the rc file must sit directly under {root}"
            )
        if not _within(real, root):
            raise RcPathError(f"refusing to edit {lexical}: it is a symlink leading outside {root}")
        real = root / os.path.basename(os.fspath(real))

        if real.exists():
            if not real.is_file():
                raise RcPathError(f"refusing to edit {lexical}: it is not a regular file")
        elif real.parent.exists() and not real.parent.is_dir():
            raise RcPathError(f"cannot create {lexical}: {real.parent} is not a directory")

        return cls(path=lexical, real=real, root=root)

    # -- I/O ---------------------------------------------------------------

    def exists(self) -> bool:
        """Is there a regular file here right now?"""
        return self.real.is_file()

    def read_text(self) -> str:
        """The rc file's current text, or ``""`` when it does not exist yet."""
        if not self.exists():
            return ""
        return _read_in_dir(self.root, self.real.name)

    def write_text(self, text: str) -> None:
        """Write *text* inside the validated root, addressed by file name only."""
        self.root.mkdir(parents=True, exist_ok=True)
        _write_in_dir(self.root, self.real.name, text)

    def write_backup(self, text: str) -> Path:
        """Timestamped backup of *text* next to the rc file. See :func:`write_backup`."""
        return _write_backup(self.root, self.real.name, text)

    def newest_backup(self) -> Path | None:
        """The most recently written backup for this rc file, or ``None``."""
        pattern = f"{self.real.name}.nvsh-backup-*"
        candidates = sorted(self.root.glob(pattern))
        return candidates[-1] if candidates else None

    def __str__(self) -> str:
        return str(self.path)


# --------------------------------------------------------------------------
# backups
# --------------------------------------------------------------------------


def _write_backup(root: Path, rc_name: str, text: str) -> Path:
    """Write *text* beside an **already validated** rc file.

    Takes the validated *root* and the rc file's bare *name*, never a whole
    path, and writes through :func:`_write_in_dir` — the backup lands inside
    the same directory the rc file was validated against, by construction.

    Private on purpose: the only callers are :meth:`RcPath.write_backup` and
    the thin :func:`write_backup` shim below, both of which have run
    :meth:`RcPath.validate` first.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{rc_name}.nvsh-backup-{timestamp}"
    suffix = 1
    while (root / name).exists():
        name = f"{rc_name}.nvsh-backup-{timestamp}-{suffix}"
        suffix += 1
    _write_in_dir(root, name, text)
    return root / name


def write_backup(rc_path: "RcPath | Path | str", text: str) -> Path:
    """Write *text* to a timestamped ``<rc>.nvsh-backup-<YYYYmmdd-HHMMSS>`` sibling.

    Validates *rc_path* first (a bare ``Path`` is coerced through
    :meth:`RcPath.validate`), so there is no way to back up through an
    unvalidated path. Never collides with an existing backup (appends
    ``-1``, ``-2``, ... on a same-second re-run, which matters in fast test
    loops).
    """
    return RcPath.validate(rc_path).write_backup(text)


def newest_backup(rc_path: "RcPath | Path | str") -> Path | None:
    """The most recently written backup for *rc_path*, or ``None``."""
    return RcPath.validate(rc_path).newest_backup()
