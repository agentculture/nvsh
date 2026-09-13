"""Pure functions over rc-file text: where ``nvsh setup`` inserts its block, and how
it removes it again.

Nothing here touches the filesystem except :func:`write_backup` and
:func:`newest_backup` (both operate on an already-resolved ``Path``, never a
hard-coded ``~``). Every other function is a pure ``text -> text`` transform
so it can be tested without ever reading the real ``$HOME``.

The block is delimited by ``MARK_START_PREFIX`` (carrying a short content
hash of its own body) and ``MARK_END``. :func:`insert_block` and
:func:`remove_block` are exact inverses of each other when *block* is passed
back unchanged: ``remove_block(insert_block(text, block)) == (text, True,
False)``. That is what lets ``nvsh setup`` be idempotent (a second run
detects the identical block already present and writes nothing) and what
lets ``nvsh uninstall`` restore the rc byte-for-byte when the block was
never hand-edited.
"""

from __future__ import annotations

import hashlib
import re
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
# backups
# --------------------------------------------------------------------------


def write_backup(rc_path: Path, text: str) -> Path:
    """Write *text* to a timestamped ``<rc>.nvsh-backup-<YYYYmmdd-HHMMSS>`` sibling.

    Never collides with an existing backup (appends ``-1``, ``-2``, ... on a
    same-second re-run, which matters in fast test loops).
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = rc_path.with_name(f"{rc_path.name}.nvsh-backup-{timestamp}")
    suffix = 1
    while backup_path.exists():
        backup_path = rc_path.with_name(f"{rc_path.name}.nvsh-backup-{timestamp}-{suffix}")
        suffix += 1
    backup_path.write_text(text, encoding="utf-8")
    return backup_path


def newest_backup(rc_path: Path) -> Path | None:
    """The most recently written backup for *rc_path*, or ``None``."""
    pattern = f"{rc_path.name}.nvsh-backup-*"
    candidates = sorted(rc_path.parent.glob(pattern))
    return candidates[-1] if candidates else None
