"""Case model and case-set loader for the Tool-Jev DeepEval release gate.

A :class:`Case` is one evaluation prompt: its split tag, request text (never
populated for a held-out case, see below), the operations it was offered as
candidates, its expected outcome, and whether that expectation is read-only
or mutating.

Split files are the same ``{"header": ..., "entries": [...]}`` JSON shape
``scripts/lfm-finetune/split.py`` and ``nvsh/tiers/bench.py`` read and write:
each entry has ``id``, ``kind``, ``text``, ``expect`` (``{"operation": name,
"args": {...}}``, ``{"escalate": True}`` or ``{"explain": True}``), and
optionally ``source``, ``class`` and ``candidates`` (the missing-candidate
slices ``scripts/lfm-finetune/eval_slices.py`` produces add a ``candidates``
list and a ``source_id`` pointing back at the original entry).

The private manifest. Real split files (issue-46/53 test, test-mc, and the
sealed held-out/held-out-mc sides) live outside this repository, in the
operator's private ``lfm-train`` work tree -- never in git. Callers name
those paths in a small manifest dict (or a JSON file of the same shape) that
this module never ships a copy of:

    {"splits": {"test": "/path/to/test.json",
                "test-mc": "/path/to/test-mc.json",
                "heldout": "/path/to/held-out.json",
                "heldout-mc": "/path/to/held-out-mc.json"}}

Only ``"test"`` and ``"test-mc"`` are required to exist for an ordinary run;
``"heldout"``/``"heldout-mc"`` are read only when a caller explicitly passes
``include_heldout=True`` to :func:`load_case_set`, and even then the request
text of a held-out case is never materialized (criterion 1).

``read_only``/mutating classification is never taken from a corpus entry or
a model's output: it is looked up from :mod:`nvsh.ops.table` by the
*expected* operation's name, exactly as ``nvsh/tiers/gate.py`` does, and an
operation missing from the table is treated as mutating (conservative).

No module under ``nvsh/`` may import this one; this module may import
``nvsh.ops.table``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nvsh.ops import table

#: Split tags a Case may carry, per the plan (t2's own contract).
SPLIT_TAGS = ("test", "test-mc", "heldout", "heldout-mc")

#: Split tags whose request text is never returned by the loader, no matter
#: what the caller passes -- the sealed held-out sets stay unread as text.
_HELDOUT_TAGS = ("heldout", "heldout-mc")


@dataclass(frozen=True)
class Case:
    """One evaluation prompt, resolved from a split-file entry.

    ``text`` is the request string for a non-held-out case, and ``None`` for
    a held-out case regardless of ``include_heldout`` (criterion 1): the
    loader never returns held-out request text, only ids and expectations.

    ``candidates`` is the tuple of operation names the case offered as
    choices (missing-candidate slices carry this; an ordinary split entry
    without a ``"candidates"`` field leaves it ``None``, meaning "the full
    table", not "none offered").

    ``expect`` is kept verbatim (``{"operation": name, "args": {...}}``,
    ``{"escalate": True}`` or ``{"explain": True}``) so nothing here has to
    re-derive it; :attr:`expected_operation`, :attr:`expected_args`,
    :attr:`expects_escalate` and :attr:`expects_explain` are convenience
    reads over it.

    ``read_only`` is ``None`` for a should-decline expectation (there is no
    operation to classify) and otherwise comes from :mod:`nvsh.ops.table`:
    ``True``/``False`` for a known operation, ``False`` (mutating) for one
    missing from the table -- never from the corpus or a model's answer.
    """

    id: str
    split: str
    text: str | None
    candidates: tuple[str, ...] | None
    expect: dict
    read_only: bool | None
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_id: str | None = None

    def __post_init__(self) -> None:
        if self.split not in SPLIT_TAGS:
            raise ValueError(
                f"{self.id}: unknown split tag {self.split!r}, must be one of " f"{SPLIT_TAGS}"
            )
        if self.split in _HELDOUT_TAGS and self.text is not None:
            raise ValueError(f"{self.id}: a held-out case must never carry request text")

    @property
    def expects_escalate(self) -> bool:
        return bool(self.expect.get("escalate"))

    @property
    def expects_explain(self) -> bool:
        return bool(self.expect.get("explain"))

    @property
    def expected_operation(self) -> str | None:
        return self.expect.get("operation")

    @property
    def expected_args(self) -> dict:
        return self.expect.get("args", {})


class HeldOutAccessError(RuntimeError):
    """Raised when a held-out case set is requested without ``include_heldout``."""


class TrainingOverlapError(RuntimeError):
    """Raised by :func:`training_overlap` when scored ids appear in a train split."""

    def __init__(self, overlapping_ids: tuple[str, ...]):
        self.overlapping_ids = overlapping_ids
        super().__init__(
            "refusing to score a checkpoint on case id(s) present in its own training "
            f"split: {list(overlapping_ids)}"
        )


def _read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(manifest: Mapping[str, Any] | str | Path) -> dict[str, str]:
    """Resolve a manifest to its ``{split_tag: path}`` mapping.

    *manifest* is either an in-memory mapping already shaped like
    ``{"splits": {...}}`` (or a bare ``{split_tag: path}`` mapping), or a
    path to a JSON file of that same shape. This module never ships or
    commits a real manifest -- callers hand in their own, private, path.
    """
    raw: Mapping[str, Any]
    if isinstance(manifest, (str, Path)):
        raw = _read_json(manifest)
    else:
        raw = manifest
    splits = raw.get("splits", raw) if isinstance(raw, Mapping) else raw
    if not isinstance(splits, Mapping):
        raise ValueError(f"manifest must map split tags to paths, got {splits!r}")
    return {str(tag): str(path) for tag, path in splits.items()}


def _read_only_for(expect: dict) -> bool | None:
    """``nvsh.ops.table``-derived read-only flag for an expectation.

    ``None`` for a should-decline expectation (escalate/explain: there is no
    operation to classify). For a named operation, ``True``/``False`` per
    the table, and ``False`` (mutating) when the operation is not in the
    table at all -- the same conservative default ``nvsh/tiers/gate.py``
    uses for an operation it doesn't recognize.
    """
    if expect.get("escalate") or expect.get("explain"):
        return None
    name = expect.get("operation")
    if name is None:
        return None
    operation = table.get(name)
    if operation is None:
        return False
    return operation.read_only


def _entry_to_case(entry: dict, split: str) -> Case:
    entry_id = str(entry["id"])
    expect = entry["expect"]
    candidates = entry.get("candidates")
    tags: list[str] = []
    if entry.get("kind"):
        tags.append(str(entry["kind"]))
    if entry.get("class"):
        tags.append(str(entry["class"]))
    if entry.get("source"):
        tags.append(str(entry["source"]))
    if candidates is not None:
        tags.append("nocand")
    text = None if split in _HELDOUT_TAGS else str(entry.get("text", ""))
    return Case(
        id=entry_id,
        split=split,
        text=text,
        candidates=tuple(candidates) if candidates is not None else None,
        expect=dict(expect),
        read_only=_read_only_for(expect),
        tags=tuple(tags),
        source_id=entry.get("source_id"),
    )


def load_case_set(
    manifest: Mapping[str, Any] | str | Path,
    split: str,
    *,
    include_heldout: bool = False,
) -> tuple[Case, ...]:
    """Load every case on *split* from the split file the manifest names.

    Raises :class:`HeldOutAccessError` for ``split in {"heldout",
    "heldout-mc"}`` unless *include_heldout* is ``True`` (criterion 1). Even
    with ``include_heldout=True``, every returned :class:`Case`'s ``text``
    is ``None`` -- the loader never returns held-out request text, only ids
    and expectations, so a caller cannot accidentally leak it downstream.
    """
    if split not in SPLIT_TAGS:
        raise ValueError(f"unknown split tag {split!r}, must be one of {SPLIT_TAGS}")
    if split in _HELDOUT_TAGS and not include_heldout:
        raise HeldOutAccessError(
            f"split {split!r} is held-out; pass include_heldout=True to load it "
            "(and even then, no request text is returned)"
        )

    paths = load_manifest(manifest)
    if split not in paths:
        raise KeyError(f"manifest has no path for split {split!r}")

    raw = _read_json(paths[split])
    entries = raw.get("entries", []) if isinstance(raw, Mapping) else raw
    if not isinstance(entries, list):
        raise ValueError(f"split file {paths[split]!r} has no usable 'entries' list")

    return tuple(_entry_to_case(entry, split) for entry in entries)


def _load_ids(train_split_path: str | Path) -> frozenset[str]:
    raw = _read_json(train_split_path)
    entries = raw.get("entries", []) if isinstance(raw, Mapping) else raw
    if not isinstance(entries, list):
        raise ValueError(f"train split file {train_split_path!r} has no usable 'entries' list")
    return frozenset(str(entry["id"]) for entry in entries)


def training_overlap(
    case_ids: Iterable[str],
    train_split_path: str | Path,
) -> tuple[str, ...]:
    """Refuse to score a checkpoint on any id present in its own train split.

    Reads *train_split_path* (the checkpoint's train-side split file, the
    same shape as any other split) and compares its ids against
    *case_ids*. Raises :class:`TrainingOverlapError` naming every
    overlapping id, sorted, when the two sets intersect. Returns the
    (empty) overlap tuple when they don't, so a caller can also use this as
    a pure check without wrapping every call in try/except.
    """
    train_ids = _load_ids(train_split_path)
    overlap = tuple(sorted(set(case_ids) & train_ids))
    if overlap:
        raise TrainingOverlapError(overlap)
    return overlap
