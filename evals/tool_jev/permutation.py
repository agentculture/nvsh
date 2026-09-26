"""Permutation stability entries for the issue-64 report (task t23, issue #64/#53).

This module never runs a model, never runs ``scripts/lfm-finetune/permutation_probe.py``,
and never touches a GPU. It is a pure loader/transform: given a probe JSON that some
other process (a development-machine run of ``permutation_probe.py``, e.g. issue 53's
``final/probe-scorer-r3b-test.json``) already wrote to disk, it produces exactly the
``"<subject>__<policy>": {...opaque, numeric-only permutation figures...}`` entries that
``evals/tool_jev/report.py``'s :func:`report.load_permutation` reads back from
``<run_dir>/permutation.json`` (``report.PERMUTATION_FILENAME``).

Why a3-heal (Track A, tool-call/generative) cannot be probed by
``permutation_probe.py``: every kind it draws (``order``/``letters``/``subset``/
``paraphrase``/``all``) is built through ``scorer.permute``/``scorer.labels_for`` and
scored through ``scorer.score``/``scorer.same_choice`` -- all of it Track B's
one-position, candidate-lettered "choice" interface (a fixed A-R alphabet a served or
in-process *scorer* answers by picking one letter). Its CLI's ``_build_real_scorer``
only ever calls ``measure.build_scorer`` with ``scorer_kind`` ``"served"`` or
``"in-process"``; there is no seam anywhere in the module for a generative/tool-call
run (``measure.generative_predictions``/``generative_candidates``), which answers by
emitting a tool call naming an operation directly -- there is no candidate listing or
letter map on that path to permute in the first place. So a3-heal's permutation is
:func:`not_measurable_entry`, never estimated by, say, reusing r3b's numbers or by
permuting the *tools* array a3-heal was offered (a different question the probe was
never built to ask). If ``permutation_probe.py`` grows a generative seam later, this
module's docstring is the place to update.

Privacy: :func:`probe_to_entry` never copies a probe JSON's dict through verbatim. It
reads only the specific, whitelisted numeric fields documented on
:data:`_KIND_NUMERIC_FIELDS` out of each ``kinds[]`` entry, plus a handful of top-level
run-shape integers/strings (``per_entry``, ``seed``, ``entries``). A probe JSON has no
case/request text in its own schema (see ``permutation_probe.py``'s ``kind_report``),
but this module does not rely on that being true of every future probe file -- it reads
fields by name, not by iterating the input's keys, so an unexpected extra string field
in some other probe JSON is simply never reached.

Held-out guard: a probe JSON's own schema (``per_entry``, ``seed``, ``entries``,
``canonicalised_order``, ``reasons``, ``kinds``) records no split or source field at
all -- confirmed against the real issue-53 output. :func:`resolve_split` therefore
requires the caller to pass ``split=`` explicitly, and refuses one that looks like a
held-out split (``"held-out"``, ``"held_out"``, ``"heldout"``, ``"heldout-mc"``, any
case). If some other probe JSON ever does carry a recorded split/source field, this
module prefers that recorded value over the caller's ``split=`` argument (the file is
the more trustworthy witness) and applies the same refusal to it.

Nothing under ``nvsh/`` may import this module. No network, no subprocess, no model or
GPU work.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

#: Fields copied verbatim (they are already numbers, or short enum-like strings with no
#: case/request text) out of each ``kinds[]`` entry of a probe JSON, per issue 53's
#: ``permutation_probe.kind_report``. Anything else in that dict (e.g. its free-text
#: ``bootstrap_note``) is never copied through.
_KIND_NUMERIC_FIELDS = ("trials", "changes", "entries", "rate", "ci_low", "ci_high")

#: Fields copied out of a probe JSON's top level (all short, non-case-text values).
_TOP_LEVEL_FIELDS = ("per_entry", "seed", "entries")

#: Field names a probe JSON (or some other caller-supplied record) might record its
#: source split under. None of these exist in ``permutation_probe.py``'s own schema
#: today; kept as a forward-compatible list in case a future probe file adds one.
_SPLIT_FIELDS = ("split", "source_split", "split_source", "source")

#: Matches "held-out", "held_out", "heldout", "heldout-mc", any case, anywhere in the
#: string -- mirrors ``scripts/lfm-finetune/calibration_fit.py``'s ``split_markers``
#: intent (refuse test/held-out input) but only the held-out half: a *test* split (like
#: issue 53's own ``probe-scorer-r3b-test.json``) is exactly what this module is for.
_HELDOUT_RE = re.compile(r"held[-_]?out", re.IGNORECASE)


class PermutationError(ValueError):
    """A refused input: a missing/held-out split, or a malformed probe JSON."""


# ---------------------------------------------------------------------------
# Held-out guard
# ---------------------------------------------------------------------------


def _recorded_split(probe: Mapping[str, Any]) -> str | None:
    for key in _SPLIT_FIELDS:
        value = probe.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def resolve_split(probe: Mapping[str, Any], split: str | None = None) -> str:
    """The split this probe JSON is for, refusing anything held-out.

    Prefers a split/source field the probe JSON itself records (see
    :data:`_SPLIT_FIELDS`) over *split*; when the probe records none (true of every
    ``permutation_probe.py`` output today), *split* is required. Raises
    :class:`PermutationError` when neither is given, or when the resolved value looks
    held-out (:data:`_HELDOUT_RE`).
    """
    recorded = _recorded_split(probe)
    resolved = recorded if recorded is not None else split
    if resolved is None:
        raise PermutationError("probe JSON records no split/source field; pass split= explicitly")
    if _HELDOUT_RE.search(resolved):
        raise PermutationError(f"refusing a held-out split for a permutation probe: {resolved!r}")
    return resolved


# ---------------------------------------------------------------------------
# Loading a saved probe JSON
# ---------------------------------------------------------------------------


def load_probe_json(path: str | Path) -> dict:
    """Read a ``permutation_probe.py`` report JSON from disk. Never runs the probe."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PermutationError(f"cannot read probe JSON {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("kinds"), list):
        raise PermutationError(f"{path} is not a permutation_probe.py report (missing 'kinds')")
    return raw


def _kind_entry(kind_report: Mapping[str, Any]) -> dict:
    entry = {field: kind_report.get(field) for field in _KIND_NUMERIC_FIELDS}
    incomplete = kind_report.get("incomplete")
    if isinstance(incomplete, Mapping):
        entry["incomplete_rate"] = incomplete.get("rate")
    return entry


def probe_to_entry(probe: Mapping[str, Any], *, split: str | None = None) -> dict:
    """One report-permutation-slice entry (measurable) from a loaded probe JSON.

    *probe* is a dict already read (:func:`load_probe_json`, or an equivalent synthetic
    fixture in a test) -- this function itself does no file I/O. Raises
    :class:`PermutationError` via :func:`resolve_split` when *split* is missing/held-out,
    or when the probe JSON is missing its ``"kinds"`` list.
    """
    if not isinstance(probe.get("kinds"), list):
        raise PermutationError("probe is missing a 'kinds' list")
    resolved_split = resolve_split(probe, split)
    kinds = {}
    for kind_report in probe["kinds"]:
        name = kind_report.get("kind")
        if not isinstance(name, str) or not name:
            raise PermutationError("a probe kind entry is missing its 'kind' name")
        kinds[name] = _kind_entry(kind_report)
    result: dict[str, Any] = {"measurable": True, "split": resolved_split}
    for field in _TOP_LEVEL_FIELDS:
        result[field] = probe.get(field)
    result["kinds"] = kinds
    return result


def not_measurable_entry(reason: str) -> dict:
    """The report-permutation-slice entry for a subject/policy the probe cannot cover.

    *reason* must be a short, human-readable, non-case-text explanation (e.g. a3-heal's:
    "permutation_probe.py only probes a Track B one-position scorer (scorer.permute /
    scorer.same_choice); a3-heal is a Track A tool-call checkpoint with no candidate
    listing or letter map on its answer path"). Never estimated from another subject's
    numbers.
    """
    if not reason:
        raise PermutationError("not_measurable_entry requires a non-empty reason")
    return {"measurable": False, "reason": reason}


def permutation_key(subject: str, policy: str) -> str:
    """The ``"<subject>__<policy>"`` key ``report.py``'s permutation slice looks up."""
    return f"{subject}__{policy}"


def build_permutation_file(entries: Mapping[str, Mapping[str, Any]]) -> dict:
    """The full ``<run_dir>/permutation.json`` payload from ``{key: entry}`` pairs.

    *entries* keys should already be :func:`permutation_key` outputs; each value should
    already be a :func:`probe_to_entry` or :func:`not_measurable_entry` result. This
    function itself performs no validation beyond returning a plain ``dict`` copy --
    the per-entry builders above are where refusal/validation happens.
    """
    return dict(entries)
