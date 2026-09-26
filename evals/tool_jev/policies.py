"""Harness decision policies: versioned JSON configs, applied offline (issue 64, t8).

A "policy" here is a small JSON document that says how to turn one already
recorded scorer prediction (``metrics.py``'s ``candidates`` distribution,
same schema ``scripts/lfm-finetune/gate.py`` and ``calibration_fit.py``
already consume) into a final decision, without ever calling a model again:

- an optional ``calibration`` block (``{"temperature": ..., "vector": ...}``)
  rescaled onto the recorded ``candidates`` via
  ``calibration_fit.apply_scaling`` -- the same function
  ``calibration_fit.py apply`` uses, not a reimplementation;
- an optional ``gate`` block (``gate.Thresholds.from_json``'s own shape)
  that gates the (possibly rescaled) distribution's argmax via
  ``gate.decide`` -- also reused, not reimplemented.

Both blocks are optional. A policy with neither (see ``policies/raw.json``)
reduces to the bare argmax: ``gate.decide`` with an all-``None``
``Thresholds()`` already reproduces that (see its own module docstring),
including the *unconditional* semantic escalate/explain checks -- those
fire on the argmax itself, not on any threshold, so "no gate configured"
still correctly reports ``escalate``/``explain`` when the argmax names one
of those controls. It never introduces abstention on its own.

``read_only`` vs. ``mutating`` dispatch is exactly ``gate.decide``'s: read
from ``nvsh.ops.table``'s ``read_only`` flag for the argmax operation, never
from the operation's name, and an operation missing from the table is
gated as mutating (the stricter set). A policy JSON is therefore free to
tune ``gate.read_only`` and ``gate.mutating`` independently, and the two
classes are reported as separate ``ThresholdSet`` blocks in the JSON and
separate reasons in the returned :class:`~gate.Decision` (via ``reason``),
so callers can tell which class fired an abstention.

**What a policy here can *not* do**: filter or restrict the set of
operations offered to the scorer before it scored a request. That kind of
"allowed-operation filtering" changes what the model actually saw and
scored -- a different, smaller ``offered`` would produce a different
probability distribution over a different candidate set, not just a
different decision over the *same* one. It cannot be replayed offline from
a single saved ``candidates`` distribution, and this module does not
attempt it: :func:`apply` only ever rescales and gates the distribution
that was actually recorded.

Applying a policy never calls a model: it is pure function of
(*policy*, *raw_record_like*, *offered*), and every code path below only
reads dict-like input already sitting in memory -- no I/O, no subprocess,
no network. A record with no recorded distribution (``candidates`` absent
or falsy, e.g. a line ``calibration_fit.py`` itself would count under
``skipped_null``) returns the explicit ``"not_gateable"`` decision rather
than guessing at an argmax that was never computed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LFM_FINETUNE_DIR = _REPO_ROOT / "scripts" / "lfm-finetune"
_POLICIES_DIR = Path(__file__).resolve().parent / "policies"


def _load_sibling_script(path: Path, module_name: str):
    """Load a scripts/lfm-finetune/*.py module by path (never copy its code).

    Matches the pattern ``tests/test_lfm_finetune_gate.py`` and
    ``tests/test_lfm_finetune_metrics.py`` already use:
    ``importlib.util.spec_from_file_location`` plus registering the module
    in ``sys.modules`` under its own name so the module's own dataclasses
    (which look themselves up there) resolve correctly.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_sibling_script(_LFM_FINETUNE_DIR / "gate.py", "evals_lfm_finetune_gate")
calibration_fit = _load_sibling_script(
    _LFM_FINETUNE_DIR / "calibration_fit.py", "evals_lfm_finetune_calibration_fit"
)


class PolicyError(ValueError):
    """A malformed policy payload (missing ``name``/``version``, bad JSON shape)."""


#: The decision string returned when a record carries no distribution to gate.
NOT_GATEABLE = "not_gateable"


def load_policy(path: Path | str) -> dict:
    """Read and validate one policy JSON file (does not call a model)."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_policy(data)


def validate_policy(policy: Mapping) -> dict:
    """Check *policy* carries ``name`` and ``version``; return it as a plain dict.

    Raises :class:`PolicyError` otherwise. ``calibration`` and ``gate`` are
    validated lazily, by :func:`apply` itself (via ``gate.Thresholds.from_json``),
    since a policy may omit either or both.
    """
    if not isinstance(policy, Mapping):
        raise PolicyError(f"a policy must be a JSON object, got {policy!r}")
    if not policy.get("name"):
        raise PolicyError("policy must carry a non-empty 'name'")
    if policy.get("version") in (None, ""):
        raise PolicyError("policy must carry a non-empty 'version'")
    return dict(policy)


def builtin_policy_path(name: str) -> Path:
    """Path of one of this package's own ``policies/<name>.json`` files."""
    return _POLICIES_DIR / f"{name}.json"


def apply(
    policy: Mapping,
    raw_record_like: Mapping,
    offered: Sequence[str],
) -> tuple[str, str | None, str, object]:
    """Apply *policy* to one already-recorded prediction. Never calls a model.

    *raw_record_like* is anything carrying a ``candidates`` mapping in
    ``metrics.py``'s schema (a full prediction line, or a minimal
    ``{"candidates": {...}}``). *offered* is the sequence of labels that
    were actually offered to the scorer for this request, in their
    original order (see ``gate.decide``'s docstring for why order matters
    for tie-breaking).

    Returns ``(decision, reason, policy_name, policy_version)``:

    - *decision* is ``"not_gateable"`` when *raw_record_like* has no usable
      distribution, otherwise one of ``gate.OUTCOMES``
      (``"propose"``, ``"explain"``, ``"escalate"``, ``"abstain_uncertain"``).
    - *reason* is ``None`` for every decision except a not-gateable record
      (``"no_distribution"``) or ``abstain_uncertain``/``escalate``, which
      carry whatever ``gate.Decision.reason`` reports.
    - *policy_name* / *policy_version* are *policy*'s own ``"name"`` /
      ``"version"`` fields, unchanged, so a caller building a decision
      trace can record exactly which policy produced this entry.
    """
    validated = validate_policy(policy)
    name = validated["name"]
    version = validated["version"]

    candidates = raw_record_like.get("candidates") if isinstance(raw_record_like, Mapping) else None
    if not candidates:
        return NOT_GATEABLE, "no_distribution", name, version

    scaled = dict(candidates)
    calibration = validated.get("calibration")
    if calibration:
        temperature = float(calibration.get("temperature", 1.0))
        vector = calibration.get("vector") or None
        scaled = calibration_fit.apply_scaling(scaled, temperature=temperature, vector=vector)

    gate_json = validated.get("gate")
    thresholds = gate.Thresholds.from_json(gate_json) if gate_json else gate.Thresholds()

    decision = gate.decide(scaled, offered, thresholds)
    return decision.outcome, decision.reason, name, version
