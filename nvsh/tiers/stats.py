"""Pure aggregation over :meth:`nvsh.tiers.records.TierRecords.read_all` output.

Task t20 (``nvsh tiers stats``). Takes a plain ``list[dict]`` (as produced by
``TierRecords.read_all()``) and an optional ``dropped`` count, and returns a
JSON-serialisable summary: per-tier counts and latency percentiles, an
escalation/decline-reason histogram, and operator approve/decline rates.

Deliberately dumb about tier names: whatever value the ``tier`` field holds
in a record becomes a group, so a new tier needs no change here (mirrors the
operation-table rule: nothing here switches on a specific tier name).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence


def _nearest_rank_percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile over an already-sorted sequence. Empty-safe.

    Deterministic: ``rank = ceil(pct / 100 * n)``, clamped to ``[1, n]``.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    rank = math.ceil(pct / 100.0 * n)
    rank = min(max(rank, 1), n)
    return float(sorted_values[rank - 1])


def _latency(record: Mapping[str, object]) -> float:
    """A record's latency, or 0.0 for a missing, non-numeric or non-finite value."""
    value = record.get("latency_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if math.isfinite(value) else 0.0


def _tier_stats(records: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    by_tier: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        tier = str(record.get("tier") or "unknown")
        by_tier.setdefault(tier, []).append(record)

    result: dict[str, dict[str, object]] = {}
    for tier in sorted(by_tier):
        recs = by_tier[tier]
        latencies = sorted(_latency(r) for r in recs)
        result[tier] = {
            "count": len(recs),
            "latency_p50_ms": _nearest_rank_percentile(latencies, 50),
            "latency_p95_ms": _nearest_rank_percentile(latencies, 95),
        }
    return result


def _escalation_reasons(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    reasons = Counter(
        str(record["decline_reason"]) for record in records if record.get("decline_reason")
    )
    return dict(sorted(reasons.items()))


def _operator_decisions(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    decisions = Counter(
        str(record["operator_decision"]) for record in records if record.get("operator_decision")
    )
    approved = decisions.get("approved", 0)
    declined = decisions.get("declined", 0)
    total = approved + declined
    return {
        "approved": approved,
        "declined": declined,
        "approve_rate": (approved / total) if total else 0.0,
        "decline_rate": (declined / total) if total else 0.0,
    }


def compute_stats(records: Sequence[Mapping[str, object]], *, dropped: int = 0) -> dict:
    """Aggregate *records* into the shape ``nvsh tiers stats`` reports.

    ``records`` is untrusted, already-redacted data read back from disk: no
    field is assumed present, and an empty list produces zeroed-out stats
    rather than raising. ``TierRecords.read_all()`` accepts any JSON value
    that parses, so a hand-edited or torn line can decode to ``null``, a
    list or a scalar; those are counted in ``total`` (they were valid JSON)
    but excluded from every ``.get()``-based aggregate below, which would
    otherwise raise ``AttributeError``.
    """
    well_formed = [record for record in records if isinstance(record, Mapping)]
    return {
        "total": len(records),
        "dropped": dropped,
        "tiers": _tier_stats(well_formed),
        "escalation_reasons": _escalation_reasons(well_formed),
        "operator_decisions": _operator_decisions(well_formed),
    }
