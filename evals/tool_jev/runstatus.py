"""``status`` for the gate runner (issue #64, t17): read-only, never takes the run lock.

Counts come from ``ledger.json``; spend comes from ``billing.jsonl`` (priced
at answer time, codex review P1-5) plus the reservations in ``run.json``.
Every file read here is replaced atomically by its writer, so a read while a
drive loop holds the lock sees a whole version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import runstate
from .runplan import RunError

_COUNTS = ("done", "submitted", "pending", "invalid")


def status(run_dir: Path) -> dict:
    """Per-provider and per-model done/submitted/pending/invalid counts and spend so far."""
    run_dir = Path(run_dir)
    state = runstate.read_json(run_dir / runstate.RUN_FILE, {}) or {}
    if not state:
        raise RunError(f"{run_dir} holds no run")
    ledger_doc = runstate.read_json(run_dir / "ledger.json", {}) or {}
    models = state.get("models", {})
    by_spec: dict[tuple[str, str], str] = {}
    for label, info in models.items():
        for provider_name, model_id in info.get("spec_keys", []):
            by_spec[(provider_name, model_id)] = label
    billing = runstate.Billing(run_dir).entries()
    label_spend = runstate.sums(billing, "label")
    provider_spend = runstate.sums(billing, "provider")
    label_provider = {e["label"]: e["provider"] for e in billing}
    for label, info in models.items():
        label_provider.setdefault(label, info.get("provider"))
    per_model: dict[str, dict[str, Any]] = {}
    for rec in ledger_doc.get("entries", {}).values():
        spec = rec.get("spec", {})
        label = by_spec.get((spec.get("provider"), spec.get("model")), "(unknown)")
        row = per_model.setdefault(label, {name: 0 for name in _COUNTS})
        row[rec["state"]] = row.get(rec["state"], 0) + 1
    reserved: dict[str, float] = {}
    for held in state.get("reserved", {}).values():
        reserved[held["provider"]] = reserved.get(held["provider"], 0.0) + held["usd"]
    per_provider: dict[str, dict[str, Any]] = {}
    for label, row in per_model.items():
        info = models.get(label, {})
        row["truncated"] = info.get("truncated", 0)
        row["answers"] = info.get("answers", 0)
        row["spend_usd"] = round(label_spend.get(label, 0.0), 6)
        kind = label_provider.get(label, "(unknown)")
        total = per_provider.setdefault(kind, {name: 0 for name in _COUNTS})
        for name in _COUNTS:
            total[name] += row[name]
    for kind in set(provider_spend) | set(reserved):
        per_provider.setdefault(kind, {name: 0 for name in _COUNTS})
    budgets = state.get("budgets", {})
    for kind, total in per_provider.items():
        total["spend_usd"] = round(provider_spend.get(kind, 0.0), 6)
        total["reserved_usd"] = round(reserved.get(kind, 0.0), 6)
        total["usd_cap"] = budgets.get(kind, {}).get("usd_cap")
    return {
        "run_id": state.get("run_id"),
        "date": state.get("date"),
        "mode": state.get("mode"),
        "scope": state.get("scope"),
        "status": state.get("status", "not started"),
        "providers": dict(sorted(per_provider.items())),
        "models": dict(sorted(per_model.items())),
        "stops": state.get("stops", {}),
        "capabilities": state.get("capabilities", {}),
        "hosts": state.get("hosts", {}),
        "uncertain_charges": state.get("uncertain_charges", []),
        "batch_failures": state.get("batch_failures", {}),
        "messages": state.get("messages", []),
        "smoke": state.get("smoke"),
    }


def render_status(doc: Mapping[str, Any]) -> list[str]:
    mode = f" [{doc['mode']}]" if doc.get("mode") else ""
    lines = [f"run {doc['run_id']} ({doc['date']}){mode}: {doc['status']}"]
    for kind, row in doc["providers"].items():
        cap = row.get("usd_cap")
        cap_text = f" of ${cap:.2f}" if isinstance(cap, (int, float)) else ""
        lines.append(
            f"provider {kind}: done {row['done']} submitted {row['submitted']} "
            f"pending {row['pending']} invalid {row['invalid']} | "
            f"spend ${row['spend_usd']:.4f}{cap_text} (reserved ${row['reserved_usd']:.4f})"
        )
    stops = doc.get("stops", {})
    for label, row in doc["models"].items():
        line = (
            f"  model {label}: done {row['done']} submitted {row['submitted']} "
            f"pending {row['pending']} invalid {row['invalid']} "
            f"truncated {row.get('truncated', 0)}/{row.get('answers', 0)} "
            f"spend ${row.get('spend_usd', 0.0):.4f}"
        )
        stop = stops.get("models", {}).get(label)
        if stop:
            line += f" | STOPPED ({stop['kind']}): {stop['message']}"
        lines.append(line)
    for kind, stop in stops.get("providers", {}).items():
        lines.append(f"provider {kind} STOPPED ({stop['kind']}): {stop['message']}")
    for charge in doc.get("uncertain_charges", []):
        lines.append(
            f"uncertain charge: {charge['label']} call {charge['key'][:12]} "
            f"~${charge['cost_usd']:.4f} (in flight at a crash, resent)"
        )
    hosts = doc.get("hosts", {})
    if hosts:
        shown = ", ".join(f"{host} ({', '.join(names)})" for host, names in sorted(hosts.items()))
        lines.append(f"hosts that received case text: {shown}")
    else:
        lines.append("hosts that received case text: none yet")
    return lines
