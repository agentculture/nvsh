"""Durable run state and the append-only billing record (issue #64, t17).

Everything the runner keeps besides the ledger lives here:

- ``run.json`` -- the run record (:func:`load_state` / :func:`save_state`),
  read and written only while the run lock (the ledger's ``.lock``) is held;
- ``billing.jsonl`` -- one line per answered call (and per call that was in
  flight at a crash), carrying the provider, model, prices, route, discount,
  usage and cost *at the time of the answer*. Spend is summed from these
  lines and never recomputed from the current manifest, so replacing a model
  or changing a price never gives budget back (codex review P1-5);
- ``smoke.json`` -- the smoke run's summary.

Every JSON document is written with :func:`write_json_durable`: a unique
temporary file in the same directory, fsync, ``os.replace``, fsync of the
directory -- so two writers never share a temporary name and a crash leaves
the previous or the next version, never a torn one (codex review P2-12).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

RUN_FILE = "run.json"
BILLING_FILE = "billing.jsonl"
SMOKE_FILE = "smoke.json"

BILL_ANSWER = "answer"
BILL_UNCERTAIN = "uncertain"


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_durable(path: Path, data: Any) -> None:
    """Write *data* as sorted JSON: unique temp file, fsync, replace, fsync the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_state(run_dir: Path) -> dict:
    """``run.json``, or ``{}``. Callers hold the run lock."""
    return read_json(Path(run_dir) / RUN_FILE, {}) or {}


def save_state(run_dir: Path, state: Mapping[str, Any]) -> None:
    """Write ``run.json`` durably. Callers hold the run lock."""
    write_json_durable(Path(run_dir) / RUN_FILE, state)


class Billing:
    """``billing.jsonl``: append-only, one fsync'd line per charge."""

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / BILLING_FILE

    def append(self, entry: Mapping[str, Any]) -> None:
        line = json.dumps(dict(entry), sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def billed_keys(self) -> set[str]:
        return {e["key"] for e in self.entries() if e.get("kind") == BILL_ANSWER}


def sums(entries: list[Mapping[str, Any]], field: str) -> dict[str, float]:
    """Total ``cost_usd`` of *entries* grouped by *field* (``provider`` or ``label``)."""
    out: dict[str, float] = {}
    for entry in entries:
        name = entry.get(field)
        out[name] = out.get(name, 0.0) + float(entry.get("cost_usd", 0.0))
    return out
