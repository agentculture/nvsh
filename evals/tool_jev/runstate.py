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


class BillingTorn(Exception):
    """A line in the middle of ``billing.jsonl`` does not parse: stop and ask."""


class Billing:
    """``billing.jsonl``: append-only, one fsync'd line per charge.

    Every line carries an ``attempt_id`` (an answer: ledger key + response
    id; an uncertain charge: the reservation id persisted when the call was
    claimed). Reading counts each ``(kind, attempt_id)`` once, so a line
    appended again by a replayed pass never charges twice (codex review
    items 14, 15).
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / BILLING_FILE

    def append(self, entry: Mapping[str, Any]) -> None:
        line = json.dumps(dict(entry), sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def repair(self, now: float, log=None) -> Path | None:
        """Move a torn last line aside; a torn line anywhere else is :class:`BillingTorn`.

        Why moving the tail aside is correct: a charge is appended *before*
        the ledger caches the answer and before ``run.json`` is saved. A line
        that never finished (no newline, or not parseable) therefore belongs
        to an answer that was never recorded; that call is fetched (or sent)
        again and billed again by a whole line. The torn bytes are kept in
        ``billing.torn.<timestamp>`` for the operator, never deleted.
        Called only under the run lock.
        """
        if not self.path.exists():
            return None
        data = self.path.read_bytes()
        if not data:
            return None
        body, _, tail = data.rpartition(b"\n")
        lines = body.split(b"\n") if body else []
        bad = [i for i, line in enumerate(lines) if line.strip() and not _parses(line)]
        if tail:  # an unterminated last line
            torn, keep = tail, data[: len(data) - len(tail)]
            if bad:
                raise BillingTorn(self._middle_message(bad[0]))
        elif bad:
            if bad != [len(lines) - 1]:
                raise BillingTorn(self._middle_message(bad[0]))
            torn = lines[-1]
            keep = data[: len(data) - len(torn) - 1]
        else:
            return None
        aside = self.run_dir / f"billing.torn.{int(now)}"
        n = 0
        while aside.exists():
            n += 1
            aside = self.run_dir / f"billing.torn.{int(now)}.{n}"
        aside.write_bytes(torn)
        with open(self.path, "r+b") as handle:
            handle.truncate(len(keep))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(self.run_dir)
        if log is not None:
            log("billing_torn_tail", [], moved_to=aside.name, size=len(torn))
        return aside

    def _middle_message(self, index: int) -> str:
        return (
            f"{self.path}: line {index + 1} does not parse and is not the last line; a charge "
            "may be lost or garbled -- repair billing.jsonl by hand, then continue"
        )

    def entries(self, *, tolerant: bool = False) -> list[dict]:
        """Every charge, each ``(kind, attempt_id)`` once.

        *tolerant* skips an unparseable last line (``status`` reads without
        the lock); otherwise the journal must already be repaired.
        """
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8").splitlines()
        out, seen = [], set()
        for index, line in enumerate(raw):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                if tolerant and index == len(raw) - 1:
                    continue
                raise BillingTorn(self._middle_message(index)) from None
            attempt = entry.get("attempt_id")
            if attempt is not None:
                if (entry.get("kind"), attempt) in seen:
                    continue
                seen.add((entry.get("kind"), attempt))
            out.append(entry)
        return out

    def billed_keys(self) -> set[str]:
        return {e["key"] for e in self.entries() if e.get("kind") == BILL_ANSWER}


def _parses(line: bytes) -> bool:
    try:
        json.loads(line)
    except ValueError:
        return False
    return True


def sums(entries: list[Mapping[str, Any]], field: str) -> dict[str, float]:
    """Total ``cost_usd`` of *entries* grouped by *field* (``provider`` or ``label``)."""
    out: dict[str, float] = {}
    for entry in entries:
        name = entry.get(field)
        out[name] = out.get(name, 0.0) + float(entry.get("cost_usd", 0.0))
    return out
