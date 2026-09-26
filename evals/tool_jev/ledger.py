"""Durable call ledger and response cache for the Tool-Jev evals (issue #64, task t9).

A run must survive the operator stopping it (Ctrl+C), a provider budget
running out, and a hard reset or power-off. After any of these, ``continue``
resumes where it stopped and never pays twice for a call whose answer is
already on disk. This module stores only *state*; the runner drives the
providers (see ``run_until_settled`` in ``tests/test_ledger.py`` for the
reference loop).

On-disk layout under ``run_dir``::

    ledger.json          every call's state (one JSON document, sorted keys)
    cache/<key>.json     the provider's answer for one call (raw bytes + meta)
    events.jsonl         append-only, timestamped audit trail (never results)
    .lock                advisory flock: one live ``Ledger`` per run dir

Every write of ``ledger.json`` or a cache file goes through
:func:`_atomic_write`: write ``<name>.tmp``, ``fsync`` it, ``os.replace``
it over the target and ``fsync`` the directory. A crash at any point leaves
either the previous or the next version, never a torn one; leftover
``*.tmp`` files are deleted on open.

A cache file is always written *before* the state change that says the call
finished. If the process dies between the two, :class:`Ledger` finds the
cached answer on open and promotes the call to its final state, so the call
is not sent again.

States:

- ``pending``   — not yet sent, or returned by an infrastructure stop. May
  carry a ``submit_token`` meaning "a submission was being made when we
  stopped": the runner must ask the provider whether a batch with that token
  exists (re-attach) before sending again (:meth:`Ledger.abandon_submit`).
- ``submitted`` — sent in batch ``batch_id``; on continue that batch is
  re-attached and polled, never resubmitted while it is valid.
- ``done``      — answer cached and accepted.
- ``invalid``   — the *model's* answer was bad (reason recorded); only a bad
  answer lands here, never an infrastructure failure.

:meth:`Ledger.results_json` is deterministic: sorted keys, stable ordering,
no timestamps, batch ids or submission tokens — two runs that got the same
answers produce byte-identical results no matter how often they stopped.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "PENDING",
    "SUBMITTED",
    "DONE",
    "INVALID",
    "STATES",
    "SCHEMA",
    "CallSpec",
    "CachedResponse",
    "Entry",
    "ContinuePlan",
    "Ledger",
    "LedgerError",
    "LedgerCorrupt",
    "LedgerLocked",
    "LedgerStateError",
    "ledger_key",
    "prompt_hash",
    "canonical_json",
]

PENDING = "pending"
SUBMITTED = "submitted"
DONE = "done"
INVALID = "invalid"
STATES = (PENDING, SUBMITTED, DONE, INVALID)
SCHEMA = 1

_LEDGER_FILE = "ledger.json"
_CACHE_DIR = "cache"
_EVENTS_FILE = "events.jsonl"
_LOCK_FILE = ".lock"
_TMP_SUFFIX = ".tmp"


class LedgerError(Exception):
    """Base class for ledger errors."""


class LedgerCorrupt(LedgerError):
    """A ledger or cache file on disk cannot be trusted (never silently reset)."""


class LedgerLocked(LedgerError):
    """Another live ``Ledger`` holds this run directory."""


class LedgerStateError(LedgerError):
    """A state transition that the ledger's state machine does not allow."""


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, ASCII, NaN/Inf refused."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def prompt_hash(prompt: str | bytes) -> str:
    """sha256 hex of the exact prompt text (UTF-8) or bytes sent to the provider."""
    data = prompt.encode("utf-8") if isinstance(prompt, str) else prompt
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CallSpec:
    """Everything that identifies one provider call; its hash is the ledger key.

    ``target`` is the interface id (for a subject call) or the judge id (for
    a judge call). ``params`` holds every sampling/request parameter that can
    change the answer (temperature, max tokens, reasoning effort, ...) and
    must be JSON-serialisable.
    """

    provider: str
    model: str
    subject_role: str
    case_id: str
    target: str
    prompt_hash: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def ledger_key(spec: CallSpec) -> str:
    """sha256 over (provider, model, subject role, case id, target, prompt hash, params)."""
    try:
        text = canonical_json({"v": SCHEMA, **spec.to_json()})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"call spec is not canonical JSON: {exc}") from exc
    return hashlib.sha256(text.encode("ascii")).hexdigest()


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedResponse:
    """One provider answer, exactly as returned (what a warm rerun replays)."""

    raw: bytes
    model_id: str
    response_id: str
    usage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.raw, bytes):
            raise TypeError("CachedResponse.raw must be bytes")
        for name, value in self.usage.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"usage[{name!r}] must be an int, got {value!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "raw_b64": base64.b64encode(self.raw).decode("ascii"),
            "raw_sha256": hashlib.sha256(self.raw).hexdigest(),
            "response_id": self.response_id,
            "usage": dict(self.usage),
        }


@dataclass(frozen=True)
class Entry:
    """One call's current state (a read-only snapshot)."""

    key: str
    spec: dict[str, Any]
    state: str
    batch_id: str | None = None
    reason: str | None = None
    submit_token: str | None = None


@dataclass(frozen=True)
class ContinuePlan:
    """What ``continue`` has to do, computed from disk only.

    - ``done``/``invalid``: answered already; served from the cache.
    - ``batches``: batch id -> its still-submitted keys; re-attach and poll.
    - ``orphans``: submit token -> pending keys whose submission may or may
      not have reached the provider; look the token up before resending.
    - ``pending``: never sent (or requeued); send these.
    """

    done: list[str]
    invalid: list[str]
    batches: dict[str, list[str]]
    orphans: dict[str, list[str]]
    pending: list[str]


# --------------------------------------------------------------------------
# Durable writes
# --------------------------------------------------------------------------


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write-temp-then-rename with fsync of the file and its directory."""
    path = Path(path)
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _dump(doc: dict[str, Any]) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------


class Ledger:
    """Durable per-run call ledger + response cache. Use as a context manager.

    Only one live ``Ledger`` may hold a run directory (``flock`` on
    ``.lock``; released automatically if the process dies).
    """

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run_dir = Path(run_dir)
        self.cache_dir = self.run_dir / _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock_fh = open(self.run_dir / _LOCK_FILE, "a+b")
        try:
            fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_fh.close()
            raise LedgerLocked(f"{self.run_dir} is in use by another run") from exc
        try:
            self._remove_leftover_temps()
            self._entries: dict[str, dict[str, Any]] = {}
            self._submit_seq = 0
            self._load()
            self._reconcile_with_cache()
        except BaseException:
            self.close()
            raise

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        fh = getattr(self, "_lock_fh", None)
        if fh is not None and not fh.closed:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- loading -----------------------------------------------------------

    def _remove_leftover_temps(self) -> None:
        for directory in (self.run_dir, self.cache_dir):
            for tmp in directory.glob("*" + _TMP_SUFFIX):
                tmp.unlink(missing_ok=True)

    def _load(self) -> None:
        path = self.run_dir / _LEDGER_FILE
        if not path.exists():
            return
        try:
            doc = json.loads(path.read_bytes())
        except ValueError as exc:
            raise LedgerCorrupt(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise LedgerCorrupt(f"{path} has an unknown schema")
        entries = doc.get("entries")
        if not isinstance(entries, dict):
            raise LedgerCorrupt(f"{path} has no entries table")
        for key, rec in entries.items():
            if not isinstance(rec, dict) or rec.get("state") not in STATES:
                raise LedgerCorrupt(f"{path}: entry {key} has no valid state")
        self._entries = entries
        self._submit_seq = int(doc.get("submit_seq", 0))

    def _reconcile_with_cache(self) -> None:
        """Promote calls whose answer was cached before their state was written."""
        changed = []
        for key, rec in self._entries.items():
            if rec["state"] in (DONE, INVALID):
                continue
            meta = self._read_cache_doc(key)
            if meta is None:
                continue
            verdict = meta.get("verdict", DONE)
            self._set(key, verdict, reason=meta.get("reason"))
            changed.append(key)
        if changed:
            self._commit("recovered_from_cache", changed)

    # -- persistence -------------------------------------------------------

    def _commit(self, event: str, keys: Iterable[str], **extra: Any) -> None:
        doc = {"schema": SCHEMA, "submit_seq": self._submit_seq, "entries": self._entries}
        _atomic_write(self.run_dir / _LEDGER_FILE, _dump(doc))
        self._log(event, sorted(keys), **extra)

    def _log(self, event: str, keys: list[str], **extra: Any) -> None:
        """Best-effort audit trail; timestamps live here, never in results."""
        line = json.dumps({"ts": time.time(), "event": event, "keys": keys, **extra})
        try:
            with open(self.run_dir / _EVENTS_FILE, "a", encoding="ascii") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache_doc(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            doc = json.loads(path.read_bytes())
        except ValueError as exc:
            raise LedgerCorrupt(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(doc, dict) or doc.get("key") != key:
            raise LedgerCorrupt(f"{path} does not belong to key {key}")
        return doc

    def _write_cache(self, key: str, response: CachedResponse, verdict: str, reason: str | None):
        doc = {"key": key, "reason": reason, "verdict": verdict, **response.to_json()}
        _atomic_write(self._cache_path(key), _dump(doc))

    # -- state helpers -----------------------------------------------------

    def _rec(self, key: str) -> dict[str, Any]:
        try:
            return self._entries[key]
        except KeyError:
            raise LedgerStateError(f"unknown ledger key {key}") from None

    def _require(self, key: str, allowed: tuple[str, ...], action: str) -> dict[str, Any]:
        rec = self._rec(key)
        if rec["state"] not in allowed:
            raise LedgerStateError(f"cannot {action} {key}: it is {rec['state']}")
        return rec

    def _set(self, key: str, state: str, **fields: Any) -> None:
        rec = self._entries[key]
        rec["state"] = state
        for name in ("batch_id", "reason", "submit_token"):
            rec.pop(name, None)
        for name, value in fields.items():
            if value is not None:
                rec[name] = value

    # -- public API: registration -----------------------------------------

    def register(self, spec: CallSpec) -> str:
        """Add one call as pending (no-op if already known). Returns its key."""
        return self.register_many([spec])[0]

    def register_many(self, specs: Iterable[CallSpec]) -> list[str]:
        """Add calls as pending in one durable write; known keys keep their state."""
        keys, new = [], []
        for spec in specs:
            key = ledger_key(spec)
            keys.append(key)
            if key not in self._entries:
                self._entries[key] = {"spec": spec.to_json(), "state": PENDING}
                new.append(key)
        if new:
            self._commit("registered", new)
        return keys

    # -- public API: reading ----------------------------------------------

    def entry(self, key: str) -> Entry:
        rec = self._rec(key)
        return Entry(
            key=key,
            spec=dict(rec["spec"]),
            state=rec["state"],
            batch_id=rec.get("batch_id"),
            reason=rec.get("reason"),
            submit_token=rec.get("submit_token"),
        )

    def entries(self) -> list[Entry]:
        return [self.entry(key) for key in sorted(self._entries)]

    def keys(self, state: str | None = None) -> list[str]:
        """Sorted keys, optionally only those in ``state``."""
        return sorted(k for k, r in self._entries.items() if state is None or r["state"] == state)

    def submitted_batches(self) -> dict[str, list[str]]:
        """batch id -> its keys still in ``submitted`` (the batches to poll)."""
        out: dict[str, list[str]] = {}
        for key in self.keys(SUBMITTED):
            out.setdefault(self._entries[key]["batch_id"], []).append(key)
        return out

    def continue_plan(self) -> ContinuePlan:
        orphans: dict[str, list[str]] = {}
        fresh = []
        for key in self.keys(PENDING):
            token = self._entries[key].get("submit_token")
            if token:
                orphans.setdefault(token, []).append(key)
            else:
                fresh.append(key)
        return ContinuePlan(
            done=self.keys(DONE),
            invalid=self.keys(INVALID),
            batches=self.submitted_batches(),
            orphans=orphans,
            pending=fresh,
        )

    def cached(self, key: str) -> CachedResponse | None:
        """The cached answer for ``key`` (checked against its sha256), or None."""
        doc = self._read_cache_doc(key)
        if doc is None:
            return None
        try:
            raw = base64.b64decode(doc["raw_b64"], validate=True)
            if hashlib.sha256(raw).hexdigest() != doc["raw_sha256"]:
                raise LedgerCorrupt(f"cache for {key} fails its sha256 check")
            return CachedResponse(
                raw=raw,
                model_id=doc["model_id"],
                response_id=doc["response_id"],
                usage=dict(doc["usage"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerCorrupt(f"cache for {key} is malformed: {exc}") from exc

    # -- public API: transitions ------------------------------------------

    def begin_submit(self, keys: list[str]) -> str:
        """Record the intent to submit ``keys`` and return a unique submit token.

        Call before the provider's submit and pass the token to the provider
        (as batch metadata / request ids) so a batch accepted just before a
        crash can be found again instead of being paid for twice.
        """
        for key in keys:
            self._require(key, (PENDING,), "submit")
        self._submit_seq += 1
        digest = hashlib.sha256(
            canonical_json({"seq": self._submit_seq, "keys": sorted(keys)}).encode("ascii")
        ).hexdigest()
        token = f"tj-{digest[:32]}"
        for key in keys:
            self._set(key, PENDING, submit_token=token)
        self._commit("submit_begun", keys, token=token)
        return token

    def abandon_submit(self, keys: list[str]) -> None:
        """The provider has no batch for these keys' token: make them plain pending."""
        for key in keys:
            self._require(key, (PENDING,), "abandon submission of")
            self._set(key, PENDING)
        self._commit("submit_abandoned", keys)

    def mark_submitted(self, keys: list[str], batch_id: str) -> None:
        """``keys`` are now in provider batch ``batch_id``."""
        if not batch_id:
            raise LedgerStateError("batch_id must be non-empty")
        for key in keys:
            self._require(key, (PENDING,), "mark submitted")
        for key in keys:
            self._set(key, SUBMITTED, batch_id=batch_id)
        self._commit("submitted", keys, batch_id=batch_id)

    def record_done(self, key: str, response: CachedResponse) -> None:
        """Cache the answer (first) and mark ``key`` done (second)."""
        self._require(key, (PENDING, SUBMITTED), "record done for")
        self._write_cache(key, response, DONE, None)
        self._set(key, DONE)
        self._commit("done", [key])

    def mark_invalid(self, key: str, reason: str, response: CachedResponse | None = None) -> None:
        """The model's answer was bad. Only bad answers go here, never infra stops.

        Pass ``response`` so the bad answer is cached and not paid for again.
        """
        if not reason:
            raise LedgerStateError("an invalid call needs a reason")
        self._require(key, (PENDING, SUBMITTED), "mark invalid")
        if response is not None:
            self._write_cache(key, response, INVALID, reason)
        self._set(key, INVALID, reason=reason)
        self._commit("invalid", [key], reason=reason)

    def return_to_pending(self, keys: list[str], reason: str | None = None) -> None:
        """Infrastructure stop (or operator re-grade): submitted/invalid -> pending.

        Also drops any cached answer so the call is really sent again.
        """
        for key in keys:
            self._require(key, (PENDING, SUBMITTED, INVALID), "return to pending")
        for key in keys:
            self._cache_path(key).unlink(missing_ok=True)
            self._set(key, PENDING)
        if keys:
            _fsync_dir(self.cache_dir)
        self._commit("returned_to_pending", keys, reason=reason)

    def requeue_batch(self, batch_id: str, reason: str | None = None) -> list[str]:
        """Batch ended/expired/failed/cancelled: its *unfinished* keys go back to pending.

        Keys already done or invalid are untouched. Returns the requeued keys.
        """
        keys = self.submitted_batches().get(batch_id, [])
        for key in keys:
            self._set(key, PENDING)
        if keys:
            self._commit("batch_requeued", keys, batch_id=batch_id, reason=reason)
        return keys

    # -- public API: results ----------------------------------------------

    def results(self) -> dict[str, Any]:
        """Deterministic results document (no timestamps, batch ids or tokens)."""
        out: dict[str, Any] = {}
        for key in self.keys():
            rec = self._entries[key]
            item: dict[str, Any] = {"spec": rec["spec"], "state": rec["state"]}
            if rec.get("reason"):
                item["reason"] = rec["reason"]
            if rec["state"] in (DONE, INVALID):
                response = self.cached(key)
                if response is not None:
                    item["response"] = response.to_json()
            out[key] = item
        return {"schema": SCHEMA, "calls": out}

    def results_json(self) -> bytes:
        """:meth:`results` as canonical bytes (sorted keys, indent 2, trailing newline)."""
        return _dump(self.results())
