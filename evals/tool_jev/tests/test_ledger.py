"""Tests for the durable call ledger and response cache (task t9, issue #64).

Acceptance criteria covered here:

1. a fake run killed between two ledger writes (simulated by raising
   mid-update, at *every* write index in turn) continues to byte-identical
   results as an uninterrupted run, and never pays twice for a call whose
   answer it already has;
2. a warm-cache rerun makes zero provider calls and yields identical JSON;
3. a submitted batch id is re-attached and polled on continue, never
   resubmitted; an expired batch requeues only its unfinished keys.

``providers/base.py`` (task t10) is written in parallel, so the provider
here is a local, minimal fake with ``submit`` / ``poll`` / ``fetch`` /
``find`` that counts calls. ``run_until_settled`` below is the reference
runner showing how a real runner drives :class:`Ledger` — the ledger itself
never talks to a provider. No network, no subprocesses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from evals.tool_jev import ledger as ledger_mod
from evals.tool_jev.ledger import (
    DONE,
    INVALID,
    PENDING,
    SUBMITTED,
    CachedResponse,
    CallSpec,
    Ledger,
    LedgerCorrupt,
    LedgerLocked,
    LedgerStateError,
    ledger_key,
    prompt_hash,
)

# --------------------------------------------------------------------------
# Fake provider + reference runner
# --------------------------------------------------------------------------


class Crash(Exception):
    """Stands in for Ctrl+C / power-off in the middle of a ledger update."""


@dataclass
class FakeBatchProvider:
    """A batch provider fake: every call is counted, nothing touches a network.

    ``plan`` maps a submission index (0-based) to the status that batch ends
    in (``"ended"`` by default; ``"expired"`` ends with only
    ``partial`` of its keys answered). Answers are a pure function of the
    key, so two runs produce byte-identical responses. Keys in ``bad``
    produce an answer the grader rejects (-> INVALID).
    """

    plan: dict[int, str] = field(default_factory=dict)
    partial: int = 1
    bad: frozenset[str] = frozenset()
    submits: list[list[str]] = field(default_factory=list)
    polls: list[str] = field(default_factory=list)
    fetches: list[str] = field(default_factory=list)
    finds: list[str] = field(default_factory=list)
    _batches: dict[str, dict] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        return len(self.submits) + len(self.polls) + len(self.fetches) + len(self.finds)

    @property
    def paid_keys(self) -> list[str]:
        return [k for batch in self.submits for k in batch]

    def submit(self, keys: list[str], token: str) -> str:
        index = len(self.submits)
        self.submits.append(list(keys))
        batch_id = f"batch_{index:03d}"
        self._batches[batch_id] = {
            "keys": list(keys),
            "token": token,
            "status": self.plan.get(index, "ended"),
        }
        return batch_id

    def find(self, token: str) -> str | None:
        self.finds.append(token)
        for batch_id, batch in self._batches.items():
            if batch["token"] == token:
                return batch_id
        return None

    def poll(self, batch_id: str) -> str:
        self.polls.append(batch_id)
        return self._batches[batch_id]["status"]

    def fetch(self, batch_id: str) -> dict[str, CachedResponse]:
        self.fetches.append(batch_id)
        batch = self._batches[batch_id]
        keys = batch["keys"]
        if batch["status"] == "expired":
            keys = keys[: self.partial]
        return {key: _answer(key, key in self.bad) for key in keys}


def _answer(key: str, bad: bool) -> CachedResponse:
    text = "not json" if bad else json.dumps({"op": "disk_usage", "key": key[:8]})
    return CachedResponse(
        raw=json.dumps({"id": f"resp_{key[:12]}", "output": text}).encode(),
        model_id="fake-model-2026",
        response_id=f"resp_{key[:12]}",
        usage={"input_tokens": 11, "output_tokens": 7, "reasoning_tokens": 3},
    )


def _grade(response: CachedResponse) -> str | None:
    """Return an INVALID reason for a bad model answer, else None."""
    output = json.loads(response.raw)["output"]
    try:
        json.loads(output)
    except ValueError:
        return "answer is not JSON"
    return None


def run_until_settled(led: Ledger, provider: FakeBatchProvider, batch_size: int = 2) -> None:
    """Reference ``continue`` loop: done from cache, re-attach, then send."""
    plan = led.continue_plan()
    rounds = 0
    # Submissions whose batch id never reached disk: ask the provider first.
    for token, keys in plan.orphans.items():
        batch_id = provider.find(token)
        if batch_id is None:
            led.abandon_submit(keys)
        else:
            led.mark_submitted(keys, batch_id)
    while True:
        rounds += 1
        assert rounds < 100, "runner did not settle"
        for batch_id, keys in sorted(led.submitted_batches().items()):
            status = provider.poll(batch_id)
            if status not in ("ended", "expired"):
                continue
            for key, response in sorted(provider.fetch(batch_id).items()):
                if key not in keys:
                    continue
                reason = _grade(response)
                if reason is None:
                    led.record_done(key, response)
                else:
                    led.mark_invalid(key, reason, response=response)
            led.requeue_batch(batch_id)
        pending = led.keys(PENDING)
        if not pending:
            if not led.submitted_batches():
                return
            continue
        chunk = pending[:batch_size]
        token = led.begin_submit(chunk)
        batch_id = provider.submit(chunk, token)
        led.mark_submitted(chunk, batch_id)


def _specs(n: int = 5) -> list[CallSpec]:
    return [
        CallSpec(
            provider="fake",
            model="fake-model",
            subject_role="subject" if i % 2 == 0 else "judge",
            case_id=f"case-{i:03d}",
            target="interface:tool" if i % 2 == 0 else "judge:correctness",
            prompt_hash=prompt_hash(f"synthetic prompt {i}"),
            params={"temperature": 0.0, "max_tokens": 256},
        )
        for i in range(n)
    ]


def _fresh_run(run_dir: Path, provider: FakeBatchProvider, specs: list[CallSpec]) -> bytes:
    with Ledger(run_dir) as led:
        led.register_many(specs)
        run_until_settled(led, provider)
        return led.results_json()


# --------------------------------------------------------------------------
# Key
# --------------------------------------------------------------------------


def test_key_is_stable_and_covers_every_field():
    spec = _specs(1)[0]
    base = ledger_key(spec)
    assert base == ledger_key(CallSpec(**spec.__dict__))
    assert len(base) == 64 and int(base, 16) >= 0
    variants = {
        "provider": "other",
        "model": "other",
        "subject_role": "other",
        "case_id": "other",
        "target": "other",
        "prompt_hash": prompt_hash("other"),
        "params": {"temperature": 0.5, "max_tokens": 256},
    }
    for name, value in variants.items():
        changed = CallSpec(**{**spec.__dict__, name: value})
        assert ledger_key(changed) != base, name


def test_key_ignores_param_dict_order_and_rejects_nan():
    a = CallSpec("p", "m", "r", "c", "t", "h", {"a": 1, "b": 2})
    b = CallSpec("p", "m", "r", "c", "t", "h", {"b": 2, "a": 1})
    assert ledger_key(a) == ledger_key(b)
    with pytest.raises(ValueError):
        ledger_key(CallSpec("p", "m", "r", "c", "t", "h", {"a": float("nan")}))


def test_prompt_hash_accepts_text_and_bytes():
    assert prompt_hash("x") == prompt_hash(b"x") == hashlib.sha256(b"x").hexdigest()


# --------------------------------------------------------------------------
# States and durability
# --------------------------------------------------------------------------


def test_state_transitions_persist_across_reopen(tmp_path):
    specs = _specs(3)
    with Ledger(tmp_path) as led:
        k0, k1, k2 = led.register_many(specs)
        assert led.keys(PENDING) == sorted([k0, k1, k2])
        token = led.begin_submit([k0, k1])
        led.mark_submitted([k0, k1], "batch_x")
        assert token
        led.record_done(k0, _answer(k0, False))
        led.mark_invalid(k2, "answer is not JSON")
    with Ledger(tmp_path) as led:
        assert led.entry(k0).state == DONE
        assert led.entry(k1).state == SUBMITTED
        assert led.entry(k1).batch_id == "batch_x"
        assert led.entry(k2).state == INVALID
        assert led.entry(k2).reason == "answer is not JSON"
        assert led.cached(k0) == _answer(k0, False)
        assert led.submitted_batches() == {"batch_x": [k1]}
        led.return_to_pending([k2])
        assert led.entry(k2).state == PENDING
        assert led.entry(k2).reason is None


def test_register_is_idempotent_and_keeps_state(tmp_path):
    spec = _specs(1)[0]
    with Ledger(tmp_path) as led:
        (key,) = led.register_many([spec])
        led.record_done(key, _answer(key, False))
        assert led.register(spec) == key
        assert led.entry(key).state == DONE


def test_illegal_transitions_are_refused(tmp_path):
    with Ledger(tmp_path) as led:
        (key,) = led.register_many(_specs(1))
        with pytest.raises(LedgerStateError):
            led.mark_submitted(["unknown-key"], "b")
        assert led.requeue_batch("no-such-batch") == []
        led.record_done(key, _answer(key, False))
        with pytest.raises(LedgerStateError):
            led.mark_submitted([key], "b")
        with pytest.raises(LedgerStateError):
            led.return_to_pending([key])
        with pytest.raises(LedgerStateError):
            led.mark_invalid(key, "late")


def test_second_open_is_locked_out(tmp_path):
    with Ledger(tmp_path):
        with pytest.raises(LedgerLocked):
            Ledger(tmp_path)
    Ledger(tmp_path).close()


def test_leftover_temp_files_are_ignored(tmp_path):
    with Ledger(tmp_path) as led:
        (key,) = led.register_many(_specs(1))
    (tmp_path / "ledger.json.tmp").write_bytes(b'{"schema": 1, "entr')
    (tmp_path / "cache" / f"{key}.json.tmp").write_bytes(b"\x00\x01")
    with Ledger(tmp_path) as led:
        assert led.entry(key).state == PENDING
        assert led.cached(key) is None
    assert not (tmp_path / "ledger.json.tmp").exists()


def test_torn_ledger_is_reported_not_silently_reset(tmp_path):
    with Ledger(tmp_path) as led:
        led.register_many(_specs(1))
    (tmp_path / "ledger.json").write_bytes(b'{"schema": 1, "entr')
    with pytest.raises(LedgerCorrupt):
        Ledger(tmp_path)


def test_tampered_cache_is_reported(tmp_path):
    with Ledger(tmp_path) as led:
        (key,) = led.register_many(_specs(1))
        led.record_done(key, _answer(key, False))
    path = tmp_path / "cache" / f"{key}.json"
    doc = json.loads(path.read_text())
    doc["raw_sha256"] = "0" * 64
    path.write_text(json.dumps(doc))
    with Ledger(tmp_path) as led:
        with pytest.raises(LedgerCorrupt):
            led.cached(key)


def test_cache_written_but_state_not_is_promoted_on_open(tmp_path, monkeypatch):
    """Crash after the cache write, before the state write: never pay again."""
    with Ledger(tmp_path) as led:
        (key,) = led.register_many(_specs(1))
    real = ledger_mod._atomic_write

    def cache_then_crash(path, data):
        if Path(path).name == "ledger.json":
            raise Crash  # the cache write went through; the state write never happens
        real(path, data)

    led = Ledger(tmp_path)
    monkeypatch.setattr(ledger_mod, "_atomic_write", cache_then_crash)
    with pytest.raises(Crash):
        led.record_done(key, _answer(key, False))
    led.close()
    monkeypatch.setattr(ledger_mod, "_atomic_write", real)
    with Ledger(tmp_path) as led:
        assert led.entry(key).state == DONE
        assert led.continue_plan().pending == []


def test_results_json_is_deterministic_and_timestamp_free(tmp_path):
    specs = _specs(3)
    out = _fresh_run(tmp_path / "a", FakeBatchProvider(bad=frozenset()), specs)
    doc = json.loads(out)
    assert out == json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=True).encode() + b"\n"
    text = out.decode()
    assert "batch_" not in text and "2026-" not in text
    events = (tmp_path / "a" / "events.jsonl").read_text().splitlines()
    assert events and all("ts" in json.loads(line) for line in events)


# --------------------------------------------------------------------------
# Acceptance criterion 1: killed between two writes -> byte-identical results
# --------------------------------------------------------------------------


class _TornFile:
    """A binary file whose write stores only half the bytes, then dies."""

    def __init__(self, fh):
        self._fh = fh

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._fh.close()

    def write(self, data):
        self._fh.write(data[: len(data) // 2])
        self._fh.flush()
        raise Crash


def _tearing_open(path, mode="r", *args, **kwargs):
    fh = open(path, mode, *args, **kwargs)
    return _TornFile(fh) if "w" in mode and "b" in mode else fh


def _count_writes(tmp_path: Path, monkeypatch, specs, make_provider) -> int:
    real = ledger_mod._atomic_write
    count = 0

    def counting(path, data):
        nonlocal count
        count += 1
        real(path, data)

    monkeypatch.setattr(ledger_mod, "_atomic_write", counting)
    _fresh_run(tmp_path / "count", make_provider(), specs)
    monkeypatch.setattr(ledger_mod, "_atomic_write", real)
    return count


@pytest.mark.parametrize("torn", [False, True], ids=["before-rename", "torn-temp"])
def test_crash_at_every_write_continues_to_identical_results(tmp_path, monkeypatch, torn):
    specs = _specs(5)
    bad = frozenset({ledger_key(specs[3])})

    def make_provider():
        return FakeBatchProvider(bad=bad)

    baseline = _fresh_run(tmp_path / "baseline", make_provider(), specs)
    total = _count_writes(tmp_path, monkeypatch, specs, make_provider)
    assert total > 10
    real = ledger_mod._atomic_write

    for crash_at in range(1, total + 1):
        run_dir = tmp_path / f"crash{crash_at}"
        provider = make_provider()
        seen = 0

        def crashing(path, data):
            nonlocal seen
            seen += 1
            if seen == crash_at:
                if not torn:
                    raise Crash
                # power-off halfway through the bytes of this very write
                monkeypatch.setattr(ledger_mod, "open", _tearing_open, raising=False)
                try:
                    real(path, data)
                finally:
                    monkeypatch.delattr(ledger_mod, "open", raising=False)
            real(path, data)

        monkeypatch.setattr(ledger_mod, "_atomic_write", crashing)
        led = Ledger(run_dir)
        with pytest.raises(Crash):
            led.register_many(specs)
            run_until_settled(led, provider)
        led.close()  # the process died: its lock goes with it
        monkeypatch.setattr(ledger_mod, "_atomic_write", real)

        with Ledger(run_dir) as led:
            led.register_many(specs)
            run_until_settled(led, provider)
            assert led.results_json() == baseline, f"crash at write {crash_at}"
        paid = provider.paid_keys
        assert len(paid) == len(set(paid)) == len(specs), f"paid twice (crash {crash_at})"


def test_operator_stop_leaves_calls_pending_not_invalid(tmp_path):
    """Ctrl+C / budget stop between submissions: infrastructure keeps PENDING."""
    specs = _specs(4)

    class StopAfterFirst(FakeBatchProvider):
        def submit(self, keys, token):
            if self.submits:
                raise KeyboardInterrupt
            return super().submit(keys, token)

    provider = StopAfterFirst()
    with Ledger(tmp_path) as led:
        led.register_many(specs)
        with pytest.raises(KeyboardInterrupt):
            run_until_settled(led, provider)
        assert led.keys(INVALID) == []
        orphan_keys = [k for keys in led.continue_plan().orphans.values() for k in keys]
        assert len(orphan_keys) == 2  # the interrupted submission, not yet known
    resumed = FakeBatchProvider()
    resumed.submits = list(provider.submits)
    resumed._batches = dict(provider._batches)
    with Ledger(tmp_path) as led:
        run_until_settled(led, resumed)
        assert len(led.keys(DONE)) == 4
    assert sorted(resumed.paid_keys) == sorted(ledger_key(s) for s in specs)


# --------------------------------------------------------------------------
# Acceptance criterion 2: warm cache -> zero provider calls, identical JSON
# --------------------------------------------------------------------------


def test_warm_cache_rerun_makes_zero_provider_calls(tmp_path):
    specs = _specs(5)
    bad = frozenset({ledger_key(specs[1])})
    first = _fresh_run(tmp_path, FakeBatchProvider(bad=bad), specs)
    rerun_provider = FakeBatchProvider(bad=bad)
    second = _fresh_run(tmp_path, rerun_provider, specs)
    assert rerun_provider.calls == 0
    assert second == first
    with Ledger(tmp_path) as led:
        plan = led.continue_plan()
        assert sorted(plan.done) == sorted(led.keys(DONE))
        assert plan.pending == [] and plan.batches == {} and plan.orphans == {}
        invalid_key = ledger_key(specs[1])
        assert led.cached(invalid_key) is not None  # a bad answer is not paid for twice


# --------------------------------------------------------------------------
# Acceptance criterion 3: re-attach by batch id; expired -> requeue unfinished
# --------------------------------------------------------------------------


def test_submitted_batch_is_reattached_and_polled_not_resubmitted(tmp_path):
    specs = _specs(2)
    provider = FakeBatchProvider()
    with Ledger(tmp_path) as led:
        keys = led.register_many(specs)
        token = led.begin_submit(keys)
        batch_id = provider.submit(keys, token)
        led.mark_submitted(keys, batch_id)
    # --- stop / reset here; the batch keeps running server-side ---
    with Ledger(tmp_path) as led:
        plan = led.continue_plan()
        assert plan.batches == {batch_id: sorted(keys)}
        assert plan.pending == []
        run_until_settled(led, provider)
        assert led.keys(DONE) == sorted(keys)
    assert len(provider.submits) == 1
    assert provider.polls == [batch_id]


def test_expired_batch_requeues_only_its_unfinished_keys(tmp_path):
    specs = _specs(4)
    provider = FakeBatchProvider(plan={0: "expired"}, partial=1)
    with Ledger(tmp_path) as led:
        keys = led.register_many(specs)
        first = sorted(keys)[:3]
        other = sorted(keys)[3]
        token = led.begin_submit(first)
        led.mark_submitted(first, provider.submit(first, token))
        led.mark_submitted([other], "batch_other")
        answered = provider.fetch("batch_000")
        for key, response in answered.items():
            led.record_done(key, response)
        requeued = led.requeue_batch("batch_000")
        assert requeued == first[1:]
        assert led.keys(PENDING) == first[1:]
        assert led.entry(first[0]).state == DONE
        assert led.entry(other).state == SUBMITTED
        assert led.entry(other).batch_id == "batch_other"
        assert "batch_000" not in led.submitted_batches()


def test_orphaned_submission_is_found_by_token_not_resent(tmp_path):
    """Crash after the provider accepted a batch but before its id was saved."""
    specs = _specs(2)
    provider = FakeBatchProvider()
    with Ledger(tmp_path) as led:
        keys = led.register_many(specs)
        token = led.begin_submit(keys)
        provider.submit(keys, token)  # accepted server-side, then power-off
    with Ledger(tmp_path) as led:
        plan = led.continue_plan()
        assert plan.orphans == {token: sorted(keys)}
        run_until_settled(led, provider)
        assert led.keys(DONE) == sorted(keys)
    assert len(provider.submits) == 1
    assert provider.finds == [token]
