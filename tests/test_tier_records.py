"""Tests for nvsh.tiers.records: JSONL tier measurement records with rotation.

Covers spec targets c6, c35, h30: tier measurement records written as JSONL
under XDG_STATE_HOME/nvsh, redacted through nvsh/redact.py, rotation by size
cap, and file permissions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from nvsh.tiers.records import DEFAULT_CAP_BYTES, ROTATED_FILES, TierRecord, TierRecords

# A secret token that the redactor should catch and replace.
_SECRET_TEXT = "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456"


def _write_records(
    tmp_path: Path,
    *,
    cap_bytes: int = DEFAULT_CAP_BYTES,
    store_request_text: bool = False,
    clock: Callable[[], float] | None = None,
) -> TierRecords:
    """Create a TierRecords backed by a tmp_path fixture (never touches real home)."""
    parent = tmp_path / "nvsh"
    parent.mkdir(parents=True)
    path = parent / "tiers.jsonl"
    if clock is None:
        clock = time.time  # type: ignore[assignment]
    return TierRecords(
        path=path, cap_bytes=cap_bytes, store_request_text=store_request_text, clock=clock
    )


# -- fields and permissions --


def test_record_fields_and_permissions(tmp_path):
    """File mode 0o600, directory mode 0o700, all fields present, confidence None is null."""
    recs = _write_records(tmp_path)

    record = TierRecord(
        tier="needle",
        request_kind="failure",
        operation="diagnose",
        args={"foo": "bar"},
        confidence=0.95,
        latency_ms=123.4,
        decline_reason=None,
        escalated_to="lfm",
        operator_decision="approved",
        request_text=_SECRET_TEXT,
    )
    recs.write(record)

    # Directory permissions
    dir_path = Path(recs.path.parent)
    dir_stat = dir_path.stat()
    assert dir_stat.st_mode & 0o777 == 0o700

    # File permissions
    file_stat = recs.path.stat()
    assert file_stat.st_mode & 0o777 == 0o600

    # Read back and check all fields
    entries = recs.read_all()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["tier"] == "needle"
    assert entry["request_kind"] == "failure"
    assert entry["operation"] == "diagnose"
    assert entry["args"] == {"foo": "bar"}
    assert entry["confidence"] == 0.95
    assert entry["latency_ms"] == 123.4
    assert entry["decline_reason"] is None
    assert entry["escalated_to"] == "lfm"
    assert entry["operator_decision"] == "approved"
    assert "request_text" not in entry  # default: not stored
    assert entry["request_chars"] == len(_SECRET_TEXT)
    assert "ts" in entry
    assert isinstance(entry["ts"], float)


def test_record_fields_with_none_tier(tmp_path):
    """A TierRecord where optional fields are None still writes cleanly."""
    recs = _write_records(tmp_path)

    record = TierRecord(tier="agent", request_kind="slash")
    recs.write(record)

    entries = recs.read_all()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["tier"] == "agent"
    assert entry["request_kind"] == "slash"
    assert entry["operation"] is None
    assert entry["args"] == {}
    assert entry["confidence"] is None
    assert entry["latency_ms"] == 0.0
    assert entry["decline_reason"] is None
    assert entry["escalated_to"] is None
    assert entry["operator_decision"] is None


# -- request_text not stored by default --


def test_request_text_not_stored_by_default(tmp_path):
    """Only request_chars appears when store_request_text is False (the default)."""
    recs = _write_records(tmp_path, store_request_text=False)

    record = TierRecord(
        tier="needle",
        request_kind="explicit",
        request_text="hello world",
    )
    recs.write(record)

    # Read raw file — ensure "request_text" key is absent
    raw = recs.path.read_text(encoding="utf-8")
    line = raw.strip()
    entry = json.loads(line)
    assert "request_text" not in entry
    assert entry["request_chars"] == len("hello world")


# -- request_text stored and redacted when opted in --


def test_request_text_stored_redacted_when_opted_in(tmp_path):
    """When store_request_text is True, the token value is not present in the file."""
    recs = _write_records(tmp_path, store_request_text=True)

    record = TierRecord(
        tier="lfm",
        request_kind="failure",
        request_text=_SECRET_TEXT,
    )
    recs.write(record)

    raw = recs.path.read_text(encoding="utf-8")
    assert _SECRET_TEXT not in raw
    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in raw

    # The key must be present with the redacted value; env_assignment redacts
    # the whole KEY=value form, so <REDACTED:env_assignment> is present.
    entry = json.loads(raw.strip())
    assert "request_text" in entry
    assert "<REDACTED:env_assignment>" in entry["request_text"]


# -- args are redacted --


def test_args_are_redacted(tmp_path):
    """String values inside args dict are redacted."""
    recs = _write_records(tmp_path, store_request_text=True)

    # Assembled at runtime so scripts/scan-secrets.py sees no key-shaped literal.
    planted = "sk-" + "secretvalue" + "12345678901234567890"
    record = TierRecord(
        tier="needle",
        request_kind="failure",
        args={"service": planted},
        request_text="run deploy",
    )
    recs.write(record)

    raw = recs.path.read_text(encoding="utf-8")
    assert planted not in raw
    entry = json.loads(raw.strip())
    assert "<REDACTED:openai_key>" in entry["args"]["service"]


# -- rotation keeps total under cap --


def test_rotation_keeps_total_under_cap(tmp_path):
    """cap_bytes=4000, 500 records: total <= 4000, <= 4 files, newest present."""
    recs = _write_records(tmp_path, cap_bytes=4000)

    for i in range(500):
        recs.write(
            TierRecord(
                tier="needle",
                request_kind="failure",
                request_text=f"operation {i}" * 10,
            )
        )

    # Count files that exist
    tier_dir = recs.path.parent
    tier_files = sorted(tier_dir.glob("tiers.jsonl*"))
    assert len(tier_files) <= 4  # ROTATED_FILES

    # Total size under cap
    total_size = sum(f.stat().st_size for f in tier_files)
    assert total_size <= 4000

    # The newest record must be in the primary file (request_chars when
    # store_request_text=False, the default).
    newest_raw = recs.path.read_text(encoding="utf-8")
    newest_entry = json.loads(newest_raw.strip().split("\n")[-1])
    assert newest_entry["request_chars"] == len("operation 499" * 10)


# -- read_all is oldest-first across rotated files --


def test_read_all_is_oldest_first_across_rotated_files(tmp_path):
    """Records read across rotated files are returned oldest first."""
    clock_vals = [100.0, 200.0, 300.0, 400.0, 500.0]
    clock_idx = [0]

    def monotonic_clock() -> float:
        val = clock_vals[clock_idx[0]]
        clock_idx[0] += 1
        return val

    recs = _write_records(tmp_path, cap_bytes=600, clock=monotonic_clock)

    # Write 5 records with cap=600 to force rotation
    for i in range(5):
        recs.write(
            TierRecord(
                tier="needle",
                request_kind="failure",
                request_text=f"record {i} with enough padding to fill lines quickly",
            )
        )

    all_entries = recs.read_all()
    timestamps = [e["ts"] for e in all_entries]
    assert timestamps == sorted(timestamps), "read_all must return entries oldest first"


# -- write swallows OSError --


def test_write_swallows_oserror(tmp_path):
    """Writing to a path inside a directory that is itself a regular file never raises."""
    # Create a regular file where the parent directory should be
    trap_file = tmp_path / "notadir"
    trap_file.write_text("blocked", encoding="utf-8")

    # Point TierRecords at a path whose parent is the trap file
    bad_path = trap_file / "tiers.jsonl"
    recs = TierRecords(path=bad_path)

    # write() must NOT raise
    record = TierRecord(tier="needle", request_kind="failure")
    recs.write(record)


# -- DEFAULT_CAP_BYTES and ROTATED_FILES constants --


def test_constants_are_correct():
    assert DEFAULT_CAP_BYTES == 8 * 1024 * 1024
    assert ROTATED_FILES == 4


def test_cap_smaller_than_one_record_terminates(tmp_path):
    """A zero cap must drop records, never spin (config allows records_cap_mb = 0)."""
    records = TierRecords(tmp_path / "state" / "tiers.jsonl", cap_bytes=0)
    for _ in range(3):
        records.write(TierRecord(tier="needle", request_kind="explicit"))
    assert records.read_all() == []


def test_read_all_skips_a_torn_line(tmp_path):
    path = tmp_path / "state" / "tiers.jsonl"
    records = TierRecords(path)
    records.write(TierRecord(tier="needle", request_kind="explicit"))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    records.write(TierRecord(tier="lfm", request_kind="failure"))
    assert [e["tier"] for e in records.read_all()] == ["needle", "lfm"]
