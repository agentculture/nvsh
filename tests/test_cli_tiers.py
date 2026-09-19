"""Tests for ``nvsh tiers`` — stats/export/prefetch (task t20).

Acceptance criteria covered:
- ``nvsh tiers stats [--json]`` prints per-tier counts, latency p50/p95,
  escalation reasons and approve/decline rates from local records.
- ``nvsh tiers export <file>`` writes a redacted bundle to a local path
  only and opens no socket; ``nvsh tiers prefetch`` shows sizes and asks
  before downloading.
- every verb supports --json and has an explain catalog entry.
"""

from __future__ import annotations

import json
import socket
import stat

import pytest

from nvsh.cli import main
from nvsh.explain.catalog import ENTRIES
from nvsh.tiers.records import TierRecord, TierRecords


@pytest.fixture(autouse=True)
def xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _records_path(tmp_path):
    return tmp_path / "state" / "nvsh" / "tiers.jsonl"


def _write_records(tmp_path, records):
    store = TierRecords(path=_records_path(tmp_path))
    for record in records:
        store.write(record)
    return store


class _SocketGuard:
    """Raises if anything under it touches socket.socket/create_connection."""

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch

    def __enter__(self):
        def _boom(*_args, **_kwargs):
            raise AssertionError("socket access attempted")

        self._monkeypatch.setattr(socket, "socket", _boom)
        self._monkeypatch.setattr(socket, "create_connection", _boom)
        return self

    def __exit__(self, *_exc):
        return False


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_tiers_stats_empty_json(tmp_path, capsys):
    rc = main(["tiers", "stats", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 0
    assert payload["dropped"] == 0
    assert payload["tiers"] == {}
    assert payload["escalation_reasons"] == {}
    assert payload["operator_decisions"] == {
        "approved": 0,
        "declined": 0,
        "approve_rate": 0.0,
        "decline_rate": 0.0,
    }


def test_tiers_stats_per_tier_counts_and_latency(tmp_path, capsys):
    records = [
        TierRecord(tier="needle", request_kind="failure", latency_ms=10.0),
        TierRecord(tier="needle", request_kind="failure", latency_ms=20.0),
        TierRecord(tier="needle", request_kind="failure", latency_ms=30.0),
        TierRecord(tier="needle", request_kind="failure", latency_ms=40.0),
        TierRecord(tier="agent", request_kind="explicit", latency_ms=500.0),
    ]
    _write_records(tmp_path, records)

    rc = main(["tiers", "stats", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 5
    assert payload["tiers"]["needle"]["count"] == 4
    # nearest-rank p50 of [10,20,30,40] (rank = ceil(0.5*4)=2) -> 20.0
    assert payload["tiers"]["needle"]["latency_p50_ms"] == 20.0
    # nearest-rank p95 of [10,20,30,40] (rank = ceil(0.95*4)=4) -> 40.0
    assert payload["tiers"]["needle"]["latency_p95_ms"] == 40.0
    assert payload["tiers"]["agent"]["count"] == 1


def test_tiers_stats_escalation_reasons_and_decision_rates(tmp_path, capsys):
    records = [
        TierRecord(
            tier="needle",
            request_kind="failure",
            decline_reason="low_confidence",
            escalated_to="lfm",
            operator_decision="approved",
        ),
        TierRecord(
            tier="needle",
            request_kind="failure",
            decline_reason="low_confidence",
            escalated_to="lfm",
            operator_decision="declined",
        ),
        TierRecord(
            tier="lfm",
            request_kind="failure",
            decline_reason="unsupported_operation",
            escalated_to="agent",
            operator_decision="approved",
        ),
    ]
    _write_records(tmp_path, records)

    rc = main(["tiers", "stats", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["escalation_reasons"] == {
        "low_confidence": 2,
        "unsupported_operation": 1,
    }
    decisions = payload["operator_decisions"]
    assert decisions["approved"] == 2
    assert decisions["declined"] == 1
    assert decisions["approve_rate"] == pytest.approx(2 / 3)
    assert decisions["decline_rate"] == pytest.approx(1 / 3)


def test_tiers_stats_text_mode_is_readable(tmp_path, capsys):
    _write_records(
        tmp_path,
        [TierRecord(tier="needle", request_kind="failure", latency_ms=15.0)],
    )
    rc = main(["tiers", "stats"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "needle" in out
    assert "p50=" in out
    assert "p95=" in out


def test_tiers_stats_no_hardcoded_tier_names(tmp_path, capsys):
    """A tier name never seen before ('needle'/'lfm'/'agent') still groups correctly."""
    _write_records(
        tmp_path,
        [TierRecord(tier="future-tier", request_kind="failure", latency_ms=1.0)],
    )
    rc = main(["tiers", "stats", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "future-tier" in payload["tiers"]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_tiers_export_writes_local_file_mode_0600(tmp_path, capsys):
    _write_records(
        tmp_path,
        [TierRecord(tier="needle", request_kind="failure", latency_ms=1.0)],
    )
    dest = tmp_path / "bundle.json"
    rc = main(["tiers", "export", str(dest), "--json"])
    assert rc == 0
    assert dest.exists()
    mode = stat.S_IMODE(dest.stat().st_mode)
    assert mode == 0o600
    bundle = json.loads(dest.read_text(encoding="utf-8"))
    assert "nvsh_version" in bundle
    assert "platform" in bundle
    assert len(bundle["records"]) == 1


def test_tiers_export_redacts_secret_shaped_content(tmp_path):
    secret_env_var = "HF_TOKEN"
    fake_token = "hf_" + ("x" * 32)
    store = TierRecords(path=_records_path(tmp_path), store_request_text=True)
    store.write(
        TierRecord(
            tier="needle",
            request_kind="failure",
            request_text=f"{secret_env_var}={fake_token} failed",
        )
    )
    dest = tmp_path / "bundle.json"
    rc = main(["tiers", "export", str(dest)])
    assert rc == 0
    contents = dest.read_text(encoding="utf-8")
    assert fake_token not in contents


def test_tiers_export_refuses_url_target(capsys):
    dest = "https://example.com/bundle.json"
    rc = main(["tiers", "export", dest, "--json"])
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "URL" in err["message"]


def test_tiers_export_refuses_scp_style_remote_target(capsys):
    dest = "operator@example.com:/tmp/bundle.json"
    rc = main(["tiers", "export", dest, "--json"])
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "--force" not in err["remediation"]


def test_tiers_export_refuses_overwrite_without_force(tmp_path, capsys):
    dest = tmp_path / "bundle.json"
    dest.write_text("{}", encoding="utf-8")
    rc = main(["tiers", "export", str(dest), "--json"])
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "--force" in err["remediation"]


def test_tiers_export_overwrite_with_force(tmp_path):
    dest = tmp_path / "bundle.json"
    dest.write_text("{}", encoding="utf-8")
    rc = main(["tiers", "export", str(dest), "--force"])
    assert rc == 0
    bundle = json.loads(dest.read_text(encoding="utf-8"))
    assert "records" in bundle


def test_tiers_export_opens_no_socket(tmp_path, monkeypatch):
    dest = tmp_path / "bundle.json"
    with _SocketGuard(monkeypatch):
        rc = main(["tiers", "export", str(dest)])
    assert rc == 0


# ---------------------------------------------------------------------------
# prefetch
# ---------------------------------------------------------------------------


def test_tiers_prefetch_json_without_yes_refuses_and_names_flag(capsys):
    rc = main(["tiers", "prefetch", "--json"])
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "--yes" in err["remediation"]


def test_tiers_prefetch_text_without_yes_refuses(capsys):
    # Under pytest, stdin is not a tty, so this hits the same non-interactive
    # refusal path as --json without --yes.
    rc = main(["tiers", "prefetch"])
    assert rc != 0
    out = capsys.readouterr()
    assert out.out == ""


def test_tiers_prefetch_yes_shows_sizes_and_uses_injected_prefetch(monkeypatch, capsys):
    calls = []

    def _fake_prefetch(items, **kwargs):
        calls.append((list(items), kwargs))
        return []

    monkeypatch.setattr("nvsh.tiers.fetch.prefetch", _fake_prefetch)

    rc = main(["tiers", "prefetch", "--yes", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "items" in payload
    assert payload["items"], "expected at least one pinned item from pins.json"
    assert payload["problems"] == []
    for row in payload["items"]:
        assert "size_bytes" in row
        assert "present" in row
    assert len(calls) == 1


def test_tiers_prefetch_yes_reports_problems_from_injected_prefetch(monkeypatch, capsys):
    from dataclasses import dataclass

    @dataclass
    class _FakeProblem:
        item: str
        code: str
        message: str

    def _fake_prefetch(items, **kwargs):
        name = items[0].name if items else "unknown"
        return [_FakeProblem(item=name, code="download_failed", message="simulated failure")]

    monkeypatch.setattr("nvsh.tiers.fetch.prefetch", _fake_prefetch)

    rc = main(["tiers", "prefetch", "--yes", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["problems"][0]["code"] == "download_failed"


def test_tiers_prefetch_never_touches_network_without_yes(monkeypatch):
    with _SocketGuard(monkeypatch):
        rc = main(["tiers", "prefetch"])
    assert rc != 0


# ---------------------------------------------------------------------------
# --json + explain catalog coverage
# ---------------------------------------------------------------------------


def test_tiers_stats_json_flag_accepted(capsys):
    rc = main(["tiers", "stats", "--json"])
    assert rc == 0
    json.loads(capsys.readouterr().out)


def test_tiers_export_json_flag_accepted(tmp_path, capsys):
    dest = tmp_path / "bundle.json"
    rc = main(["tiers", "export", str(dest), "--json"])
    assert rc == 0
    json.loads(capsys.readouterr().out)


def test_tiers_prefetch_json_flag_accepted(capsys):
    rc = main(["tiers", "prefetch", "--json"])
    assert rc != 0
    json.loads(capsys.readouterr().err)


def test_explain_catalog_has_every_tiers_entry():
    for path in [("tiers",), ("tiers", "stats"), ("tiers", "export"), ("tiers", "prefetch")]:
        assert path in ENTRIES
        assert ENTRIES[path].strip()
