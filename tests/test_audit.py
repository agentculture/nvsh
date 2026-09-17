"""Tests for nvsh.agent.audit (task t18): the resolved target on every line.

Covers acceptance criterion 2 -- ``audit.py`` records the resolved target on
every proposal/decision/outcome line -- and checks that existing call sites
(``nvsh/agent/loop.py``, ``nvsh/installers.py``), which never pass ``target``,
keep working: they get ``"target": None`` rather than a KeyError or a
required-argument failure. See ``tests/test_approval_loop.py`` and
``tests/test_installers.py`` for those call sites' own behavioral coverage,
left untouched by this task.

Also covers task t3 (reliable-agent-stop): the ``EXIT_DECLINED`` exit-code
constant in ``nvsh/cli/_errors.py`` and ``AuditLog.record_stop`` -- the
dedicated stop/cancel/steer/replace/busy-exit/declined/doctor-apply audit
event that later tasks (t17, t18) call from the client and doctor paths.
"""

from __future__ import annotations

import pytest

from nvsh.agent.audit import AuditLog
from nvsh.agent.base import Target
from nvsh.cli._errors import EXIT_DECLINED, EXIT_ENV_ERROR, EXIT_SUCCESS, EXIT_USER_ERROR


def test_record_without_a_target_defaults_to_none(tmp_path):
    """Existing callers (loop.py, installers.py) never pass ``target``."""
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    entry = audit.record(event="proposal", proposal="ls -la")
    assert entry["target"] is None
    assert audit.read_all()[0]["target"] is None


def test_record_with_a_target_dataclass_is_encoded_via_target_to_dict(tmp_path):
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    target = Target(backend="pi", model="associate", effort="high", alias="assoc")
    entry = audit.record(event="proposal", proposal="ls -la", target=target)
    assert entry["target"] == {
        "backend": "pi",
        "model": "associate",
        "effort": "high",
        "alias": "assoc",
    }


def test_record_with_a_bare_backend_target_omits_none_fields(tmp_path):
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    target = Target(backend="claude")
    entry = audit.record(event="decision", decision=True, target=target)
    assert entry["target"] == {"backend": "claude"}


def test_record_accepts_a_plain_dict_target(tmp_path):
    """A caller that already decoded a wire ``target`` may pass the dict."""
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    entry = audit.record(event="outcome", outcome=0, target={"backend": "codex"})
    assert entry["target"] == {"backend": "codex"}


def test_the_resolved_target_is_recorded_on_proposal_decision_and_outcome(tmp_path):
    """The full proposal/decision/outcome triple all carry the same target."""
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    target = Target(backend="pi", model="associate")
    audit.record(event="proposal", proposal="df -h", target=target)
    audit.record(event="decision", proposal="df -h", decision=True, target=target)
    audit.record(event="outcome", proposal="df -h", outcome=0, target=target)

    entries = audit.read_all()
    assert [e["event"] for e in entries] == ["proposal", "decision", "outcome"]
    for entry in entries:
        assert entry["target"] == {"backend": "pi", "model": "associate"}


def test_record_round_trips_through_json_on_disk(tmp_path):
    """The written JSONL line, read back fresh, still carries the target."""
    path = tmp_path / "audit.jsonl"
    AuditLog(path=path).record(event="proposal", target=Target(backend="fake"))
    reloaded = AuditLog(path=path).read_all()
    assert reloaded[0]["target"] == {"backend": "fake"}


def test_exit_declined_is_3_and_distinct_from_the_other_exit_codes():
    """EXIT_DECLINED is the first of the reserved 3+ range (t3 criterion 1)."""
    assert EXIT_DECLINED == 3
    assert EXIT_DECLINED not in (EXIT_SUCCESS, EXIT_USER_ERROR, EXIT_ENV_ERROR, 130)


def test_record_stop_writes_one_json_line_with_the_stop_shape(tmp_path):
    """record_stop writes event='stop' plus kind, shell, target, elapsed, outcome."""
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    target = Target(backend="pi", model="associate")
    entry = audit.record_stop(
        kind="cancel",
        shell="12345",
        target=target,
        elapsed=1.5,
        outcome="cancelled",
    )
    assert entry["event"] == "stop"
    assert entry["kind"] == "cancel"
    assert entry["shell"] == "12345"
    assert entry["target"] == {"backend": "pi", "model": "associate"}
    assert entry["elapsed"] == 1.5
    assert entry["outcome"] == "cancelled"

    entries = audit.read_all()
    assert len(entries) == 1
    assert entries[0] == entry


@pytest.mark.parametrize(
    "kind",
    ["cancel", "force_kill", "steer", "replace", "busy_exit", "declined", "doctor_apply"],
)
def test_record_stop_accepts_every_documented_kind(tmp_path, kind):
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    entry = audit.record_stop(kind=kind, shell="1", target=None, elapsed=0.1, outcome=None)
    assert entry["kind"] == kind


def test_record_stop_rejects_an_unknown_kind(tmp_path):
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    with pytest.raises(ValueError):
        audit.record_stop(kind="bogus", shell="1", target=None, elapsed=0.1, outcome=None)


def test_record_stop_accepts_a_plain_dict_target(tmp_path):
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    entry = audit.record_stop(
        kind="force_kill",
        shell="7",
        target={"backend": "codex"},
        elapsed=3.0,
        outcome=1,
    )
    assert entry["target"] == {"backend": "codex"}


def test_record_stop_defaults_target_to_none(tmp_path):
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    entry = audit.record_stop(kind="declined", shell="7", target=None, elapsed=0.0, outcome=None)
    assert entry["target"] is None
