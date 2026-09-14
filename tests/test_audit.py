"""Tests for nvsh.agent.audit (task t18): the resolved target on every line.

Covers acceptance criterion 2 -- ``audit.py`` records the resolved target on
every proposal/decision/outcome line -- and checks that existing call sites
(``nvsh/agent/loop.py``, ``nvsh/installers.py``), which never pass ``target``,
keep working: they get ``"target": None`` rather than a KeyError or a
required-argument failure. See ``tests/test_approval_loop.py`` and
``tests/test_installers.py`` for those call sites' own behavioral coverage,
left untouched by this task.
"""

from __future__ import annotations

from nvsh.agent.audit import AuditLog
from nvsh.agent.base import Target


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
