"""Wire-codec tests for the NvshAgent contract extensions (task t4).

Covers: EventKind.THINKING, AgentRequest.target (backend/model/effort/alias),
and the richer Capabilities fields. The wire codec must keep omitting
default-valued fields so old daemons/clients stay compatible, and decoding
must tolerate dicts that predate these fields.
"""

from nvsh.agent.base import (
    AgentRequest,
    Capabilities,
    EventKind,
    RequestKind,
    Target,
    event_from_dict,
    event_to_dict,
    request_from_dict,
    request_to_dict,
    target_from_dict,
    target_to_dict,
)


def test_event_kind_has_thinking():
    assert EventKind.THINKING == "thinking"
    assert EventKind("thinking") is EventKind.THINKING


def test_unknown_event_kind_degrades_to_status():
    event = event_from_dict({"kind": "some_future_kind", "text": "hi"})
    assert event.kind is EventKind.STATUS
    assert event.text == "hi"


def test_thinking_event_round_trips():
    from nvsh.agent.base import AgentEvent

    event = AgentEvent(kind=EventKind.THINKING, text="pondering...")
    data = event_to_dict(event)
    assert data == {"kind": "thinking", "text": "pondering..."}
    restored = event_from_dict(data)
    assert restored.kind is EventKind.THINKING
    assert restored.text == "pondering..."


def test_target_defaults_to_none_fields():
    target = Target(backend="claude")
    assert target.model is None
    assert target.effort is None
    assert target.alias is None


def test_target_is_frozen():
    target = Target(backend="claude")
    try:
        target.backend = "codex"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("Target should be frozen/immutable")


def test_target_to_dict_omits_defaults():
    target = Target(backend="claude")
    assert target_to_dict(target) == {"backend": "claude"}


def test_target_to_dict_includes_set_fields():
    target = Target(backend="claude", model="sonnet", effort="high", alias="assoc")
    assert target_to_dict(target) == {
        "backend": "claude",
        "model": "sonnet",
        "effort": "high",
        "alias": "assoc",
    }


def test_target_round_trips():
    target = Target(backend="codex", model="gpt-5", effort="low", alias="pi")
    restored = target_from_dict(target_to_dict(target))
    assert restored == target


def test_target_from_dict_none_and_missing():
    assert target_from_dict(None) is None
    assert target_from_dict({}) is None


def test_agent_request_target_defaults_none():
    request = AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi")
    assert request.target is None


def test_request_to_dict_omits_target_when_none():
    request = AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi")
    data = request_to_dict(request)
    assert "target" not in data
    assert data == {"kind": "explicit", "prompt": "hi"}


def test_request_round_trips_with_target():
    request = AgentRequest(
        kind=RequestKind.FAILURE,
        prompt="explain",
        command="ls -z",
        exit_code=127,
        failure_id="abc123",
        ask="what happened?",
        target=Target(backend="claude", model="sonnet", effort="high", alias="assoc"),
    )
    data = request_to_dict(request)
    assert data["target"] == {
        "backend": "claude",
        "model": "sonnet",
        "effort": "high",
        "alias": "assoc",
    }
    restored = request_from_dict(data)
    assert restored == request


def test_request_from_dict_without_target_still_parses():
    # A pre-change dict, exactly as an old daemon/client would have sent it.
    legacy = {
        "kind": "failure",
        "prompt": "explain",
        "command": "ls -z",
        "exit_code": 127,
        "failure_id": "abc123",
        "ask": "",
    }
    request = request_from_dict(legacy)
    assert request.target is None
    assert request.kind is RequestKind.FAILURE
    assert request.command == "ls -z"


def test_request_from_dict_tolerates_junk():
    request = request_from_dict(None)
    assert request.kind is RequestKind.EXPLICIT
    assert request.target is None

    request2 = request_from_dict({"target": "not-a-mapping"})
    assert request2.target is None


def test_capabilities_new_fields_have_safe_defaults():
    caps = Capabilities()
    assert caps.thinking is False
    assert caps.effort is False
    assert caps.path == ""
    assert caps.approval == "none"
    assert caps.unmediated_file_access is False


def test_capabilities_accepts_new_fields():
    caps = Capabilities(
        thinking=True,
        effort=True,
        path="/usr/bin/codex",
        approval="harness",
        unmediated_file_access=True,
    )
    assert caps.thinking is True
    assert caps.effort is True
    assert caps.path == "/usr/bin/codex"
    assert caps.approval == "harness"
    assert caps.unmediated_file_access is True
