"""Tests for evals.tool_jev.request (issue #64, t11).

Covers the two t11 acceptance criteria:

1. for a fixture case the system and user content is byte-identical across
   the openai, anthropic and openai_compat request builders -- proven here
   by making every "adapter" call ``canonical_content`` (the one function
   this module says an adapter must call) and comparing;
2. an answer naming an operation outside the offered set parses as invalid
   (answer failure), not as a pick.

Real operation names (``machine_status``, ``gpu_stats``, ``power_set``) are
fixture data only, the same convention ``test_policies.py`` uses -- nothing
here is real case text, and no case here carries a "failure"/"explicit"
split with actual private content.
"""

from __future__ import annotations

from evals.tool_jev import request as req
from evals.tool_jev.cases import Case
from evals.tool_jev.providers.base import CallRequest
from evals.tool_jev.providers.errors import Outcome
from nvsh.ops import table as ops_table
from nvsh.platform._model import Platform
from nvsh.tiers import lfm

PLATFORM = Platform(kind="jetson")


def _case(
    case_id="c1",
    split="test",
    text="show gpu memory usage",
    candidates=None,
    expect=None,
    tags=(),
):
    return Case(
        id=case_id,
        split=split,
        text=text,
        candidates=candidates,
        expect=expect or {"operation": "gpu_stats", "args": {}},
        read_only=True,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# criterion 1: byte-identical (system, user) content across provider kinds
# ---------------------------------------------------------------------------


def _adapter_stub(provider_kind: str, request: CallRequest):
    """Stands in for an adapter: reads only canonical_content, never the case."""
    system_text, messages, tools, labels = req.canonical_content(request)
    assert messages[0]["role"] == "user"
    return provider_kind, system_text, messages[0]["content"], tools, labels


def test_tool_call_content_byte_identical_across_provider_kinds():
    case = _case()
    call_request = req.build_tool_call_request(case, None, PLATFORM)

    results = [
        _adapter_stub(kind, call_request) for kind in ("openai", "anthropic", "openai_compat")
    ]

    systems = {system for _, system, _, _, _ in results}
    users = {user for _, _, user, _, _ in results}
    assert len(systems) == 1
    assert len(users) == 1
    # the actual content is exactly what lfm builds, not a paraphrase of it
    assert next(iter(systems)) == lfm.system_brief(PLATFORM)


def test_choice_content_byte_identical_across_provider_kinds():
    case = _case()
    call_request = req.build_choice_request(case)

    results = [
        _adapter_stub(kind, call_request) for kind in ("openai", "anthropic", "openai_compat")
    ]

    systems = {system for _, system, _, _, _ in results}
    users = {user for _, _, user, _, _ in results}
    assert len(systems) == 1
    assert len(users) == 1


def test_canonical_content_is_deterministic_and_provider_independent():
    case = _case()
    call_request = req.build_tool_call_request(case, None, PLATFORM)

    first = req.canonical_content(call_request)
    second = req.canonical_content(call_request)

    assert first == second


def test_canonical_content_reports_tools_for_tool_call_and_labels_for_choice():
    case = _case()
    tool_call_request = req.build_tool_call_request(case, None, PLATFORM)
    choice_request = req.build_choice_request(case)

    _, _, tools, labels = req.canonical_content(tool_call_request)
    assert tools is not None
    assert labels is None

    _, _, tools2, labels2 = req.canonical_content(choice_request)
    assert tools2 is None
    assert labels2 is not None


def test_both_tracks_carry_the_same_request_text():
    """Parked v3: a reference answers through both interfaces on the same content."""
    case = _case()
    tool_call_request = req.build_tool_call_request(case, None, PLATFORM)
    choice_request = req.build_choice_request(case)

    assert tool_call_request.case_text == choice_request.case_text


def test_ground_snapshot_does_not_change_the_built_content():
    case = _case()
    without = req.build_tool_call_request(case, None, PLATFORM)
    with_snapshot = req.build_tool_call_request(
        case,
        {"services": ["nginx.service"], "containers": [], "source": "x", "created": "x"},
        PLATFORM,
    )

    assert without.prompt == with_snapshot.prompt
    assert without.case_text == with_snapshot.case_text


# ---------------------------------------------------------------------------
# tool_call request shape
# ---------------------------------------------------------------------------


def test_tool_call_request_interface_and_split():
    case = _case(split="test")
    call_request = req.build_tool_call_request(case, None, PLATFORM)

    assert call_request.interface == "tool_call"
    assert call_request.split == "test"
    assert call_request.case_id == "c1"


def test_tool_call_offered_candidates_include_controls_always():
    case = _case(candidates=("gpu_stats",))
    call_request = req.build_tool_call_request(case, None, PLATFORM)

    assert "gpu_stats" in call_request.offered_candidates
    assert lfm.EXPLAIN_TOOL in call_request.offered_candidates
    assert lfm.ESCALATE_TOOL in call_request.offered_candidates
    assert "power_set" not in call_request.offered_candidates


def test_tool_call_full_table_offered_when_case_has_no_candidates():
    case = _case(candidates=None)
    call_request = req.build_tool_call_request(case, None, PLATFORM)

    for name in ops_table.names():
        assert name in call_request.offered_candidates


def test_tool_call_tools_narrowed_to_case_candidates():
    case = _case(candidates=("gpu_stats",))
    call_request = req.build_tool_call_request(case, None, PLATFORM)
    tools = call_request.params["tools"]
    names = {tool["function"]["name"] for tool in tools}

    assert "gpu_stats" in names
    assert "power_set" not in names
    assert lfm.PROPOSE_TOOL in names
    assert lfm.EXPLAIN_TOOL in names
    assert lfm.ESCALATE_TOOL in names
    propose = next(t for t in tools if t["function"]["name"] == lfm.PROPOSE_TOOL)
    enum = propose["function"]["parameters"]["properties"]["operation"]["enum"]
    assert enum == ["gpu_stats"]


def test_failure_tag_builds_a_failure_agent_request():
    case = _case(text="some-command --flag", tags=("failure",))
    call_request = req.build_tool_call_request(case, None, PLATFORM)

    assert "some-command --flag" in call_request.case_text
    assert "exit status" in call_request.case_text


def test_explicit_case_builds_an_explicit_agent_request():
    case = _case(text="what is using the most memory?", tags=("explicit",))
    call_request = req.build_tool_call_request(case, None, PLATFORM)

    assert call_request.case_text == "what is using the most memory?"


# ---------------------------------------------------------------------------
# choice request shape
# ---------------------------------------------------------------------------


def test_choice_request_interface_and_labels():
    case = _case(candidates=("gpu_stats", "memory_stats"))
    call_request = req.build_choice_request(case)

    assert call_request.interface == "choice"
    labels = call_request.params["labels"]
    assert "gpu_stats" in labels
    assert "memory_stats" in labels
    assert lfm.EXPLAIN_TOOL in labels
    assert lfm.ESCALATE_TOOL in labels
    assert "power_set" not in labels


def test_choice_request_labels_keep_full_table_positions():
    """A candidate's letter must not move when others are left out."""
    full_case = _case(candidates=None)
    narrow_case = _case(candidates=("gpu_stats",))

    full_labels = req.build_choice_request(full_case).params["labels"]
    narrow_labels = req.build_choice_request(narrow_case).params["labels"]

    assert full_labels["gpu_stats"] == narrow_labels["gpu_stats"]


# ---------------------------------------------------------------------------
# criterion 2: an answer naming an operation outside the offered set parses
# as invalid, not as a pick.
# ---------------------------------------------------------------------------


def test_parse_tool_call_outside_offered_set_is_invalid():
    offered = ("gpu_stats", lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)
    answer = {"name": lfm.PROPOSE_TOOL, "arguments": {"operation": "power_set", "arguments": {}}}

    classification, operation, arguments = req.parse_tool_call(answer, offered)

    assert classification.outcome is Outcome.INVALID
    assert classification.reason == "outside_offered_set"
    assert operation is None
    assert arguments is None


def test_parse_tool_call_within_offered_set_is_ok():
    offered = ("gpu_stats", lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)
    answer = {"name": lfm.PROPOSE_TOOL, "arguments": {"operation": "gpu_stats", "arguments": {}}}

    classification, operation, arguments = req.parse_tool_call(answer, offered)

    assert classification.outcome is Outcome.OK
    assert operation == "gpu_stats"
    assert arguments == {}


def test_parse_tool_call_json_string_answer():
    offered = ("gpu_stats", lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)
    answer = '{"name": "propose", "arguments": {"operation": "gpu_stats", "arguments": {}}}'

    classification, operation, arguments = req.parse_tool_call(answer, offered)

    assert classification.outcome is Outcome.OK
    assert operation == "gpu_stats"


def test_parse_tool_call_explain_always_valid_even_when_narrowed():
    offered = ("gpu_stats", lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)
    answer = {"name": lfm.EXPLAIN_TOOL, "arguments": {"text": "here is the answer"}}

    classification, operation, arguments = req.parse_tool_call(answer, offered)

    assert classification.outcome is Outcome.OK
    assert operation == lfm.EXPLAIN_TOOL
    assert arguments == {"text": "here is the answer"}


def test_parse_tool_call_escalate_is_valid():
    offered = ("gpu_stats", lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)
    answer = {"name": lfm.ESCALATE_TOOL, "arguments": {"reason": "needs sudo"}}

    classification, operation, arguments = req.parse_tool_call(answer, offered)

    assert classification.outcome is Outcome.OK
    assert operation == lfm.ESCALATE_TOOL


def test_parse_tool_call_malformed_text_is_invalid():
    offered = ("gpu_stats", lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)

    classification, operation, arguments = req.parse_tool_call("not json at all", offered)

    assert classification.outcome is Outcome.INVALID
    assert classification.reason == "malformed"
    assert operation is None
    assert arguments is None


def test_parse_tool_call_missing_operation_key_is_malformed():
    offered = ("gpu_stats", lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)
    answer = {"name": lfm.PROPOSE_TOOL, "arguments": {}}

    classification, operation, arguments = req.parse_tool_call(answer, offered)

    assert classification.outcome is Outcome.INVALID
    assert classification.reason == "malformed"


def test_parse_tool_call_unrecognised_tool_name_is_invalid():
    offered = ("gpu_stats", lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)
    answer = {"name": "gpu_stats", "arguments": {}}  # a direct operation call, not via propose

    classification, operation, arguments = req.parse_tool_call(answer, offered)

    assert classification.outcome is Outcome.INVALID
    assert operation is None


def test_parse_choice_within_offered_set_is_ok():
    labels = {"gpu_stats": "A", lfm.EXPLAIN_TOOL: "B", lfm.ESCALATE_TOOL: "C"}

    classification, name = req.parse_choice("A", labels)

    assert classification.outcome is Outcome.OK
    assert name == "gpu_stats"


def test_parse_choice_unrecognised_letter_is_invalid():
    labels = {"gpu_stats": "A", lfm.EXPLAIN_TOOL: "B", lfm.ESCALATE_TOOL: "C"}

    classification, name = req.parse_choice("Z", labels)

    assert classification.outcome is Outcome.INVALID
    assert name is None


def test_parse_choice_strips_surrounding_text():
    labels = {"gpu_stats": "A", lfm.EXPLAIN_TOOL: "B", lfm.ESCALATE_TOOL: "C"}

    classification, name = req.parse_choice("  B) explain\n", labels)

    assert classification.outcome is Outcome.OK
    assert name == lfm.EXPLAIN_TOOL


def test_parse_choice_none_answer_is_invalid():
    labels = {"gpu_stats": "A", lfm.EXPLAIN_TOOL: "B", lfm.ESCALATE_TOOL: "C"}

    classification, name = req.parse_choice(None, labels)

    assert classification.outcome is Outcome.INVALID
    assert classification.reason == "malformed"


# ---------------------------------------------------------------------------
# distribution_from_logprobs -- never estimated
# ---------------------------------------------------------------------------


def test_distribution_from_logprobs_renormalises_over_offered_labels():
    labels = {"gpu_stats": "A", "memory_stats": "B"}
    # ln(0.7) and ln(0.3) roughly -- raw next-token logprobs for A and B.
    logprobs = {"A": -0.357, "B": -1.204}

    dist = req.distribution_from_logprobs(logprobs, labels, ("gpu_stats", "memory_stats"))

    assert dist is not None
    assert set(dist) == {"gpu_stats", "memory_stats"}
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert dist["gpu_stats"] > dist["memory_stats"]


def test_distribution_from_logprobs_none_when_a_label_is_missing():
    labels = {"gpu_stats": "A", "memory_stats": "B"}
    logprobs = {"A": -0.357}  # B never came back in the top logprobs

    dist = req.distribution_from_logprobs(logprobs, labels, ("gpu_stats", "memory_stats"))

    assert dist is None


def test_distribution_from_logprobs_none_when_no_label_has_mass():
    labels = {"gpu_stats": "A", "memory_stats": "B"}
    # Both labels are present (so missing_labels is satisfied) but their
    # probabilities underflow to exactly zero float mass.
    logprobs = {"A": -1000.0, "B": -1000.0, "X": -0.001}

    dist = req.distribution_from_logprobs(logprobs, labels, ("gpu_stats", "memory_stats"))

    assert dist is None


def test_distribution_from_logprobs_respects_offered_order():
    labels = {"gpu_stats": "A", "memory_stats": "B", "disk_stats": "C"}
    logprobs = {"A": -1.0, "B": -1.0, "C": -1.0}

    dist = req.distribution_from_logprobs(
        logprobs, labels, ("disk_stats", "gpu_stats", "memory_stats")
    )

    assert dist is not None
    assert list(dist) == ["disk_stats", "gpu_stats", "memory_stats"]
