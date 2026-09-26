"""The request contract shared by every reference-provider adapter (issue #64, t11).

There is exactly one way to turn a :class:`~evals.tool_jev.cases.Case` into a
provider call, for each of the two interfaces the plan defines:

- **Track A** (``interface="tool_call"``): :func:`build_tool_call_request`
  builds the same messages :mod:`nvsh.tiers.lfm` builds for the real Tier-2
  candidates -- ``lfm.system_brief`` for the system message, and
  ``lfm.request_message`` (fed an :class:`~nvsh.agent.base.AgentRequest`
  / :class:`~nvsh.agent.base.AgentContext` pair derived from the case) for
  the user message -- with the tool list narrowed to the case's offered
  candidates exactly as ``scripts/lfm-finetune/measure.py``'s
  ``restrict_tools`` narrows it for a missing-candidate slice (read
  independently here, never copied).
- **Track B** (``interface="choice"``): :func:`build_choice_request` calls
  ``scorer.prompt_messages`` (loaded by path from
  ``scripts/lfm-finetune/scorer.py``, the way
  ``tests/test_lfm_finetune_scorer.py`` does) with the *same* request text
  Track A used, so a reference model sees byte-identical case content
  through both interfaces (the parked-v3 contract: every reference answers
  through both).

Every adapter (OpenAI, Anthropic, an OpenAI-compatible local server, ...)
MUST build its provider payload by calling :func:`canonical_content` on the
:class:`~evals.tool_jev.providers.base.CallRequest` this module returns,
never by re-deriving system/user text itself -- that is what makes
criterion 1 (byte-identical system/user content across provider kinds) hold
by construction rather than by convention.

Answers are parsed back with :func:`parse_tool_call` (Track A) and
:func:`parse_choice` (Track B), both routed through
``evals.tool_jev.providers.errors.classify_answer`` so "invalid" is
classified in exactly one place across the whole tree. Where a served
model returned per-label log-probabilities, :func:`distribution_from_logprobs`
gives the candidate distribution -- never an estimate when a label's mass
did not come back.

``ground_snapshot`` (deviation d1's fixed machine snapshot) is accepted by
:func:`build_tool_call_request` for parity with ``measure.py``'s call, but
this module builds only the first-turn request (no inspection rounds have
happened yet): ``lfm.system_brief`` and ``lfm.request_message`` never read
the snapshot, and ``measure.py`` only ever hands a snapshot to the
*grounding runner* invoked while a candidate's tool-use loop is under way,
never to the initial system/user text. So the returned request's content
does not depend on *ground_snapshot* today; it is threaded through so a
later multi-round reference call can be added here without moving the
request-contract surface again.

No module under ``nvsh/`` may import this one; this module may import
``nvsh`` (``nvsh.tiers.lfm``, ``nvsh.ops.table``, ``nvsh.agent.base``) and
``evals.tool_jev`` siblings (``cases``, ``providers.base``,
``providers.errors``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nvsh.agent.base import AgentContext, AgentRequest, RequestKind
from nvsh.ops import table as ops_table
from nvsh.platform._model import Platform
from nvsh.tiers import lfm

from .cases import Case
from .providers.base import CallRequest
from .providers.errors import Classification, Outcome, classify_answer

#: The reasoning/output-length knobs every request carries in ``params``.
#: Fixed here (not per-adapter) so every provider is asked the same thing.
DEFAULT_REASONING = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 512

#: A split-file entry's ``"kind"`` (see ``cases._entry_to_case``) that marks a
#: failed-command case; it survives onto ``Case.tags`` and is the only signal
#: this module has for choosing ``AgentRequest.kind`` the way
#: ``nvsh/tiers/bench.py``'s ``request_for``/``context_for`` do for a
#: ``CorpusEntry`` (``Case`` carries no ``kind`` field of its own).
_FAILURE_TAG = "failure"

#: scripts/lfm-finetune/scorer.py, loaded by path -- never copied (COMMON2).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCORER_PATH = _REPO_ROOT / "scripts" / "lfm-finetune" / "scorer.py"


def _load_scorer() -> Any:
    spec = importlib.util.spec_from_file_location("_tool_jev_scorer", _SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


scorer = _load_scorer()


# ---------------------------------------------------------------------------
# Case -> (AgentRequest, AgentContext), the way nvsh/tiers/bench.py does it
# for a CorpusEntry -- Case's own shape, independently.
# ---------------------------------------------------------------------------


def _agent_request_and_context(case: Case) -> tuple[AgentRequest, AgentContext]:
    if case.text is None:
        raise ValueError(f"{case.id}: a held-out case has no request text to build a request from")
    if _FAILURE_TAG in case.tags:
        return (
            AgentRequest(kind=RequestKind.FAILURE, command=case.text, exit_code=1),
            AgentContext(output=case.text),
        )
    return AgentRequest(kind=RequestKind.EXPLICIT, prompt=case.text), AgentContext(output="")


def _request_text(case: Case) -> str:
    """The one request-text rendering both tracks share (Track A's user message)."""
    request, context = _agent_request_and_context(case)
    return lfm.request_message(request, context)


# ---------------------------------------------------------------------------
# Track A: tool_call
# ---------------------------------------------------------------------------


def _operation_names(case: Case) -> tuple[str, ...]:
    if case.candidates is not None:
        return tuple(case.candidates)
    return ops_table.names()


def _tools_for_case(case: Case) -> list[dict]:
    """``lfm.tools_for()`` narrowed to *case*'s offered operations.

    Independent of (never calling into) ``measure.py``'s ``restrict_tools``:
    the control tools (``propose``/``explain``/``escalate``) always stay,
    every other tool stays only if its name is an offered operation (or
    *case.candidates* is ``None``, meaning the full table), and ``propose``'s
    own ``operation`` enum is narrowed the same way.
    """
    tools = lfm.tools_for()
    if case.candidates is None:
        return tools
    allowed = set(case.candidates)
    kept: list[dict] = []
    for tool in tools:
        name = tool["function"]["name"]
        if ops_table.get(name) is not None and name not in allowed:
            continue
        if name == lfm.PROPOSE_TOOL:
            tool = json.loads(json.dumps(tool))
            spec = tool["function"]["parameters"]["properties"]["operation"]
            spec["enum"] = [op for op in spec.get("enum", []) if op in allowed]
        kept.append(tool)
    return kept


def build_tool_call_request(
    case: Case,
    ground_snapshot: Mapping[str, Any] | None,
    platform: Platform,
) -> CallRequest:
    """Track A's ``CallRequest`` for *case*: the model must call one tool.

    ``prompt`` is ``lfm.system_brief(platform)``; ``case_text`` is
    ``lfm.request_message(...)`` fed the case's own
    (:class:`AgentRequest`, :class:`AgentContext`) pair. ``offered_candidates``
    is every name the model may end on: the case's offered operations (or
    every table operation) plus ``explain``/``escalate``, which are never
    narrowed by *case.candidates*.

    See the module docstring for why *ground_snapshot* does not change the
    returned content.
    """
    del ground_snapshot  # not read: see module docstring
    system = lfm.system_brief(platform)
    ask = _request_text(case)
    offered = _operation_names(case) + (lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)
    params = {
        "tools": _tools_for_case(case),
        "reasoning": DEFAULT_REASONING,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
    }
    return CallRequest(
        case_id=case.id,
        split=case.split,
        case_text=ask,
        prompt=system,
        offered_candidates=offered,
        params=params,
        interface="tool_call",
    )


def _as_tool_call(value: Any) -> tuple[str, Mapping[str, Any]] | None:
    """``(name, arguments)`` from a tool-call-shaped dict or JSON string, or ``None``."""
    candidate: Any = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except ValueError:
            return None
    if not isinstance(candidate, Mapping):
        return None
    name = candidate.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments: Any = candidate.get("arguments", {})
    if isinstance(arguments, str):
        if not arguments:
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except ValueError:
                return None
    if not isinstance(arguments, Mapping):
        return None
    return name, arguments


def parse_tool_call(
    answer_json_or_text: Any, offered: tuple[str, ...]
) -> tuple[Classification, str | None, dict | None]:
    """Parse one Track A answer into ``(classification, operation, arguments)``.

    *answer_json_or_text* is a tool call, either as a dict shaped
    ``{"name": ..., "arguments": {...}}`` (``arguments`` a dict or a JSON
    string) or that same shape serialised to a JSON string. Plain text
    that is not that shape is never a tool call and is classified
    malformed -- Track A requires a tool call.

    *offered* is exactly the ``CallRequest.offered_candidates``
    :func:`build_tool_call_request` built: the case's offered operations
    plus ``lfm.EXPLAIN_TOOL``/``lfm.ESCALATE_TOOL``. A ``propose`` call
    naming an operation outside *offered* classifies ``invalid`` with
    reason ``outside_offered_set`` (acceptance criterion 2) -- it is never
    read back as a pick.

    Returns ``operation``/``arguments`` as ``None`` unless the
    classification's ``outcome`` is ``Outcome.OK``.
    """
    call = _as_tool_call(answer_json_or_text)
    if call is None:
        return classify_answer(None, offered, malformed=True), None, None
    name, arguments = call
    if name == lfm.PROPOSE_TOOL:
        operation = arguments.get("operation")
        if not isinstance(operation, str) or not operation:
            return classify_answer(None, offered, malformed=True), None, None
        classification = classify_answer(operation, offered)
        if classification.outcome is not Outcome.OK:
            return classification, None, None
        raw_args = arguments.get("arguments", {})
        return classification, operation, dict(raw_args) if isinstance(raw_args, Mapping) else {}
    if name in (lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL):
        classification = classify_answer(name, offered)
        if classification.outcome is not Outcome.OK:
            return classification, None, None
        return classification, name, dict(arguments)
    # Any other tool name (e.g. an operation tool called directly, outside
    # `propose`) is not a final answer this contract recognises.
    return classify_answer(None, offered, malformed=True), None, None


# ---------------------------------------------------------------------------
# Track B: choice
# ---------------------------------------------------------------------------


def build_choice_request(case: Case) -> CallRequest:
    """Track B's ``CallRequest`` for *case*: the model answers with one label.

    Built exactly the way ``measure.py``'s ``scorer_request`` (default,
    non-``--reasons`` mode) builds it: ``scorer.prompt_messages`` is handed
    the *same* request text :func:`build_tool_call_request` used, and the
    case's offered operations plus ``scorer.CONTROLS`` (``explain`` is a
    scorer control here, and ``escalate`` its bare form -- reasons-mode
    splitting escalate into named reasons is out of this module's scope).
    """
    ask = _request_text(case)
    offered_ops = None if case.candidates is None else tuple(case.candidates)
    candidates = None if offered_ops is None else offered_ops + scorer.CONTROLS
    messages = scorer.prompt_messages(ask, candidates)
    names = list(candidates) if candidates is not None else list(scorer.candidates())
    labels = scorer.labels_for(names)
    system_text = messages[0]["content"]
    user_text = messages[1]["content"]
    params = {
        "labels": labels,
        "reasoning": DEFAULT_REASONING,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
    }
    return CallRequest(
        case_id=case.id,
        split=case.split,
        case_text=user_text,
        prompt=system_text,
        offered_candidates=tuple(labels),
        params=params,
        interface="choice",
    )


def parse_choice(
    answer_text: str | None, labels: Mapping[str, str]
) -> tuple[Classification, str | None]:
    """Parse one Track B answer into ``(classification, chosen candidate name)``.

    *labels* is the candidate -> letter map :func:`build_choice_request`
    put in ``params["labels"]`` (already narrowed to the case's offered
    candidates). The answer is read as "the first non-whitespace
    character is the label" (the prompt instructs "answer with the
    action's letter only"); a character that is not one of *labels*'s
    letters is malformed, never guessed into a pick.
    """
    offered = tuple(labels)
    if not isinstance(answer_text, str):
        return classify_answer(None, offered, malformed=True), None
    stripped = answer_text.strip()
    token = stripped[:1]
    by_label = {label: name for name, label in labels.items()}
    name = by_label.get(token) if token else None
    if name is None:
        return classify_answer(None, offered, malformed=True), None
    classification = classify_answer(name, offered)
    if classification.outcome is not Outcome.OK:
        return classification, None
    return classification, name


def distribution_from_logprobs(
    top_logprobs_of_first_answer_token: Mapping[str, float],
    labels: Mapping[str, str],
    offered_order: tuple[str, ...],
) -> dict[str, float] | None:
    """The candidate distribution over *labels*, in *offered_order*, or ``None``.

    Delegates the one readout definition to ``scorer.missing_labels`` /
    ``scorer.distribution`` (loaded by path, never copied): ``None`` when
    any offered label's mass did not come back at all (never renormalised
    over the labels that did -- that would turn a sliver of raw mass into
    a false certainty) or when no label had any mass. Otherwise every
    offered candidate's renormalised probability, ordered as
    *offered_order* asks.
    """
    missing = scorer.missing_labels(top_logprobs_of_first_answer_token, labels)
    if missing:
        return None
    distribution, mass = scorer.distribution(top_logprobs_of_first_answer_token, labels)
    if mass <= 0 or not distribution:
        return None
    return {name: distribution[name] for name in offered_order if name in distribution}


# ---------------------------------------------------------------------------
# The one thing every adapter must call: canonical_content
# ---------------------------------------------------------------------------


def canonical_content(
    request: CallRequest,
) -> tuple[str, str, list[dict] | None, dict[str, str] | None]:
    """``(system_text, user_text, tools, labels)`` -- the only content an adapter may send.

    A pure read of *request*'s own fields: ``system_text`` is
    ``request.prompt``, ``user_text`` is ``request.case_text``, ``tools``
    is ``request.params["tools"]`` for a ``"tool_call"`` request (``None``
    for ``"choice"``), and ``labels`` is ``request.params["labels"]`` for a
    ``"choice"`` request (``None`` for ``"tool_call"``).

    Every adapter (OpenAI, Anthropic, an OpenAI-compatible server, ...)
    builds its provider-specific payload from exactly this tuple, never by
    re-deriving system/user text or tool/label shapes itself -- that is
    what makes the system/user content byte-identical across provider
    kinds by construction (acceptance criterion 1), independent of which
    adapter calls it, transport framing, or call order.
    """
    tools = request.params.get("tools") if request.interface == "tool_call" else None
    labels = request.params.get("labels") if request.interface == "choice" else None
    return request.prompt, request.case_text, tools, labels
