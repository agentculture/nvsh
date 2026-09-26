"""DeepEval layer over Tool-Jev traces (issue 64, task t15).

One :class:`deepeval.test_case.LLMTestCase` per ``(subject, case, policy)``:
the case's raw model output plus one policy's final decision on it, exactly
as :class:`evals.tool_jev.trace.Trace` already carries them. Four custom
``BaseMetric`` subclasses grade each test case with an *exact* check — no
LLM judge, no network, no randomness:

- :class:`RightActionMetric` — the raw prediction's top-1/right
  operation+arguments call, per ``scripts/lfm-finetune/metrics.py``'s own
  conventions (``metrics_bridge.top1_correct`` when the row carries a
  candidate distribution, otherwise ``metrics.py``'s own
  ``_right_proposal``).
- :class:`WrongMutatingMetric` — the negation of
  ``metrics_bridge``'s row-level ``wrong_mutating`` flag
  (``metrics._wrong_operation_mutating`` OR ``_wrong_arguments_mutating``).
- :class:`CorrectAbstainEscalateMetric` — for an escalate/explain-expected
  case, whether *this policy's* final decision correctly declined
  (``metrics._escalated`` semantics for escalate-expected, a bare
  ``"explain"`` decision for explain-expected). Not applicable to an
  operation-expected case.
- :class:`MissingCandidateHandledMetric` — for a case
  ``metrics.is_missing_candidate`` flags (its gold operation was not among
  the offered candidates), whether the raw prediction escalated
  (``metrics._escalated``) rather than guessing. Not applicable otherwise.

Every metric's verdict is computed by calling the same
``scripts/lfm-finetune/metrics.py`` functions ``evals.tool_jev.metrics_bridge``
already calls (never a re-derived copy of the rule), from the trace fields
this module put in the test case's ``additional_metadata`` — so a metric's
pass/fail is reproducible from the metadata alone, with no need to keep the
original :class:`Trace` object around.

Corpus-level figures (ECE, Brier, coverage, abstain precision/recall,
per-slice) are never computed here or by DeepEval: :func:`evaluate_traces`
returns ``evals.tool_jev.metrics_bridge.compute``'s own output unchanged,
alongside DeepEval's own per-case ``EvaluationResult``.

No network: this module only calls DeepEval's synchronous, local
``evaluate()`` (``async_config=AsyncConfig(run_async=False)``), and every
metric here is a pure function of already-recorded fields. Import order
matters — ``evals.tool_jev`` (this package) must be imported before
``deepeval`` anywhere in the process, which is why this module's first
import is ``evals.tool_jev`` itself (see that package's docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import deepeval  # noqa: F401  (forces deepeval's own import to run after the guard above)
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

import evals.tool_jev  # noqa: F401  (env guard: must precede any deepeval import)
from evals.tool_jev import metrics_bridge
from evals.tool_jev.trace import Trace

#: Outcomes that count as a decline, exactly the vocabulary
#: ``scripts/lfm-finetune/metrics.py``'s own ``_escalated`` checks (private,
#: so reused via a tiny stand-in object rather than imported by name).
_ESCALATED_OUTCOMES = ("escalate", "abstain_uncertain")


class DeepevalLayerError(ValueError):
    """A trace this module cannot build a scorable test case from."""


# ---------------------------------------------------------------------------
# Trace -> Prediction / LLMTestCase
# ---------------------------------------------------------------------------


def _prediction_row(
    *,
    case_id: str,
    expected: Mapping[str, Any],
    outcome: str,
    operation: str | None,
    arguments: dict | None,
    candidates: dict | None,
    tokens: int | None,
    ttfd_ms: float | None,
    latency_ms: float | None,
    invalid_reason: str | None,
) -> dict:
    """A ``metrics.Prediction.from_dict``-ready row.

    ``tokens``/``ttfd_ms``/``latency_ms`` are ``None`` on a ``RawRecord``
    built from a provider answer (see ``trace.py``'s module docstring) but
    ``metrics.Prediction.from_dict`` requires non-negative numbers; this
    module only ever uses these three fields for validation, never for a
    timing figure, so a missing one defaults to zero rather than failing to
    build a Prediction at all.
    """
    return {
        "id": case_id,
        "expected": dict(expected),
        "outcome": outcome,
        "operation": operation,
        "arguments": None if arguments is None else dict(arguments),
        "candidates": None if candidates is None else dict(candidates),
        "tokens": 0 if tokens is None else tokens,
        "ttfd_ms": 0.0 if ttfd_ms is None else ttfd_ms,
        "latency_ms": 0.0 if latency_ms is None else latency_ms,
        "invalid_reason": invalid_reason,
    }


def prediction_from_trace(trace: Trace, metrics_mod) -> Any:
    """A ``metrics_mod.Prediction`` built from *trace*'s raw record and ground truth.

    Raises :class:`DeepevalLayerError` when *trace* carries no
    ``ground_truth`` (there is nothing to score it against).
    """
    if trace.ground_truth is None:
        raise DeepevalLayerError(
            f"trace {trace.case_id!r} carries no ground_truth to score against"
        )
    row = _prediction_row(
        case_id=trace.case_id,
        expected=trace.ground_truth,
        outcome=trace.raw.outcome,
        operation=trace.raw.operation,
        arguments=trace.raw.arguments,
        candidates=trace.raw.candidates,
        tokens=trace.raw.tokens,
        ttfd_ms=trace.raw.ttfd_ms,
        latency_ms=trace.raw.latency_ms,
        invalid_reason=trace.raw.invalid_reason,
    )
    return metrics_mod.Prediction.from_dict(row)


def build_test_case(trace: Trace, policy: str, *, metrics_mod=None) -> LLMTestCase:
    """One ``LLMTestCase`` for *trace*'s ``policy`` final decision.

    ``input`` is a case-id placeholder (never the case text: real request
    text must never reach DeepEval's own results files).
    ``actual_output``/``expected_output`` are the final decision string and
    the gold calibration label. Every field the four metrics need to
    recompute their verdict independently of the original ``Trace`` object
    lives in ``additional_metadata``.
    """
    metrics_mod = metrics_mod or metrics_bridge.load_metrics_module()
    if trace.ground_truth is None:
        raise DeepevalLayerError(
            f"trace {trace.case_id!r} carries no ground_truth to score against"
        )
    final = trace.final.get(policy)
    if final is None:
        raise DeepevalLayerError(
            f"trace {trace.case_id!r} carries no final decision for policy {policy!r}"
        )
    gold_label = metrics_mod.expected_label(trace.ground_truth)
    return LLMTestCase(
        input=f"case:{trace.case_id}",
        actual_output=final.decision,
        expected_output=gold_label,
        additional_metadata={
            "case_id": trace.case_id,
            "split": trace.split,
            "subject": trace.subject,
            "policy": policy,
            "expected": dict(trace.ground_truth),
            "raw_outcome": trace.raw.outcome,
            "raw_operation": trace.raw.operation,
            "raw_arguments": trace.raw.arguments,
            "raw_candidates": trace.raw.candidates,
            "raw_tokens": trace.raw.tokens,
            "raw_ttfd_ms": trace.raw.ttfd_ms,
            "raw_latency_ms": trace.raw.latency_ms,
            "raw_invalid_reason": trace.raw.invalid_reason,
            "final_decision": final.decision,
            "final_reason": final.reason,
        },
    )


def _prediction_from_metadata(metadata: Mapping[str, Any], metrics_mod) -> Any:
    """Rebuild a ``metrics_mod.Prediction`` from a test case's ``additional_metadata``.

    This is what makes a metric's verdict reproducible from the test case
    alone: the same row :func:`build_test_case` derived from the original
    ``Trace`` is derived again here, from the metadata it stored.
    """
    row = _prediction_row(
        case_id=metadata["case_id"],
        expected=metadata["expected"],
        outcome=metadata["raw_outcome"],
        operation=metadata["raw_operation"],
        arguments=metadata["raw_arguments"],
        candidates=metadata["raw_candidates"],
        tokens=metadata["raw_tokens"],
        ttfd_ms=metadata["raw_ttfd_ms"],
        latency_ms=metadata["raw_latency_ms"],
        invalid_reason=metadata["raw_invalid_reason"],
    )
    return metrics_mod.Prediction.from_dict(row)


# ---------------------------------------------------------------------------
# The four exact metrics
# ---------------------------------------------------------------------------


class _ExactMetric(BaseMetric):
    """Shared plumbing for an exact, non-LLM, deterministic check.

    ``threshold = 1.0`` makes DeepEval's own ``is_successful()`` (``score >=
    threshold``) agree with the boolean ``_verdict`` computes: a match
    scores ``1.0`` and passes, a mismatch scores ``0.0`` and fails. Async
    mode is off — there is nothing to await, this is a pure function of
    already-recorded fields.
    """

    threshold: float = 1.0
    async_mode: bool = False

    def __init__(self, metrics_mod=None):
        self.metrics_mod = metrics_mod or metrics_bridge.load_metrics_module()
        self.threshold = 1.0
        self.async_mode = False
        self.score = None
        self.success = None
        self.reason = None
        self.error = None

    def _verdict(self, test_case: LLMTestCase) -> tuple[bool, str]:
        """Return ``(success, reason)`` for *test_case*. Overridden per metric."""
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.error = None
        success, reason = self._verdict(test_case)
        self.success = success
        self.reason = reason
        self.score = 1.0 if success else 0.0
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool | None:
        if self.error is not None:
            self.success = False
            return False
        return self.success


class RightActionMetric(_ExactMetric):
    """Right action: the raw prediction's top-1 (or right proposal) call.

    Not applicable to an escalate/explain-expected case (there is no
    "action" to grade there); such a case always passes with a neutral
    reason so it never drags down the pass rate for a check that doesn't
    apply to it.
    """

    @property
    def __name__(self) -> str:  # noqa: N802 - DeepEval's own metric-name protocol
        return "right_action"

    def _verdict(self, test_case: LLMTestCase) -> tuple[bool, str]:
        metadata = test_case.metadata or {}
        prediction = _prediction_from_metadata(metadata, self.metrics_mod)
        if self.metrics_mod.expect_kind(prediction.expected) != "operation":
            return True, "not applicable: case does not expect an operation"
        gold = self.metrics_mod.expected_label(prediction.expected)
        if prediction.candidates is not None:
            correct = metrics_bridge.top1_correct(self.metrics_mod, prediction)
            reason = f"top-1 candidate {'matches' if correct else 'differs from'} gold {gold!r}"
            return bool(correct), reason
        correct = self.metrics_mod._right_proposal(prediction)
        reason = (
            f"raw proposal {'matches' if correct else 'differs from'} gold "
            f"operation+arguments {gold!r}"
        )
        return bool(correct), reason


class WrongMutatingMetric(_ExactMetric):
    """Wrong mutating: the row must not be a wrong mutating call.

    Exactly ``metrics_bridge.row_metrics``'s ``wrong_mutating`` flag,
    negated (a wrong mutating call is a failure, not a pass).
    """

    @property
    def __name__(self) -> str:  # noqa: N802
        return "wrong_mutating"

    def _verdict(self, test_case: LLMTestCase) -> tuple[bool, str]:
        metadata = test_case.metadata or {}
        prediction = _prediction_from_metadata(metadata, self.metrics_mod)
        wrong = self.metrics_mod._wrong_operation_mutating(
            prediction
        ) or self.metrics_mod._wrong_arguments_mutating(prediction)
        if wrong:
            return False, "raw prediction is a wrong mutating call"
        return True, "raw prediction is not a wrong mutating call"


class CorrectAbstainEscalateMetric(_ExactMetric):
    """Correct abstain/escalate: this policy's final decision, on a decline-expected case.

    Not applicable to an operation-expected case (there is nothing to
    decline there); such a case always passes with a neutral reason.
    """

    @property
    def __name__(self) -> str:  # noqa: N802
        return "correct_abstain_escalate"

    def _verdict(self, test_case: LLMTestCase) -> tuple[bool, str]:
        metadata = test_case.metadata or {}
        expected = metadata["expected"]
        kind = self.metrics_mod.expect_kind(expected)
        final_decision = metadata["final_decision"]
        if kind == "escalate":
            escalated = final_decision in _ESCALATED_OUTCOMES
            reason = (
                f"final decision {final_decision!r} "
                f"{'correctly escalates' if escalated else 'fails to escalate'} "
                "an escalate-expected case"
            )
            return escalated, reason
        if kind == "explain":
            explained = final_decision == "explain"
            reason = (
                f"final decision {final_decision!r} "
                f"{'correctly explains' if explained else 'fails to explain'} "
                "an explain-expected case"
            )
            return explained, reason
        return True, "not applicable: case expects an operation, not a decline"


class MissingCandidateHandledMetric(_ExactMetric):
    """Missing-candidate handled: escalate when the gold op was never offered.

    ``metrics.is_missing_candidate`` decides whether this row's offered
    candidates ever included the gold label; when they didn't, the raw
    prediction must have escalated (``metrics._escalated``) rather than
    guessing at an operation it was never offered. Not applicable to a row
    whose candidates did include gold.
    """

    @property
    def __name__(self) -> str:  # noqa: N802
        return "missing_candidate_handled"

    def _verdict(self, test_case: LLMTestCase) -> tuple[bool, str]:
        metadata = test_case.metadata or {}
        prediction = _prediction_from_metadata(metadata, self.metrics_mod)
        if not self.metrics_mod.is_missing_candidate(prediction):
            return True, "not applicable: gold operation was offered"
        escalated = self.metrics_mod._escalated(prediction)
        reason = (
            f"missing-candidate case {'correctly escalated' if escalated else 'did not escalate'}"
        )
        return escalated, reason


#: Fresh instances of every exact metric, in a stable order, sharing one
#: already-loaded ``metrics.py`` module (never re-loaded per metric).
def build_metrics(metrics_mod=None) -> list[BaseMetric]:
    metrics_mod = metrics_mod or metrics_bridge.load_metrics_module()
    return [
        RightActionMetric(metrics_mod=metrics_mod),
        WrongMutatingMetric(metrics_mod=metrics_mod),
        CorrectAbstainEscalateMetric(metrics_mod=metrics_mod),
        MissingCandidateHandledMetric(metrics_mod=metrics_mod),
    ]


# ---------------------------------------------------------------------------
# evaluate() wrapper + corpus metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationOutcome:
    """DeepEval's own per-case result, alongside ``metrics_bridge``'s corpus figures.

    ``deepeval_result`` is exactly what ``deepeval.evaluate.evaluate()``
    returned (per-test-case pass/fail, scores, reasons — also written to
    ``results_folder`` as a timestamped ``test_run_*.json``).
    ``corpus_metrics`` is exactly ``metrics_bridge.compute()``'s own output
    for the same predictions (ECE, Brier, coverage, abstain
    precision/recall, wrong-mutating, per-slice) — never recomputed or
    approximated by DeepEval.
    """

    deepeval_result: EvaluationResult
    corpus_metrics: dict


def evaluate_traces(
    traces: Sequence[Trace],
    policy: str,
    *,
    results_folder: str | Path,
    metrics_mod=None,
    gate_mod=None,
    metrics: Sequence[BaseMetric] | None = None,
    display_config: DisplayConfig | None = None,
    async_config: AsyncConfig | None = None,
    cache_config: CacheConfig | None = None,
) -> EvaluationOutcome:
    """Score *traces* under *policy* with DeepEval, and report corpus metrics.

    Runs synchronously (``AsyncConfig(run_async=False)`` by default) and
    never calls a model or the network: every metric here is a pure
    function of the trace fields already recorded. ``results_folder`` is
    passed straight to ``DisplayConfig`` (or merged into a caller-supplied
    one) so the timestamped ``test_run_*.json`` lands exactly where the
    caller asked, never inside this repository — callers running this
    under test point it at a ``tmp_path``-backed directory outside any git
    worktree (see this module's test file).
    """
    metrics_mod = metrics_mod or metrics_bridge.load_metrics_module()
    gate_mod = gate_mod or metrics_bridge.load_gate_module()
    metric_list = list(metrics) if metrics is not None else build_metrics(metrics_mod)

    predictions = [prediction_from_trace(trace, metrics_mod) for trace in traces]
    test_cases = [build_test_case(trace, policy, metrics_mod=metrics_mod) for trace in traces]

    if display_config is None:
        display_config = DisplayConfig(
            results_folder=str(results_folder),
            print_results=False,
            show_indicator=False,
        )
    if async_config is None:
        async_config = AsyncConfig(run_async=False)
    if cache_config is None:
        cache_config = CacheConfig(write_cache=False, use_cache=False)

    deepeval_result = evaluate(
        test_cases,
        metric_list,
        async_config=async_config,
        display_config=display_config,
        cache_config=cache_config,
    )
    corpus_metrics = metrics_bridge.compute(predictions, metrics_mod=metrics_mod, gate_mod=gate_mod)
    return EvaluationOutcome(deepeval_result=deepeval_result, corpus_metrics=corpus_metrics)
