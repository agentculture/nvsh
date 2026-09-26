"""DeepEval layer over Tool-Jev traces (issue 64, task t15).

One :class:`deepeval.test_case.LLMTestCase` per ``(subject, case, policy)``:
the case's raw model output, *replayed through one harness policy*, exactly
as :func:`apply_policy_to_prediction` builds it. Four custom ``BaseMetric``
subclasses grade each test case with an *exact* check — no LLM judge, no
network, no randomness:

- :class:`RightActionMetric` — ``metrics.py``'s own ``_right_proposal`` on
  the **policy-applied** prediction: did this policy's final decision
  actually propose the right operation+arguments. A policy that gates a
  correct raw pick away into an abstention does not get right-action
  credit for it — that is the whole point of grading per policy.
- :class:`WrongMutatingMetric` — the negation of ``metrics.py``'s
  ``_wrong_operation_mutating`` OR ``_wrong_arguments_mutating`` on the
  **policy-applied** prediction. A raw wrong-mutating pick that a policy
  gates away into an abstention (``operation`` becomes ``None``) is no
  longer a wrong-mutating call *under that policy* — this is issue 64's
  core distinction between "the model improved" and "the harness
  prevented the model's mistake".
- :class:`CorrectAbstainEscalateMetric` — for an escalate/explain-expected
  case, whether the **policy-applied** prediction's outcome correctly
  declined (``metrics._escalated`` semantics for escalate-expected, a bare
  ``"explain"`` outcome for explain-expected). Not applicable to an
  operation-expected case.
- :class:`MissingCandidateHandledMetric` — for a case
  ``metrics.is_missing_candidate`` flags (its gold operation was not among
  the offered candidates), whether the **policy-applied** prediction
  escalated (``metrics._escalated``) rather than guessing. Not applicable
  otherwise.

:func:`apply_policy_to_prediction` is the single place a "policy-applied"
``metrics.Prediction`` is built from an already-recorded raw one — the
corpus-level runner (t17) reuses it to feed ``metrics_bridge.compute`` per
``(subject, policy)``, so the per-case metrics here and the corpus figures
there always agree. It replays the same calibration/gate math
``evals.tool_jev.policies.apply`` uses (via that module's own already-loaded
``calibration_fit``/``gate``), because ``policies.apply`` itself only
returns the decision string and reason, not the scaled distribution or
decided label this module also needs.

Every metric's verdict is computed by calling the same
``scripts/lfm-finetune/metrics.py`` functions ``evals.tool_jev.metrics_bridge``
already calls (never a re-derived copy of the rule) on that policy-applied
prediction, rebuilt fresh from the trace fields and policy JSON this module
put in the test case's ``additional_metadata`` — so a metric's pass/fail is
reproducible from the metadata alone, with no need to keep the original
:class:`Trace` object around.

Corpus-level figures (ECE, Brier, coverage, abstain precision/recall,
per-slice) are never computed here or by DeepEval: :func:`evaluate_traces`
returns ``evals.tool_jev.metrics_bridge.compute``'s own output, computed
over the same policy-applied predictions, unchanged, alongside DeepEval's
own per-case ``EvaluationResult``.

No network: this module only calls DeepEval's synchronous, local
``evaluate()`` (``async_config=AsyncConfig(run_async=False)``), and every
metric here is a pure function of already-recorded fields. Import order
matters — ``evals.tool_jev`` (this package) must be imported before
``deepeval`` anywhere in the process, which is why this module's first
import is ``evals.tool_jev`` itself (see that package's docstring).

``evaluate_traces`` also keeps DeepEval's own local state directory
(``.deepeval/``, resolved *relative to the process's current working
directory* at the moment DeepEval touches disk — see
``deepeval.config.constants.HIDDEN_DIR``) out of this repository by running
``deepeval.evaluate.evaluate()`` inside ``contextlib.chdir()`` pointed at a
directory under ``results_folder``, restoring the previous cwd afterwards.
``os.chdir`` is process-global, so this is safe only because this evals
suite is single-threaded/single-process per run (no concurrent
``evaluate_traces`` call in another thread of the same process); a future
concurrent runner would need a different isolation strategy (e.g. a
subprocess) rather than relying on this chdir.
"""

from __future__ import annotations

import contextlib
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
from evals.tool_jev import policies as policy_module
from evals.tool_jev.trace import Trace, _is_inside_git_worktree

#: ``invalid_reason`` of a reply the provider cut at the output budget
#: (``track_a_loop.TRUNCATED``; repeated here so this module stays import-light).
TRUNCATED = "truncated"


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


def _resolve_policy(policy: Mapping[str, Any] | str) -> dict:
    """*policy* as a validated policy dict.

    A ``str`` names one of this package's own ``policies/<name>.json``
    files (``policies.builtin_policy_path`` + ``policies.load_policy``); a
    mapping is validated as-is (``policies.validate_policy``) so a caller
    building its own policy JSON in memory (e.g. a test) never has to write
    it to disk first.
    """
    if isinstance(policy, str):
        return policy_module.load_policy(policy_module.builtin_policy_path(policy))
    return policy_module.validate_policy(policy)


def apply_policy_to_prediction(
    policy: Mapping[str, Any] | str,
    prediction: Any,
    *,
    metrics_mod,
    offered: Sequence[str] | None = None,
) -> Any:
    """The ``metrics.Prediction`` *policy* actually produces on *prediction*.

    This is the one place a "policy-applied" prediction is built from an
    already-recorded raw one, shared by this module's four metrics and by
    the corpus-level runner (t17), so both layers always agree. It never
    reimplements ``evals.tool_jev.policies.apply``'s calibration/gate math:
    it calls the exact same ``calibration_fit.apply_scaling`` /
    ``gate.decide`` functions ``policies.apply`` uses internally (via that
    module's own already-loaded copies, ``policy_module.calibration_fit`` /
    ``policy_module.gate``), because ``policies.apply`` itself only returns
    the decision string and reason -- not the scaled distribution or the
    decided label this module also needs to build a full ``Prediction``.

    *prediction*'s ``outcome`` becomes the policy's final decision.
    ``operation``/``arguments`` are kept only when that decision is
    ``"propose"`` (``None`` otherwise -- ``metrics.Prediction`` requires
    this): the decided operation is ``gate.Decision.label``, and its
    arguments are *prediction*'s own recorded arguments when that label
    matches *prediction*'s own operation, else an empty dict (the gate
    picked a different top candidate than the raw record's own proposal;
    there are no recorded arguments for that hypothetical). ``candidates``
    is the policy's calibrated distribution when it carries a calibration
    block, otherwise *prediction*'s own distribution, unchanged.

    A row with no recorded distribution at all (``prediction.candidates``
    falsy) is returned **unchanged**: ``policies.apply``'s own
    ``NOT_GATEABLE`` case -- there is nothing for any policy to gate, so
    every policy agrees with the raw record on such a row.
    """
    validated = _resolve_policy(policy)
    if prediction.outcome == "invalid" and prediction.invalid_reason == TRUNCATED:
        # A reply cut at the output budget is never an answer, under any
        # policy: its distribution is not rebuilt into a decision (issue 64,
        # codex review of t17 item 13). Other invalid reasons keep the
        # designed behaviour below (a harness policy may use a saved
        # distribution).
        return prediction
    candidates = prediction.candidates
    if not candidates:
        return prediction

    scaled = dict(candidates)
    calibration = validated.get("calibration")
    if calibration:
        temperature = float(calibration.get("temperature", 1.0))
        vector = calibration.get("vector") or None
        scaled = policy_module.calibration_fit.apply_scaling(
            scaled, temperature=temperature, vector=vector
        )

    gate_json = validated.get("gate")
    thresholds = (
        policy_module.gate.Thresholds.from_json(gate_json)
        if gate_json
        else policy_module.gate.Thresholds()
    )
    offered_labels = list(offered) if offered is not None else list(candidates.keys())
    decision = policy_module.gate.decide(scaled, offered_labels, thresholds)

    if decision.outcome == "propose":
        operation = decision.label
        arguments = (
            dict(prediction.arguments)
            if (prediction.operation == operation and prediction.arguments is not None)
            else {}
        )
    else:
        operation = None
        arguments = None

    row = _prediction_row(
        case_id=prediction.id,
        expected=prediction.expected,
        outcome=decision.outcome,
        operation=operation,
        arguments=arguments,
        candidates=scaled,
        tokens=prediction.tokens,
        ttfd_ms=prediction.ttfd_ms,
        latency_ms=prediction.latency_ms,
        invalid_reason=prediction.invalid_reason,
    )
    return metrics_mod.Prediction.from_dict(row)


def build_test_case(
    trace: Trace, policy: Mapping[str, Any] | str, *, metrics_mod=None
) -> LLMTestCase:
    """One ``LLMTestCase`` for *trace* replayed through *policy*.

    ``input`` is a case-id placeholder (never the case text: real request
    text must never reach DeepEval's own results files).
    ``actual_output``/``expected_output`` are the policy-applied
    prediction's outcome and the gold calibration label. Every field the
    four metrics need to rebuild that same policy-applied prediction (and
    recompute their verdict) independently of the original ``Trace``
    object lives in ``additional_metadata`` -- including the resolved
    policy JSON itself, so a metric never has to re-resolve a policy name
    against disk.
    """
    metrics_mod = metrics_mod or metrics_bridge.load_metrics_module()
    if trace.ground_truth is None:
        raise DeepevalLayerError(
            f"trace {trace.case_id!r} carries no ground_truth to score against"
        )
    policy_dict = _resolve_policy(policy)
    raw_prediction = prediction_from_trace(trace, metrics_mod)
    policy_applied = apply_policy_to_prediction(
        policy_dict, raw_prediction, metrics_mod=metrics_mod
    )
    gold_label = metrics_mod.expected_label(trace.ground_truth)
    return LLMTestCase(
        input=f"case:{trace.case_id}",
        actual_output=policy_applied.outcome,
        expected_output=gold_label,
        additional_metadata={
            "case_id": trace.case_id,
            "split": trace.split,
            "subject": trace.subject,
            "policy": policy_dict["name"],
            "policy_json": policy_dict,
            "expected": dict(trace.ground_truth),
            "raw_outcome": trace.raw.outcome,
            "raw_operation": trace.raw.operation,
            "raw_arguments": trace.raw.arguments,
            "raw_candidates": trace.raw.candidates,
            "raw_tokens": trace.raw.tokens,
            "raw_ttfd_ms": trace.raw.ttfd_ms,
            "raw_latency_ms": trace.raw.latency_ms,
            "raw_invalid_reason": trace.raw.invalid_reason,
            "final_decision": policy_applied.outcome,
            "final_operation": policy_applied.operation,
            "final_arguments": policy_applied.arguments,
            "final_candidates": policy_applied.candidates,
        },
    )


def _raw_prediction_from_metadata(metadata: Mapping[str, Any], metrics_mod) -> Any:
    """Rebuild the *raw* ``metrics_mod.Prediction`` from a test case's ``additional_metadata``."""
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


def _policy_applied_prediction_from_metadata(metadata: Mapping[str, Any], metrics_mod) -> Any:
    """Rebuild the **policy-applied** ``metrics_mod.Prediction`` a metric grades.

    Recomputed fresh via :func:`apply_policy_to_prediction` from the raw
    fields and the resolved ``policy_json`` :func:`build_test_case` stored
    in ``additional_metadata`` -- so a metric's verdict is reproducible
    from the test case alone, and is always the *same* policy-applied
    prediction :func:`build_test_case` used to pick ``actual_output``.
    """
    raw_prediction = _raw_prediction_from_metadata(metadata, metrics_mod)
    return apply_policy_to_prediction(
        metadata["policy_json"], raw_prediction, metrics_mod=metrics_mod
    )


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
    """Right action: the **policy-applied** prediction's right proposal call.

    ``metrics.py``'s own ``_right_proposal`` already requires
    ``outcome == "propose"`` and an exact operation+arguments match, so a
    policy that gates a correct raw pick away into an abstention correctly
    loses right-action credit for it under this metric -- grading is per
    policy, on the decision that policy actually made, never on the raw
    model's distribution alone.

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
        prediction = _policy_applied_prediction_from_metadata(metadata, self.metrics_mod)
        if self.metrics_mod.expect_kind(prediction.expected) != "operation":
            return True, "not applicable: case does not expect an operation"
        gold = self.metrics_mod.expected_label(prediction.expected)
        correct = self.metrics_mod._right_proposal(prediction)
        reason = (
            f"policy-applied outcome {prediction.outcome!r} "
            f"{'proposes' if correct else 'does not propose'} gold {gold!r}"
        )
        return bool(correct), reason


class WrongMutatingMetric(_ExactMetric):
    """Wrong mutating: the **policy-applied** prediction must not be a wrong mutating call.

    Exactly ``metrics.py``'s ``_wrong_operation_mutating`` OR
    ``_wrong_arguments_mutating``, negated (a wrong mutating call is a
    failure, not a pass), on the policy-applied prediction: when a policy
    gates a raw wrong-mutating pick away into an abstention (``operation``
    becomes ``None``), that check is no longer true *under this policy* --
    issue 64's "the harness prevented the model's mistake" case.
    """

    @property
    def __name__(self) -> str:  # noqa: N802
        return "wrong_mutating"

    def _verdict(self, test_case: LLMTestCase) -> tuple[bool, str]:
        metadata = test_case.metadata or {}
        prediction = _policy_applied_prediction_from_metadata(metadata, self.metrics_mod)
        wrong = self.metrics_mod._wrong_operation_mutating(
            prediction
        ) or self.metrics_mod._wrong_arguments_mutating(prediction)
        if wrong:
            return False, "policy-applied outcome is a wrong mutating call"
        return True, "policy-applied outcome is not a wrong mutating call"


class CorrectAbstainEscalateMetric(_ExactMetric):
    """Correct abstain/escalate: the **policy-applied** outcome, on a decline-expected case.

    Not applicable to an operation-expected case (there is nothing to
    decline there); such a case always passes with a neutral reason.
    """

    @property
    def __name__(self) -> str:  # noqa: N802
        return "correct_abstain_escalate"

    def _verdict(self, test_case: LLMTestCase) -> tuple[bool, str]:
        metadata = test_case.metadata or {}
        prediction = _policy_applied_prediction_from_metadata(metadata, self.metrics_mod)
        kind = self.metrics_mod.expect_kind(prediction.expected)
        if kind == "escalate":
            escalated = self.metrics_mod._escalated(prediction)
            reason = (
                f"policy-applied outcome {prediction.outcome!r} "
                f"{'correctly escalates' if escalated else 'fails to escalate'} "
                "an escalate-expected case"
            )
            return escalated, reason
        if kind == "explain":
            explained = prediction.outcome == "explain"
            reason = (
                f"policy-applied outcome {prediction.outcome!r} "
                f"{'correctly explains' if explained else 'fails to explain'} "
                "an explain-expected case"
            )
            return explained, reason
        return True, "not applicable: case expects an operation, not a decline"


class MissingCandidateHandledMetric(_ExactMetric):
    """Missing-candidate handled: escalate when the gold op was never offered.

    ``metrics.is_missing_candidate`` decides whether this row's offered
    candidates ever included the gold label (invariant to a policy's
    calibration, which only reweights the same labels); when they didn't,
    the **policy-applied** prediction must have escalated
    (``metrics._escalated``) rather than guessing at an operation it was
    never offered. Not applicable to a row whose candidates did include
    gold.
    """

    @property
    def __name__(self) -> str:  # noqa: N802
        return "missing_candidate_handled"

    def _verdict(self, test_case: LLMTestCase) -> tuple[bool, str]:
        metadata = test_case.metadata or {}
        prediction = _policy_applied_prediction_from_metadata(metadata, self.metrics_mod)
        if not self.metrics_mod.is_missing_candidate(prediction):
            return True, "not applicable: gold operation was offered"
        escalated = self.metrics_mod._escalated(prediction)
        reason = (
            f"missing-candidate case {'correctly escalated' if escalated else 'did not escalate'} "
            f"under this policy"
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


def _refuse_inside_repo(path: Path, what: str) -> None:
    """Refuse a deepeval destination inside a git worktree, before anything is created.

    deepeval writes per-case results (and its relative ``.deepeval/`` state)
    under these folders; the repository must never receive them.
    """
    if _is_inside_git_worktree(path):
        raise DeepevalLayerError(
            f"{what} {path} is inside a git worktree; deepeval output must go to a "
            f"private run directory outside the repository"
        )


def evaluate_traces(
    traces: Sequence[Trace],
    policy: Mapping[str, Any] | str,
    *,
    results_folder: str | Path,
    metrics_mod=None,
    gate_mod=None,
    metrics: Sequence[BaseMetric] | None = None,
    display_config: DisplayConfig | None = None,
    async_config: AsyncConfig | None = None,
    cache_config: CacheConfig | None = None,
) -> EvaluationOutcome:
    """Score *traces* replayed through *policy* with DeepEval, and report corpus metrics.

    *policy* is either a builtin policy name (``"raw"``,
    ``"scorer-r3b-shipped"``, ...) or an in-memory policy dict; it is
    resolved once here and the *same* resolved dict is used to build every
    test case's policy-applied prediction (:func:`apply_policy_to_prediction`)
    and to feed ``metrics_bridge.compute`` for ``corpus_metrics`` below --
    so the per-case metrics and the corpus figures always agree on what
    "this policy's decision" was for a given row.

    Runs synchronously (``AsyncConfig(run_async=False)`` by default) and
    never calls a model or the network: every metric here is a pure
    function of the trace fields already recorded. ``results_folder`` is
    passed straight to ``DisplayConfig`` (or merged into a caller-supplied
    one) so the timestamped ``test_run_*.json`` lands exactly where the
    caller asked.

    DeepEval also keeps its own local state relative to the process's
    *current working directory* at the moment it touches disk (see this
    module's docstring). To keep that out of this repository regardless of
    the caller's own cwd, the actual ``deepeval.evaluate.evaluate()`` call
    below runs inside ``contextlib.chdir()`` pointed at a directory created
    under ``results_folder``, restoring the previous cwd on exit (even on
    error). ``os.chdir`` is process-global -- see the module docstring for
    why that is safe here.
    """
    metrics_mod = metrics_mod or metrics_bridge.load_metrics_module()
    gate_mod = gate_mod or metrics_bridge.load_gate_module()
    policy_dict = _resolve_policy(policy)
    metric_list = list(metrics) if metrics is not None else build_metrics(metrics_mod)

    predictions = [
        apply_policy_to_prediction(
            policy_dict, prediction_from_trace(trace, metrics_mod), metrics_mod=metrics_mod
        )
        for trace in traces
    ]
    test_cases = [build_test_case(trace, policy_dict, metrics_mod=metrics_mod) for trace in traces]

    # Resolved to an absolute path *before* the chdir below, since
    # DisplayConfig.results_folder (a relative string, if given one) would
    # otherwise be resolved against the temporary run_dir instead of the
    # caller's own cwd.
    results_folder = Path(results_folder).resolve()
    _refuse_inside_repo(results_folder, "results_folder")
    if display_config is not None and getattr(display_config, "results_folder", None):
        _refuse_inside_repo(
            Path(display_config.results_folder).resolve(), "display_config.results_folder"
        )
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

    run_dir = results_folder / ".deepeval-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(run_dir):
        deepeval_result = evaluate(
            test_cases,
            metric_list,
            async_config=async_config,
            display_config=display_config,
            cache_config=cache_config,
        )
    corpus_metrics = metrics_bridge.compute(predictions, metrics_mod=metrics_mod, gate_mod=gate_mod)
    return EvaluationOutcome(deepeval_result=deepeval_result, corpus_metrics=corpus_metrics)
