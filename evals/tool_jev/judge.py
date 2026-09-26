"""Blind, all-to-all judge panel for explain text, in two DeepEval passes (issue 64, t16).

Every subject that produced explain text (both candidates, the baselines and
every reference model) is scored by every judge on the panel. The operator's
decision, quoted from the spec: *"a judge's scores of its own answers are
kept apart from the panel score, which is aggregated over the other judges;
panel scores stay outside the release bars"*.

Two passes, so judge calls go into the same batches as subject calls and a
rerun never pays twice:

- **Pass 1 (record)** -- :func:`record_prompts` runs DeepEval's G-Eval metric
  against a :class:`RecordingModel`. The model captures the exact prompt
  G-Eval would send, turns it into a :class:`~evals.tool_jev.ledger.CallSpec`
  (``target="judge:<provider>/<model>"``, ``prompt_hash`` of the exact
  prompt) and raises a private sentinel so no score is produced. The runner
  registers :func:`unique_call_specs` in its ledger and sends the prompts
  through the provider batches like any other call.
- **Pass 2 (replay)** -- :func:`replay_scores` re-runs the *same* metric
  against a :class:`ReplayModel` that answers from a per-judge ``{judge
  name: {prompt_hash: reply text}}`` cache (see :func:`replies_from_ledger`;
  per judge because a prompt never names its judge) and raises
  :class:`ReplayMiss` on a miss. DeepEval computes the scores; no live call
  is made. Pass 2's prompts equal pass 1's byte for byte (tested).

**One scoring call per answer (verified against deepeval 4.2.6).**
``GEval.measure`` with fixed ``evaluation_steps`` skips
``_generate_evaluation_steps`` and makes exactly one model call, through
``deepeval.metrics.utils.decision.generate_rubric_score``, which asks for a
``ReasonScore`` (score *and* reason in the same JSON answer -- the reason is
not a second call, whatever ``verbose_mode`` says). That helper first tries
``model.generate_raw_response(prompt, top_logprobs=...)`` for the
logprob-weighted G-Eval score; :class:`DeepEvalBaseLLM` has no such method
and this module's models do not add one, so the ``AttributeError`` sends it
down the plain path, ``generate_with_schema`` -> ``generate(prompt,
schema=ReasonScore)``, and the score is the judge's integer answer, never a
logprob-weighted sum. ``generate_with_schema`` accepts a string reply and
parses it with ``trimAndLoadJson``. G-Eval's default range is 0..10 and
``measure`` returns ``score / 10``; ``strict_mode`` stays off.

**Guard for parked v6.** A metric that needs a second call whose prompt
depends on the first reply (G-Eval drafting its own steps from ``criteria``
is the concrete case: a ``Steps`` call first, then the scoring call) cannot
be batched in two passes, so it is refused with :class:`DependentCallError`:
:func:`build_metric` refuses a rubric with no fixed steps; the recording
model refuses a first call that is not the final ``ReasonScore`` call; the
replay model refuses any second call inside one ``measure``.

**Blinding.** A judge sees only the operator's request (G-Eval "Input") and
the answer's explain text ("Actual Output"), both passed through
:func:`nvsh.redact.redact` and then :func:`scrub`, which replaces every
roster model id, its path segments, its family token (``claude``, ``gpt``,
``kimi``, ...), known vendor names for that family, every non-domain
provider name and every subject name with ``[model]``. Two provider names
are exempt, :data:`DOMAIN_TERMS`: ``nvidia`` is the platform every case is
about (Jetson, DGX Spark, ``nvidia-smi``), so scrubbing it would change the
question without hiding an author, and ``local`` is an ordinary English
word. Answers get opaque ids (``item-00017``) assigned after a seeded
shuffle, and the (judge, answer) order is shuffled with the same seeded
:class:`random.Random`; the seed is recorded in ``judge_results.json``.

**Self-scores.** :func:`is_self` matches a judge to a subject by the last
path segment of the model id, case-insensitive, so the same model served by
two providers (``kimi-k3`` via build.nvidia.com and via OpenRouter) counts
as the judge's own answer. A judge's score of its own answers is reported
as its own row (verdict :data:`VERDICT_SELF`) and left out of that
subject's panel score, which is the mean over the *other* judges of each
judge's mean score across the subject's cases.

**Agreement.** Mean pairwise absolute difference (MAD) of the 0..1 scores
two judges gave the same answer, over answers neither of the two wrote,
per judge pair and overall (weighted by item count). 0 is perfect
agreement, 1 the worst. Chosen over Krippendorff's alpha because it is
defined for any two judges with one shared item and reads directly on the
score scale; it does not correct for chance agreement.

**Not applicable.** A subject whose answer carries no explain text (the
track-B scorers answer with a label; a saved prediction may carry no prose)
or whose case has no request text (held-out) is listed as
:data:`VERDICT_NOT_APPLICABLE` with score ``None`` -- never given invented
text.

**Output.** :func:`aggregate` returns the ``judge_results.json`` document
``evals/tool_jev/report.py``'s ``load_judge_results`` reads:
``{"judges": [...], "results": [{"subject", "policy", "judge", "verdict",
"score"}, ...]}``; everything else in it (per-case scores, agreement, seed,
rubric version and hash, ``counts_toward_release_bars: false``) is an
addition. No panel score is read by any release bar.

DeepEval keeps local state (``.deepeval/``) relative to the process cwd;
both passes run inside ``contextlib.chdir()`` into ``<run_dir>/.deepeval-judge``
exactly as ``deepeval_layer.evaluate_traces`` does (that module exposes no
reusable helper). ``os.chdir`` is process-global, so the passes must not
run concurrently in threads of one process.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import deepeval  # noqa: F401  (runs after the evals.tool_jev env guard below)
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

import evals.tool_jev  # noqa: F401  (env guard: must precede any deepeval import)
from evals.tool_jev.ledger import CallSpec, Ledger, ledger_key, prompt_hash
from evals.tool_jev.report import JUDGE_RESULTS_FILENAME
from nvsh.redact import redact

__all__ = [
    "DEFAULT_RUBRIC",
    "DOMAIN_TERMS",
    "PANEL_JUDGE",
    "VERDICT_SCORED",
    "VERDICT_SELF",
    "VERDICT_INVALID",
    "VERDICT_NOT_APPLICABLE",
    "VERDICT_PANEL",
    "VERDICT_NO_INDEPENDENT_JUDGE",
    "JudgeError",
    "DependentCallError",
    "ReplayMiss",
    "Rubric",
    "load_rubric",
    "JudgeId",
    "ExplainAnswer",
    "JudgeItem",
    "JudgePlan",
    "blind_terms",
    "scrub",
    "is_self",
    "plan_panel",
    "build_metric",
    "RecordingModel",
    "ReplayModel",
    "RecordedPrompt",
    "record_prompts",
    "unique_call_specs",
    "replies_from_ledger",
    "JudgeScore",
    "ReplayOutcome",
    "replay_scores",
    "aggregate",
    "write_judge_results",
]

RUBRIC_DIR = Path(__file__).resolve().parent / "rubric"
DEFAULT_RUBRIC = "explain-v1"
METRIC_NAME = "explain"

#: Provider names left in judged text: ``nvidia`` is the platform every case
#: is about, ``local`` an ordinary word. Neither identifies an author.
DOMAIN_TERMS = ("nvidia", "local")

#: Every provider name the manifest allows (``manifest.ALLOWED_PROVIDERS``),
#: blinded whether or not a roster member uses it.
_PROVIDER_NAMES = ("openai", "anthropic", "openrouter", "nvidia", "local")

#: Vendor names that identify a model family even when the id does not say them.
_VENDOR_ALIASES = {
    "claude": ("anthropic",),
    "gpt": ("openai", "chatgpt"),
    "kimi": ("moonshot", "moonshotai"),
    "qwen": ("alibaba", "tongyi"),
    "nemotron": (),
}

_BLIND_TOKEN = "[model]"
_MIN_TERM = 3
_SCORE_SCHEMA = "ReasonScore"
_SCORE_RANGE = (0, 10)
_EVAL_PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]

PANEL_JUDGE = "panel"
VERDICT_SCORED = "scored"
VERDICT_SELF = "self_score"
VERDICT_INVALID = "invalid"
VERDICT_NOT_APPLICABLE = "not_applicable"
VERDICT_PANEL = "panel_mean_excluding_self"
VERDICT_NO_INDEPENDENT_JUDGE = "no_independent_judge"


class JudgeError(ValueError):
    """The judge panel cannot run as configured."""


class DependentCallError(JudgeError):
    """A metric needed a second judge call that depends on the first reply."""


class ReplayMiss(JudgeError):
    """Pass 2 met a prompt with no cached reply."""


def _dependent_call_message(detail: str) -> str:
    return (
        f"judge metric needs a second, dependent judge call ({detail}); only metrics that "
        "score with exactly one call per answer can be batched in two passes. Multi-call "
        "judge metrics are parked v6: use a rubric with fixed evaluation steps."
    )


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rubric:
    """A versioned rubric: its id, its fixed G-Eval steps and the file's sha256."""

    version: str
    evaluation_steps: tuple[str, ...]
    sha256: str


_STEP_RE = re.compile(r"^\d+\.\s+(.*\S)\s*$")


def load_rubric(version: str = DEFAULT_RUBRIC) -> Rubric:
    """Read ``rubric/<version>.md``; its numbered "Evaluation steps" list, verbatim."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", version):
        raise JudgeError(f"bad rubric version {version!r}")
    path = RUBRIC_DIR / f"{version}.md"
    if not path.is_file():
        raise JudgeError(f"no rubric {version!r} at {path}")
    data = path.read_bytes()
    steps: list[str] = []
    in_steps = False
    for line in data.decode("utf-8").splitlines():
        if line.startswith("## "):
            in_steps = line.strip().lower() == "## evaluation steps"
            continue
        match = _STEP_RE.match(line) if in_steps else None
        if match:
            steps.append(match.group(1))
    if not steps:
        raise JudgeError(f"rubric {version!r} has no numbered '## Evaluation steps' list")
    return Rubric(version=version, evaluation_steps=tuple(steps), sha256=prompt_hash(data))


# ---------------------------------------------------------------------------
# Subjects, judges, blinding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeId:
    """One panel member, keyed like ``manifest.Judge`` by (provider, model)."""

    provider: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def target(self) -> str:
        return f"judge:{self.name}"


@dataclass(frozen=True)
class ExplainAnswer:
    """One subject's explain text for one case under one policy.

    ``provider``/``model`` identify a hosted subject (``None`` for a local
    candidate or baseline); they decide self-scores and blinding.
    ``explain_text`` or ``request_text`` of ``None`` makes the answer
    not applicable.
    """

    subject: str
    policy: str
    case_id: str
    request_text: str | None
    explain_text: str | None
    provider: str | None = None
    model: str | None = None

    @property
    def applicable(self) -> bool:
        return bool(self.explain_text and self.explain_text.strip()) and bool(
            self.request_text and self.request_text.strip()
        )


@dataclass(frozen=True)
class JudgeItem:
    """One answer as a judge will see it: opaque id, blinded and redacted text."""

    item_id: str
    answer: ExplainAnswer
    judged_input: str
    judged_output: str


@dataclass(frozen=True)
class JudgePlan:
    """Everything both passes need, fixed once: items, shuffled order, seed."""

    seed: int
    rubric: Rubric
    judges: tuple[JudgeId, ...]
    items: tuple[JudgeItem, ...]
    order: tuple[tuple[int, str], ...]
    not_applicable: tuple[ExplainAnswer, ...]
    params: Mapping[str, Any] = field(default_factory=dict)

    def item(self, item_id: str) -> JudgeItem:
        for it in self.items:
            if it.item_id == item_id:
                return it
        raise KeyError(item_id)


def _last_segment(model: str) -> str:
    return model.rsplit("/", 1)[-1].lower()


def _family(model: str) -> str | None:
    match = re.match(r"[a-z]+", _last_segment(model))
    return match.group(0) if match else None


def blind_terms(
    judges: Iterable[JudgeId],
    answers: Iterable[ExplainAnswer],
    extra: Iterable[str] = (),
) -> tuple[str, ...]:
    """Every string that could name an answer's author, longest first."""
    terms: set[str] = set()
    models: list[str] = [j.model for j in judges]
    providers: set[str] = set(_PROVIDER_NAMES) | {j.provider for j in judges}
    for answer in answers:
        terms.add(answer.subject)
        terms.update(answer.subject.split("/"))
        if answer.model:
            models.append(answer.model)
        if answer.provider:
            providers.add(answer.provider)
    for model in models:
        terms.add(model)
        terms.update(model.split("/"))
        family = _family(model)
        if family:
            terms.add(family)
            terms.update(_VENDOR_ALIASES.get(family, ()))
    terms.update(providers)
    terms.update(extra)
    domain = set(DOMAIN_TERMS)
    kept = {t.strip() for t in terms if t and t.strip()}
    kept = {t for t in kept if len(t) >= _MIN_TERM and t.lower() not in domain}
    return tuple(sorted(kept, key=lambda t: (-len(t), t.lower())))


def scrub(text: str, terms: Sequence[str]) -> str:
    """Replace each term (case-insensitive, whole-token) with ``[model]``."""
    if not terms:
        return text
    pattern = "|".join(re.escape(t) for t in terms)
    return re.sub(rf"(?<![A-Za-z0-9])(?:{pattern})(?![A-Za-z0-9])", _BLIND_TOKEN, text, flags=re.I)


def is_self(judge: JudgeId, answer: ExplainAnswer) -> bool:
    """True when *answer* was written by the judge's own model (any provider)."""
    if not answer.model:
        return False
    return _last_segment(answer.model) == _last_segment(judge.model)


def _redacted(text: str) -> str:
    return redact(text.encode("utf-8")).decode("utf-8", errors="replace")


def plan_panel(
    answers: Iterable[ExplainAnswer],
    judges: Sequence[JudgeId],
    *,
    seed: int,
    rubric: Rubric | None = None,
    extra_blind_terms: Iterable[str] = (),
    params: Mapping[str, Any] | None = None,
) -> JudgePlan:
    """Blind, redact and shuffle the answers; fix every (judge, answer) pair's order.

    ``params`` are the judge sampling parameters that go into every judge
    call's ledger key (e.g. ``{"reasoning": "medium", "max_output_tokens": 512}``).
    """
    judges = tuple(judges)
    if not judges:
        raise JudgeError("the judge panel is empty")
    if len(set(judges)) != len(judges):
        raise JudgeError("a judge appears twice on the panel")
    rubric = rubric or load_rubric()
    answers = list(answers)
    terms = blind_terms(judges, answers, extra_blind_terms)
    rng = random.Random(seed)
    applicable = [a for a in answers if a.applicable]
    not_applicable = tuple(a for a in answers if not a.applicable)
    rng.shuffle(applicable)
    items = tuple(
        JudgeItem(
            item_id=f"item-{index:05d}",
            answer=answer,
            judged_input=scrub(_redacted(answer.request_text or ""), terms),
            judged_output=scrub(_redacted(answer.explain_text or ""), terms),
        )
        for index, answer in enumerate(applicable)
    )
    order = [(j, item.item_id) for j in range(len(judges)) for item in items]
    rng.shuffle(order)
    return JudgePlan(
        seed=seed,
        rubric=rubric,
        judges=judges,
        items=items,
        order=tuple(order),
        not_applicable=not_applicable,
        params=dict(params or {}),
    )


# ---------------------------------------------------------------------------
# DeepEval models and metric
# ---------------------------------------------------------------------------


def _make_geval(rubric: Rubric, model: DeepEvalBaseLLM) -> GEval:
    return GEval(
        name=METRIC_NAME,
        evaluation_params=list(_EVAL_PARAMS),
        evaluation_steps=list(rubric.evaluation_steps),
        model=model,
        threshold=0.5,
        strict_mode=False,
        async_mode=False,
        verbose_mode=False,
    )


def build_metric(rubric: Rubric, model: DeepEvalBaseLLM) -> GEval:
    """G-Eval over (input, explain text) with the rubric's fixed steps: one call per answer."""
    if not rubric.evaluation_steps:
        raise DependentCallError(
            _dependent_call_message(
                f"rubric {rubric.version!r} has no fixed evaluation steps, so G-Eval would "
                "first ask the judge to draft them"
            )
        )
    return _make_geval(rubric, model)


class _Recorded(Exception):
    """Pass 1 sentinel: the prompt is captured, no score is produced."""


def _schema_name(schema: Any) -> str | None:
    return None if schema is None else getattr(schema, "__name__", str(schema))


class _JudgeModel(DeepEvalBaseLLM):
    """Shared base: a panel judge's identity, no network client, no logprobs.

    It deliberately has no ``generate_raw_response``, so G-Eval takes its
    plain ``generate(prompt, schema=ReasonScore)`` path (see module docstring).
    """

    def __init__(self, judge: JudgeId) -> None:
        self.judge = judge
        self.name = judge.name
        self.model = None

    def load_model(self, *args: Any, **kwargs: Any) -> "_JudgeModel":
        return self

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self.judge.name

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        return self.generate(prompt, schema=schema, **kwargs)


class RecordingModel(_JudgeModel):
    """Pass 1: capture the exact prompt DeepEval would send, then stop the metric."""

    def __init__(self, judge: JudgeId) -> None:
        super().__init__(judge)
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        name = _schema_name(schema)
        self.calls.append((prompt, name))
        if name != _SCORE_SCHEMA:
            raise DependentCallError(
                _dependent_call_message(
                    f"its first call asks for {name!r}, not the final {_SCORE_SCHEMA!r} score"
                )
            )
        raise _Recorded()


class ReplayModel(_JudgeModel):
    """Pass 2: answer from this judge's ``{prompt_hash: reply text}``; one call per metric run.

    The cache is per judge: a judge prompt never names the judge, so the
    same prompt sent to two judges has one hash and two different replies.
    """

    def __init__(self, judge: JudgeId, replies: Mapping[str, str]) -> None:
        super().__init__(judge)
        self.replies = replies
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        if self.prompts:
            raise DependentCallError(
                _dependent_call_message(
                    f"judge {self.judge.name} was asked a second time in one measure"
                )
            )
        self.prompts.append(prompt)
        digest = prompt_hash(prompt)
        try:
            return self.replies[digest]
        except KeyError:
            raise ReplayMiss(
                f"no cached reply from judge {self.judge.name} for prompt {digest[:12]}...: "
                "pass 2 answers only from pass 1's recorded prompts; run pass 1, send its "
                "prompts through the batch, and retry"
            ) from None


def _test_case(item: JudgeItem) -> LLMTestCase:
    return LLMTestCase(input=item.judged_input, actual_output=item.judged_output)


@contextlib.contextmanager
def _deepeval_cwd(run_dir: str | Path):
    work = Path(run_dir).resolve() / ".deepeval-judge"
    work.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(work):
        yield


# ---------------------------------------------------------------------------
# Pass 1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedPrompt:
    """One judge prompt captured in pass 1, with the ledger spec that sends it."""

    judge_index: int
    judge: JudgeId
    item_id: str
    prompt: str
    prompt_hash: str
    schema: str
    spec: CallSpec


def _call_spec(plan: JudgePlan, judge: JudgeId, item: JudgeItem, prompt: str) -> CallSpec:
    return CallSpec(
        provider=judge.provider,
        model=judge.model,
        subject_role="judge",
        case_id=item.answer.case_id,
        target=judge.target,
        prompt_hash=prompt_hash(prompt),
        params={
            "metric": METRIC_NAME,
            "rubric": plan.rubric.version,
            "rubric_sha256": plan.rubric.sha256,
            "schema": _SCORE_SCHEMA,
            **dict(plan.params),
        },
    )


def record_prompts(plan: JudgePlan, run_dir: str | Path) -> list[RecordedPrompt]:
    """Pass 1: every (judge, answer) prompt in the plan's shuffled order; no score."""
    recorded: list[RecordedPrompt] = []
    items = {it.item_id: it for it in plan.items}
    with _deepeval_cwd(run_dir):
        for judge_index, item_id in plan.order:
            judge = plan.judges[judge_index]
            item = items[item_id]
            model = RecordingModel(judge)
            metric = _make_geval(plan.rubric, model)
            try:
                metric.measure(_test_case(item), _show_indicator=False)
            except _Recorded:
                pass
            else:
                raise JudgeError(
                    f"G-Eval produced a score without asking judge {judge.name}; "
                    "the recording pass cannot trust this deepeval version"
                )
            if len(model.calls) != 1:
                raise DependentCallError(
                    _dependent_call_message(f"{len(model.calls)} calls in one measure")
                )
            prompt, schema = model.calls[0]
            recorded.append(
                RecordedPrompt(
                    judge_index=judge_index,
                    judge=judge,
                    item_id=item_id,
                    prompt=prompt,
                    prompt_hash=prompt_hash(prompt),
                    schema=schema or "",
                    spec=_call_spec(plan, judge, item, prompt),
                )
            )
    return recorded


def unique_call_specs(recorded: Iterable[RecordedPrompt]) -> list[CallSpec]:
    """The pass-1 specs, one per ledger key, in first-seen order."""
    seen: set[str] = set()
    out: list[CallSpec] = []
    for rec in recorded:
        key = ledger_key(rec.spec)
        if key not in seen:
            seen.add(key)
            out.append(rec.spec)
    return out


def replies_from_ledger(
    ledger: Ledger,
    recorded: Iterable[RecordedPrompt],
    *,
    extract_text: Callable[[bytes], str],
) -> dict[str, dict[str, str]]:
    """``{judge name: {prompt_hash: reply text}}`` for every recorded prompt with a cached answer.

    ``extract_text`` turns a provider's cached raw bytes into the judge's
    reply text (it is provider-specific and lives with the adapter).
    """
    replies: dict[str, dict[str, str]] = {}
    for rec in recorded:
        cached = ledger.cached(ledger_key(rec.spec))
        if cached is not None:
            replies.setdefault(rec.judge.name, {})[rec.prompt_hash] = extract_text(cached.raw)
    return replies


# ---------------------------------------------------------------------------
# Pass 2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeScore:
    """One judge's score of one answer (0..1), or an invalid reply."""

    judge: JudgeId
    item_id: str
    subject: str
    policy: str
    case_id: str
    self_score: bool
    verdict: str
    score: float | None
    raw_score: float | None
    reason: str | None
    prompt_hash: str


@dataclass(frozen=True)
class ReplayOutcome:
    scores: list[JudgeScore]
    prompts: list[str]


def replay_scores(
    plan: JudgePlan, replies: Mapping[str, Mapping[str, str]], run_dir: str | Path
) -> ReplayOutcome:
    """Pass 2: DeepEval scores every pair from cached replies; no live call is made.

    ``replies`` is ``{judge name: {prompt_hash: reply text}}``, as
    :func:`replies_from_ledger` returns it.
    """
    items = {it.item_id: it for it in plan.items}
    scores: list[JudgeScore] = []
    prompts: list[str] = []
    low, high = _SCORE_RANGE
    with _deepeval_cwd(run_dir):
        for judge_index, item_id in plan.order:
            judge = plan.judges[judge_index]
            item = items[item_id]
            model = ReplayModel(judge, replies.get(judge.name, {}))
            metric = _make_geval(plan.rubric, model)
            verdict = VERDICT_SELF if is_self(judge, item.answer) else VERDICT_SCORED
            score = raw = reason = None
            try:
                metric.measure(_test_case(item), _show_indicator=False)
            except JudgeError:
                raise
            except (ValueError, KeyError, TypeError) as exc:
                verdict, reason = VERDICT_INVALID, f"unparseable judge reply: {exc}"
            else:
                raw = metric.score * (high - low) + low
                if not low <= raw <= high:
                    verdict, reason, raw = VERDICT_INVALID, f"score {raw:g} out of range", None
                else:
                    score, reason = float(metric.score), metric.reason
            prompts.extend(model.prompts)
            scores.append(
                JudgeScore(
                    judge=judge,
                    item_id=item_id,
                    subject=item.answer.subject,
                    policy=item.answer.policy,
                    case_id=item.answer.case_id,
                    self_score=is_self(judge, item.answer),
                    verdict=verdict,
                    score=score,
                    raw_score=None if raw is None else round(raw, 6),
                    reason=reason,
                    prompt_hash=prompt_hash(model.prompts[0]) if model.prompts else "",
                )
            )
    return ReplayOutcome(scores=scores, prompts=prompts)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _agreement(plan: JudgePlan, scores: Sequence[JudgeScore]) -> dict[str, Any]:
    by_item: dict[str, dict[int, float]] = {}
    for s in scores:
        if s.score is not None:
            by_item.setdefault(s.item_id, {})[plan.judges.index(s.judge)] = s.score
    items = {it.item_id: it for it in plan.items}
    pairs = []
    total, count = 0.0, 0
    for a in range(len(plan.judges)):
        for b in range(a + 1, len(plan.judges)):
            diffs = [
                abs(got[a] - got[b])
                for item_id, got in by_item.items()
                if a in got
                and b in got
                and not is_self(plan.judges[a], items[item_id].answer)
                and not is_self(plan.judges[b], items[item_id].answer)
            ]
            pairs.append(
                {
                    "judges": [plan.judges[a].name, plan.judges[b].name],
                    "mad": _mean(diffs),
                    "n": len(diffs),
                }
            )
            total += sum(diffs)
            count += len(diffs)
    return {
        "method": "mean_pairwise_absolute_difference",
        "scale": "0..1 score difference; 0 = perfect agreement",
        "excludes": "answers written by either judge of the pair",
        "pairs": pairs,
        "overall": total / count if count else None,
    }


def aggregate(plan: JudgePlan, scores: Sequence[JudgeScore]) -> dict[str, Any]:
    """The ``judge_results.json`` document (report.py shape plus additions)."""
    groups: dict[tuple[str, str], dict[JudgeId, list[JudgeScore]]] = {}
    for s in scores:
        groups.setdefault((s.subject, s.policy), {}).setdefault(s.judge, []).append(s)

    results: list[dict[str, Any]] = []
    for (subject, policy), per_judge in groups.items():
        independent: list[float] = []
        excluded: list[str] = []
        for judge in plan.judges:
            rows = per_judge.get(judge, [])
            if not rows:
                continue
            valid = [r.score for r in rows if r.score is not None]
            is_own = any(r.self_score for r in rows)
            mean = _mean(valid)
            results.append(
                {
                    "subject": subject,
                    "policy": policy,
                    "judge": judge.name,
                    "verdict": (
                        VERDICT_SELF if is_own else VERDICT_SCORED if valid else VERDICT_INVALID
                    ),
                    "score": mean,
                    "n_cases": len(valid),
                    "n_invalid": len(rows) - len(valid),
                }
            )
            if is_own:
                excluded.append(judge.name)
            elif mean is not None:
                independent.append(mean)
        panel = _mean(independent)
        results.append(
            {
                "subject": subject,
                "policy": policy,
                "judge": PANEL_JUDGE,
                "verdict": VERDICT_PANEL if panel is not None else VERDICT_NO_INDEPENDENT_JUDGE,
                "score": panel,
                "n_judges": len(independent),
                "excluded_self_judges": excluded,
            }
        )

    judged = set(groups)
    na_seen: dict[tuple[str, str], int] = {}
    for answer in plan.not_applicable:
        key = (answer.subject, answer.policy)
        na_seen[key] = na_seen.get(key, 0) + 1
    for (subject, policy), n in na_seen.items():
        if (subject, policy) in judged:
            continue
        results.append(
            {
                "subject": subject,
                "policy": policy,
                "judge": None,
                "verdict": VERDICT_NOT_APPLICABLE,
                "score": None,
                "n_cases": n,
                "reason": "no explain text (or no request text) to judge",
            }
        )

    return {
        "judges": [j.name for j in plan.judges],
        "results": results,
        "counts_toward_release_bars": False,
        "seed": plan.seed,
        "rubric": {"version": plan.rubric.version, "sha256": plan.rubric.sha256},
        "metric": {"name": METRIC_NAME, "kind": "G-Eval", "score_range": list(_SCORE_RANGE)},
        "agreement": _agreement(plan, scores),
        "items": [
            {
                "item_id": it.item_id,
                "subject": it.answer.subject,
                "policy": it.answer.policy,
                "case_id": it.answer.case_id,
            }
            for it in plan.items
        ],
        "scores": [
            {
                "judge": s.judge.name,
                "item_id": s.item_id,
                "subject": s.subject,
                "policy": s.policy,
                "case_id": s.case_id,
                "self_score": s.self_score,
                "verdict": s.verdict,
                "score": s.score,
                "raw_score": s.raw_score,
                "reason": s.reason,
            }
            for s in scores
        ],
        "not_applicable_cases": [
            {"subject": a.subject, "policy": a.policy, "case_id": a.case_id}
            for a in plan.not_applicable
        ],
    }


def write_judge_results(run_dir: str | Path, results: Mapping[str, Any]) -> Path:
    """Write ``<run_dir>/judge_results.json`` (sorted keys) and return its path."""
    path = Path(run_dir) / JUDGE_RESULTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
