"""Runner for the Tool-Jev DeepEval release gate (issue #64, task t17).

One run directory holds one gate run: the call ledger and response cache
(``ledger.py``), the run record (``run.json``), and -- once every call is
answered -- the traces, per-(subject, policy) metrics, permutation entries,
judge results, ``result.json`` and ``report.md`` that ``report.py`` reads and
writes. The run directory is PRIVATE and must sit outside every git worktree
(refused otherwise); only the aggregate page may later be copied into
``docs/`` by the operator. The runner never writes into the repository.

Run-dir layout::

    run.json                     run record: run id/date, manifest sha256 history,
                                 hosts that received case text, stops, capability
                                 entries, per-model counts, spend, smoke summary
    ledger.json cache/ events.jsonl .lock    the call ledger (ledger.py)
    traces/<subject>.jsonl       raw record + each policy's final decision
    metrics/<subject>__<policy>.json         metrics_bridge.compute() output
    deepeval/<subject>__<policy>/            deepeval's own per-case results
    .deepeval-judge/             deepeval state of the judge passes
    permutation.json judge_results.json manifest.json   what report.py reads
    result.json report.md        the result and the page (report.py)
    smoke.json                   smoke only: tokens, cost, projection, OK/CAPPED
    drive.json drive.log         drive only: money-stop clock, one line per step

Subject names are file-safe: ``<checkpoint>.<case set>`` for a candidate or
baseline, ``<provider>.<model>.<A|B>.<case set>`` for a reference.

Subjects
--------
- **candidates and baselines** replay their SAVED per-case prediction files
  (``RunEntry.predictions_for(case_set)``), via
  ``trace.Trace.from_prediction_line``: no GPU, no model call. Each row is
  labelled with its exact artifact (h1): the predictions file's sha256 plus
  the manifest's repo id / revision.
- **references, Track A** run through the candidates' own multi-round
  ``LfmTier`` loop (deviation d1, ``track_a_loop.run_round``), one round at a
  time over a case set: a batch provider gets ONE batch per (model, case set,
  round), a sync provider gets the round's calls under its concurrency cap.
- **references, Track B** send ``request.build_choice_request`` once per case.
- **judges** (``judge.py``'s two passes) run once every subject's answers are
  final: pass 1 records the G-Eval prompts, which go out as free-text
  (``interface="text"``) ledger calls, batched for OpenAI / Anthropic; pass 2
  replays the cached replies (``Provider.reply_text``; a truncated reply is an
  invalid judge answer) with zero network.

Stops (c41 c42 c45 h28 h29 h31 h32; plan risks r6, r10)
-------------------------------------------------------
Everything goes through one :class:`~evals.tool_jev.ledger.Ledger`, so a run
survives Ctrl+C, a reset or a money stop and ``continue`` never pays twice.

- **money stop** (HTTP 402 / insufficient credit or quota) stops one
  *provider*; its calls stay pending; other providers keep going.
  ``continue`` retries it; the ``drive`` loop probes it with one call every
  30 minutes.
- **budget cap** (``[budget.<provider>] usd_cap`` reached by the spend the
  ledger's cached usage x the manifest's prices adds up to) stops one
  provider until the cap is raised.
- **transient stop** (429, timeout, network, a missing key) pauses one
  provider for the rest of the pass.
- **rejected request** (400/401/403/404/422, unsupported parameter) stops one
  *model* until its request parameters change; it is recorded as a
  capability entry, never a crash.
- **truncation stop** (r10): once a model has ``[stops] min_answers`` answers
  and ``max_truncated_share`` of them were cut at the output budget, its
  calls stop until the operator raises ``max_output_tokens``, lowers
  ``reasoning`` or replaces the model.
- **stop and ask**: a batch whose submission may have reached the provider
  but cannot be found or ruled out (``BatchLookupUnresolved``) stops the run
  with a message naming the provider, the submit ref and the keys. It is
  never resubmitted.

Model-answer failures are ``invalid`` and counted; infrastructure stops stay
``pending``.

Every host that received case text is recorded in ``run.json`` before the
call leaves (h27). Nothing here reads a key: adapters read their key from the
environment variable the manifest names, at call time.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import evals.tool_jev  # noqa: F401  (env guard before anything imports deepeval)
from nvsh.ops import table as ops_table
from nvsh.tiers import bench as tier_bench

from . import permutation as perm
from . import request as contract
from . import track_a_loop as loop
from .cases import Case, TrainingOverlapError, load_case_set, training_overlap
from .ledger import (
    DONE,
    INVALID,
    PENDING,
    SUBMITTED,
    CallSpec,
    Ledger,
    LedgerCorrupt,
    LedgerLocked,
    canonical_json,
    prompt_hash,
)
from .manifest import CaseSet, Manifest, ManifestError, Reference, RunEntry, load_manifest
from .providers import anthropic as anthropic_mod
from .providers import openai as openai_mod
from .providers import openai_compat
from .providers.base import (
    BatchHandle,
    BatchLookupUnresolved,
    CallRequest,
    CallResult,
    MissingProviderKey,
    Provider,
    ProviderCapabilities,
)
from .providers.errors import Classification, Outcome, classify_transport, stop_message
from .trace import RawRecord, Trace, _is_inside_git_worktree, write_traces

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_USER = 1
EXIT_ENV = 2
EXIT_ASK = 3
EXIT_STOPPED = 4
EXIT_WAITING = 5
EXIT_INTERRUPTED = 130

STATUS_COMPLETE = "complete"
STATUS_WAITING = "waiting"
STATUS_STOPPED = "stopped"

#: The private data root case-set paths, the ground snapshot and relative
#: prediction paths are joined onto.
ENV_PRIVATE_ROOT = "NVSH_EVALS_PRIVATE_ROOT"
#: What the manifest's ``${NVSH_EVALS_PRIVATE}`` placeholder expands to
#: (falls back to ``NVSH_EVALS_PRIVATE_ROOT``).
ENV_PRIVATE = "NVSH_EVALS_PRIVATE"
PLACEHOLDER = "${NVSH_EVALS_PRIVATE}"
#: Optional per-kind base URL override for an OpenAI-compatible host, e.g.
#: ``NVSH_EVALS_BASE_URL_LOCAL`` for the lobes gateway's port. Validated by
#: the adapter (``local`` must be localhost, hosted kinds must not be).
ENV_BASE_URL = "NVSH_EVALS_BASE_URL_{kind}"

RUN_FILE = "run.json"
SMOKE_FILE = "smoke.json"
RESULT_FILE = "result.json"
PAGE_FILE = "report.md"
DEEPEVAL_DIR = "deepeval"

SUBJECT_ROLE = loop.DEFAULT_SUBJECT_ROLE
CHOICE_TARGET = "choice"
JUDGE_REQUEST_PREFIX = "j-"

#: Reasons that make a stop a *money* stop (one provider; probed by ``drive``).
MONEY_REASONS = frozenset({"insufficient_credit", "budget_cap_reached"})

#: The truncation stop's operator question (plan risk r10).
TRUNCATION_DECISION = (
    "needs operator decision: raise max_output_tokens, lower reasoning, or replace the model"
)

#: Why a Track A subject has no permutation figure (``permutation.py``'s docstring).
TRACK_A_PERMUTATION_REASON = (
    "permutation_probe.py only probes a Track B one-position scorer (scorer.permute / "
    "scorer.same_choice); a Track A tool-call answer has no candidate listing or letter "
    "map to permute"
)

#: How each provider is asked for ``reasoning = "none"`` (every other level is
#: passed through as the provider's own effort value). ``None`` omits the
#: parameter entirely.
NONE_REASONING: dict[str, str | None] = {
    # OpenAI Responses API ``reasoning.effort``: "minimal" is the lowest
    # effort the reasoning guide documents for reasoning models; omitting the
    # field would mean the model default ("medium").
    # https://platform.openai.com/docs/guides/reasoning
    "openai": "minimal",
    # Anthropic ``output_config.effort`` takes low / medium / high; "low" is
    # its minimum (the adapter leaves ``thinking`` unset on purpose).
    # https://docs.claude.com/en/docs/build-with-claude/effort
    "anthropic": "low",
    # OpenRouter's unified ``reasoning.effort`` accepts "none" to turn
    # reasoning off. https://openrouter.ai/docs/use-cases/reasoning-tokens
    "openrouter": "none",
    # build.nvidia.com and local servers: the ``reasoning_effort`` field is an
    # unverified assumption (handoff, plan risk r10), sent only when the
    # manifest lists the "reasoning" capability; "none" omits it.
    "nvidia": None,
    "local": None,
}


class RunError(Exception):
    """A configuration or input problem the operator must fix (exit 1)."""


class EnvError(RunError):
    """The environment lacks something the run needs (exit 2)."""


class StopAndAsk(Exception):
    """The run cannot tell whether a batch was accepted: stop, ask, never resubmit (exit 3)."""


def provider_reasoning(provider: str, level: str) -> str | None:
    """The reasoning value sent to *provider* for manifest level *level* (``None`` = omit)."""
    if level != "none":
        return level
    return NONE_REASONING.get(provider)


# ---------------------------------------------------------------------------
# Private paths
# ---------------------------------------------------------------------------


def private_root(env: Mapping[str, str]) -> Path:
    raw = env.get(ENV_PRIVATE_ROOT) or env.get(ENV_PRIVATE)
    if not raw:
        raise EnvError(
            f"{ENV_PRIVATE_ROOT} is not set: case sets, the ground snapshot and saved "
            "predictions live in the operator's private data root, never in the repository"
        )
    return Path(raw)


def resolve_private(value: str, env: Mapping[str, str]) -> Path:
    """A manifest path: ``${NVSH_EVALS_PRIVATE}`` expanded, relative ones under the root."""
    if PLACEHOLDER in value:
        base = env.get(ENV_PRIVATE) or env.get(ENV_PRIVATE_ROOT)
        if not base:
            raise EnvError(f"{value!r} needs {ENV_PRIVATE} (or {ENV_PRIVATE_ROOT}) to be set")
        value = value.replace(PLACEHOLDER, base)
    if "${" in value:
        raise RunError(f"{value!r} carries a placeholder this runner does not expand")
    path = Path(value)
    return path if path.is_absolute() else private_root(env) / path


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise RunError(f"cannot read {path}: {exc}") from exc


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "x"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

#: ``factory(reference, budget, env) -> Provider``; tests inject FakeProviders.
ProviderFactory = Callable[..., Provider]


def default_factory(ref: Reference, budget: Any, env: Mapping[str, str]) -> Provider:
    """The real adapter for *ref* (keys are read from the environment at call time)."""
    if ref.provider == "openai":
        return openai_mod.OpenAIProvider(
            ref.model, api_key_env=ref.api_key_env or openai_mod.DEFAULT_API_KEY_ENV
        )
    if ref.provider == "anthropic":
        return anthropic_mod.AnthropicProvider(
            ref.model,
            api_key_env=ref.api_key_env or "ANTHROPIC_API_KEY",
            name=f"anthropic:{ref.model}",
        )
    capabilities = ProviderCapabilities(
        logprobs="logprobs" in ref.capabilities,
        batch=False,
        reasoning="reasoning" in ref.capabilities,
    )
    limiter = None
    if budget is not None and budget.requests_per_minute:
        limiter = openai_compat.RateLimiter(budget.requests_per_minute)
    kwargs: dict[str, Any] = {}
    if ref.api_key_env:
        kwargs["api_key_env"] = ref.api_key_env
    return openai_compat.OpenAICompatProvider(
        ref.provider,
        ref.model,
        base_url=env.get(ENV_BASE_URL.format(kind=ref.provider.upper())) or None,
        capabilities=capabilities,
        rate_limiter=limiter,
        **kwargs,
    )


def provider_host(provider: Provider) -> str:
    """The network host *provider* sends case text to."""
    host = getattr(provider, "host", None)
    if isinstance(host, str) and host:
        return host
    if isinstance(provider, openai_mod.OpenAIProvider):
        return urlsplit(openai_mod.API_BASE).hostname or openai_mod.API_BASE
    if isinstance(provider, anthropic_mod.AnthropicProvider):
        return anthropic_mod.API_HOST
    if isinstance(provider, openai_compat.OpenAICompatProvider):
        return urlsplit(provider.base_url).hostname or provider.base_url
    return provider.name


def _transport_kind(provider: str) -> str:
    return provider if provider in ("openai", "anthropic") else "openai_compat"


def classify_exception(exc: BaseException, provider: str) -> Classification | None:
    """The stop an adapter exception stands for, or ``None`` for a bug (re-raised)."""
    found = getattr(exc, "classification", None)
    if isinstance(found, Classification):
        return found
    if isinstance(exc, MissingProviderKey):
        return Classification(Outcome.PENDING, "missing_key", stop=True, retryable=True)
    status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return classify_transport(_transport_kind(provider), status_code=status)
    if isinstance(exc, (OSError, openai_mod.OpenAITransportError, openai_compat.TransportError)):
        return classify_transport(_transport_kind(provider), error_type="network_error")
    return None


def _classification_of(result: CallResult) -> Classification:
    """A ``pending`` result (batch line error, expiry) as the stop it stands for."""
    reason = result.reason or "unrecognized_infra_condition"
    rejected = reason.startswith("request_rejected:")
    return Classification(Outcome.PENDING, reason, stop=True, retryable=not rejected)


@dataclass
class Model:
    """One reference model: its adapter, where it sends, and how it is routed."""

    ref: Reference
    provider: Provider
    host: str
    route: str  # "batch" | "sync"

    @property
    def label(self) -> str:
        return f"{self.ref.provider}/{self.ref.model}"

    @property
    def kind(self) -> str:
        return self.ref.provider

    @property
    def knobs(self) -> dict[str, Any]:
        """Reasoning and output budget: sent with, and keyed into, every subject call."""
        out: dict[str, Any] = {
            "max_output_tokens": self.ref.max_output_tokens or contract.DEFAULT_MAX_OUTPUT_TOKENS
        }
        effort = provider_reasoning(self.ref.provider, self.ref.reasoning)
        if effort:
            out["reasoning"] = effort
        return out

    def cost(self, usage: Mapping[str, int], discount: float) -> float:
        tokens_in = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        tokens_out = usage.get("output_tokens", usage.get("completion_tokens", 0))
        usd = (tokens_in * self.ref.usd_per_mtok_in + tokens_out * self.ref.usd_per_mtok_out) / 1e6
        return usd * (1 - discount) if self.route == "batch" else usd


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Subject:
    """One report subject: a checkpoint or a reference track, on one case set."""

    name: str
    kind: str  # candidate | baseline | reference
    track: str
    case_set: CaseSet
    policies: tuple[str, ...]
    entry: RunEntry | None = None
    model: str | None = None  # Model.label for a reference


@dataclass
class Saved:
    """A candidate/baseline's saved predictions for one case set."""

    traces: list[Trace]
    explanations: dict[str, str]
    artifact: dict[str, Any]


@dataclass
class Plan:
    manifest: Manifest
    cases: dict[str, tuple[Case, ...]]
    subjects: list[Subject]
    saved: dict[str, Saved]
    models: dict[str, Model]
    judges: list[Model]
    snapshot: Mapping[str, object] | None
    platform: Any
    smoke: bool = False
    full_case_count: int = 0

    @property
    def references(self) -> list[Subject]:
        return [s for s in self.subjects if s.kind == "reference"]


def subject_name(base: str, case_set: str, track: str | None = None) -> str:
    parts = [_safe(base)] + ([track] if track else []) + [_safe(case_set)]
    return ".".join(parts)


def _load_cases(cs: CaseSet, env: Mapping[str, str]) -> tuple[Case, ...]:
    path = private_root(env) / cs.path
    try:
        cases = load_case_set({cs.split: str(path)}, cs.split, include_heldout=cs.include_heldout)
    except (OSError, ValueError, KeyError) as exc:
        raise RunError(f"case set {cs.name!r}: cannot load {path}: {exc}") from exc
    if len(cases) != cs.count:
        raise RunError(
            f"case set {cs.name!r}: the manifest says {cs.count} cases, the file has {len(cases)}"
        )
    if len({c.id for c in cases}) != len(cases):
        raise RunError(f"case set {cs.name!r}: case ids are not unique")
    return cases


def _load_saved(
    entry: RunEntry, cs: CaseSet, cases: Sequence[Case], name: str, env: Mapping[str, str]
) -> Saved | None:
    """The saved predictions of *entry* on *cs*, in case order; ``None`` when it has none."""
    path = resolve_private(entry.predictions_for(cs.name), env)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RunError(f"{entry.name}: cannot read saved predictions {path}: {exc}") from exc
    wanted = {c.id for c in cases}
    rows: dict[str, dict] = {}
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise RunError(f"{entry.name}: {path} line {number} is not JSON") from exc
        case_id = row.get("id") if isinstance(row, dict) else None
        if case_id in wanted:
            if case_id in rows:
                raise RunError(f"{entry.name}: {path} has case {case_id!r} twice")
            rows[case_id] = row
    if not rows:
        return None
    missing = [c.id for c in cases if c.id not in rows]
    if missing:
        raise RunError(
            f"{entry.name}: saved predictions for case set {cs.name!r} miss {len(missing)} of "
            f"{len(cases)} cases; a partial replay would bias every figure"
        )
    if entry.train_split:
        try:
            training_overlap(wanted, resolve_private(entry.train_split, env))
        except TrainingOverlapError as exc:
            raise RunError(f"{entry.name}: {exc}") from exc
        except OSError as exc:
            raise RunError(f"{entry.name}: cannot read its train split: {exc}") from exc
    traces, explanations = [], {}
    for case in cases:
        row = rows[case.id]
        try:
            traces.append(Trace.from_prediction_line(row, split=cs.split, subject=name))
        except Exception as exc:  # metrics.MetricsError names the problem
            raise RunError(f"{entry.name}: case {case.id!r}: {exc}") from exc
        text = row.get("explanation")
        if row.get("outcome") == "explain" and isinstance(text, str) and text.strip():
            explanations[case.id] = text
    artifact = {"predictions_sha256": hashlib.sha256(data).hexdigest()}
    if entry.repo_id:
        artifact["repo_id"] = entry.repo_id
    if entry.revision:
        artifact["revision"] = entry.revision
    return Saved(traces=traces, explanations=explanations, artifact=artifact)


def build_plan(
    manifest: Manifest,
    env: Mapping[str, str],
    factory: ProviderFactory,
    *,
    smoke_cases: int | None = None,
) -> Plan:
    """Everything one pass needs, from the manifest and the private data root."""
    cases: dict[str, tuple[Case, ...]] = {}
    for cs in manifest.case_sets:
        cases[cs.name] = _load_cases(cs, env)
    sendable = list(manifest.sendable_case_sets())
    full_case_count = sum(len(cases[cs.name]) for cs in sendable)
    if smoke_cases is not None:
        if not sendable:
            raise RunError("smoke needs a sendable (non-held-out) case set")
        first = sendable[0]
        cases = {first.name: cases[first.name][:smoke_cases]}
        sendable = [first]

    subjects: list[Subject] = []
    saved: dict[str, Saved] = {}
    if smoke_cases is None:
        for kind, entries in (("candidate", manifest.candidates), ("baseline", manifest.baselines)):
            for entry in entries:
                for cs in manifest.case_sets:
                    name = subject_name(entry.name, cs.name)
                    loaded = _load_saved(entry, cs, cases[cs.name], name, env)
                    if loaded is None:
                        continue
                    for policy in entry.policies:
                        if not policy_module_path(policy).is_file():
                            raise RunError(f"{entry.name}: unknown policy {policy!r}")
                    subjects.append(
                        Subject(name, kind, entry.track, cs, tuple(entry.policies), entry=entry)
                    )
                    saved[name] = loaded

    models: dict[str, Model] = {}
    for ref in manifest.references:
        budget = manifest.budget_for(ref.provider)
        if budget is None:
            raise RunError(f"{ref.provider}/{ref.model}: no [budget.{ref.provider}] table")
        provider = factory(ref, budget, env)
        if ref.batch and not provider.capabilities.batch:
            raise RunError(f"{ref.provider}/{ref.model}: batch = true but the adapter has none")
        route = "batch" if ref.batch else "sync"
        model = Model(ref=ref, provider=provider, host=provider_host(provider), route=route)
        models[model.label] = model
        for cs in sendable:
            for track in ("A", "B"):
                base = f"{ref.provider}.{ref.model}"
                subjects.append(
                    Subject(
                        subject_name(base, cs.name, track),
                        "reference",
                        track,
                        cs,
                        ("raw",),
                        model=model.label,
                    )
                )
    names = [s.name for s in subjects]
    if len(set(names)) != len(names):
        raise RunError("two subjects map to the same file name; rename a checkpoint")
    judges = [models[f"{j.provider}/{j.model}"] for j in manifest.judges]

    snapshot, platform = None, None
    if models and sendable:
        cfg = manifest.track_a
        if not cfg.snapshot:
            raise RunError("[track_a] snapshot is required to run the references' Track A loop")
        try:
            snapshot, _digest = loop.measure.load_snapshot(private_root(env) / cfg.snapshot)
        except loop.measure.MeasureError as exc:
            raise RunError(f"ground snapshot: {exc}") from exc
        platform = tier_bench.world_platform(
            {"platform": cfg.platform, "device_cli": cfg.device_cli}
        )
    return Plan(
        manifest=manifest,
        cases=cases,
        subjects=subjects,
        saved=saved,
        models=models,
        judges=judges,
        snapshot=snapshot,
        platform=platform,
        smoke=smoke_cases is not None,
        full_case_count=full_case_count,
    )


def policy_module_path(name: str) -> Path:
    from . import policies

    return policies.builtin_policy_path(name)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def choice_call(model: Model, case: Case) -> tuple[CallSpec, CallRequest]:
    """Track B: one choice call for *case*, with the model's own knobs."""
    base = contract.build_choice_request(case)
    request = dataclasses.replace(base, params={"labels": base.params["labels"], **model.knobs})
    content = canonical_json(
        {"system": request.prompt, "user": request.case_text, "labels": request.params["labels"]}
    )
    spec = CallSpec(
        provider=model.provider.name,
        model=model.ref.model,
        subject_role=SUBJECT_ROLE,
        case_id=case.id,
        target=CHOICE_TARGET,
        prompt_hash=prompt_hash(content),
        params=dict(model.knobs),
    )
    return spec, request


def judge_request_id(key: str) -> str:
    """A judge call's ``CallRequest.case_id``: unique per ledger key (fits every custom id)."""
    return f"{JUDGE_REQUEST_PREFIX}{key[:16]}"


@dataclass
class WorkItem:
    key: str
    request: CallRequest
    group: str


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


def load_state(run_dir: Path) -> dict:
    return _read_json(run_dir / RUN_FILE, {}) or {}


def save_state(run_dir: Path, state: Mapping[str, Any]) -> None:
    _write_json(run_dir / RUN_FILE, state)


def init_state(run_dir: Path, manifest_path: Path, *, run_id: str | None, date: str | None) -> dict:
    """Create ``run.json`` for a fresh run dir (or return the existing one)."""
    if _is_inside_git_worktree(run_dir):
        raise RunError(f"{run_dir} is inside a git worktree; the run dir must be private")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(run_dir)
    if state:
        return state
    digest = _sha256_file(manifest_path)
    state = {
        "schema": 1,
        "run_id": run_id or digest[:12],
        "date": date or datetime.date.today().isoformat(),
        "manifest_sha256": digest,
        "manifest_history": [digest],
        "hosts": {},
        "stops": {},
        "capabilities": {},
    }
    save_state(run_dir, state)
    return state


# ---------------------------------------------------------------------------
# The pass engine
# ---------------------------------------------------------------------------


@dataclass
class StepOutcome:
    status: str
    messages: list[str] = field(default_factory=list)
    money_stopped: set[str] = field(default_factory=set)
    exit_code: int = EXIT_OK


class Runner:
    """One pass over a run dir: resume, send what can be sent, finalize when done."""

    def __init__(
        self,
        run_dir: Path,
        plan: Plan,
        ledger: Ledger,
        state: dict,
        *,
        retry_money: bool = True,
        retry_rejected: bool = False,
        probe: Iterable[str] = (),
        out: Callable[[str], None] = print,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.plan = plan
        self.ledger = ledger
        self.state = state
        self.out = out
        self.env = os.environ if env is None else env
        self.messages: list[str] = []
        self._lock = threading.RLock()
        self._spec_model: dict[tuple[str, str], Model] = {}
        for model in plan.models.values():
            self._spec_model[(model.provider.name, model.ref.model)] = model
            self._spec_model[(model.ref.provider, model.ref.model)] = model
        self.spend: dict[str, float] = {}
        self.answers: dict[str, int] = {}
        self.truncated: dict[str, int] = {}
        self.invalid: dict[str, int] = {}
        self.model_stops: dict[str, dict] = {}
        self.provider_stops: dict[str, dict] = {}
        self.paused: dict[str, str] = {}
        self.allowance: dict[str, int] = {}
        self._judge_cache: tuple | None = None
        self._track_a: dict[tuple[str, str], loop.RoundResult] = {}
        self._restore_stops(retry_money, set(probe), retry_rejected)
        self._scan_ledger()
        for kind in sorted({m.kind for m in plan.models.values()}):
            self._check_budget(kind)
        for label in sorted(plan.models):
            self._check_truncation(plan.models[label])

    # -- bookkeeping -------------------------------------------------------

    def say(self, message: str) -> None:
        if message not in self.messages:
            self.messages.append(message)
            self.out(message)

    def _model_of_spec(self, spec: Mapping[str, Any]) -> Model | None:
        return self._spec_model.get((spec.get("provider"), spec.get("model")))

    def _judge_params(self, model: Model) -> dict[str, Any]:
        params: dict[str, Any] = {"max_output_tokens": self.plan.manifest.judging.max_output_tokens}
        effort = provider_reasoning(model.ref.provider, model.ref.reasoning)
        if effort:
            params["reasoning"] = effort
        return params

    def _current(self, model: Model, spec: Mapping[str, Any]) -> bool:
        """Whether *spec* was made with the model's current settings (stale ones never stop it)."""
        params = spec.get("params") or {}
        target = str(spec.get("target", ""))
        if spec.get("subject_role") == "judge":
            judging = self.plan.manifest.judging
            want = self._judge_params(model)
            return (
                params.get("rubric") == judging.rubric
                and params.get("max_output_tokens") == want["max_output_tokens"]
                and params.get("reasoning") == want.get("reasoning")
            )
        if target.startswith(loop.TARGET_PREFIX):
            return params == {**model.knobs, "tool_choice": loop.LOOP_TOOL_CHOICE}
        if target == CHOICE_TARGET:
            return params == model.knobs
        return False

    def _account(self, model: Model, key: str, spec: Mapping[str, Any], state: str) -> None:
        """Add one answered call to the model's spend and answer counts."""
        cached = self.ledger.cached(key)
        budget = self.plan.manifest.budget_for(model.kind)
        discount = budget.batch_discount if budget else 0.0
        if cached is not None:
            self.spend[model.kind] = self.spend.get(model.kind, 0.0) + model.cost(
                cached.usage, discount
            )
        if not self._current(model, spec):
            return
        label = model.label
        self.answers[label] = self.answers.get(label, 0) + 1
        cut = self.ledger.entry(key).reason == loop.TRUNCATED
        if cached is not None and not cut:
            try:
                cut = model.provider.reply_text(cached.raw).truncated
            except Exception:  # noqa: BLE001 -- an unreadable reply is not a cut one
                cut = False
        if cut:
            self.truncated[label] = self.truncated.get(label, 0) + 1
        if state == INVALID:
            self.invalid[label] = self.invalid.get(label, 0) + 1

    def _scan_ledger(self) -> None:
        for entry in self.ledger.entries():
            if entry.state not in (DONE, INVALID):
                continue
            model = self._model_of_spec(entry.spec)
            if model is not None:
                self._account(model, entry.key, entry.spec, entry.state)

    def _restore_stops(self, retry_money: bool, probe: set[str], retry_rejected: bool) -> None:
        """Carry stops over from the last pass.

        A rejected model stays stopped while its request parameters are
        unchanged, unless the operator asks to retry it (e.g. after fixing a
        key behind a 401). A money stop is retried by ``continue`` and
        probed by ``drive``; budget-cap and truncation stops are recomputed.
        """
        stops = self.state.get("stops", {})
        for label, stop in stops.get("models", {}).items():
            model = self.plan.models.get(label)
            if model is None or stop.get("kind") != "rejected" or retry_rejected:
                continue
            if stop.get("params") == canonical_json(model.knobs):
                self.model_stops[label] = stop
        for kind, stop in stops.get("providers", {}).items():
            if stop.get("kind") != "money" or retry_money:
                continue
            if kind in probe:
                self.allowance[kind] = 1
                continue
            self.provider_stops[kind] = stop

    def _persist(self) -> None:
        with self._lock:
            self._persist_locked()

    def _persist_locked(self) -> None:
        stops = {
            "models": {k: v for k, v in sorted(self.model_stops.items())},
            "providers": {k: v for k, v in sorted(self.provider_stops.items())},
        }
        self.state["stops"] = stops
        self.state["models"] = {
            label: {
                "provider": model.kind,
                "host": model.host,
                "route": model.route,
                "spec_keys": sorted([list(k) for k, m in self._spec_model.items() if m is model]),
                "usd_per_mtok_in": model.ref.usd_per_mtok_in,
                "usd_per_mtok_out": model.ref.usd_per_mtok_out,
                "answers": self.answers.get(label, 0),
                "truncated": self.truncated.get(label, 0),
                "invalid": self.invalid.get(label, 0),
            }
            for label, model in sorted(self.plan.models.items())
        }
        self.state["budgets"] = {
            b.provider: {"usd_cap": b.usd_cap, "batch_discount": b.batch_discount}
            for b in self.plan.manifest.budgets
        }
        self.state["spend"] = {k: round(v, 6) for k, v in sorted(self.spend.items())}
        save_state(self.run_dir, self.state)

    def _note_host(self, model: Model) -> None:
        """Record *model*'s host as having received case text, before anything is sent."""
        with self._lock:
            hosts = self.state.setdefault("hosts", {})
            names = hosts.setdefault(model.host, [])
            if model.label not in names:
                names.append(model.label)
                names.sort()
                save_state(self.run_dir, self.state)

    def _remaining(self, keys_of: Callable[[Mapping[str, Any]], bool]) -> int:
        count = 0
        for entry in self.ledger.entries():
            if entry.state in (PENDING, SUBMITTED) and keys_of(entry.spec):
                count += 1
        return count

    def _model_remaining(self, model: Model) -> int:
        return self._remaining(lambda spec: self._model_of_spec(spec) is model)

    def _provider_remaining(self, kind: str) -> int:
        return self._remaining(
            lambda spec: (m := self._model_of_spec(spec)) is not None and m.kind == kind
        )

    # -- stops -------------------------------------------------------------

    def blocked(self, model: Model) -> bool:
        with self._lock:
            return (
                model.label in self.model_stops
                or model.kind in self.provider_stops
                or model.kind in self.paused
                or self.allowance.get(model.kind, 1) <= 0
            )

    def _take(self, model: Model, count: int = 1) -> int:
        """Up to *count* calls this provider may still send in this pass."""
        with self._lock:
            if model.kind not in self.allowance:
                return count
            granted = min(count, self.allowance[model.kind])
            self.allowance[model.kind] -= granted
            return granted

    def apply_stop(self, model: Model, classification: Classification) -> None:
        if classification.rejected or not classification.retryable:
            message = stop_message(model.label, classification, self._model_remaining(model))
            with self._lock:
                self.model_stops[model.label] = {
                    "kind": "rejected",
                    "reason": classification.reason,
                    "message": message,
                    "params": canonical_json(model.knobs),
                }
            capabilities = self.state.setdefault("capabilities", {})
            entry = {"request_rejected": classification.reason, "params": model.knobs}
            if entry not in capabilities.setdefault(model.label, []):
                capabilities[model.label].append(entry)
        elif classification.reason in MONEY_REASONS:
            message = stop_message(model.kind, classification, self._provider_remaining(model.kind))
            with self._lock:
                self.provider_stops[model.kind] = {
                    "kind": "money",
                    "reason": classification.reason,
                    "message": message,
                }
        else:
            message = stop_message(model.kind, classification, self._provider_remaining(model.kind))
            with self._lock:
                self.paused[model.kind] = message
        self.say(message)
        self._persist()

    def _check_budget(self, kind: str) -> None:
        budget = self.plan.manifest.budget_for(kind)
        if budget is None:
            return
        spent = self.spend.get(kind, 0.0)
        if spent >= budget.usd_cap and not (budget.usd_cap == 0 and spent == 0):
            message = (
                f"{kind}: stopping cleanly (budget_cap_reached: ${spent:.4f} of "
                f"${budget.usd_cap:.2f}); {self._provider_remaining(kind)} call(s) left "
                "pending; raise [budget." + kind + "] usd_cap, then continue"
            )
            with self._lock:
                self.provider_stops[kind] = {
                    "kind": "budget_cap",
                    "reason": "budget_cap_reached",
                    "message": message,
                }
            self.say(message)

    def _check_truncation(self, model: Model) -> None:
        rules = self.plan.manifest.stops
        answers = self.answers.get(model.label, 0)
        cut = self.truncated.get(model.label, 0)
        if answers < rules.min_answers or cut / answers < rules.max_truncated_share:
            return
        if self.model_stops.get(model.label, {}).get("kind") == "truncation":
            return
        message = (
            f"{model.label}: truncation stop ({cut} of {answers} answers cut at "
            f"max_output_tokens={model.knobs['max_output_tokens']}, "
            f"{self.invalid.get(model.label, 0)} invalid); "
            f"{self._model_remaining(model)} call(s) left pending; {TRUNCATION_DECISION}"
        )
        with self._lock:
            self.model_stops[model.label] = {
                "kind": "truncation",
                "reason": "truncated",
                "message": message,
                "truncated": cut,
                "answers": answers,
                "invalid": self.invalid.get(model.label, 0),
            }
        self.say(message)

    # -- recording ---------------------------------------------------------

    def record(self, model: Model, key: str, result: CallResult) -> None:
        """One answer into the ledger (main thread only), then spend and stop checks."""
        entry = self.ledger.entry(key)
        if entry.state not in (PENDING, SUBMITTED):
            return
        if result.outcome is Outcome.PENDING:
            self.apply_stop(model, _classification_of(result))
            return
        cached = loop.cached_response(result)
        if result.outcome is Outcome.OK:
            if entry.spec.get("subject_role") == "judge" and _cut(model, result.raw):
                self.ledger.mark_invalid(key, loop.TRUNCATED, cached)
            else:
                self.ledger.record_done(key, cached)
        else:
            self.ledger.mark_invalid(key, result.reason or "invalid", cached)
        self._account(model, key, entry.spec, self.ledger.entry(key).state)
        self._check_budget(model.kind)
        self._check_truncation(model)

    # -- batches -----------------------------------------------------------

    def resolve_orphans(self) -> None:
        """Submissions that may or may not have reached the provider: find them first."""
        for token, keys in sorted(self.ledger.continue_plan().orphans.items()):
            model = self._model_of_spec(self.ledger.entry(keys[0]).spec)
            if model is None:
                raise StopAndAsk(
                    f"submit ref {token} ({len(keys)} key(s)) belongs to no model in the "
                    "manifest; restore that model or ask before resending: " + ", ".join(keys)
                )
            try:
                handle = model.provider.find_batch(token)
            except BatchLookupUnresolved as exc:
                raise StopAndAsk(
                    f"{model.label}: cannot tell whether batch submission {token} reached the "
                    f"provider ({exc}); it is NOT resubmitted. Check the provider's batch list, "
                    f"then ask the operator. Keys: {', '.join(keys)}"
                ) from exc
            if handle is None:
                self.ledger.abandon_submit(keys)
            else:
                self.ledger.mark_submitted(keys, handle.batch_id)

    def poll_batches(self) -> bool:
        progress = False
        for batch_id, keys in sorted(self.ledger.submitted_batches().items()):
            model = self._model_of_spec(self.ledger.entry(keys[0]).spec)
            if model is None or model.kind in self.paused:
                continue
            handle = BatchHandle(batch_id=batch_id, provider=model.provider.name)
            try:
                status = model.provider.poll_batch(handle)
                if not status.complete:
                    continue
                if status.expired:
                    self.ledger.requeue_batch(batch_id, "expired")
                    progress = True
                    continue
                results = model.provider.fetch_batch(handle)
            except (KeyboardInterrupt, BatchLookupUnresolved):
                raise
            except Exception as exc:  # noqa: BLE001 -- classified or re-raised
                classification = classify_exception(exc, model.kind)
                if classification is None:
                    raise
                self.apply_stop(model, classification)
                continue
            by_request = {self._request_id(key): key for key in keys}
            for result in results:
                key = by_request.get(result.case_id)
                if key is not None:
                    self.record(model, key, result)
            self.ledger.requeue_batch(batch_id, "unanswered")
            progress = True
        return progress

    def live_batches(self) -> dict[str, list[str]]:
        """Submitted batches of models in this manifest (a removed model's are reported)."""
        live = {}
        for batch_id, keys in self.ledger.submitted_batches().items():
            if self._model_of_spec(self.ledger.entry(keys[0]).spec) is None:
                self.say(
                    f"batch {batch_id} ({len(keys)} call(s)) belongs to a model no longer in "
                    "the manifest; it is left alone"
                )
                continue
            live[batch_id] = keys
        return live

    def _request_id(self, key: str) -> str:
        spec = self.ledger.entry(key).spec
        if spec.get("subject_role") == "judge":
            return judge_request_id(key)
        return str(spec["case_id"])

    def submit_batches(self, model: Model, items: list[WorkItem]) -> bool:
        progress = False
        groups: dict[str, list[WorkItem]] = {}
        for item in items:
            groups.setdefault(item.group, []).append(item)
        for group in sorted(groups):
            if self.blocked(model):
                break
            chunk = groups[group][: self._take(model, len(groups[group]))]
            if not chunk:
                break
            keys = [item.key for item in chunk]
            submit_ref = self.ledger.begin_submit(keys)
            self._note_host(model)
            try:
                handle = model.provider.submit_batch([item.request for item in chunk], submit_ref)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 -- left as an orphan, looked up next pass
                classification = classify_exception(exc, model.kind)
                if classification is None:
                    raise
                self.apply_stop(model, classification)
                continue
            self.ledger.mark_submitted(keys, handle.batch_id)
            progress = True
        return progress

    # -- sync --------------------------------------------------------------

    def run_sync(self, work: list[tuple[Model, WorkItem]]) -> bool:
        """Send *work* synchronously, each provider under its concurrency cap.

        A worker holds its provider's semaphore through the send, the ledger
        write and the stop checks, so with a cap of N at most N calls are in
        flight when a stop lands, and none starts after it. Ledger writes are
        serialized under the runner's lock.
        """
        if not work:
            return False
        semaphores: dict[str, threading.Semaphore] = {}
        workers = 0
        for model, _item in work:
            if model.kind not in semaphores:
                budget = self.plan.manifest.budget_for(model.kind)
                cap = budget.concurrency_cap if budget else 1
                semaphores[model.kind] = threading.Semaphore(cap)
                workers += cap

        def task(model: Model, item: WorkItem) -> bool:
            with semaphores[model.kind]:
                if self.blocked(model) or not self._take(model):
                    return False
                self._note_host(model)
                try:
                    result = model.provider.submit_sync(item.request)
                except (KeyboardInterrupt, BatchLookupUnresolved):
                    raise
                except Exception as exc:  # noqa: BLE001 -- classified or re-raised
                    classification = classify_exception(exc, model.kind)
                    if classification is None:
                        raise
                    with self._lock:
                        self.apply_stop(model, classification)
                    return False
                with self._lock:
                    self.record(model, item.key, result)
                return True

        progress = False
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers))
        futures = [pool.submit(task, model, item) for model, item in work]
        failure: BaseException | None = None
        try:
            for future in concurrent.futures.as_completed(futures):
                if future.cancelled():
                    continue
                try:
                    progress |= future.result()
                except BaseException as exc:  # noqa: BLE001 -- re-raised after the drain
                    if failure is None:
                        failure = exc
                        for other in futures:
                            other.cancel()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if failure is not None:
            raise failure
        return progress

    # -- work --------------------------------------------------------------

    def _fresh(self, key: str) -> bool:
        entry = self.ledger.entry(key)
        return entry.state == PENDING and not entry.submit_token

    def track_a(self, model: Model, cs: CaseSet) -> loop.RoundResult:
        return loop.run_round(
            self.plan.cases[cs.name],
            provider=model.provider,
            model=model.ref.model,
            ledger=self.ledger,
            snapshot=self.plan.snapshot,
            platform=self.plan.platform,
            params=model.knobs,
        )

    def _choice_calls(self, model: Model, cs: CaseSet) -> list[tuple[str, CallSpec, CallRequest]]:
        calls = [choice_call(model, case) for case in self.plan.cases[cs.name]]
        keys = self.ledger.register_many([spec for spec, _request in calls])
        return [(key, spec, request) for key, (spec, request) in zip(keys, calls)]

    def subject_work(self) -> tuple[dict[str, list[WorkItem]], bool]:
        """Every subject call that can go out now, per model; and whether all are final."""
        work: dict[str, list[WorkItem]] = {}
        final = True
        for subject in self.plan.references:
            model = self.plan.models[subject.model]
            cs = subject.case_set
            if subject.track == "A":
                round_result = self.track_a(model, cs)
                self._track_a[(model.label, cs.name)] = round_result
                if round_result.pending:
                    final = False
                for call in round_result.pending:
                    if self._fresh(call.key):
                        work.setdefault(model.label, []).append(
                            WorkItem(call.key, call.request, f"A:{cs.name}:r{call.round}")
                        )
            else:
                for key, _spec, request in self._choice_calls(model, cs):
                    state = self.ledger.entry(key).state
                    if state not in (DONE, INVALID):
                        final = False
                    if self._fresh(key):
                        work.setdefault(model.label, []).append(
                            WorkItem(key, request, f"B:{cs.name}")
                        )
        return work, final

    # -- records per subject -------------------------------------------------

    def choice_record(self, model: Model, case: Case, key: str, request: CallRequest) -> RawRecord:
        """A Track B answer as the scorer's own prediction mapping reads it."""
        cached = self.ledger.cached(key)
        entry = self.ledger.entry(key)

        def invalid(reason: str, result: CallResult | None = None) -> RawRecord:
            return RawRecord.from_provider_answer(
                provider=model.provider.name,
                model=model.ref.model,
                returned_model=result.returned_model if result else None,
                interface="choice",
                outcome="invalid",
                candidates=result.candidates if result else None,
                invalid_reason=reason,
            )

        if cached is None:
            return invalid(entry.reason or "invalid")
        result = model.provider.result_from_raw(request, cached.raw)
        spoken = model.provider.reply_text(cached.raw)
        labels = request.params["labels"]
        classification, name = contract.parse_choice(result.answer, labels)
        if result.outcome is not Outcome.OK or name is None:
            reason = result.reason if result.outcome is not Outcome.OK else classification.reason
            if spoken.truncated and reason in ("malformed", "empty_answer"):
                reason = loop.TRUNCATED
            return invalid(reason, result)
        base = {
            "provider": model.provider.name,
            "model": model.ref.model,
            "returned_model": result.returned_model,
            "interface": "choice",
            "candidates": result.candidates,
        }
        if name == loop.lfm.EXPLAIN_TOOL:
            return RawRecord.from_provider_answer(outcome="explain", **base)
        if name == loop.lfm.ESCALATE_TOOL:
            return RawRecord.from_provider_answer(outcome="escalate", **base)
        operation = ops_table.get(name)
        grounded = (
            contract.scorer.ground_arguments(
                operation,
                contract._request_text(case),
                loop.measure.snapshot_runner(self.plan.snapshot),
            )
            if operation is not None
            else "unknown operation"
        )
        if isinstance(grounded, str):
            return RawRecord.from_provider_answer(
                outcome="invalid", invalid_reason="not_grounded", **base
            )
        return RawRecord.from_provider_answer(
            outcome="propose", operation=name, arguments=dict(grounded), **base
        )

    def subject_answers(self) -> dict[str, tuple[list[Trace], dict[str, str]]]:
        """Every subject's traces (case order) and explain texts; call only when final."""
        out: dict[str, tuple[list[Trace], dict[str, str]]] = {}
        for subject in self.plan.subjects:
            if subject.kind != "reference":
                saved = self.plan.saved[subject.name]
                out[subject.name] = (saved.traces, saved.explanations)
                continue
            model = self.plan.models[subject.model]
            cs = subject.case_set
            cases = self.plan.cases[cs.name]
            traces: list[Trace] = []
            explanations: dict[str, str] = {}
            if subject.track == "A":
                round_result = self._track_a.get((model.label, cs.name)) or self.track_a(model, cs)
                for case in cases:
                    traces.append(self._trace(subject, case, round_result.finished[case.id]))
                explanations = dict(round_result.explanations)
            else:
                for (key, _spec, request), case in zip(self._choice_calls(model, cs), cases):
                    record = self.choice_record(model, case, key, request)
                    traces.append(self._trace(subject, case, record))
            out[subject.name] = (traces, explanations)
        return out

    def _trace(self, subject: Subject, case: Case, record: RawRecord) -> Trace:
        return Trace(
            case_id=case.id,
            split=case.split,
            raw=record,
            ground_truth=dict(case.expect),
            subject=subject.name,
        )

    # -- judges --------------------------------------------------------------

    def judge_plan(self, answers: Mapping[str, tuple[list[Trace], dict[str, str]]]):
        from . import judge as judge_mod

        if self._judge_cache is not None:
            return self._judge_cache
        case_of = {case.id: case for cases in self.plan.cases.values() for case in cases}
        explain_answers = []
        for subject in self.plan.subjects:
            traces, explanations = answers[subject.name]
            model = self.plan.models.get(subject.model) if subject.model else None
            for trace in traces:
                if trace.raw.outcome != "explain":
                    continue
                case = case_of.get(trace.case_id)
                request_text = (
                    contract._request_text(case) if case is not None and case.text else None
                )
                explain_answers.append(
                    judge_mod.ExplainAnswer(
                        subject=subject.name,
                        policy="raw",
                        case_id=trace.case_id,
                        request_text=request_text,
                        explain_text=explanations.get(trace.case_id),
                        provider=model.ref.provider if model else None,
                        model=model.ref.model if model else None,
                    )
                )
        judging = self.plan.manifest.judging
        extra = [s.name for s in self.plan.subjects]
        extra += [e.name for e in (*self.plan.manifest.candidates, *self.plan.manifest.baselines)]
        plan = judge_mod.plan_panel(
            explain_answers,
            [judge_mod.JudgeId(m.ref.provider, m.ref.model) for m in self.plan.judges],
            seed=judging.seed,
            rubric=judge_mod.load_rubric(judging.rubric),
            extra_blind_terms=extra,
            params={"max_output_tokens": judging.max_output_tokens},
        )
        recorded = []
        for rec in judge_mod.record_prompts(plan, self.run_dir):
            model = self.plan.models[f"{rec.judge.provider}/{rec.judge.model}"]
            params = {**rec.spec.params, **self._judge_params(model)}
            recorded.append(
                dataclasses.replace(rec, spec=dataclasses.replace(rec.spec, params=params))
            )
        self.ledger.register_many(judge_mod.unique_call_specs(recorded))
        split_of = {case.id: case.split for cases in self.plan.cases.values() for case in cases}
        requests: dict[str, tuple[Model, CallRequest]] = {}
        for rec in recorded:
            key = _spec_key(rec.spec)
            if key in requests:
                continue
            model = self.plan.models[f"{rec.judge.provider}/{rec.judge.model}"]
            params = {
                name: value
                for name, value in rec.spec.params.items()
                if name in ("max_output_tokens", "reasoning")
            }
            requests[key] = (
                model,
                CallRequest(
                    case_id=judge_request_id(key),
                    split=split_of.get(rec.spec.case_id, "test"),
                    case_text=rec.prompt,
                    prompt="",
                    interface="text",
                    params=params,
                ),
            )
        self._judge_cache = (plan, recorded, requests)
        return self._judge_cache

    def judge_work(self, answers) -> tuple[dict[str, list[WorkItem]], bool]:
        if not self.plan.judges:
            return {}, True
        _plan, _recorded, requests = self.judge_plan(answers)
        work: dict[str, list[WorkItem]] = {}
        final = True
        for key, (model, request) in requests.items():
            if self.ledger.entry(key).state not in (DONE, INVALID):
                final = False
            if self._fresh(key):
                work.setdefault(model.label, []).append(WorkItem(key, request, "judge"))
        return work, final

    def judge_results(self, answers) -> dict | None:
        from . import judge as judge_mod

        if not self.plan.judges:
            return None
        plan, recorded, _requests = self.judge_plan(answers)
        replies: dict[str, dict[str, str]] = {}
        for rec in recorded:
            key = _spec_key(rec.spec)
            cached = self.ledger.cached(key)
            if cached is None:
                continue
            model = self.plan.models[f"{rec.judge.provider}/{rec.judge.model}"]
            text = ""
            if self.ledger.entry(key).state == DONE:
                text = model.provider.reply_text(cached.raw).text
            replies.setdefault(rec.judge.name, {})[rec.prompt_hash] = text
        outcome = judge_mod.replay_scores(plan, replies, self.run_dir)
        return judge_mod.aggregate(plan, outcome.scores)

    # -- the pass ------------------------------------------------------------

    def dispatch(self, work: Mapping[str, list[WorkItem]]) -> bool:
        progress = False
        sync: list[tuple[Model, WorkItem]] = []
        for label in sorted(work):
            model = self.plan.models[label]
            if self.blocked(model):
                continue
            if model.route == "batch":
                progress |= self.submit_batches(model, work[label])
            else:
                sync.extend((model, item) for item in work[label])
        progress |= self.run_sync(sync)
        return progress

    def run_pass(self) -> StepOutcome:
        self.resolve_orphans()
        answers = None
        while True:
            progress = self.poll_batches()
            work, final = self.subject_work()
            if final:
                answers = answers or self.subject_answers()
                judge_work, judges_final = self.judge_work(answers)
                work = {**work, **judge_work}
                final = judges_final
            if final and not self.live_batches():
                break
            if work:
                progress |= self.dispatch(work)
            if not progress:
                break
        self._persist()
        for kind in sorted(self.allowance):
            if kind not in self.provider_stops:
                self.say(f"{kind}: the probe call went through; its money stop is cleared")
        if final and not self.live_batches():
            if self.plan.smoke:
                self.write_smoke()
            else:
                self.finalize(answers)
            return StepOutcome(STATUS_COMPLETE, list(self.messages), exit_code=EXIT_OK)
        money = {k for k, v in self.provider_stops.items() if v.get("kind") == "money"}
        waiting = self.live_batches()
        if waiting:
            self.say(
                f"waiting on {len(waiting)} submitted batch(es) "
                f"({sum(len(v) for v in waiting.values())} call(s)); continue later"
            )
            return StepOutcome(STATUS_WAITING, list(self.messages), money, EXIT_WAITING)
        if not self.messages:
            self.say("no call could be sent; see status")

        return StepOutcome(STATUS_STOPPED, list(self.messages), money, EXIT_STOPPED)

    # -- outputs -------------------------------------------------------------

    def finalize(self, answers) -> None:
        """Traces, metrics, permutation, judge results, result.json and the page."""
        from . import deepeval_layer, policies, report

        answers = answers or self.subject_answers()
        run_dir = self.run_dir
        loaded_policies = {
            name: policies.load_policy(policies.builtin_policy_path(name))
            for subject in self.plan.subjects
            for name in subject.policies
        }
        subjects_doc = []
        permutation: dict[str, dict] = {}
        for subject in self.plan.subjects:
            traces, _explanations = answers[subject.name]
            final_traces = []
            for trace in traces:
                offered = list(trace.raw.candidates or [])
                for policy in subject.policies:
                    decision, reason, _n, _v = policies.apply(
                        loaded_policies[policy], trace.raw.to_dict(), offered
                    )
                    trace = trace.with_policy(policy, decision, reason or "")
                final_traces.append(trace)
            write_traces(report.traces_path(run_dir, subject.name), final_traces)
            for policy in subject.policies:
                folder = run_dir / DEEPEVAL_DIR / f"{subject.name}__{policy}"
                folder.mkdir(parents=True, exist_ok=True)
                # deepeval prints its own summary; keep stdout for the runner's lines.
                with open(folder / "deepeval.log", "w", encoding="utf-8") as sink:
                    with contextlib.redirect_stdout(sink):
                        outcome = deepeval_layer.evaluate_traces(
                            final_traces, policy, results_folder=folder
                        )
                _write_json(
                    report.metrics_path(run_dir, subject.name, policy), outcome.corpus_metrics
                )
            doc: dict[str, Any] = {
                "name": subject.name,
                "kind": subject.kind,
                "policies": list(subject.policies),
            }
            if subject.kind != "reference":
                doc["artifact"] = self.plan.saved[subject.name].artifact
            subjects_doc.append(doc)
            key = perm.permutation_key(subject.name, "raw")
            probe = (
                subject.entry.permutation_probes.get(subject.case_set.name)
                if subject.entry
                else None
            )
            if subject.track == "A":
                permutation[key] = perm.not_measurable_entry(TRACK_A_PERMUTATION_REASON)
            elif probe:
                loaded_probe = perm.load_probe_json(resolve_private(probe, self.env))
                permutation[key] = perm.probe_to_entry(loaded_probe, split=subject.case_set.split)
        if permutation:
            _write_json(
                run_dir / report.PERMUTATION_FILENAME, perm.build_permutation_file(permutation)
            )
        results = self.judge_results(answers)
        if results is not None:
            from . import judge as judge_mod

            judge_mod.write_judge_results(run_dir, results)
        _write_json(
            run_dir / report.MANIFEST_FILENAME,
            {"run_id": self.state["run_id"], "date": self.state["date"], "subjects": subjects_doc},
        )
        report.generate(
            run_dir, result_path=run_dir / RESULT_FILE, markdown_path=run_dir / PAGE_FILE
        )
        self.state["status"] = STATUS_COMPLETE
        self._persist()
        self.say(f"complete: {run_dir / RESULT_FILE} and {run_dir / PAGE_FILE}")

    def write_smoke(self) -> None:
        """Per-model tokens, cost, truncation and a projected full-run cost (smoke run)."""
        smoke_ids = {case.id for cases in self.plan.cases.values() for case in cases}
        smoke_count = len(smoke_ids)
        scale = self.plan.full_case_count / smoke_count if smoke_count else 0.0
        rows: dict[str, dict[str, Any]] = {}
        for entry in self.ledger.entries():
            model = self._model_of_spec(entry.spec)
            if model is None or entry.spec.get("case_id") not in smoke_ids:
                continue
            if entry.state not in (DONE, INVALID) or not self._current(model, entry.spec):
                continue
            row = rows.setdefault(
                model.label,
                {
                    "provider": model.kind,
                    "calls": 0,
                    "invalid": 0,
                    "truncated": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            cached = self.ledger.cached(entry.key)
            row["calls"] += 1
            row["invalid"] += entry.state == INVALID
            usage = cached.usage if cached else {}
            row["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0))
            row["output_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0))
            row["reasoning_tokens"] += usage.get(
                "reasoning_tokens", usage.get("thinking_tokens", 0)
            )
            budget = self.plan.manifest.budget_for(model.kind)
            row["cost_usd"] += model.cost(usage, budget.batch_discount if budget else 0.0)
            cut = entry.reason == loop.TRUNCATED
            if cached is not None and not cut:
                cut = model.provider.reply_text(cached.raw).truncated
            row["truncated"] += bool(cut)
        for label, row in rows.items():
            model = self.plan.models[label]
            row["cost_usd"] = round(row["cost_usd"], 6)
            row["projected_full_run_usd"] = round(row["cost_usd"] * scale, 4)
            row["max_output_tokens"] = model.knobs["max_output_tokens"]
            row["reasoning"] = model.ref.reasoning
            row["flag"] = "CAPPED" if row["truncated"] else "OK"
            stop = self.model_stops.get(label)
            if stop:
                row["stop"] = stop["kind"]
        providers: dict[str, dict[str, float]] = {}
        for row in rows.values():
            total = providers.setdefault(
                row["provider"], {"cost_usd": 0.0, "projected_full_run_usd": 0.0}
            )
            total["cost_usd"] = round(total["cost_usd"] + row["cost_usd"], 6)
            total["projected_full_run_usd"] = round(
                total["projected_full_run_usd"] + row["projected_full_run_usd"], 4
            )
        smoke = {
            "cases": smoke_count,
            "full_run_cases": self.plan.full_case_count,
            "models": dict(sorted(rows.items())),
            "providers": dict(sorted(providers.items())),
        }
        _write_json(self.run_dir / SMOKE_FILE, smoke)
        self.state["smoke"] = smoke
        self._persist()
        for label, row in sorted(rows.items()):
            self.out(
                f"smoke {label}: {row['flag']} calls={row['calls']} truncated={row['truncated']} "
                f"invalid={row['invalid']} tokens in/out/reasoning={row['input_tokens']}/"
                f"{row['output_tokens']}/{row['reasoning_tokens']} cost=${row['cost_usd']:.4f} "
                f"projected full run=${row['projected_full_run_usd']:.2f}"
            )


def _cut(model: Model, raw: bytes) -> bool:
    try:
        return model.provider.reply_text(raw).truncated
    except Exception:  # noqa: BLE001 -- an unreadable reply is not a cut one
        return False


def _spec_key(spec: CallSpec) -> str:
    from .ledger import ledger_key

    return ledger_key(spec)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> Manifest:
    try:
        return load_manifest(manifest_path)
    except (OSError, ManifestError, ValueError) as exc:
        raise RunError(f"manifest {manifest_path}: {exc}") from exc


def step(
    run_dir: Path,
    manifest_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    factory: ProviderFactory | None = None,
    retry_money: bool = True,
    retry_rejected: bool = False,
    probe: Iterable[str] = (),
    smoke_cases: int | None = None,
    out: Callable[[str], None] = print,
    mark_started: bool = False,
) -> StepOutcome:
    """One pass over *run_dir* (``continue``); the run dir must already be initialized."""
    env = os.environ if env is None else env
    run_dir = Path(run_dir)
    if _is_inside_git_worktree(run_dir):
        raise RunError(f"{run_dir} is inside a git worktree; the run dir must be private")
    state = load_state(run_dir)
    if not state:
        raise RunError(f"{run_dir} holds no run; start one with `run` (or `smoke`)")
    digest = _sha256_file(manifest_path)
    if digest != state.get("manifest_sha256"):
        state["manifest_sha256"] = digest
        state.setdefault("manifest_history", []).append(digest)
        out(f"manifest changed since the last pass (now {digest[:12]})")
    manifest = _load_manifest(manifest_path)
    factory = factory or default_factory
    plan = build_plan(manifest, env, factory, smoke_cases=smoke_cases)
    state["mode"] = "smoke" if smoke_cases is not None else "full"
    if mark_started:
        state["started_full"] = True
        save_state(run_dir, state)
    try:
        ledger = Ledger(run_dir)
    except LedgerLocked as exc:
        raise RunError(f"{run_dir} is in use by another runner (a drive loop?): {exc}") from exc
    except LedgerCorrupt as exc:
        raise RunError(f"{run_dir}: the ledger cannot be trusted: {exc}") from exc
    with ledger:
        runner = Runner(
            run_dir,
            plan,
            ledger,
            state,
            retry_money=retry_money,
            retry_rejected=retry_rejected,
            probe=probe,
            out=out,
            env=env,
        )
        try:
            outcome = runner.run_pass()
        finally:
            runner._persist()
    state["status"] = outcome.status
    state["messages"] = outcome.messages
    save_state(run_dir, state)
    return outcome


def start(
    run_dir: Path,
    manifest_path: Path,
    *,
    run_id: str | None = None,
    date: str | None = None,
    **kwargs: Any,
) -> StepOutcome:
    """``run`` (and ``smoke``): initialize *run_dir* and take the first pass.

    A full run is marked started only once its plan has been built, so a
    configuration error can be fixed and ``run`` repeated.
    """
    state = init_state(Path(run_dir), Path(manifest_path), run_id=run_id, date=date)
    full = not kwargs.get("smoke_cases")
    if full and state.get("started_full"):
        raise RunError(f"{run_dir} already holds a run; use `continue`")
    return step(Path(run_dir), Path(manifest_path), mark_started=full, **kwargs)


# ---------------------------------------------------------------------------
# Status (read-only: never takes the ledger lock)
# ---------------------------------------------------------------------------


def status(run_dir: Path) -> dict:
    """Per-provider and per-model done/submitted/pending/invalid counts and spend so far."""
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    if not state:
        raise RunError(f"{run_dir} holds no run")
    ledger_doc = _read_json(run_dir / "ledger.json", {"entries": {}}) or {"entries": {}}
    models = state.get("models", {})
    by_spec: dict[tuple[str, str], str] = {}
    for label, info in models.items():
        for provider_name, model_id in info.get("spec_keys", []):
            by_spec[(provider_name, model_id)] = label
    budgets = state.get("budgets", {})
    per_model: dict[str, dict[str, Any]] = {}
    for key, rec in sorted(ledger_doc.get("entries", {}).items()):
        spec = rec.get("spec", {})
        label = by_spec.get((spec.get("provider"), spec.get("model")), "(unknown)")
        row = per_model.setdefault(
            label, {"done": 0, "submitted": 0, "pending": 0, "invalid": 0, "spend_usd": 0.0}
        )
        row[rec["state"]] = row.get(rec["state"], 0) + 1
        if rec["state"] in (DONE, INVALID) and label in models:
            cache = _read_json(run_dir / "cache" / f"{key}.json", None)
            usage = (cache or {}).get("usage", {})
            info = models[label]
            tokens_in = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            tokens_out = usage.get("output_tokens", usage.get("completion_tokens", 0))
            usd = (
                tokens_in * info["usd_per_mtok_in"] + tokens_out * info["usd_per_mtok_out"]
            ) / 1e6
            if info.get("route") == "batch":
                usd *= 1 - budgets.get(info["provider"], {}).get("batch_discount", 0.0)
            row["spend_usd"] += usd
    per_provider: dict[str, dict[str, Any]] = {}
    for label, row in per_model.items():
        kind = models.get(label, {}).get("provider", "(unknown)")
        total = per_provider.setdefault(
            kind, {"done": 0, "submitted": 0, "pending": 0, "invalid": 0, "spend_usd": 0.0}
        )
        for name in ("done", "submitted", "pending", "invalid", "spend_usd"):
            total[name] += row[name]
        info = models.get(label, {})
        row["truncated"] = info.get("truncated", 0)
        row["answers"] = info.get("answers", 0)
        row["spend_usd"] = round(row["spend_usd"], 6)
    for kind, total in per_provider.items():
        total["spend_usd"] = round(total["spend_usd"], 6)
        total["usd_cap"] = budgets.get(kind, {}).get("usd_cap")
    return {
        "run_id": state.get("run_id"),
        "date": state.get("date"),
        "mode": state.get("mode"),
        "status": state.get("status", "not started"),
        "providers": dict(sorted(per_provider.items())),
        "models": dict(sorted(per_model.items())),
        "stops": state.get("stops", {}),
        "capabilities": state.get("capabilities", {}),
        "hosts": state.get("hosts", {}),
        "messages": state.get("messages", []),
        "smoke": state.get("smoke"),
    }


def render_status(doc: Mapping[str, Any]) -> list[str]:
    mode = f" [{doc['mode']}]" if doc.get("mode") else ""
    lines = [f"run {doc['run_id']} ({doc['date']}){mode}: {doc['status']}"]
    for kind, row in doc["providers"].items():
        cap = row.get("usd_cap")
        cap_text = f" of ${cap:.2f}" if isinstance(cap, (int, float)) else ""
        lines.append(
            f"provider {kind}: done {row['done']} submitted {row['submitted']} "
            f"pending {row['pending']} invalid {row['invalid']} | "
            f"spend ${row['spend_usd']:.4f}{cap_text}"
        )
    stops = doc.get("stops", {})
    for label, row in doc["models"].items():
        line = (
            f"  model {label}: done {row['done']} submitted {row['submitted']} "
            f"pending {row['pending']} invalid {row['invalid']} "
            f"truncated {row.get('truncated', 0)}/{row.get('answers', 0)} "
            f"spend ${row['spend_usd']:.4f}"
        )
        stop = stops.get("models", {}).get(label)
        if stop:
            line += f" | STOPPED ({stop['kind']}): {stop['message']}"
        lines.append(line)
    for kind, stop in stops.get("providers", {}).items():
        lines.append(f"provider {kind} STOPPED ({stop['kind']}): {stop['message']}")
    hosts = doc.get("hosts", {})
    if hosts:
        shown = ", ".join(f"{host} ({', '.join(names)})" for host, names in sorted(hosts.items()))
        lines.append(f"hosts that received case text: {shown}")
    else:
        lines.append("hosts that received case text: none yet")
    return lines
