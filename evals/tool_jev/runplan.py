"""Plan-building for the gate runner (issue #64, t17): errors, private paths,
providers, subjects, and the calls each subject makes.

Split out of ``run.py``; ``run.py`` re-exports every public name here, so
``evals.tool_jev.run`` stays the one import the CLI and tests use.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import evals.tool_jev  # noqa: F401  (env guard before anything imports deepeval)
from nvsh.tiers import bench as tier_bench

from . import request as contract
from . import track_a_loop as loop
from .cases import Case, TrainingOverlapError, load_case_set, training_overlap
from .ledger import CallSpec, canonical_json, prompt_hash
from .manifest import CaseSet, Manifest, Reference, RunEntry
from .providers import anthropic as anthropic_mod
from .providers import openai as openai_mod
from .providers import openai_compat
from .providers.base import (
    CallRequest,
    CallResult,
    HeldoutSplitRefused,
    MissingProviderKey,
    Provider,
    ProviderCapabilities,
)
from .providers.errors import Classification, Outcome, classify_transport
from .trace import Trace

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

RESULT_FILE = "result.json"
PAGE_FILE = "report.md"
DEEPEVAL_DIR = "deepeval"

SUBJECT_ROLE = loop.DEFAULT_SUBJECT_ROLE
CHOICE_TARGET = "choice"
JUDGE_REQUEST_PREFIX = "j-"

#: Reasons that make a stop a *money* stop (one provider; probed by ``drive``).
MONEY_REASONS = frozenset({"insufficient_credit", "budget_cap_reached"})

#: Consecutive failed (expired / failed / cancelled) batches of one model
#: before its calls stop for an operator decision (codex review P1-1).
MAX_BATCH_FAILURES = 3
#: Longest wait between batch resubmissions after a failed batch.
MAX_BACKOFF_SECONDS = 1800.0

#: Uncertain sync attempts (sent, maybe billed, no answer) per key before
#: the model stops for an operator decision (codex review item 2).
MAX_UNCERTAIN_ATTEMPTS = 2
#: Classification reasons that mean a sync request may have been accepted.
UNCERTAIN_REASONS = frozenset({"timeout", "network_loss", "machine_reset"})

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
        limiter = shared_limiter(ref.provider, budget.requests_per_minute)
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


#: One synchronized limiter per (provider account, rate), shared by every
#: model of that provider and kept for the life of the process, so a drive
#: loop's passes keep one schedule (codex review P2-10).
_LIMITERS: dict[tuple[str, float], openai_compat.RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_limiter(provider: str, requests_per_minute: float) -> openai_compat.RateLimiter:
    with _LIMITERS_LOCK:
        key = (provider, float(requests_per_minute))
        if key not in _LIMITERS:
            _LIMITERS[key] = openai_compat.RateLimiter(requests_per_minute)
        return _LIMITERS[key]


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
    """The stop an adapter exception stands for.

    An adapter's own classification wins. An HTTP error carrying its body is
    read the way the adapter reads a sync error, so ``insufficient_quota`` in
    a 429 body is a money stop, not a rate limit (codex review P2-8). A
    transport failure is network loss. Anything else pauses that provider as
    ``unexpected_error:<type>`` so one provider's surprise never stops the
    others (P2-7) -- except the held-out guard and test assertions, which
    are re-raised (``None``).
    """
    if isinstance(exc, (HeldoutSplitRefused, AssertionError)):
        return None
    found = getattr(exc, "classification", None)
    if isinstance(found, Classification):
        return found
    if isinstance(exc, MissingProviderKey):
        return Classification(Outcome.PENDING, "missing_key", stop=True, retryable=True)
    status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        body = getattr(exc, "body", None)
        if provider == "openai" and isinstance(body, bytes):
            return openai_mod._classify_http(status, body)
        error_type = openai_compat._error_type_from_body(body) if isinstance(body, bytes) else None
        return classify_transport(
            _transport_kind(provider), status_code=status, error_type=error_type
        )
    if isinstance(exc, (OSError, openai_mod.OpenAITransportError, openai_compat.TransportError)):
        return classify_transport(_transport_kind(provider), error_type="network_error")
    return Classification(
        Outcome.PENDING, f"unexpected_error:{type(exc).__name__}", stop=True, retryable=True
    )


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

    def rate(self, tokens_in: int, tokens_out: int, discount: float) -> float:
        usd = (tokens_in * self.ref.usd_per_mtok_in + tokens_out * self.ref.usd_per_mtok_out) / 1e6
        return usd * (1 - discount) if self.route == "batch" else usd

    def cost(self, usage: Mapping[str, int], discount: float) -> float:
        """What an answer cost, from its reported usage at the current prices."""
        return self.rate(tokens_in(usage), tokens_out(usage), discount)

    def estimate(self, request: CallRequest, discount: float) -> float:
        """A call's cost before it is sent: prompt size plus its whole output budget.

        Conservative on purpose (about three characters per token, every
        output token used): the reservation must not undershoot the bill.
        """
        system, messages, tools, labels = contract.canonical_content(request)
        size = len(json.dumps([system, messages, tools, labels], ensure_ascii=False))
        budget = int(request.params.get("max_output_tokens") or contract.DEFAULT_MAX_OUTPUT_TOKENS)
        return self.rate(size // 3 + 1, budget, discount)


def tokens_in(usage: Mapping[str, int]) -> int:
    return int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))


def tokens_out(usage: Mapping[str, int]) -> int:
    return int(usage.get("output_tokens", usage.get("completion_tokens", 0)))


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
    scope: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {"mode": "full"})

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
    scope: Mapping[str, Any] | None = None,
) -> Plan:
    """Everything one pass needs, from the manifest and the private data root.

    *scope* is the run's persisted scope: ``{"mode": "full"}`` or a smoke
    selection ``{"mode": "smoke", "case_set": name, "case_ids": [...]}``.
    A smoke run never sends a case outside its selection, however it is
    resumed (codex review P1-6).
    """
    scope = dict(scope or {"mode": "full"})
    cases: dict[str, tuple[Case, ...]] = {}
    for cs in manifest.case_sets:
        cases[cs.name] = _load_cases(cs, env)
    sendable = list(manifest.sendable_case_sets())
    full_case_count = sum(len(cases[cs.name]) for cs in sendable)
    smoke = scope.get("mode") == "smoke"
    if smoke:
        chosen = [cs for cs in sendable if cs.name == scope.get("case_set")]
        if not chosen:
            raise RunError(f"the smoke case set {scope.get('case_set')!r} is not in the manifest")
        wanted = list(scope.get("case_ids", []))
        by_id = {case.id: case for case in cases[chosen[0].name]}
        missing = [case_id for case_id in wanted if case_id not in by_id]
        if missing:
            raise RunError(f"smoke cases {missing} are no longer in {chosen[0].name!r}")
        cases = {chosen[0].name: tuple(by_id[case_id] for case_id in wanted)}
        sendable = chosen

    subjects: list[Subject] = []
    saved: dict[str, Saved] = {}
    if not smoke:
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
        smoke=smoke,
        full_case_count=full_case_count,
        scope=scope,
    )


def smoke_scope(manifest: Manifest, env: Mapping[str, str], count: int) -> dict[str, Any]:
    """The smoke selection persisted at init: the first *count* cases of the first sendable set."""
    sendable = list(manifest.sendable_case_sets())
    if not sendable:
        raise RunError("smoke needs a sendable (non-held-out) case set")
    first = sendable[0]
    ids = [case.id for case in _load_cases(first, env)[:count]]
    return {"mode": "smoke", "case_set": first.name, "case_ids": ids}


def policy_module_path(name: str) -> Path:
    from . import policies

    return policies.builtin_policy_path(name)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def backoff_delay(base: float, attempts: int, cap: float = MAX_BACKOFF_SECONDS) -> float:
    """``base x 2^attempts``, capped; the exponent is clamped so it never overflows (item 18)."""
    return min(base * 2 ** max(0, min(int(attempts), 30)), cap)


def refused_before_send(exc: BaseException) -> bool:
    """True when the request certainly never reached the provider (refused, unknown host)."""
    import socket
    import urllib.error

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ConnectionRefusedError, socket.gaierror)):
            return True
        reason = getattr(current, "reason", None)
        if isinstance(current, urllib.error.URLError) and isinstance(
            reason, (ConnectionRefusedError, socket.gaierror)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def request_kind(role: str, interface: str) -> tuple[str, str]:
    return ("judge" if role == "judge" or interface == "text" else "reference", interface)


def spec_interface(spec: Mapping[str, Any]) -> str:
    target = str(spec.get("target", ""))
    if spec.get("subject_role") == "judge":
        return "text"
    return "choice" if target == CHOICE_TARGET else "tool_call"


def request_fingerprint(params: Mapping[str, Any]) -> str:
    """The request parameters a rejection is about (codex review P2-11)."""
    kept = {k: params[k] for k in ("max_output_tokens", "reasoning", "tool_choice") if k in params}
    return canonical_json(kept)


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
