"""Run manifest: candidates, baselines, references, judges, case sets.

The manifest is the single, data-driven description of one release-gate
run: which checkpoints (candidates and baselines) to score, which hosted
reference models to compare them against, which of those references sit on
the judge panel, which case sets to run, and the per-provider budget caps.
Every list is an array-of-tables, so adding a checkpoint, a reference model
or a case set is a data change (one ``[[candidate]]``/``[[reference]]``/
``[[case_set]]`` table) — never a code change here or in a runner that
consumes :func:`load_manifest`.

Two manifests exist:

- ``evals/tool_jev/manifest.example.toml`` — committed, no private paths,
  no non-localhost URLs (``scripts/scan-secrets.py`` and this module's own
  tests both check it).
- the operator's real manifest, which lives outside this repo (in the
  operator's private lfm-train work tree) and is never read from a
  hard-coded path. Its location comes from the ``NVSH_EVALS_MANIFEST``
  environment variable (:data:`ENV_MANIFEST_PATH`), read via
  :func:`resolve_manifest_path` / :func:`load_manifest_from_env`.

Only environment variable *names* are ever stored here (``api_key_env``),
never secret values, and provider default base URLs belong in the provider
adapters (``evals/tool_jev/providers/``), not in the manifest — this module
does not even define a base-URL field.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from evals.tool_jev.cases import SPLIT_TAGS

#: Name of the environment variable that points at the operator's private,
#: real manifest file. The file itself is never committed and never lives
#: at a hard-coded path.
ENV_MANIFEST_PATH = "NVSH_EVALS_MANIFEST"

#: Split tags whose case sets must never be sent to a provider (``cases.py``
#: enforces the same tags -- ``load_case_set`` refuses to load either
#: without ``include_heldout=True``). Every ``[[case_set]]`` here must set
#: ``include_heldout`` to exactly this membership test.
_HELDOUT_SPLITS = frozenset({"heldout", "heldout-mc"})

#: Providers this manifest schema knows about. Anything else is rejected at
#: parse time so a typo'd provider name fails loudly instead of silently
#: routing to no adapter.
ALLOWED_PROVIDERS = frozenset({"openai", "anthropic", "openrouter", "nvidia", "local"})

#: The two tool_jev tracks: A = generative (heals a failing command), B =
#: scorer (grades a proposed fix).
ALLOWED_TRACKS = frozenset({"A", "B"})

#: Reasoning levels a ``[[reference]]`` may ask for (c44's default is
#: ``"medium"``). ``"none"`` is mapped per provider by the runner
#: (``evals/tool_jev/run.py``'s ``provider_reasoning``): some providers take
#: a minimal value, others get the parameter omitted.
REASONING_LEVELS = ("none", "low", "medium", "high")

#: Batch-API price discount applied when a provider routes to its batch API.
DEFAULT_BATCH_DISCOUNT = 0.5


class ManifestError(ValueError):
    """A manifest file failed validation."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunEntry:
    """One candidate or baseline checkpoint: a name plus its saved predictions.

    ``train_split`` is the checkpoint's own training-side split file path
    (same private-path convention as ``predictions_path``: an operator
    string, possibly carrying a ``${...}`` placeholder this module never
    expands), passed to :func:`evals.tool_jev.cases.training_overlap` so a
    runner can refuse to score a checkpoint on any case id present in its
    own training data. ``None`` when no such check applies (e.g. a hosted
    reference model with no local training split).
    """

    name: str
    track: str
    predictions_path: str
    train_split: str | None = None
    #: case set name -> that case set's saved predictions file (overrides
    #: ``predictions_path`` for that case set only).
    predictions: Mapping[str, str] = field(default_factory=dict)
    #: Harness policies scored for this checkpoint; ``"raw"`` is model-only.
    policies: tuple[str, ...] = ("raw",)
    #: The exact artifact (h1): repo id and revision, when the operator has them.
    repo_id: str | None = None
    revision: str | None = None
    #: case set name -> a saved ``permutation_probe.py`` report for that set.
    permutation_probes: Mapping[str, str] = field(default_factory=dict)

    def predictions_for(self, case_set: str) -> str:
        """The saved predictions file for *case_set* (the override, else the default)."""
        return self.predictions.get(case_set, self.predictions_path)


@dataclass(frozen=True)
class Reference:
    """One hosted reference model in the roster.

    ``(provider, model)`` is the roster's uniqueness key — the same model id
    may legitimately appear under two providers (e.g. ``kimi-k3`` via both
    ``openrouter`` and ``nvidia``), so uniqueness is on the pair, not on
    ``model`` alone.
    """

    provider: str
    model: str
    reasoning: str = "medium"
    batch: bool = False
    capabilities: tuple[str, ...] = ()
    api_key_env: str | None = None
    quantization: str | None = None
    #: Output budget per call; ``None`` = the request contract's default.
    max_output_tokens: int | None = None
    #: Price per million input / output tokens, for spend tracking (c26).
    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.model)


@dataclass(frozen=True)
class Judge:
    """One member of the judge panel. Must be a roster member (same key)."""

    provider: str
    model: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.model)


@dataclass(frozen=True)
class CaseSet:
    """One named slice of cases to run.

    ``split`` is one of :data:`evals.tool_jev.cases.SPLIT_TAGS` (``"test"``,
    ``"test-mc"``, ``"heldout"``, ``"heldout-mc"``) -- the split-file tag
    :func:`evals.tool_jev.cases.load_case_set` loads this case set under.
    ``path`` is that split file's location, RELATIVE to a private data root
    the caller supplies out of band (the ``NVSH_EVALS_PRIVATE_ROOT``
    environment variable; see
    :func:`evals.tool_jev.cases.case_sets_from_manifest`) -- never an
    absolute or home-shorthand path, so the committed example manifest
    stays free of private paths.

    ``include_heldout`` marks a saved-only, held-out set: a runner must
    never send these cases to any provider (they are scored against saved
    predictions only). It is required to be ``True`` for ``split in
    {"heldout", "heldout-mc"}`` and ``False`` otherwise -- parsing rejects
    any other combination -- so the flag can never silently drift from the
    split tag it must agree with.
    """

    name: str
    count: int
    split: str
    path: str
    include_heldout: bool = False


@dataclass(frozen=True)
class Budget:
    """Per-provider spend and concurrency cap."""

    provider: str
    usd_cap: float
    concurrency_cap: int
    #: Fraction taken off list prices when this provider routes to its batch API.
    batch_discount: float = DEFAULT_BATCH_DISCOUNT
    #: Client-side request spacing for sync calls (e.g. a free tier's RPM).
    requests_per_minute: float | None = None


@dataclass(frozen=True)
class TrackAConfig:
    """Grounding for references' Track A loop (deviation d1).

    ``snapshot`` is the recorded ground snapshot (``measure.py snapshot``
    output), RELATIVE to the private data root like a case set's path.
    ``platform``/``device_cli`` rebuild the tier platform the candidates
    ran with (``nvsh.tiers.bench.world_platform``'s two fields).
    """

    snapshot: str | None = None
    platform: str = "unknown"
    device_cli: str | None = None


@dataclass(frozen=True)
class StopRules:
    """When a model's own answers stop its calls (plan risk r10).

    Once a model has ``min_answers`` answers and at least
    ``max_truncated_share`` of them were cut at the output budget, the
    runner stops sending that model's calls until the operator decides.
    """

    min_answers: int = 5
    max_truncated_share: float = 0.3


@dataclass(frozen=True)
class JudgingConfig:
    """Judge panel knobs: shuffle seed, output budget per judge call, rubric."""

    seed: int = 64
    max_output_tokens: int = 1024
    rubric: str = "explain-v1"


@dataclass(frozen=True)
class RunTarget:
    """A candidate or baseline as it appears in the planned run."""

    name: str
    track: str
    predictions_path: str
    kind: str  # "candidate" | "baseline"
    train_split: str | None = None
    policies: tuple[str, ...] = ("raw",)


@dataclass(frozen=True)
class Manifest:
    candidates: tuple[RunEntry, ...]
    baselines: tuple[RunEntry, ...]
    references: tuple[Reference, ...]
    judges: tuple[Judge, ...]
    case_sets: tuple[CaseSet, ...]
    budgets: tuple[Budget, ...]
    track_a: TrackAConfig = TrackAConfig()
    stops: StopRules = StopRules()
    judging: JudgingConfig = JudgingConfig()

    def budget_for(self, provider: str) -> Budget | None:
        """The ``[budget.<provider>]`` table, or ``None`` when absent."""
        for budget in self.budgets:
            if budget.provider == provider:
                return budget
        return None

    def plan_run(self) -> tuple[RunTarget, ...]:
        """The candidates and baselines a run must score, in manifest order.

        Purely a projection of ``candidates``/``baselines`` into one
        sequence: a new ``[[candidate]]`` or ``[[baseline]]`` table shows up
        here with no code change anywhere.
        """
        targets = [
            RunTarget(c.name, c.track, c.predictions_path, "candidate", c.train_split, c.policies)
            for c in self.candidates
        ]
        targets.extend(
            RunTarget(b.name, b.track, b.predictions_path, "baseline", b.train_split, b.policies)
            for b in self.baselines
        )
        return tuple(targets)

    def sendable_case_sets(self) -> tuple[CaseSet, ...]:
        """Case sets that may be sent to providers (excludes held-out sets)."""
        return tuple(cs for cs in self.case_sets if not cs.include_heldout)

    def roster_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(ref.key for ref in self.references)


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------


def _require_str(table: Mapping, key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: missing or empty required string field {key!r}")
    return value


def _require_provider(table: Mapping, where: str) -> str:
    provider = _require_str(table, "provider", where)
    if provider not in ALLOWED_PROVIDERS:
        raise ManifestError(
            f"{where}: unknown provider {provider!r} "
            f"(must be one of {sorted(ALLOWED_PROVIDERS)})"
        )
    return provider


def _parse_run_entry(table: Mapping, where: str) -> RunEntry:
    name = _require_str(table, "name", where)
    track = _require_str(table, "track", where)
    if track not in ALLOWED_TRACKS:
        raise ManifestError(
            f"{where} (name={name!r}): unknown track {track!r} "
            f"(must be one of {sorted(ALLOWED_TRACKS)})"
        )
    predictions_path = _require_str(table, "predictions_path", where)
    train_split = table.get("train_split")
    if train_split is not None and not isinstance(train_split, str):
        raise ManifestError(f"{where} (name={name!r}): train_split must be a string")
    where = f"{where} (name={name!r})"
    policies = table.get("policies", ["raw"])
    if (
        not isinstance(policies, list)
        or not policies
        or not all(isinstance(p, str) and p for p in policies)
    ):
        raise ManifestError(f"{where}: policies must be a non-empty list of policy names")
    return RunEntry(
        name=name,
        track=track,
        predictions_path=predictions_path,
        train_split=train_split,
        predictions=_str_table(table, "predictions", where),
        policies=tuple(policies),
        repo_id=_optional_str(table, "repo_id", where),
        revision=_optional_str(table, "revision", where),
        permutation_probes=_str_table(table, "permutation_probes", where),
    )


def _optional_str(table: Mapping, key: str, where: str) -> str | None:
    value = table.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ManifestError(f"{where}: {key} must be a non-empty string")
    return value


def _str_table(table: Mapping, key: str, where: str) -> dict[str, str]:
    value = table.get(key, {})
    if not isinstance(value, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in value.items()
    ):
        raise ManifestError(f"{where}: {key} must be a table of case-set name -> path strings")
    return dict(value)


def _number(table: Mapping, key: str, where: str, default: float) -> float:
    value = table.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ManifestError(f"{where}: {key} must be a non-negative number")
    return float(value)


def _positive_int(table: Mapping, key: str, where: str, default: int | None) -> int | None:
    value = table.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ManifestError(f"{where}: {key} must be a positive integer")
    return value


def _parse_reference(table: Mapping, where: str) -> Reference:
    provider = _require_provider(table, where)
    model = _require_str(table, "model", where)
    reasoning = table.get("reasoning", "medium")
    if not isinstance(reasoning, str) or not reasoning:
        raise ManifestError(f"{where}: reasoning must be a non-empty string")
    batch = table.get("batch", False)
    if not isinstance(batch, bool):
        raise ManifestError(f"{where}: batch must be a boolean")
    raw_capabilities = table.get("capabilities", ())
    if not isinstance(raw_capabilities, (list, tuple)) or not all(
        isinstance(c, str) for c in raw_capabilities
    ):
        raise ManifestError(f"{where}: capabilities must be a list of strings")
    api_key_env = table.get("api_key_env")
    if api_key_env is not None and not isinstance(api_key_env, str):
        raise ManifestError(f"{where}: api_key_env must be a string (an env var NAME, not a value)")
    quantization = table.get("quantization")
    if quantization is not None and not isinstance(quantization, str):
        raise ManifestError(f"{where}: quantization must be a string")
    if reasoning not in REASONING_LEVELS:
        raise ManifestError(f"{where}: reasoning must be one of {list(REASONING_LEVELS)}")
    return Reference(
        provider=provider,
        model=model,
        reasoning=reasoning,
        batch=batch,
        capabilities=tuple(raw_capabilities),
        api_key_env=api_key_env,
        quantization=quantization,
        max_output_tokens=_positive_int(table, "max_output_tokens", where, None),
        usd_per_mtok_in=_number(table, "usd_per_mtok_in", where, 0.0),
        usd_per_mtok_out=_number(table, "usd_per_mtok_out", where, 0.0),
    )


def _parse_judge(table: Mapping, where: str, roster: frozenset[tuple[str, str]]) -> Judge:
    provider = _require_provider(table, where)
    model = _require_str(table, "model", where)
    key = (provider, model)
    if key not in roster:
        raise ManifestError(
            f"{where}: judge {provider}/{model} is not a member of the reference roster "
            "(every judge must also be a [[reference]])"
        )
    return Judge(provider=provider, model=model)


def _parse_case_set(table: Mapping, where: str) -> CaseSet:
    name = _require_str(table, "name", where)
    count = table.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ManifestError(f"{where} (name={name!r}): count must be a non-negative integer")
    split = _require_str(table, "split", where)
    if split not in SPLIT_TAGS:
        raise ManifestError(
            f"{where} (name={name!r}): unknown split {split!r} " f"(must be one of {SPLIT_TAGS})"
        )
    path = _require_str(table, "path", where)
    if path.startswith("/") or "~" in path:
        raise ManifestError(
            f"{where} (name={name!r}): path must be relative to the private data root "
            f"(NVSH_EVALS_PRIVATE_ROOT), never absolute or home-shaped, got {path!r}"
        )
    include_heldout = table.get("include_heldout", False)
    if not isinstance(include_heldout, bool):
        raise ManifestError(f"{where} (name={name!r}): include_heldout must be a boolean")
    expected_heldout = split in _HELDOUT_SPLITS
    if include_heldout != expected_heldout:
        raise ManifestError(
            f"{where} (name={name!r}): include_heldout must be {expected_heldout} "
            f"for split {split!r}"
        )
    return CaseSet(name=name, count=count, split=split, path=path, include_heldout=include_heldout)


def _parse_budgets(raw: Mapping, where: str) -> tuple[Budget, ...]:
    budgets = []
    for provider, table in raw.items():
        entry_where = f"{where}.{provider}"
        if provider not in ALLOWED_PROVIDERS:
            raise ManifestError(
                f"{entry_where}: unknown provider {provider!r} "
                f"(must be one of {sorted(ALLOWED_PROVIDERS)})"
            )
        usd_cap = table.get("usd_cap")
        if not isinstance(usd_cap, (int, float)) or isinstance(usd_cap, bool) or usd_cap < 0:
            raise ManifestError(f"{entry_where}: usd_cap must be a non-negative number")
        concurrency_cap = table.get("concurrency_cap")
        if (
            not isinstance(concurrency_cap, int)
            or isinstance(concurrency_cap, bool)
            or concurrency_cap < 1
        ):
            raise ManifestError(f"{entry_where}: concurrency_cap must be a positive integer")
        batch_discount = _number(table, "batch_discount", entry_where, DEFAULT_BATCH_DISCOUNT)
        if batch_discount > 1:
            raise ManifestError(f"{entry_where}: batch_discount must be between 0 and 1")
        rpm = table.get("requests_per_minute")
        if rpm is not None and (
            not isinstance(rpm, (int, float)) or isinstance(rpm, bool) or rpm <= 0
        ):
            raise ManifestError(f"{entry_where}: requests_per_minute must be a positive number")
        budgets.append(
            Budget(
                provider=provider,
                usd_cap=float(usd_cap),
                concurrency_cap=concurrency_cap,
                batch_discount=batch_discount,
                requests_per_minute=None if rpm is None else float(rpm),
            )
        )
    return tuple(budgets)


def _private_relative(value: object, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where} must be a non-empty string")
    if value.startswith("/") or "~" in value:
        raise ManifestError(
            f"{where} must be relative to the private data root (NVSH_EVALS_PRIVATE_ROOT), "
            f"never absolute or home-shaped, got {value!r}"
        )
    return value


def _parse_track_a(raw: Mapping) -> TrackAConfig:
    if not isinstance(raw, Mapping):
        raise ManifestError("track_a must be a table")
    platform = raw.get("platform", "unknown")
    if not isinstance(platform, str) or not platform:
        raise ManifestError("track_a.platform must be a non-empty string")
    return TrackAConfig(
        snapshot=_private_relative(raw.get("snapshot"), "track_a.snapshot"),
        platform=platform,
        device_cli=_optional_str(raw, "device_cli", "track_a"),
    )


def _parse_stops(raw: Mapping) -> StopRules:
    if not isinstance(raw, Mapping):
        raise ManifestError("stops must be a table")
    min_answers = _positive_int(raw, "min_answers", "stops", StopRules.min_answers)
    share = raw.get("max_truncated_share", StopRules.max_truncated_share)
    if not isinstance(share, (int, float)) or isinstance(share, bool) or not 0 < float(share) <= 1:
        raise ManifestError("stops.max_truncated_share must be a number in (0, 1]")
    return StopRules(min_answers=int(min_answers), max_truncated_share=float(share))


def _parse_judging(raw: Mapping) -> JudgingConfig:
    if not isinstance(raw, Mapping):
        raise ManifestError("judging must be a table")
    seed = raw.get("seed", JudgingConfig.seed)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ManifestError("judging.seed must be an integer")
    rubric = raw.get("rubric", JudgingConfig.rubric)
    if not isinstance(rubric, str) or not rubric:
        raise ManifestError("judging.rubric must be a non-empty string")
    return JudgingConfig(
        seed=seed,
        max_output_tokens=int(
            _positive_int(raw, "max_output_tokens", "judging", JudgingConfig.max_output_tokens)
        ),
        rubric=rubric,
    )


def parse_manifest(data: Mapping) -> Manifest:
    """Validate and build a :class:`Manifest` from a parsed TOML mapping.

    Raises :class:`ManifestError` for: an unknown provider anywhere
    (reference, judge or budget table), a duplicate ``(provider, model)``
    pair in the reference roster, or a judge whose ``(provider, model)`` is
    not in the reference roster.
    """
    candidates = tuple(
        _parse_run_entry(t, f"candidate[{i}]") for i, t in enumerate(data.get("candidate", []))
    )
    baselines = tuple(
        _parse_run_entry(t, f"baseline[{i}]") for i, t in enumerate(data.get("baseline", []))
    )

    references: list[Reference] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, table in enumerate(data.get("reference", [])):
        where = f"reference[{i}]"
        ref = _parse_reference(table, where)
        if ref.key in seen_pairs:
            raise ManifestError(
                f"{where}: duplicate (provider, model) pair {ref.key!r} in reference roster"
            )
        seen_pairs.add(ref.key)
        references.append(ref)

    roster = frozenset(seen_pairs)
    judges = tuple(
        _parse_judge(t, f"judge[{i}]", roster) for i, t in enumerate(data.get("judge", []))
    )

    case_sets = tuple(
        _parse_case_set(t, f"case_set[{i}]") for i, t in enumerate(data.get("case_set", []))
    )

    budgets = _parse_budgets(data.get("budget", {}), "budget")

    return Manifest(
        candidates=candidates,
        baselines=baselines,
        references=tuple(references),
        judges=judges,
        case_sets=case_sets,
        budgets=budgets,
        track_a=_parse_track_a(data.get("track_a", {})),
        stops=_parse_stops(data.get("stops", {})),
        judging=_parse_judging(data.get("judging", {})),
    )


def load_manifest(path: str | os.PathLike) -> Manifest:
    """Load and validate a manifest TOML file at ``path``."""
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return parse_manifest(data)


def resolve_manifest_path(env: Mapping[str, str] | None = None) -> Path:
    """Return the operator's private manifest path from ``NVSH_EVALS_MANIFEST``.

    Raises :class:`ManifestError` if the environment variable is unset or
    empty — this module never falls back to a hard-coded or repo-relative
    path for the real, private manifest.
    """
    env = os.environ if env is None else env
    raw = env.get(ENV_MANIFEST_PATH)
    if not raw:
        raise ManifestError(
            f"{ENV_MANIFEST_PATH} is not set. The private run manifest lives outside this "
            f"repo; point {ENV_MANIFEST_PATH} at the operator's manifest file."
        )
    return Path(raw)


def load_manifest_from_env(env: Mapping[str, str] | None = None) -> Manifest:
    """Load the operator's private manifest via ``NVSH_EVALS_MANIFEST``."""
    return load_manifest(resolve_manifest_path(env))
