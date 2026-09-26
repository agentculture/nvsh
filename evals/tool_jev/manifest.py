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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: Name of the environment variable that points at the operator's private,
#: real manifest file. The file itself is never committed and never lives
#: at a hard-coded path.
ENV_MANIFEST_PATH = "NVSH_EVALS_MANIFEST"

#: Providers this manifest schema knows about. Anything else is rejected at
#: parse time so a typo'd provider name fails loudly instead of silently
#: routing to no adapter.
ALLOWED_PROVIDERS = frozenset({"openai", "anthropic", "openrouter", "nvidia", "local"})

#: The two tool_jev tracks: A = generative (heals a failing command), B =
#: scorer (grades a proposed fix).
ALLOWED_TRACKS = frozenset({"A", "B"})


class ManifestError(ValueError):
    """A manifest file failed validation."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunEntry:
    """One candidate or baseline checkpoint: a name plus its saved predictions."""

    name: str
    track: str
    predictions_path: str


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

    ``include_heldout`` marks a saved-only, held-out set: a runner must
    never send these cases to any provider (they are scored against saved
    predictions only). This module only carries the flag; enforcing it is a
    runner's job.
    """

    name: str
    count: int
    include_heldout: bool = False


@dataclass(frozen=True)
class Budget:
    """Per-provider spend and concurrency cap."""

    provider: str
    usd_cap: float
    concurrency_cap: int


@dataclass(frozen=True)
class RunTarget:
    """A candidate or baseline as it appears in the planned run."""

    name: str
    track: str
    predictions_path: str
    kind: str  # "candidate" | "baseline"


@dataclass(frozen=True)
class Manifest:
    candidates: tuple[RunEntry, ...]
    baselines: tuple[RunEntry, ...]
    references: tuple[Reference, ...]
    judges: tuple[Judge, ...]
    case_sets: tuple[CaseSet, ...]
    budgets: tuple[Budget, ...]

    def plan_run(self) -> tuple[RunTarget, ...]:
        """The candidates and baselines a run must score, in manifest order.

        Purely a projection of ``candidates``/``baselines`` into one
        sequence: a new ``[[candidate]]`` or ``[[baseline]]`` table shows up
        here with no code change anywhere.
        """
        targets = [
            RunTarget(c.name, c.track, c.predictions_path, "candidate") for c in self.candidates
        ]
        targets.extend(
            RunTarget(b.name, b.track, b.predictions_path, "baseline") for b in self.baselines
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
    return RunEntry(name=name, track=track, predictions_path=predictions_path)


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
    return Reference(
        provider=provider,
        model=model,
        reasoning=reasoning,
        batch=batch,
        capabilities=tuple(raw_capabilities),
        api_key_env=api_key_env,
        quantization=quantization,
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
    include_heldout = table.get("include_heldout", False)
    if not isinstance(include_heldout, bool):
        raise ManifestError(f"{where} (name={name!r}): include_heldout must be a boolean")
    return CaseSet(name=name, count=count, include_heldout=include_heldout)


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
        budgets.append(
            Budget(
                provider=provider,
                usd_cap=float(usd_cap),
                concurrency_cap=concurrency_cap,
            )
        )
    return tuple(budgets)


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
