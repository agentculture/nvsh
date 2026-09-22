#!/usr/bin/env python3
"""Stock-versus-tuned Tier 2 (LFM2.5) measurement on a named split file.

A development-machine tool for ``docs/lfm-finetune.md`` (part of #39). It is
NEVER imported by the nvsh package -- nothing under nvsh/ may depend on it.
Issue #40 later replaces it with ``nvsh tiers bench --tier2 lfm``.

Usage (one invocation measures every ``--model`` back to back, with identical
settings except the model id)::

    uv run python scripts/lfm-finetune/measure.py --split out/val.json \\
        --model LiquidAI/LFM2.5-350M --revision <commit> \\
        --model <tuned-repo> --revision <commit> --label val-350m

For each model it builds an :class:`~nvsh.tiers.lfm.LfmTier` exactly the way
nvsh's daemon does (:func:`nvsh.tiers.runtime_docker.build_runtime` over the
operator's ``[tiers.lfm]`` settings with only ``model`` overridden, floor-
checked with ``[tiers] memory_floor_mb``), times the runtime's start-up, then
calls :func:`nvsh.tiers.bench.bench` with ``tier1=None`` and
``options.tier2`` set to that tier. Scoring is the bench's own
(``_is_correct``, ``compute_escalation``, ``compute_false_mutating``); this
script only adds the per-source vote (one vote per ``source_id``: the majority
over its variations, a tie counts against the model), the count of explain
outcomes on explain entries, and a separate count of mutating proposals of the
expected operation with wrong arguments (bench counts only a wrong operation
name as a wrong mutating pick; the use-case bar is judged on both rows).

Guards:

* the split's side is read from BOTH its file name and its header (split.py's
  ``Split '<side>' of <corpus>`` note, or held-out.json's own header), so a
  renamed file keeps its side: held-out needs ``--acceptance``, test needs
  ``--final`` (each flag is refused on any other file), and a file whose side
  neither names is refused unless it is the plain dev corpus;
* ``--revision`` is verified, not just recorded: for an engine that downloads
  by repo id into ``[tiers.lfm] hf_cache_dir`` the cache's ``refs/main`` must
  equal it before that model runs (exit 2 otherwise); a mounted model file or
  an attached endpoint is recorded as operator-supplied, not verified;
* in managed mode it refuses to start while a container named
  ``nvsh-tier2-<uid>`` is already running -- it never stops one it did not
  start;
* it refuses to overwrite an existing results file unless ``--force``.

Before each run it records ``docker ps`` and ``nvidia-smi``; the results go to
``docs/benchmarks/<YYYY-MM-DD>-lfm-<label>.md`` (or ``--out``) with the command
line, the split's seed, every model's repo id and revision, and nvsh's commit.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import subprocess  # nosec B404 - fixed argv lists, never a shell
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh import __version__  # noqa: E402
from nvsh import config as nvsh_config  # noqa: E402
from nvsh.ops import table as ops_table  # noqa: E402
from nvsh.platform._model import Platform  # noqa: E402
from nvsh.redact import redact  # noqa: E402
from nvsh.tiers import bench as tier_bench  # noqa: E402
from nvsh.tiers.base import Tier  # noqa: E402
from nvsh.tiers.router import AGENT, TierOutcome  # noqa: E402
from nvsh.tiers.runtime import Runtime, RuntimeUnavailable  # noqa: E402
from nvsh.tiers.runtime_docker import (  # noqa: E402
    DEFAULT_ENGINE,
    HF_CACHE_NAME,
    container_name,
    engine_template,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARKS_DIR = _REPO_ROOT / "docs" / "benchmarks"
_SCRIPT_NAME = "scripts/lfm-finetune/measure.py"

MANAGED = "managed"
FINAL_MARKER = "- Final run: yes"

EXIT_OK = 0
EXIT_USER = 1
EXIT_ENV = 2

_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_SEED_RE = re.compile(r"seed=(\d+)")
_RUN_TIMEOUT = 30.0

WRONG_MUTATING_ROW = "Wrong mutating proposals, per source / per variation"
WRONG_ARGUMENTS_ROW = "Mutating proposals with wrong arguments, per source / per variation"

#: ``[tiers.lfm]`` keys recorded in the results file. ``base_url`` is left
#: out on purpose: no endpoint is ever written into a committed file.
_RECORDED_SETTINGS = (
    "engine",
    "mode",
    "image",
    "gpu",
    "ctx",
    "gpu_memory_fraction",
    "tool_call_parser",
    "model_dir",
    "port",
)

RunFn = Callable[[list[str], float], "tuple[int, str]"]


class MeasureError(Exception):
    """A refusal: printed as one line plus a hint, never a traceback."""

    def __init__(self, code: int, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


# ---------------------------------------------------------------------------
# Seams: everything that touches the machine, injectable for tests
# ---------------------------------------------------------------------------


def default_run(argv: list[str], timeout: float) -> tuple[int, str]:  # pragma: no cover
    """Run *argv* (a fixed list, no shell) and return ``(exit code, output)``."""
    try:
        completed = subprocess.run(  # nosec B603 - fixed argv list, no shell=True
            argv, check=False, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        return (127, f"{argv[0]}: not found")
    except (OSError, subprocess.SubprocessError) as exc:
        return (1, f"{type(exc).__name__}: {exc}")
    return (completed.returncode, (completed.stdout or "") + (completed.stderr or ""))


@dataclass(frozen=True)
class TierSpec:
    """What :func:`build_lfm_tier` needs for one run: identical across runs but ``model``."""

    lfm_settings: Mapping[str, object]
    runtime_platform: Platform
    tier_platform: Platform
    runner: RunFn
    memory_floor_mb: int


def build_lfm_tier(spec: TierSpec) -> tuple[Tier, Runtime]:
    """The Tier 2 nvsh's daemon builds (``nvsh.tiers.manager.TierManager._lfm``).

    Same launcher (:func:`~nvsh.tiers.runtime_docker.build_runtime` over the
    ``[tiers.lfm]`` settings), same floor check, same :class:`LfmTier`; the
    only difference is that grounding and inspection use *spec.runner*
    (the corpus's fixture world unless ``--live``), as ``nvsh tiers bench``
    does for Tier 1. Constructing either object starts nothing.
    """
    from nvsh.tiers.lfm import LfmTier
    from nvsh.tiers.memfloor import check_floor
    from nvsh.tiers.runtime_docker import build_runtime

    floor_mb = int(spec.memory_floor_mb)

    def floor_check():
        return check_floor(floor_mb)

    runtime = build_runtime(spec.lfm_settings, spec.runtime_platform, floor_check=floor_check)
    tier = LfmTier(
        runtime,
        spec.tier_platform,
        model=str(spec.lfm_settings["model"]),
        runner=spec.runner,
        floor_check=floor_check,
    )
    return tier, runtime


def _detect_platform() -> Platform:  # pragma: no cover - reads the real machine
    from nvsh import platform as platform_mod

    return platform_mod.detect()


def _today() -> str:  # pragma: no cover - wall clock
    return datetime.date.today().isoformat()


@dataclass
class Seams:
    run: RunFn = default_run
    build_tier: Callable[[TierSpec], tuple[Tier, Runtime]] = build_lfm_tier
    detect_platform: Callable[[], Platform] = _detect_platform
    today: Callable[[], str] = _today
    clock: Callable[[], float] = time.monotonic
    uid: Callable[[], int] = os.getuid
    load_config: Callable[[Path | None], object] = nvsh_config.load


# ---------------------------------------------------------------------------
# Split file: refusals, seed, source ids
# ---------------------------------------------------------------------------


#: The sides a split file can be. ``held-out`` and ``test`` need a flag.
HELD_OUT = "held-out"
TEST = "test"
DEV = "dev"
_NAMED_SIDES = ("train", "val", TEST)
#: ``split.py`` appends ``Split '<side>' of <corpus> (seed=<n>).`` to the header.
_SPLIT_NOTE_RE = re.compile(r"Split '([A-Za-z0-9_-]+)' of (\S+?) \(seed=")
#: The first words of the committed corpora's own headers.
_HELD_OUT_HEADER_START = "held-out split"
_DEV_HEADER_START = "development split"
_UNREAD = object()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def sides_from_name(path: Path) -> set[str]:
    """What the file name says: ``test``/``val``/``train`` as a word, or held-out."""
    stem = path.stem.lower()
    words = {word for word in re.split(r"[^a-z0-9]+", stem) if word}
    sides = {side for side in _NAMED_SIDES if side in words}
    if _compact(HELD_OUT) in _compact(stem):
        sides.add(HELD_OUT)
    return sides


def _header_text(header: object) -> str:
    if header is None:
        return ""
    if isinstance(header, str):
        return header
    return json.dumps(header, sort_keys=True)


def sides_from_header(header: object) -> set[str]:
    """What the header says: every ``split.py`` note, and the committed corpora's own."""
    text = _header_text(header)
    sides: set[str] = set()
    notes = _SPLIT_NOTE_RE.findall(text)
    for side, corpus in notes:
        sides.add(side.lower())
        if _compact(HELD_OUT) in _compact(corpus):
            sides.add(HELD_OUT)
    start = text.lstrip().lower()
    if start.startswith(_HELD_OUT_HEADER_START):
        sides.add(HELD_OUT)
    if start.startswith(_DEV_HEADER_START) and not notes:
        sides.add(DEV)
    return sides


def _read_header(path: Path) -> object:
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return None
    return raw.get("header") if isinstance(raw, dict) else None


def _is_dev_corpus(path: Path) -> bool:
    try:
        return path.resolve() == tier_bench.dev_corpus_path().resolve()
    except OSError:
        return False


def check_split_allowed(
    path: Path, *, acceptance: bool, final: bool, header: object = _UNREAD
) -> None:
    """Refuse the held-out file without ``--acceptance`` and the test side without ``--final``.

    The side comes from BOTH the file name and its header (read from *path*
    when *header* is not given), so renaming a file does not change what it
    is: either one marking it test needs ``--final``, either one marking it
    held-out needs ``--acceptance``. A file neither one places is refused
    unless it is the plain dev corpus.
    """
    if header is _UNREAD:
        header = _read_header(path)
    name = path.name
    by_name, by_header = sides_from_name(path), sides_from_header(header)
    sides = by_name | by_header
    if _is_dev_corpus(path):
        sides.add(DEV)

    def origin(side: str) -> str:
        return " and ".join(
            where for where, found in (("name", by_name), ("header", by_header)) if side in found
        )

    if not sides:
        raise MeasureError(
            EXIT_USER,
            f"cannot tell which side {name} is: neither its name nor its header "
            "names train, val, test, held-out or the dev corpus",
            "measure a file written by split.py, or nvsh/tiers/corpus/dev.json",
        )
    if HELD_OUT in sides and not acceptance:
        raise MeasureError(
            EXIT_USER,
            f"{name} is the acceptance split (by its {origin(HELD_OUT)}); "
            "refusing to measure it without --acceptance",
            "pass --acceptance only for the one adoption measurement",
        )
    if TEST in sides and not final:
        raise MeasureError(
            EXIT_USER,
            f"{name} is the test side (by its {origin(TEST)}); "
            "it is only measured on final runs (--final)",
            "iterate on val.json; pass --final for a final run",
        )
    if acceptance and HELD_OUT not in sides:
        raise MeasureError(
            EXIT_USER, f"--acceptance applies to the held-out split only, not {name}"
        )
    if final and TEST not in sides:
        raise MeasureError(EXIT_USER, f"--final applies to the test side only, not {name}")


def read_split(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError) as exc:
        raise MeasureError(EXIT_USER, f"cannot read split file {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise MeasureError(EXIT_USER, f"{path} is not a split file ({{header, entries}})")
    return raw


def seed_from_header(header: object) -> int | None:
    """The seed ``split.py`` wrote into the header (``... (seed=39).``), if any."""
    if isinstance(header, dict):
        seed = header.get("seed")
        return seed if isinstance(seed, int) else None
    if isinstance(header, str):
        match = _SEED_RE.search(header)
        return int(match.group(1)) if match else None
    return None


def source_ids(raw_entries: Sequence[object]) -> dict[str, str]:
    """``entry id -> source_id`` (an entry without one is its own source)."""
    mapping: dict[str, str] = {}
    for item in raw_entries:
        if isinstance(item, dict) and "id" in item:
            entry_id = str(item["id"])
            mapping[entry_id] = str(item.get("source_id") or entry_id)
    return mapping


# ---------------------------------------------------------------------------
# Scoring on top of nvsh.tiers.bench's own helpers
# ---------------------------------------------------------------------------


def items_from_result(
    result: Mapping[str, object], entries: Sequence[tier_bench.CorpusEntry]
) -> list[tier_bench.ItemResult]:
    """Rebuild the bench's per-entry outcomes from its ``items`` rows.

    A row carries ``handled_by``/``operation``/``args``; the router sets
    exactly one of ``handled_by`` and ``escalated_to``, so a row with no
    ``handled_by`` escalated, and one handled with no operation was an
    explanation. That is everything ``_is_correct``, ``compute_escalation``
    and ``compute_false_mutating``'s count read.
    """
    by_id = {entry.id: entry for entry in entries}
    items = []
    for row in result.get("items", []):  # type: ignore[union-attr]
        entry = by_id[row["id"]]
        handled_by = row.get("handled_by")
        operation = row.get("operation")
        outcome = TierOutcome(
            handled_by=handled_by,
            escalated_to=None if handled_by else AGENT,
            explanation="" if handled_by and operation is None else None,
            operation=operation,
            args=dict(row.get("args") or {}),
        )
        items.append(tier_bench.ItemResult(entry=entry, outcome=outcome, latency_ms=0.0))
    return items


def detail_rows(model: str, result: Mapping[str, object], entries) -> list[dict]:
    """One row per entry: what was expected, what the tier did, and whether it was right."""
    rows = []
    for item in items_from_result(result, entries):
        outcome = item.outcome
        if outcome.escalated_to is not None:
            did = "escalate"
        elif outcome.operation is None:
            did = "explain"
        else:
            did = "propose"
        rows.append(
            {
                "model": model,
                "id": item.entry.id,
                "expect": item.entry.expect,
                "did": did,
                "operation": outcome.operation,
                "args": dict(outcome.args),
                "correct": tier_bench._is_correct(item),
                "text": item.entry.text,
            }
        )
    return rows


def _expect_kind(entry: tier_bench.CorpusEntry) -> str:
    if entry.expect.get("escalate"):
        return "escalate"
    if entry.expect.get("explain"):
        return "explain"
    return "operation"


def _escalated(item: tier_bench.ItemResult) -> bool:
    return tier_bench.compute_escalation([item])["tp"] == 1


def _false_mutating(item: tier_bench.ItemResult) -> bool:
    return tier_bench.compute_false_mutating([item])["count"] == 1


def _wrong_arguments_mutating(item: tier_bench.ItemResult) -> bool:
    """A mutating operation proposed where that very operation was expected, but wrong.

    ``bench._is_false_mutating`` compares operation names only, so
    ``container_restart(container="inference")`` where ``container="trainer"``
    was expected is not a wrong mutating pick there. Bench's scoring is left as
    it is (honesty h20); this is the separate count for exactly that case --
    decided from the operation table's ``read_only`` flag, never from a name.
    """
    if not _is_mutating_proposal(item):
        return False
    expected = item.entry.expect.get("operation")
    return expected == item.outcome.operation and not tier_bench._is_correct(item)  # type: ignore


def _explained(item: tier_bench.ItemResult) -> bool:
    outcome = item.outcome
    return outcome is not None and outcome.handled_by is not None and outcome.operation is None


def _proposed(item: tier_bench.ItemResult) -> bool:
    return item.outcome is not None and item.outcome.operation is not None


def _is_mutating_proposal(item: tier_bench.ItemResult) -> bool:
    if not _proposed(item):
        return False
    operation = ops_table.get(item.outcome.operation)  # type: ignore[union-attr]
    return operation is not None and not operation.read_only


def _majority(votes: Sequence[bool]) -> bool:
    """Strict majority; a tie is not a majority."""
    return sum(votes) * 2 > len(votes)


def _group(items: Sequence[tier_bench.ItemResult], sources: Mapping[str, str]) -> dict:
    groups: dict[str, list[tier_bench.ItemResult]] = {}
    for item in items:
        groups.setdefault(sources.get(item.entry.id, item.entry.id), []).append(item)
    return groups


def _per_source(groups: Mapping[str, list], good: Callable[[tier_bench.ItemResult], bool]) -> int:
    """Sources whose majority of variations were *good*; a tie counts as not good."""
    return sum(1 for group in groups.values() if _majority([good(item) for item in group]))


def _per_source_bad(
    groups: Mapping[str, list], bad: Callable[[tier_bench.ItemResult], bool]
) -> int:
    """Sources whose variations were NOT mostly free of *bad*; a tie counts as bad."""
    return sum(1 for group in groups.values() if not _majority([not bad(item) for item in group]))


def score(
    result: Mapping[str, object],
    entries: Sequence[tier_bench.CorpusEntry],
    sources: Mapping[str, str],
) -> dict:
    """Per-variation and per-source figures for one bench run."""
    items = items_from_result(result, entries)
    kinds: dict[str, list[tier_bench.ItemResult]] = {"operation": [], "escalate": [], "explain": []}
    for item in items:
        kinds[_expect_kind(item.entry)].append(item)
    op_groups = _group(kinds["operation"], sources)
    esc_groups = _group(kinds["escalate"], sources)
    exp_groups = _group(kinds["explain"], sources)
    all_groups = _group(items, sources)
    escalation = tier_bench.compute_escalation(items)
    return {
        "right": {
            "variation": sum(1 for item in kinds["operation"] if tier_bench._is_correct(item)),
            "variation_total": len(kinds["operation"]),
            "source": _per_source(op_groups, tier_bench._is_correct),
            "source_total": len(op_groups),
        },
        "escalated": {
            "variation": escalation["tp"],
            "variation_total": escalation["tp"] + escalation["fn"],
            "source": _per_source(esc_groups, _escalated),
            "source_total": len(esc_groups),
        },
        "wrong_mutating": {
            "variation": tier_bench.compute_false_mutating(items)["count"],
            "source": _per_source_bad(all_groups, _false_mutating),
        },
        "wrong_arguments_mutating": {
            "variation": sum(1 for item in items if _wrong_arguments_mutating(item)),
            "source": _per_source_bad(all_groups, _wrong_arguments_mutating),
        },
        "explain": {
            "total": len(kinds["explain"]),
            "explained": sum(1 for item in kinds["explain"] if _explained(item)),
            "proposed": sum(1 for item in kinds["explain"] if _proposed(item)),
            "escalated": sum(1 for item in kinds["explain"] if _escalated_any(item)),
            "mutating": sum(1 for item in kinds["explain"] if _is_mutating_proposal(item)),
            "source": _per_source(exp_groups, _explained),
            "source_total": len(exp_groups),
        },
    }


def _escalated_any(item: tier_bench.ItemResult) -> bool:
    return item.outcome is not None and item.outcome.escalated_to is not None


# ---------------------------------------------------------------------------
# Machine state: the container guard and the before-run captures
# ---------------------------------------------------------------------------


def guard_container(run: RunFn, uid: int) -> None:
    """Refuse (never stop) while ``nvsh-tier2-<uid>`` is running."""
    name = container_name(uid)
    code, output = run(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"], _RUN_TIMEOUT
    )
    if code != 0:
        raise MeasureError(
            EXIT_ENV,
            f"cannot check whether {name} is running: docker ps exited {code}",
            "make docker usable here, or measure an attached endpoint ([tiers.lfm] mode)",
        )
    if name in output.split():
        raise MeasureError(
            EXIT_ENV,
            f"container {name} is already running; refusing to start (it was not stopped)",
            f"stop it yourself when nothing needs it: docker stop {name}",
        )


def _capture(run: RunFn, argv: list[str]) -> str:
    code, output = run(argv, _RUN_TIMEOUT)
    text = redact(output.encode("utf-8", "replace")).decode("utf-8", "replace").rstrip()
    return text if code == 0 else f"(exit {code}) {text}".rstrip()


def background_set(run: RunFn, own: str) -> tuple[str, ...] | None:
    """Running container names other than ours, or ``None`` when unknown."""
    code, output = run(["docker", "ps", "--format", "{{.Names}}"], _RUN_TIMEOUT)
    if code != 0:
        return None
    return tuple(sorted(name for name in output.split() if name != own))


def container_memory(run: RunFn, name: str) -> str:
    def run_stats(container: str) -> str | None:
        code, output = run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            _RUN_TIMEOUT,
        )
        return output.strip() if code == 0 else None

    reading = tier_bench.read_docker_stats(name, run_stats=run_stats)
    if reading is None or reading.docker_used_mib is None:
        return "not measured"
    return f"{reading.docker_used_mib / 1024.0:.1f} GiB"


def nvsh_commit(run: RunFn) -> str:
    code, output = run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], _RUN_TIMEOUT)
    return output.strip() if code == 0 and output.strip() else "unknown"


# ---------------------------------------------------------------------------
# --revision: verified against what the engine will serve, or said not to be
# ---------------------------------------------------------------------------

REVISION_VERIFIED = "revision verified from the cache"
REVISION_UNVERIFIED = "operator-supplied, not verified"


def hf_cache_dir(settings: Mapping[str, object]) -> Path:
    """The host HF cache the managed launcher mounts as ``HF_HOME``.

    ``[tiers.lfm] hf_cache_dir``, or the launcher's own default
    (``runtime_docker._with_cache_dir``: the tier cache's ``hf`` directory).
    """
    configured = settings.get("hf_cache_dir")
    if configured is not None:
        return Path(str(configured))
    from nvsh.tiers.fetch import default_cache_dir

    return default_cache_dir() / HF_CACHE_NAME


def cached_revision(cache: Path, repo_id: str) -> tuple[Path, str | None]:
    """``(refs/main path, commit)`` for *repo_id* in *cache*; commit ``None`` if absent.

    The launcher sets ``HF_HOME`` to the mount of *cache*, so the hub cache
    is ``<cache>/hub``; ``<cache>`` itself is tried too, for a cache laid out
    as a bare hub directory. The first ``refs/main`` found is the one read.
    """
    folder = "models--" + repo_id.replace("/", "--")
    candidates = [cache / "hub" / folder / "refs" / "main", cache / folder / "refs" / "main"]
    for ref in candidates:
        try:
            return ref, ref.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return candidates[0], None


def verify_revision(settings: Mapping[str, object], model: str, revision: str) -> str:
    """How *revision* is known to be what the engine serves for *model*.

    An engine that downloads by repo id serves whatever the host cache's
    ``refs/main`` resolves to, so that ref must equal *revision* or the run is
    refused (exit 2). An engine handed a mounted model file, or an endpoint
    nvsh did not launch, gives nothing to check: the revision is recorded as
    operator-supplied.
    """
    if str(settings.get("mode") or MANAGED) != MANAGED:
        return f"{REVISION_UNVERIFIED}: attached endpoint"
    try:
        template = engine_template(str(settings.get("engine") or DEFAULT_ENGINE))
    except RuntimeUnavailable as exc:
        raise MeasureError(EXIT_ENV, f"cannot verify --revision for {model}: {exc}") from exc
    if template.needs_model_mount or not template.downloads_model:
        return REVISION_UNVERIFIED
    ref, found = cached_revision(hf_cache_dir(settings), model)
    if found is None:
        raise MeasureError(
            EXIT_ENV,
            f"cannot verify --revision {revision} for {model}: {ref} does not exist",
            "fetch the pinned revision into [tiers.lfm] hf_cache_dir first, so refs/main "
            "names it; the engine serves whatever refs/main resolves to",
        )
    if found != revision:
        raise MeasureError(
            EXIT_ENV,
            f"{model}: the cache's refs/main is {found}, not --revision {revision}; "
            f"the engine would serve {found}",
            f"re-fetch {model} at {revision} into the cache, or pass the revision it holds",
        )
    return REVISION_VERIFIED


# ---------------------------------------------------------------------------
# One run per model
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    model: str
    revision: str
    revision_status: str = ""
    docker_ps: str = ""
    nvidia_smi: str = ""
    background: tuple[str, ...] | None = None
    startup_s: float | None = None
    failure: str = ""
    result: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    memory: str = "not measured"


@dataclass(frozen=True)
class RunPlan:
    entries: Sequence[tier_bench.CorpusEntry]
    problems: Sequence[str]
    sources: Mapping[str, str]
    split: str
    tiers: Mapping[str, object]
    lfm_settings: Mapping[str, object]
    router_platform: Platform
    runtime_platform: Platform
    runner: RunFn
    grounding: str


def measure_one(plan: RunPlan, model: str, revision: str, seams: Seams) -> RunRecord:
    record = RunRecord(model=model, revision=revision)
    managed = str(plan.lfm_settings.get("mode") or MANAGED) == MANAGED
    uid = int(seams.uid())
    own = container_name(uid)
    if managed:
        guard_container(seams.run, uid)
    settings = {**plan.lfm_settings, "model": model}
    record.revision_status = verify_revision(settings, model, revision)
    record.docker_ps = _capture(seams.run, ["docker", "ps"])
    record.nvidia_smi = _capture(seams.run, ["nvidia-smi"])
    record.background = background_set(seams.run, own)

    spec = TierSpec(
        lfm_settings=settings,
        runtime_platform=plan.runtime_platform,
        tier_platform=plan.router_platform,
        runner=plan.runner,
        memory_floor_mb=int(plan.tiers.get("memory_floor_mb", 1024)),  # type: ignore[arg-type]
    )
    tier, runtime = seams.build_tier(spec)
    try:
        started = seams.clock()
        try:
            runtime.ensure()
        except RuntimeUnavailable as exc:
            record.failure = f"start-up failed: {exc}"
            return record
        record.startup_s = seams.clock() - started
        record.result = tier_bench.bench(
            plan.entries,
            split=plan.split,
            tier1=None,
            platform=plan.router_platform,
            options=tier_bench.BenchOptions(
                tier2=tier,
                clock=seams.clock,
                runner=plan.runner,
                nvsh_version=__version__,
                engine=str(settings.get("engine") or ""),
                mode=str(settings.get("mode") or MANAGED),
                grounding=plan.grounding,
                corpus_problems=plan.problems,
            ),
        )
        record.scores = score(record.result, plan.entries, plan.sources)
        if managed:
            record.memory = container_memory(seams.run, own)
    finally:
        tier.close()
    return record


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _of(count: int, total: int) -> str:
    return f"{count} of {total}"


def _ms(value: object) -> str:
    return f"{value:.0f} ms" if isinstance(value, (int, float)) else "n/a"


def _seconds(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.1f} s"


def latency_comparable(records: Sequence[RunRecord]) -> bool:
    backgrounds = {record.background for record in records}
    return None not in backgrounds and len(backgrounds) == 1


def result_rows(records: Sequence[RunRecord]) -> list[tuple[str, list[str]]]:
    """``(metric, [one cell per model])`` in the baseline file's order."""

    def cells(fn: Callable[[RunRecord], str]) -> list[str]:
        return ["not measured" if record.failure else fn(record) for record in records]

    def s(record: RunRecord, key: str) -> dict:
        return record.scores[key]

    comparable = latency_comparable(records) or len(records) == 1
    mark = "" if comparable else " (not comparable)"
    return [
        (
            "Model revision",
            [f"`{record.revision}` ({record.revision_status})" for record in records],
        ),
        (
            "Right operation and arguments proposed, per source",
            cells(lambda r: _of(s(r, "right")["source"], s(r, "right")["source_total"])),
        ),
        (
            "Right operation and arguments proposed, per variation",
            cells(lambda r: _of(s(r, "right")["variation"], s(r, "right")["variation_total"])),
        ),
        (
            "Should-escalate asks escalated, per source",
            cells(lambda r: _of(s(r, "escalated")["source"], s(r, "escalated")["source_total"])),
        ),
        (
            "Should-escalate asks escalated, per variation",
            cells(
                lambda r: _of(s(r, "escalated")["variation"], s(r, "escalated")["variation_total"])
            ),
        ),
        (
            WRONG_MUTATING_ROW,
            cells(
                lambda r: f"{s(r, 'wrong_mutating')['source']} / "
                f"{s(r, 'wrong_mutating')['variation']}"
            ),
        ),
        (
            WRONG_ARGUMENTS_ROW,
            cells(
                lambda r: f"{s(r, 'wrong_arguments_mutating')['source']} / "
                f"{s(r, 'wrong_arguments_mutating')['variation']}"
            ),
        ),
        (
            "Explain asks explained, per source",
            cells(lambda r: _of(s(r, "explain")["source"], s(r, "explain")["source_total"])),
        ),
        (
            "Explain asks: explained / proposed / escalated, per variation",
            cells(
                lambda r: f"{s(r, 'explain')['explained']} / {s(r, 'explain')['proposed']} / "
                f"{s(r, 'explain')['escalated']} of {s(r, 'explain')['total']}"
            ),
        ),
        (
            "Mutating proposals on explain asks",
            cells(lambda r: str(s(r, "explain")["mutating"])),
        ),
        (
            "Warm latency, median / p95" + mark,
            cells(
                lambda r: f"{_ms(r.result['latency'].get('warm_median_ms'))} / "
                f"{_ms(r.result['latency'].get('warm_p95_ms'))}"
            ),
        ),
        (
            "First request after start" + mark,
            cells(lambda r: _seconds(_cold_s(r.result["latency"]))),
        ),
        ("Container memory (`docker stats`)", [record.memory for record in records]),
        (
            "Start-up, including first download",
            [record.failure or _seconds(record.startup_s) for record in records],
        ),
    ]


def _cold_s(latency: Mapping[str, object]) -> float | None:
    cold = latency.get("cold_ms")
    return cold / 1000.0 if isinstance(cold, (int, float)) else None


def render_table(records: Sequence[RunRecord]) -> str:
    header = "| Metric | " + " | ".join(f"`{record.model}`" for record in records) + " |"
    rule = "|---|" + "---|" * len(records)
    lines = [header, rule]
    for metric, row in result_rows(records):
        lines.append(f"| {metric} | " + " | ".join(row) + " |")
    return "\n".join(lines)


@dataclass(frozen=True)
class Provenance:
    date: str
    label: str
    command: str
    split_path: str
    split_count: int
    source_count: int
    problems: Sequence[str]
    seed: int | None
    seed_origin: str
    commit: str
    grounding: str
    settings: Mapping[str, object]
    final: bool
    acceptance: bool
    finals_before: int


def render_markdown(prov: Provenance, records: Sequence[RunRecord]) -> str:
    seed = f"{prov.seed} ({prov.seed_origin})" if prov.seed is not None else "not recorded"
    settings = ", ".join(
        f"{key}={prov.settings[key]}" for key in _RECORDED_SETTINGS if key in prov.settings
    )
    lines = [
        f"# Tier 2 measurement, {prov.date}: {prov.label}",
        "",
        f"- Command: `{prov.command}`",
        f"- Split: `{prov.split_path}` ({prov.split_count} entries, {prov.source_count} sources)",
        f"- Seed: {seed}",
        f"- nvsh: {__version__}, commit `{prov.commit}`",
        "- Models (repo id @ revision): "
        + "; ".join(
            f"`{record.model}` @ `{record.revision}` ({record.revision_status})"
            for record in records
        ),
        f"- Tier 2 settings, identical for every run except the model: {settings or 'defaults'}",
        f"- Grounding: {prov.grounding}",
        f"- Acceptance run: {'yes' if prov.acceptance else 'no'}",
        FINAL_MARKER if prov.final else "- Final run: no",
    ]
    if prov.final:
        lines.append(f"- Final runs on the test side, including this one: {prov.finals_before + 1}")
    if prov.problems:
        lines.append(f"- Corpus problems (entries skipped): {len(prov.problems)}")
    lines += [
        "",
        "Per-source figures (one vote per `source_id`, majority over its variations, a",
        "tie counts against the model) are the ones claims are judged on; per-variation",
        "figures show paraphrase robustness.",
        "",
        'The use-case bar\'s "0 wrong mutating proposals" is judged on the sum of both rows:',
        f'"{WRONG_MUTATING_ROW}" (bench\'s own count: a mutating operation other than the',
        f'expected one) plus "{WRONG_ARGUMENTS_ROW}" (the expected mutating',
        "operation with arguments bench does not accept, e.g. the wrong container).",
        "",
        render_table(records),
        "",
    ]
    if len(records) > 1 and not latency_comparable(records):
        lines += [
            "Latency figures are not comparable: the runs did not share the same recorded",
            "background (running containers differ, or could not be read).",
            "",
        ]
    lines += ["## Background before each run", ""]
    for record in records:
        background = ", ".join(record.background) if record.background else "none"
        if record.background is None:
            background = "unknown"
        lines += [
            f"### `{record.model}`",
            "",
            f"Other running containers: {background}",
            "",
            "`docker ps`:",
            "",
            "```text",
            record.docker_ps,
            "```",
            "",
            "`nvidia-smi`:",
            "",
            "```text",
            record.nvidia_smi,
            "```",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=_SCRIPT_NAME, description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--split", required=True, help="split file from split.py")
    parser.add_argument(
        "--model", action="append", required=True, help="[tiers.lfm] model; repeat to compare"
    )
    parser.add_argument(
        "--revision", action="append", default=[], help="pinned commit, one per --model"
    )
    parser.add_argument("--label", default="tier2", help="results file name part")
    parser.add_argument("--seed", type=int, default=None, help="default: the split header's")
    parser.add_argument("--config", default=None, help="nvsh config.toml (default: XDG path)")
    parser.add_argument("--world", default=None, help="corpus file whose fixture world to use")
    parser.add_argument("--live", action="store_true", help="ground against this machine")
    parser.add_argument("--acceptance", action="store_true", help="allow held-out.json")
    parser.add_argument("--final", action="store_true", help="allow test.json (a final run)")
    parser.add_argument("--out", default=None, help="results file (default: docs/benchmarks/)")
    parser.add_argument(
        "--details",
        default=None,
        help="also write one JSON line per entry (expected, outcome, correct) here;"
        " refused with --final or --acceptance, so test and held-out failures are never studied",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing results file")
    return parser


def _count_finals(directory: Path, exclude: Path) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for path in sorted(directory.glob("*-lfm-*.md")):
        if path.resolve() == exclude.resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if FINAL_MARKER in text.splitlines():
            count += 1
    return count


def _world(split_path: Path, world_arg: str | None) -> tuple[dict, str]:
    if world_arg:
        return tier_bench.load_world(world_arg), f"fixture world from `{world_arg}`"
    own = tier_bench.load_world(split_path)
    if own:
        return own, "fixture world from the split file"
    dev = tier_bench.dev_corpus_path()
    return tier_bench.load_world(dev), "fixture world from `nvsh/tiers/corpus/dev.json`"


def run(argv: Sequence[str], seams: Seams) -> int:
    args = _parser().parse_args(list(argv))
    if not _LABEL_RE.match(args.label):
        raise MeasureError(
            EXIT_USER, f"--label {args.label!r} must be lower-case letters, " "digits, '.' or '-'"
        )
    if len(args.revision) != len(args.model):
        raise MeasureError(
            EXIT_USER,
            f"{len(args.model)} --model but {len(args.revision)} --revision",
            "give one --revision (the pinned commit) after each --model",
        )
    split_path = Path(args.split)
    raw = read_split(split_path)
    check_split_allowed(
        split_path, acceptance=args.acceptance, final=args.final, header=raw.get("header")
    )
    loaded = tier_bench.load_corpus(split_path)
    if not loaded.entries:
        raise MeasureError(EXIT_USER, f"{split_path} has no valid entries")
    sources = source_ids(raw["entries"])
    seed, seed_origin = args.seed, "--seed"
    if seed is None:
        seed, seed_origin = seed_from_header(raw.get("header")), "from the split header"

    date = seams.today()
    out = Path(args.out) if args.out else _BENCHMARKS_DIR / f"{date}-lfm-{args.label}.md"
    if out.exists() and not args.force:
        raise MeasureError(EXIT_USER, f"{out} already exists", "pick another --label or --force")

    try:
        cfg = seams.load_config(Path(args.config) if args.config else None)
    except (OSError, ValueError) as exc:
        raise MeasureError(EXIT_USER, f"cannot load nvsh config: {exc}") from exc
    tiers = dict(getattr(cfg, "tiers", {}) or {})
    lfm = tiers.get("lfm")
    lfm_settings = dict(lfm) if isinstance(lfm, Mapping) else {}

    runtime_platform = seams.detect_platform()
    if args.live:
        router_platform, runner = runtime_platform, seams.run
        grounding = "live (this machine)"
    else:
        world, grounding = _world(split_path, args.world)
        router_platform = tier_bench.world_platform(world)
        runner = tier_bench.world_runner(world)

    plan = RunPlan(
        entries=loaded.entries,
        problems=loaded.problems,
        sources=sources,
        split=split_path.stem,
        tiers=tiers,
        lfm_settings=lfm_settings,
        router_platform=router_platform,
        runtime_platform=runtime_platform,
        runner=runner,
        grounding=grounding,
    )
    for model, revision in zip(args.model, args.revision):  # refuse before any run starts
        verify_revision({**lfm_settings, "model": model}, model, revision)
    if args.details and (args.final or args.acceptance):
        raise MeasureError(
            EXIT_USER,
            "--details is for iterating on validation, never on the test or held-out side",
            "drop --details",
        )
    finals_before = _count_finals(out.parent, out) if args.final else 0
    records = [
        measure_one(plan, model, revision, seams)
        for model, revision in zip(args.model, args.revision)
    ]

    prov = Provenance(
        date=date,
        label=args.label,
        command=shlex.join([_SCRIPT_NAME, *argv]),
        split_path=str(split_path),
        split_count=len(loaded.entries),
        source_count=len({sources.get(entry.id, entry.id) for entry in loaded.entries}),
        problems=loaded.problems,
        seed=seed,
        seed_origin=seed_origin,
        commit=nvsh_commit(seams.run),
        grounding=grounding,
        settings=lfm_settings,
        final=args.final,
        acceptance=args.acceptance,
        finals_before=finals_before,
    )
    if args.details:
        with open(args.details, "w", encoding="utf-8") as handle:
            for record in records:
                for row in detail_rows(record.model, record.result, loaded.entries):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    text = render_markdown(prov, records)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(render_table(records))
    print(f"wrote {out}")
    return EXIT_ENV if any(record.failure for record in records) else EXIT_OK


def main(argv: Sequence[str] | None = None, *, seams: Seams | None = None) -> int:
    try:
        return run(list(sys.argv[1:] if argv is None else argv), seams or Seams())
    except MeasureError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
