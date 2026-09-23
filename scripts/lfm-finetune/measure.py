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

Issue 46: the shared predictions file
-------------------------------------

Every run also writes one predictions line per entry in ``metrics.py``'s
schema and scores it with ``metrics.py`` (``read_predictions`` then
``compute``), so the stock baseline, Track A (generative) and Track B
(candidate scoring) are compared on one set of figures. The metrics go into
the results file; ``--predictions DIR`` also keeps each model's predictions
and metrics JSON there (refused with ``--final`` or ``--acceptance``, like
``--details``: a final run reports the figures, never the per-entry rows).

Generative runs (the default) drive the same :class:`LfmTier` through a
recording chat client (:class:`RecordingChat`, a non-streaming
:class:`~nvsh.tiers.toolchat.ToolChat`), which adds to every request
``chat_template_kwargs: {"enable_thinking": <value>}`` when
``--enable-thinking`` is given and ``logprobs`` / ``top_logprobs`` when
``--top-logprobs`` is above 0. Per entry it records the tokens the model
generated (``usage.completion_tokens``, summed over the tier's rounds), the
time to first decision (from the tier's start to the first model reply), the
decision time, and whether any reply carried a non-empty think block; that
count is reported and must be 0 (a run with one exits 2). The candidate
distribution comes from the deciding reply's log-probabilities
(:func:`generative_candidates`) where the engine returned them.

``--scorer served|in-process`` runs Track B instead: ``scorer.py`` scores
every candidate label once per entry and the same predictions file is
written from its results (0 tokens; arguments from the grounding path the
scorer uses). A served scorer needs ``--max-logprobs``, the value the
attached vLLM was started with, and it must cover every label plus
``scorer.TOP_MARGIN`` (risk r8); nvsh's managed launcher cannot pass that
flag, so a served scorer is refused in managed mode.

``--slice missing-candidate`` measures ``eval_slices.py``'s slice of the
split (every operation entry with its gold operation left out of the
offered candidates, expected to escalate) instead of the split itself;
``--ctx`` overrides ``[tiers.lfm] ctx`` (2048 or 4096 in issue 46) for
every model alike. ``--ground-snapshot`` grounds every proposal against one
fixed machine snapshot (deviation d1) and records its sha256; ``measure.py
snapshot`` writes one from this machine's service and container lists plus
every service/container argument value in the given split files, printing
only counts. The run record names ctx, engine, image digest and
tool_call_parser.
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess  # nosec B404 - fixed argv lists, never a shell
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh import __version__  # noqa: E402
from nvsh import config as nvsh_config  # noqa: E402
from nvsh.ops import ground as ops_ground  # noqa: E402
from nvsh.ops import table as ops_table  # noqa: E402
from nvsh.platform._model import Platform  # noqa: E402
from nvsh.redact import redact  # noqa: E402
from nvsh.tiers import bench as tier_bench  # noqa: E402
from nvsh.tiers import lfm as tier_lfm  # noqa: E402
from nvsh.tiers import toolchat  # noqa: E402
from nvsh.tiers.base import Decline, DeclineReason, Explanation, Tier, TierDecision  # noqa: E402
from nvsh.tiers.router import AGENT, TierOutcome  # noqa: E402
from nvsh.tiers.runtime import Runtime, RuntimeUnavailable  # noqa: E402
from nvsh.tiers.runtime_docker import (  # noqa: E402
    DEFAULT_CTX,
    DEFAULT_ENGINE,
    HF_CACHE_NAME,
    check_ctx,
    container_name,
    engine_template,
    resolve_image,
)

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """A script next to this one, loaded by path (they are not a package)."""
    spec = importlib.util.spec_from_file_location(f"lfm_finetune_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


metrics = _sibling("metrics")
scorer = _sibling("scorer")
eval_slices = _sibling("eval_slices")
measure_skills = _sibling("measure_skills")

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
    #: The tier's chat client factory (:class:`RecordingChat`); ``None`` is LfmTier's own.
    chat_factory: Callable[[str], object] | None = None


def build_lfm_tier(spec: TierSpec) -> tuple[Tier, Runtime]:
    """The Tier 2 nvsh's daemon builds (``nvsh.tiers.manager.TierManager._lfm``).

    Same launcher (:func:`~nvsh.tiers.runtime_docker.build_runtime` over the
    ``[tiers.lfm]`` settings), same floor check, same :class:`LfmTier`; the
    only difference is that grounding and inspection use *spec.runner*
    (the corpus's fixture world unless ``--live``), as ``nvsh tiers bench``
    does for Tier 1, and that the chat client is *spec.chat_factory*'s
    recording one when given. Constructing either object starts nothing.
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
        chat_factory=spec.chat_factory,  # type: ignore[arg-type]
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
    build_scorer: Callable[["ScorerSpec"], "ScorerHandle"] = lambda spec: build_scorer(spec)


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
# Issue 46: what each generative request carries, and what came back
# ---------------------------------------------------------------------------

#: ``--top-logprobs`` default: vLLM's own ``--max-logprobs`` default.
DEFAULT_TOP_LOGPROBS = 20


@dataclass(frozen=True)
class RequestOptions:
    """What :class:`RecordingChat` adds to every chat completion request."""

    enable_thinking: bool | None = None
    top_logprobs: int = 0

    def extra_body(self) -> dict[str, object]:
        """``chat_template_kwargs`` only when thinking is configured; logprobs when asked for."""
        body: dict[str, object] = {}
        kwargs = measure_skills.chat_template_kwargs(self.enable_thinking)
        if kwargs is not None:
            body["chat_template_kwargs"] = kwargs
        if self.top_logprobs > 0:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        return body


@dataclass(frozen=True)
class TokenLogprob:
    """One generated token: its log-probability and the top alternatives the engine returned."""

    token: str
    logprob: float
    top: Mapping[str, float]


@dataclass(frozen=True)
class ReplyRecord:
    """One model reply inside a tier's loop, as :class:`RecordingChat` saw it."""

    reply: toolchat.ChatReply
    tokens: int | None
    think: bool
    logprobs: tuple[TokenLogprob, ...] | None
    at: float


@dataclass
class SelectRecord:
    """One ``tier.select`` call: its request, its result and every reply in between."""

    request: object
    started: float
    ended: float = 0.0
    result: object = None
    replies: list[ReplyRecord] = field(default_factory=list)


def request_key(request: object) -> tuple:
    """What identifies a request across the router: its kind and its text."""
    return (
        getattr(request, "kind", None),
        getattr(request, "prompt", ""),
        getattr(request, "command", ""),
    )


def _valid_logprob(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isnan(value)
        and value <= 0
    )


def parse_logprobs(raw: object) -> tuple[TokenLogprob, ...] | None:
    """A chat completion's ``logprobs.content``, or ``None`` when absent or malformed."""
    content = raw.get("content") if isinstance(raw, dict) else None
    if not isinstance(content, list) or not content:
        return None
    tokens = []
    for item in content:
        if not isinstance(item, dict) or not isinstance(item.get("token"), str):
            return None
        if not _valid_logprob(item.get("logprob")):
            return None
        top: dict[str, float] = {}
        for alternative in item.get("top_logprobs") or []:
            if (
                isinstance(alternative, dict)
                and isinstance(alternative.get("token"), str)
                and _valid_logprob(alternative.get("logprob"))
            ):
                top[alternative["token"]] = float(alternative["logprob"])
        tokens.append(TokenLogprob(item["token"], float(item["logprob"]), top))
    return tuple(tokens)


def _completion_tokens(payload: Mapping[str, object]) -> int | None:
    usage = payload.get("usage")
    count = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    return None


def read_reply(payload: object, at: float) -> tuple[toolchat.ChatReply, ReplyRecord]:
    """The reply LfmTier reads (``ToolChat``'s own parsing) and what is recorded about it."""
    choice = toolchat._first_choice(payload)
    message = choice.get("message")
    if not isinstance(message, dict):
        raise toolchat.ToolChatError("server reply has no message")
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    reply = toolchat._reply(text, toolchat._calls_from_message(message))
    record = ReplyRecord(
        reply=reply,
        tokens=_completion_tokens(payload),  # type: ignore[arg-type]
        think=measure_skills.nonempty_think(message),
        logprobs=parse_logprobs(choice.get("logprobs")),
        at=at,
    )
    return reply, record


def restrict_tools(tools: list[dict], offered: Sequence[str] | None) -> list[dict]:
    """*tools* with every operation not in *offered* left out (the missing-candidate slice).

    The control tools stay; ``propose``'s operation enum is narrowed the
    same way. Whether a tool is an operation is read off the table.
    """
    if offered is None:
        return tools
    allowed = set(offered)
    kept = []
    for tool in tools:
        name = tool["function"]["name"]
        if ops_table.get(name) is not None and name not in allowed:
            continue
        if name == tier_lfm.PROPOSE_TOOL:
            tool = json.loads(json.dumps(tool))
            spec = tool["function"]["parameters"]["properties"]["operation"]
            spec["enum"] = [op for op in spec.get("enum", []) if op in allowed]
        kept.append(tool)
    return kept


class Recorder:
    """Per-``select`` records for one tier, filled by :class:`RecordingChat`."""

    def __init__(
        self,
        clock: Callable[[], float],
        offered: Mapping[tuple, tuple[str, ...]] | None = None,
    ) -> None:
        self._clock = clock
        self._offered = dict(offered or {})
        self.selects: list[SelectRecord] = []
        self._current: SelectRecord | None = None

    def attach(self, tier: Tier) -> None:
        """Record every ``tier.select`` call; the tier object itself is left as it is."""
        inner = tier.select

        def select(request, context):
            record = SelectRecord(request=request, started=self._clock())
            self._current = record
            try:
                record.result = inner(request, context)
            finally:
                record.ended = self._clock()
                self.selects.append(record)
                self._current = None
            return record.result

        tier.select = select  # type: ignore[method-assign]

    def offered(self) -> tuple[str, ...] | None:
        """The operations offered for the request in flight (``None``: all of them)."""
        if self._current is None:
            return None
        return self._offered.get(request_key(self._current.request))

    def add(self, reply: ReplyRecord) -> None:
        if self._current is not None:
            self._current.replies.append(reply)


class RecordingChat(toolchat.ToolChat):
    """LfmTier's non-streaming chat client, plus the issue-46 request options and records.

    Same localhost-only, no-redirect transport and the same reply parsing as
    :class:`~nvsh.tiers.toolchat.ToolChat`; each request also carries
    :meth:`RequestOptions.extra_body`, and each reply is handed to the
    :class:`Recorder` before the tier sees it.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        recorder: Recorder,
        options: RequestOptions,
        clock: Callable[[], float],
    ) -> None:
        super().__init__(base_url, model, stream=False)
        self._recorder = recorder
        self._options = options
        self._clock = clock

    def complete(self, messages: list[dict], tools: list[dict]) -> toolchat.ChatReply:
        body = {
            "model": self._model,
            "messages": messages,
            "tools": restrict_tools(tools, self._recorder.offered()),
            "stream": False,
            **self._options.extra_body(),
        }
        return self._request("/chat/completions", body, self._parse_recorded)

    def _parse_recorded(self, response) -> toolchat.ChatReply:
        reply, record = read_reply(json.loads(response.read()), self._clock())
        self._recorder.add(record)
        return reply


# ---------------------------------------------------------------------------
# Issue 46: a generative model's candidate distribution, from its log-probabilities
# ---------------------------------------------------------------------------

#: Where a printed tool call starts (``ToolChat``'s two raw shapes).
_TOOL_MARKERS = ("<tool_call>", "<|tool_call_start|>")
#: Tool-call markup left in a reply's text: the tool-call wrapper tag (open or
#: close, either raw shape) or Qwen's XML function / parameter tags.
_TOOL_MARKUP = re.compile(
    r"</?tool_call>|<\|tool_call_(?:start|end)\|>"
    r"|<(?:function|parameter)=|</(?:function|parameter)>"
)
_IDENTIFIER_CHAR = re.compile(r"[A-Za-z0-9_]")
_CONTROLS = (tier_lfm.PROPOSE_TOOL, tier_lfm.EXPLAIN_TOOL, tier_lfm.ESCALATE_TOOL)


def _is_identifier(char: str) -> bool:
    return bool(char) and bool(_IDENTIFIER_CHAR.match(char))


def _find_name(text: str, name: str, start: int) -> int:
    """The first whole-word occurrence of *name* at or after *start*, or -1."""
    at = text.find(name, start)
    while at >= 0:
        before = text[at - 1] if at > 0 else ""
        after = text[at + len(name) : at + len(name) + 1]
        if not _is_identifier(before) and not _is_identifier(after):
            return at
        at = text.find(name, at + 1)
    return -1


UNOBSERVED = "an alternative token's continuation was not observed"


def _ended_label(text: str, labels: Sequence[str]) -> str | None:
    """The label *text* spells in full and then ends (a non-name character follows), if any."""
    for label in labels:
        if text.startswith(label) and len(text) > len(label):
            if not _is_identifier(text[len(label)]):
                return label
    return None


def _walk(
    tokens: Sequence[TokenLogprob],
    offsets: Sequence[int],
    start: int,
    name: str,
    labels: Sequence[str],
) -> dict[str, float] | str:
    """Mass per label for the generated *name* at character *start*, or why there is none.

    Along the generated tokens that spell *name* and the token that ends it,
    each alternative token the engine returned is placed on a label only
    when the alternative itself spells that label and the character after
    it: then the label's probability is exact, the path up to the
    alternative times its own. An alternative that could still become a
    label, or spells one with nothing after it, needs a continuation the
    engine never scored (``gpu`` is not ``gpu_stats``, even when no other
    label starts that way), so the result is refused rather than guessed.
    Alternatives that become no label are dropped. The generated name's own
    mass is the path through the token that ends it.
    """
    index = bisect.bisect_right(offsets, start) - 1
    lead = tokens[index].token[: start - offsets[index]]
    masses = dict.fromkeys(labels, 0.0)
    path, done = 1.0, ""
    while len(done) <= len(name):
        if index >= len(tokens):
            return "the generated name runs past the recorded tokens"
        token = tokens[index]
        for alternative, logprob in token.top.items():
            if alternative == token.token or not alternative.startswith(lead):
                continue
            text = done + alternative[len(lead) :]
            ended = _ended_label(text, labels)
            if ended is not None:
                masses[ended] += path * math.exp(logprob)
            elif any(label.startswith(text) for label in labels):
                return UNOBSERVED
        path *= math.exp(token.logprob)
        done += token.token[len(lead) :]
        lead = ""
        index += 1
    masses[name] += path
    return masses


def generative_candidates(
    tokens: Sequence[TokenLogprob] | None,
    call: toolchat.ToolCall | None,
    offered_operations: Sequence[str],
) -> tuple[dict[str, float] | None, str]:
    """``(label -> probability, "")`` from the deciding reply, or ``(None, why not)``.

    Labels are metrics.py's: operation names and bench's ``(explain)`` /
    ``(escalate)``. The control tool's name is walked first (propose,
    explain, escalate; an alternative naming an inspection tool is not a
    decision and is dropped), then, for a proposal, the proposed operation's
    name, whose masses are scaled by propose's. Mass on propose that was never
    followed into an operation name cannot be split over operations, so such
    a reply has no distribution. The masses found are normalised over the
    candidates; anything outside the engine's top alternatives is 0.
    """
    if not tokens:
        return None, "no log-probabilities returned"
    if call is None:
        return None, "no tool call in the deciding reply"
    if call.name not in _CONTROLS:
        return None, "the deciding reply called no control tool"
    text = "".join(token.token for token in tokens)
    offsets = [0]
    for token in tokens[:-1]:
        offsets.append(offsets[-1] + len(token.token))
    marks = [text.find(marker) for marker in _TOOL_MARKERS if marker in text]
    at = _find_name(text, call.name, min(marks) if marks else 0)
    if at < 0:
        return None, "the tool name is not in the generated tokens"
    walked = _walk(
        tokens, offsets, at, call.name, tuple(dict.fromkeys(_CONTROLS + tuple(offered_operations)))
    )
    if isinstance(walked, str):
        return None, walked
    masses = {
        tier_bench.EXPLAIN_LABEL: walked[tier_lfm.EXPLAIN_TOOL],
        tier_bench.ESCALATE_LABEL: walked[tier_lfm.ESCALATE_TOOL],
    }
    proposed = walked[tier_lfm.PROPOSE_TOOL]
    if call.name == tier_lfm.PROPOSE_TOOL:
        operation = call.arguments.get("operation")
        if not isinstance(operation, str) or not operation:
            return None, "the proposal names no operation"
        op_at = _find_name(text, operation, at + len(call.name))
        if op_at < 0:
            return None, "the proposed operation is not in the generated tokens"
        labels = tuple(dict.fromkeys((*offered_operations, operation)))
        walked_ops = _walk(tokens, offsets, op_at, operation, labels)
        if isinstance(walked_ops, str):
            return None, walked_ops
        for name, mass in walked_ops.items():
            masses[name] = proposed * mass
    elif proposed > 0:
        return None, "proposal mass cannot be split over operations"
    total = math.fsum(masses.values())
    if total <= 0:
        return None, "no mass on any candidate"
    return {label: mass / total for label, mass in masses.items() if mass > 0}, ""


# ---------------------------------------------------------------------------
# Issue 46: predictions lines (metrics.py's schema)
# ---------------------------------------------------------------------------

THINK_BLOCKS = "think_blocks"
TOKENS_UNREPORTED = "tokens_unreported"
NOT_REACHED = "not_reached"
NO_DISTRIBUTION = "no distribution: "
#: ``invalid_reason`` for plain text that is really a tool call nothing parsed.
UNPARSED_TOOL_CALL = "unparsed_tool_call"


def _did(row: Mapping[str, object]) -> str:
    """What a bench row says the router did (``detail_rows``' three words)."""
    if not row.get("handled_by"):
        return "escalate"
    return "explain" if row.get("operation") is None else "propose"


def _unparsed_tool_call(select: SelectRecord | None) -> bool:
    """The tier explained, but in tool-call markup its chat client could not parse.

    Both the explanation and the raw text of the reply it came from are read
    (the explanation is redacted and cut short, so markup can fall out of it).
    """
    if select is None or not isinstance(select.result, Explanation):
        return False
    texts = [select.result.text]
    if select.replies and not select.replies[-1].reply.tool_calls:
        texts.append(select.replies[-1].reply.text)
    return any(_TOOL_MARKUP.search(text) for text in texts)


def _decided(
    row: Mapping[str, object] | None, select: SelectRecord | None
) -> tuple[str, str | None, dict | None, str | None]:
    """``(outcome, operation, arguments, invalid_reason)`` for one entry.

    What the router did stands when the tier answered. When the router
    escalated, the tier's own result says why: a proposal the router could
    not use is still the model's tool call, an ``escalate`` call is an
    abstention, and anything else (no usable output, out of rounds, a tier
    that failed or was unavailable) is an invalid output, not an abstention.
    An explanation written in tool-call markup is a tool call that went
    unparsed, so it is an invalid output too, never explain credit.
    """
    if _unparsed_tool_call(select):
        return "invalid", None, None, UNPARSED_TOOL_CALL
    did = _did(row) if row is not None else "escalate"
    if did == "propose":
        return "propose", str(row["operation"]), dict(row.get("args") or {}), None  # type: ignore
    if did == "explain":
        return "explain", None, None, None
    if select is None:
        return "invalid", None, None, NOT_REACHED
    result = select.result
    if isinstance(result, TierDecision):
        return "propose", result.operation, dict(result.args), None
    if isinstance(result, Explanation):
        return "explain", None, None, None
    if not isinstance(result, Decline):
        return "invalid", None, None, metrics.DEFAULT_INVALID_REASON
    if result.reason is not DeclineReason.ESCALATED:
        return "invalid", None, None, result.reason.value
    if not select.replies:
        return "escalate", None, None, None
    calls = select.replies[-1].reply.tool_calls
    if calls and calls[0].name == tier_lfm.ESCALATE_TOOL:
        return "escalate", None, None, None
    return "invalid", None, None, "no_decision"


def prediction_line(
    entry: tier_bench.CorpusEntry,
    outcome: tuple[str, str | None, dict | None, str | None],
    *,
    candidates: Mapping[str, float] | None,
    tokens: int,
    ttfd_ms: float,
    latency_ms: float,
) -> dict:
    kind, operation, arguments, reason = outcome
    line = {
        "id": entry.id,
        "expected": dict(entry.expect),
        "outcome": kind,
        "operation": operation,
        "arguments": arguments,
        "candidates": None if candidates is None else dict(candidates),
        "tokens": int(tokens),
        "ttfd_ms": max(float(ttfd_ms), 0.0),
        "latency_ms": max(float(latency_ms), 0.0),
    }
    if reason is not None:
        line["invalid_reason"] = reason
    return line


def _reply_tokens(reply: ReplyRecord) -> int | None:
    if reply.tokens is not None:
        return reply.tokens
    return len(reply.logprobs) if reply.logprobs else None


def generative_predictions(
    result: Mapping[str, object],
    entries: Sequence[tier_bench.CorpusEntry],
    selects: Sequence[SelectRecord],
    offered: Mapping[str, tuple[str, ...]],
) -> tuple[list[dict], Counter]:
    """One predictions line per entry from a bench run and the tier's recorded selects.

    Selects are matched to entries in order by their request. Tokens are the
    replies' ``completion_tokens`` summed (or their log-probability count when
    usage is missing; neither is counted as unreported); time to first
    decision is the tier's start to its first model reply, and latency the
    tier's whole decision. An entry the tier never saw is invalid.
    """
    rows = {row["id"]: row for row in result.get("items", [])}  # type: ignore[union-attr]
    notes: Counter = Counter()
    lines = []
    pointer = 0
    for entry in entries:
        select = None
        if pointer < len(selects) and request_key(selects[pointer].request) == request_key(
            tier_bench.request_for(entry)
        ):
            select, pointer = selects[pointer], pointer + 1
        outcome = _decided(rows.get(entry.id), select)
        tokens, ttfd, latency, candidates = 0, 0.0, 0.0, None
        if select is None:
            notes[NO_DISTRIBUTION + "the tier never saw the entry"] += 1
        else:
            latency = (select.ended - select.started) * 1000.0
            ttfd = (select.replies[0].at - select.started) * 1000.0 if select.replies else latency
            for reply in select.replies:
                counted = _reply_tokens(reply)
                if counted is None:
                    notes[TOKENS_UNREPORTED] += 1
                tokens += counted or 0
                notes[THINK_BLOCKS] += int(reply.think)
            candidates, why = _entry_candidates(select, outcome[0], offered.get(entry.id))
            if candidates is None:
                notes[NO_DISTRIBUTION + why] += 1
        lines.append(
            prediction_line(
                entry,
                outcome,
                candidates=candidates,
                tokens=tokens,
                ttfd_ms=ttfd,
                latency_ms=latency,
            )
        )
    return lines, notes


def _entry_candidates(
    select: SelectRecord, outcome: str, offered: tuple[str, ...] | None
) -> tuple[dict[str, float] | None, str]:
    if outcome == "invalid":
        return None, "invalid output"
    if not select.replies:
        return None, "no model reply recorded"
    last = select.replies[-1]
    call = last.reply.tool_calls[0] if last.reply.tool_calls else None
    return generative_candidates(last.logprobs, call, offered or ops_table.names())


def score_predictions(lines: Sequence[dict], path: Path) -> dict:
    """Write *lines* to *path* as JSONL, then score that file with metrics.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        return metrics.compute(metrics.read_predictions(path))
    except metrics.MetricsError as exc:
        raise MeasureError(EXIT_ENV, f"metrics.py refused the predictions file: {exc}") from exc


# ---------------------------------------------------------------------------
# Issue 46, Track B: the candidate scorer writes the same predictions file
# ---------------------------------------------------------------------------

SCORER_SERVED = "served"
SCORER_IN_PROCESS = "in-process"
REVISION_IN_PROCESS = "passed to from_pretrained in-process"


@dataclass(frozen=True)
class ScorerSpec:
    kind: str
    model: str
    revision: str
    lfm_settings: Mapping[str, object]
    runtime_platform: Platform
    memory_floor_mb: int


@dataclass(frozen=True)
class ScorerHandle:
    """A ``score_next_token`` scorer, the prompt renderer for its model, and its release."""

    scorer: object
    render: Callable[[list[dict]], str]
    close: Callable[[], None]


def scorer_labels_needed() -> int:
    """Log-probabilities a served scorer asks for: every label plus the margin (risk r8)."""
    return len(scorer.candidates()) + scorer.TOP_MARGIN


def build_scorer(spec: ScorerSpec) -> ScorerHandle:  # pragma: no cover - a model server or GPU
    """The Track B scorer for *spec*: a served model through ``ToolChat``, or in-process.

    Both render prompts with the model's own chat template at *spec.revision*
    (thinking off when the template has that switch, ``scorer.render_prompt``).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.model, revision=spec.revision)

    def render(messages: list[dict]) -> str:
        return scorer.render_prompt(tokenizer, messages)

    if spec.kind == SCORER_SERVED:
        from nvsh.tiers.memfloor import check_floor
        from nvsh.tiers.runtime_docker import build_runtime

        floor_mb = int(spec.memory_floor_mb)
        runtime = build_runtime(
            spec.lfm_settings, spec.runtime_platform, floor_check=lambda: check_floor(floor_mb)
        )
        chat = toolchat.ToolChat(runtime.ensure(), spec.model, stream=False)
        return ScorerHandle(scorer=chat, render=render, close=runtime.stop)

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        spec.model, revision=spec.revision, torch_dtype=torch.bfloat16
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    labels = scorer.labels_for(scorer.candidates())
    ids = scorer.label_token_ids(tokenizer, labels)
    in_process = scorer.TransformersScorer(model, tokenizer, labels, ids)
    return ScorerHandle(scorer=in_process, render=render, close=lambda: None)


def scorer_line(entry: tier_bench.CorpusEntry, scored, elapsed_ms: float) -> dict:
    """A predictions line from one :class:`scorer.Scored`: 0 tokens, grounded arguments."""
    choice = scored.choice
    if choice is None:
        outcome = ("invalid", None, None, "no_label_mass")
    elif choice == tier_lfm.EXPLAIN_TOOL:
        outcome = ("explain", None, None, None)
    elif choice == tier_lfm.ESCALATE_TOOL:
        outcome = ("escalate", None, None, None)
    elif scored.arguments is not None:
        outcome = ("propose", choice, dict(scored.arguments), None)
    else:
        outcome = ("invalid", None, None, DeclineReason.NOT_GROUNDED.value)
    return prediction_line(
        entry,
        outcome,
        candidates=scored.candidates,
        tokens=0,
        ttfd_ms=elapsed_ms,
        latency_ms=elapsed_ms,
    )


def scorer_predictions(
    plan: "RunPlan", handle: ScorerHandle, clock: Callable[[], float]
) -> tuple[list[dict], Counter]:
    """Score every entry once; the request text is what Track B trained on."""
    lines, notes = [], Counter()
    for entry in plan.entries:
        request_text = tier_lfm.request_message(
            tier_bench.request_for(entry), tier_bench.context_for(entry)
        )
        offered = plan.offered.get(entry.id)
        candidates = None if offered is None else tuple(offered) + scorer.CONTROLS
        prompt = handle.render(scorer.prompt_messages(request_text, candidates))
        started = clock()
        scored = scorer.score(
            handle.scorer, prompt, request_text, offered=candidates, runner=plan.runner
        )
        elapsed_ms = (clock() - started) * 1000.0
        if scored.incomplete is not None:
            notes[NO_DISTRIBUTION + "labels missing from the top log-probabilities"] += 1
        elif scored.candidates is None:
            notes[NO_DISTRIBUTION + "no label mass"] += 1
        lines.append(scorer_line(entry, scored, elapsed_ms))
    return lines, notes


# ---------------------------------------------------------------------------
# Deviation d1: one fixed machine snapshot for grounding
# ---------------------------------------------------------------------------

#: Grounded argument -> the snapshot list holding its values (nvsh.ops.ground's kinds).
SNAPSHOT_KEYS = {"service": "services", "container": "containers"}


def load_snapshot(path: Path) -> tuple[dict, str]:
    """``(snapshot, sha256 of the file)``; a malformed snapshot is refused (exit 1)."""
    hint = "write one with: measure.py snapshot --out PATH --from-split SPLIT"
    try:
        data = path.read_bytes()
        raw = json.loads(data)
    except (OSError, ValueError) as exc:
        raise MeasureError(EXIT_USER, f"cannot read ground snapshot {path}: {exc}", hint) from exc
    if not isinstance(raw, dict):
        raise MeasureError(EXIT_USER, f"{path} is not a ground snapshot object", hint)
    for key in SNAPSHOT_KEYS.values():
        values = raw.get(key)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise MeasureError(EXIT_USER, f"{path}: {key} must be a list of names", hint)
    for key in ("source", "created"):
        if not isinstance(raw.get(key), str):
            raise MeasureError(EXIT_USER, f"{path}: {key} must be a string", hint)
    return raw, hashlib.sha256(data).hexdigest()


def snapshot_runner(snapshot: Mapping[str, object]) -> RunFn:
    """A grounding runner answering the service and container lookups from *snapshot*."""
    return tier_bench.world_runner(snapshot)


def _lookup_name(argument: str, value: str) -> str:
    """How the machine's list spells *value* (a service gets its unit suffix)."""
    kind = ops_ground._KINDS[argument]
    return next((name for name in kind.wanted(value) if kind.parse(name)), value)


def build_snapshot(
    run: RunFn, splits: Sequence[Path], *, source: str, created: str
) -> tuple[dict, dict[str, int]]:
    """This machine's lists plus every grounded argument value in *splits*, and counts."""
    live: dict[str, set[str]] = {}
    for argument, key in SNAPSHOT_KEYS.items():
        kind = ops_ground._KINDS[argument]
        code, output = run(list(kind.lookup_argv), ops_ground.LOOKUP_TIMEOUT)
        if code != 0:
            raise MeasureError(
                EXIT_ENV,
                f"cannot list this machine's {kind.plural}: {kind.lookup_argv[0]} exited {code}",
                f"run the snapshot builder where {kind.lookup_argv[0]} works",
            )
        live[key] = set(kind.parse(output))
    found: dict[str, set[str]] = {key: set() for key in SNAPSHOT_KEYS.values()}
    for path in splits:
        for item in read_split(path)["entries"]:
            expect = item.get("expect") if isinstance(item, dict) else None
            args = expect.get("args") if isinstance(expect, dict) else None
            if not isinstance(args, dict):
                continue
            for argument, key in SNAPSHOT_KEYS.items():
                value = args.get(argument)
                if isinstance(value, str) and value:
                    found[key].add(_lookup_name(argument, value))
    snapshot: dict[str, object] = {
        key: sorted(live[key] | found[key]) for key in SNAPSHOT_KEYS.values()
    }
    snapshot["source"] = source
    snapshot["created"] = created
    counts = {}
    for key in SNAPSHOT_KEYS.values():
        counts[key] = len(snapshot[key])  # type: ignore[arg-type]
        counts[f"{key}_live"] = len(live[key])
        counts[f"{key}_splits_only"] = len(found[key] - live[key])
    return snapshot, counts


def run_snapshot(argv: Sequence[str], seams: Seams) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{_SCRIPT_NAME} snapshot",
        description="Write a fixed grounding snapshot (deviation d1); prints counts only.",
    )
    parser.add_argument("--out", required=True, help="snapshot JSON file to write")
    parser.add_argument(
        "--from-split",
        action="append",
        default=[],
        help="split file whose service/container argument values are added; repeat",
    )
    parser.add_argument("--source", default=None, help="what the snapshot was taken from")
    parser.add_argument("--force", action="store_true", help="overwrite an existing snapshot")
    args = parser.parse_args(list(argv))
    out = Path(args.out)
    if out.exists() and not args.force:
        raise MeasureError(EXIT_USER, f"{out} already exists", "pass --force to overwrite it")
    splits = [Path(path) for path in args.from_split]
    source = args.source or (
        "this machine's systemctl and docker lists plus the service and container argument"
        f" values of {len(splits)} split file(s)"
    )
    snapshot, counts = build_snapshot(seams.run, splits, source=source, created=seams.today())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parts = [
        f"{counts[key]} {key} ({counts[key + '_live']} from this machine, "
        f"{counts[key + '_splits_only']} only from the split files)"
        for key in SNAPSHOT_KEYS.values()
    ]
    print(f"wrote {out}: " + ", ".join(parts))
    return EXIT_OK


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
    #: Issue 46: the predictions lines, metrics.py's figures over them, and run notes.
    predictions: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    notes: Counter = field(default_factory=Counter)


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
    #: Entry id -> the operations offered to it (the missing-candidate slice); absent: all.
    offered: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    request_options: RequestOptions = RequestOptions()
    #: ``""`` for a generative run, else :data:`SCORER_SERVED` / :data:`SCORER_IN_PROCESS`.
    scorer_kind: str = ""


def _offered_by_request(plan: RunPlan) -> dict[tuple, tuple[str, ...]]:
    return {
        request_key(tier_bench.request_for(entry)): plan.offered[entry.id]
        for entry in plan.entries
        if entry.id in plan.offered
    }


def measure_one(plan: RunPlan, model: str, revision: str, seams: Seams) -> RunRecord:
    if plan.scorer_kind:
        return score_one(plan, model, revision, seams)
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

    recorder = Recorder(seams.clock, _offered_by_request(plan))

    def chat_factory(base_url: str) -> RecordingChat:
        return RecordingChat(
            base_url, model, recorder=recorder, options=plan.request_options, clock=seams.clock
        )

    spec = TierSpec(
        lfm_settings=settings,
        runtime_platform=plan.runtime_platform,
        tier_platform=plan.router_platform,
        runner=plan.runner,
        memory_floor_mb=int(plan.tiers.get("memory_floor_mb", 1024)),  # type: ignore[arg-type]
        chat_factory=chat_factory,
    )
    tier, runtime = seams.build_tier(spec)
    recorder.attach(tier)
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
        record.predictions, record.notes = generative_predictions(
            record.result, plan.entries, recorder.selects, plan.offered
        )
        if managed:
            record.memory = container_memory(seams.run, own)
    finally:
        tier.close()
    return record


def scorer_revision_status(plan: RunPlan, model: str, revision: str) -> str:
    if plan.scorer_kind == SCORER_IN_PROCESS:
        return REVISION_IN_PROCESS
    return verify_revision({**plan.lfm_settings, "model": model}, model, revision)


def score_one(plan: RunPlan, model: str, revision: str, seams: Seams) -> RunRecord:
    """Track B: the candidate scorer over every entry, into the same predictions file."""
    record = RunRecord(model=model, revision=revision)
    record.revision_status = scorer_revision_status(plan, model, revision)
    record.docker_ps = _capture(seams.run, ["docker", "ps"])
    record.nvidia_smi = _capture(seams.run, ["nvidia-smi"])
    record.background = background_set(seams.run, container_name(int(seams.uid())))
    spec = ScorerSpec(
        kind=plan.scorer_kind,
        model=model,
        revision=revision,
        lfm_settings={**plan.lfm_settings, "model": model},
        runtime_platform=plan.runtime_platform,
        memory_floor_mb=int(plan.tiers.get("memory_floor_mb", 1024)),  # type: ignore[arg-type]
    )
    started = seams.clock()
    try:
        handle = seams.build_scorer(spec)
    except (RuntimeUnavailable, toolchat.ToolChatError, OSError, ValueError, ImportError) as exc:
        record.failure = f"scorer start-up failed: {exc}"
        return record
    record.startup_s = seams.clock() - started
    try:
        record.predictions, record.notes = scorer_predictions(plan, handle, seams.clock)
    finally:
        handle.close()
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
    #: Issue 46's run record: serving (ctx, engine, image digest, parser), requests, mode, slice.
    serving: Mapping[str, str] = field(default_factory=dict)
    requests: str = ""
    decision_mode: str = ""
    slice_note: str = ""
    snapshot_note: str = ""
    scorer_run: bool = False


def serving_record(settings: Mapping[str, object]) -> dict[str, str]:
    """ctx, engine, image digest and tool_call_parser as the runs are served.

    The launcher's own defaults fill in what ``[tiers.lfm]`` leaves unset;
    an attached endpoint's image is only known when ``[tiers.lfm] image``
    names it.
    """
    engine = str(settings.get("engine") or DEFAULT_ENGINE)
    mode = str(settings.get("mode") or MANAGED)
    parser = settings.get("tool_call_parser")
    try:
        parser = parser or engine_template(engine).default_tool_parser or "none"
    except RuntimeUnavailable:
        parser = parser or "unknown"
    if settings.get("image"):
        image = str(settings["image"])
    elif mode == MANAGED:
        try:
            image = resolve_image(settings, engine)
        except RuntimeUnavailable as exc:
            image = f"not resolved ({exc})"
    else:
        image = "not recorded (attached endpoint; set [tiers.lfm] image to record it)"
    return {
        "engine": engine,
        "mode": mode,
        "ctx": str(settings.get("ctx") or DEFAULT_CTX),
        "image": image,
        "tool_call_parser": str(parser),
    }


def _pct(value: object) -> str:
    return f"{100 * value:.1f}%" if isinstance(value, (int, float)) else "n/a"


def _num(value: object, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def _timing(block: Mapping[str, object]) -> str:
    return (
        f"{_ms(block.get('cold_ms'))} / {_ms(block.get('warm_median_ms'))} / "
        f"{_ms(block.get('warm_p95_ms'))}"
    )


def metric_rows(records: Sequence[RunRecord]) -> list[tuple[str, list[str]]]:
    """``(metric, [one cell per model])`` for metrics.py's figures over each predictions file."""

    def cells(fn: Callable[[dict, RunRecord], str]) -> list[str]:
        return [
            "not measured" if record.failure or not record.metrics else fn(record.metrics, record)
            for record in records
        ]

    def invalid(m: dict, _r: RunRecord) -> str:
        reasons = ", ".join(f"{k}: {v}" for k, v in m["invalid"]["by_reason"].items())
        return _of(m["invalid"]["n"], m["invalid"]["N"]) + (f" ({reasons})" if reasons else "")

    return [
        (
            "Right proposals (metrics.py)",
            cells(lambda m, _r: _of(m["right_proposals"]["n"], m["right_proposals"]["N"])),
        ),
        (
            "Abstention recall (escalate entries escalated)",
            cells(
                lambda m, _r: f"{_pct(m['abstention']['recall'])} "
                f"({_of(m['escalation']['tp'], m['escalation']['tp'] + m['escalation']['fn'])})"
            ),
        ),
        (
            "Abstention precision, strict (deviation d2)",
            cells(lambda m, _r: _pct(m["abstention"]["precision"])),
        ),
        (
            "False-positive tool calls (proposals on explain/escalate entries)",
            cells(
                lambda m, _r: _of(
                    m["false_positive_tool_calls"]["n"], m["false_positive_tool_calls"]["N"]
                )
            ),
        ),
        (
            "Wrong mutating, total (wrong operation + wrong arguments)",
            cells(
                lambda m, _r: f"{m['wrong_mutating']['total']} "
                f"({m['wrong_mutating']['wrong_operation']} + "
                f"{m['wrong_mutating']['wrong_arguments']})"
            ),
        ),
        ("Invalid outputs", cells(invalid)),
        (
            "Lines with a candidate distribution",
            cells(
                lambda m, _r: _of(
                    m["calibration"]["n"],
                    m["calibration"]["n"] + m["calibration"]["without_distribution"],
                )
            ),
        ),
        ("ECE (10 equal-width bins)", cells(lambda m, _r: _num(m["calibration"]["ece"]))),
        ("Brier (multi-class)", cells(lambda m, _r: _num(m["calibration"]["brier"]))),
        (
            "Tokens generated per decision, mean / median",
            cells(
                lambda m, _r: f"{_num(m['tokens']['mean'], 1)} / {_num(m['tokens']['median'], 1)}"
            ),
        ),
        (
            "Time to first decision, cold / warm median / warm p95",
            cells(lambda m, _r: _timing(m["time_to_first_decision"])),
        ),
        (
            "Decision latency, cold / warm median / warm p95",
            cells(lambda m, _r: _timing(m["latency"])),
        ),
        (
            "Non-empty think blocks (must be 0)",
            cells(lambda _m, r: str(r.notes.get(THINK_BLOCKS, 0))),
        ),
    ]


def render_metrics(records: Sequence[RunRecord]) -> str:
    header = "| Metric | " + " | ".join(f"`{record.model}`" for record in records) + " |"
    lines = [header, "|---|" + "---|" * len(records)]
    for metric, row in metric_rows(records):
        lines.append(f"| {metric} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def _notes_lines(records: Sequence[RunRecord]) -> list[str]:
    lines = []
    for record in records:
        missing = sorted(
            (key[len(NO_DISTRIBUTION) :], count)
            for key, count in record.notes.items()
            if key.startswith(NO_DISTRIBUTION)
        )
        if missing:
            detail = "; ".join(f"{why}: {count}" for why, count in missing)
            lines.append(f"- `{record.model}`: lines without a candidate distribution: {detail}")
        if record.notes.get(TOKENS_UNREPORTED):
            lines.append(
                f"- `{record.model}`: replies whose token count the server did not report "
                f"(counted as 0): {record.notes[TOKENS_UNREPORTED]}"
            )
    return lines


def _notes_block(records: Sequence[RunRecord]) -> list[str]:
    notes = _notes_lines(records)
    return [*notes, ""] if notes else []


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
    ]
    if prov.serving:
        lines.append(
            "- Serving: "
            + ", ".join(
                f"{key}=`{value}`" if key == "image" else f"{key}={value}"
                for key, value in prov.serving.items()
            )
        )
    lines += [
        line
        for line in (
            f"- Requests: {prov.requests}" if prov.requests else "",
            f"- Decision mode: {prov.decision_mode}" if prov.decision_mode else "",
            f"- Slice: {prov.slice_note}" if prov.slice_note else "",
            f"- Ground snapshot: {prov.snapshot_note}" if prov.snapshot_note else "",
        )
        if line
    ]
    lines += [
        f"- Acceptance run: {'yes' if prov.acceptance else 'no'}",
        FINAL_MARKER if prov.final else "- Final run: no",
    ]
    if prov.final:
        lines.append(f"- Final runs on the test side, including this one: {prov.finals_before + 1}")
    if prov.problems:
        lines.append(f"- Corpus problems (entries skipped): {len(prov.problems)}")
    if prov.scorer_run:
        lines += [
            "",
            "A candidate-scorer run: the bench table does not apply; the figures are",
            "metrics.py's, over the predictions file the scorer's results were written to.",
            "",
        ]
    else:
        lines += _bench_section(records)
    lines += [
        "## Issue 46 metrics",
        "",
        "metrics.py over each model's predictions file (one line per entry). The candidate",
        "distribution of a generative run comes from its deciding reply's log-probabilities.",
        "",
        render_metrics(records),
        "",
        *_notes_block(records),
        metrics.issue46_mapping()["markdown"],
        "",
        metrics.ISSUE46_NOTE,
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


def _bench_section(records: Sequence[RunRecord]) -> list[str]:
    lines = [
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
    return lines


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
    parser.add_argument(
        "--predictions",
        default=None,
        help="keep each model's predictions JSONL and metrics JSON in this directory;"
        " refused with --final or --acceptance (the figures are still reported)",
    )
    parser.add_argument(
        "--enable-thinking",
        type=measure_skills.parse_bool,
        default=None,
        metavar="{true,false}",
        help="send chat_template_kwargs enable_thinking=<value> (issue 46: false); not sent"
        " when omitted",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=DEFAULT_TOP_LOGPROBS,
        help="log-probabilities asked for per generated token (0: none); default"
        f" {DEFAULT_TOP_LOGPROBS}, vLLM's default --max-logprobs",
    )
    parser.add_argument(
        "--max-logprobs",
        type=int,
        default=None,
        help="the --max-logprobs the served engine was started with (recorded; required for"
        " --scorer served)",
    )
    parser.add_argument("--ctx", type=int, default=None, help="override [tiers.lfm] ctx")
    parser.add_argument(
        "--slice",
        choices=(SLICE_FULL, SLICE_MISSING),
        default=SLICE_FULL,
        help="measure the split, or eval_slices.py's missing-candidate slice of it",
    )
    parser.add_argument(
        "--ground-snapshot",
        default=None,
        help="ground against this fixed snapshot (measure.py snapshot writes one)",
    )
    parser.add_argument(
        "--scorer",
        choices=(SCORER_SERVED, SCORER_IN_PROCESS),
        default=None,
        help="Track B: score candidate labels with scorer.py instead of generating",
    )
    return parser


SLICE_FULL = "full"
SLICE_MISSING = "missing-candidate"


def home_relative(text: str) -> str:
    """*text* with the user's home directory written as ``$HOME``.

    Reports are committed; a hard-coded ``/home/<user>/`` path is not
    portable and fails the repo's portability check.
    """
    home = str(Path.home())
    return text.replace(home + "/", "$HOME/") if home not in ("", "/") else text


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


def _sliced(raw: dict, path: Path, directory: Path) -> tuple[dict, Path]:
    """eval_slices.py's missing-candidate slice of *raw*, and a file holding it."""
    header = raw.get("header")
    header_text = header if isinstance(header, str) else json.dumps(header, sort_keys=True)
    sliced = eval_slices.missing_candidate_slice(
        {"header": header_text, "entries": raw["entries"]}, ops_table.names()
    )
    target = directory / f"{path.stem}-{SLICE_MISSING}.json"
    target.write_text(json.dumps(sliced), encoding="utf-8")
    return sliced, target


def offered_candidates(raw_entries: Sequence[object]) -> dict[str, tuple[str, ...]]:
    """Entry id -> its ``candidates`` list, for entries that carry one (a slice's)."""
    return {
        str(item["id"]): tuple(str(name) for name in item["candidates"])
        for item in raw_entries
        if isinstance(item, dict) and "id" in item and isinstance(item.get("candidates"), list)
    }


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", model.lower()).strip("-") or "model"


def _check_issue46_flags(args: argparse.Namespace, lfm_settings: Mapping[str, object]) -> None:
    """Refuse flag combinations before anything is loaded or started."""
    if args.predictions and (args.final or args.acceptance):
        raise MeasureError(
            EXIT_USER,
            "--predictions keeps per-entry rows, which are for validation, never the test or"
            " held-out side",
            "drop --predictions; the metrics are still in the results file",
        )
    if args.live and args.ground_snapshot:
        raise MeasureError(EXIT_USER, "--live and --ground-snapshot are two different groundings")
    if args.top_logprobs < 0:
        raise MeasureError(EXIT_USER, "--top-logprobs must be 0 or more")
    if args.max_logprobs is not None and args.top_logprobs > args.max_logprobs:
        raise MeasureError(
            EXIT_USER,
            f"--top-logprobs {args.top_logprobs} is above --max-logprobs {args.max_logprobs}",
        )
    if args.ctx is not None:
        try:
            check_ctx(args.ctx)
        except RuntimeUnavailable as exc:
            raise MeasureError(EXIT_USER, f"--ctx: {exc}") from exc
    if args.scorer != SCORER_SERVED:
        return
    needed = scorer_labels_needed()
    if args.max_logprobs is None or args.max_logprobs < needed:
        raise MeasureError(
            EXIT_USER,
            f"a served scorer asks for {needed} log-probabilities ({needed - scorer.TOP_MARGIN}"
            f" labels + {scorer.TOP_MARGIN}); --max-logprobs must say the engine allows that",
            f"start vLLM with --max-logprobs {needed} or more and pass the same value, or use"
            " --scorer in-process",
        )
    if str(lfm_settings.get("mode") or MANAGED) == MANAGED:
        raise MeasureError(
            EXIT_ENV,
            "nvsh's managed launcher cannot pass vLLM --max-logprobs (default 20), so a served"
            " scorer would lose labels",
            "attach to a vLLM started with --max-logprobs ([tiers.lfm] mode = attach), or use"
            " --scorer in-process",
        )


def _requests_note(args: argparse.Namespace) -> str:
    thinking = (
        "thinking not set (no chat_template_kwargs sent)"
        if args.enable_thinking is None
        else "chat_template_kwargs enable_thinking=" + ("true" if args.enable_thinking else "false")
    )
    if args.scorer:
        return thinking + " (the scorer renders its own prompts, thinking off)"
    logprobs = (
        f"log-probabilities: top {args.top_logprobs} per generated token"
        if args.top_logprobs
        else "no log-probabilities asked for"
    )
    return f"{thinking}; {logprobs}"


def _mode_note(args: argparse.Namespace) -> str:
    engine_max = (
        f"max-logprobs {args.max_logprobs} (operator-supplied)"
        if args.max_logprobs is not None
        else "max-logprobs not given (engine default)"
    )
    if args.scorer:
        return (
            f"candidate scorer ({args.scorer}) through scorer.py, {engine_max};"
            f" asks for {scorer_labels_needed()} per request"
        )
    return f"generative (LfmTier through nvsh.tiers.bench), {engine_max}"


def run(argv: Sequence[str], seams: Seams) -> int:
    if list(argv[:1]) == ["snapshot"]:
        return run_snapshot(argv[1:], seams)
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
    workdir = tempfile.TemporaryDirectory(prefix="nvsh-measure-")
    try:
        return _run(args, argv, raw, split_path, seams, Path(workdir.name))
    finally:
        workdir.cleanup()


def _run(
    args: argparse.Namespace,
    argv: Sequence[str],
    raw: dict,
    split_path: Path,
    seams: Seams,
    workdir: Path,
) -> int:
    corpus_path = split_path
    if args.slice == SLICE_MISSING:
        raw, corpus_path = _sliced(raw, split_path, workdir)
    loaded = tier_bench.load_corpus(corpus_path)
    if not loaded.entries:
        raise MeasureError(EXIT_USER, f"{corpus_path.name} has no valid entries")
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
    _check_issue46_flags(args, lfm_settings)
    if args.ctx is not None:
        lfm_settings["ctx"] = args.ctx

    snapshot_note = ""
    runtime_platform = seams.detect_platform()
    if args.live:
        router_platform, runner = runtime_platform, seams.run
        grounding = "live (this machine)"
    else:
        world, grounding = _world(split_path, args.world)
        router_platform = tier_bench.world_platform(world)
        runner = tier_bench.world_runner(world)
    if args.ground_snapshot:
        snapshot_path = Path(args.ground_snapshot)
        snapshot, digest = load_snapshot(snapshot_path)
        runner = snapshot_runner(snapshot)
        shown = home_relative(str(snapshot_path))
        grounding = f"fixed snapshot `{shown}` (platform: {grounding})"
        snapshot_note = (
            f"`{shown}` sha256 `{digest}` ({len(snapshot['services'])} services,"
            f" {len(snapshot['containers'])} containers; created {snapshot['created']})"
        )

    plan = RunPlan(
        entries=loaded.entries,
        problems=loaded.problems,
        sources=sources,
        split=split_path.stem if args.slice == SLICE_FULL else corpus_path.stem,
        tiers=tiers,
        lfm_settings=lfm_settings,
        router_platform=router_platform,
        runtime_platform=runtime_platform,
        runner=runner,
        grounding=grounding,
        offered=offered_candidates(raw["entries"]),
        request_options=RequestOptions(
            enable_thinking=args.enable_thinking, top_logprobs=args.top_logprobs
        ),
        scorer_kind=args.scorer or "",
    )
    for model, revision in zip(args.model, args.revision):  # refuse before any run starts
        if args.scorer:
            scorer_revision_status(plan, model, revision)
        else:
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
        command=home_relative(shlex.join([_SCRIPT_NAME, *argv])),
        split_path=home_relative(str(split_path)),
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
        serving=serving_record(lfm_settings),
        requests=_requests_note(args),
        decision_mode=_mode_note(args),
        slice_note=(
            "full split"
            if args.slice == SLICE_FULL
            else f"{SLICE_MISSING} (eval_slices.py: each operation entry with its gold operation"
            " left out of the offered candidates, expected to escalate)"
        ),
        snapshot_note=snapshot_note,
        scorer_run=bool(args.scorer),
    )
    keep = Path(args.predictions) if args.predictions else None
    for index, record in enumerate(records, start=1):
        if record.failure:
            continue
        name = f"{args.label}-{index}-{_slug(record.model)}"
        path = (keep or workdir) / f"{name}.predictions.jsonl"
        record.metrics = score_predictions(record.predictions, path)
        if keep is not None:
            (keep / f"{name}.metrics.json").write_text(
                json.dumps(record.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    if args.details:
        with open(args.details, "w", encoding="utf-8") as handle:
            for record in records:
                for row in detail_rows(record.model, record.result, loaded.entries):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    text = render_markdown(prov, records)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(render_metrics(records) if args.scorer else render_table(records))
    print(f"wrote {out}")
    think = sum(record.notes.get(THINK_BLOCKS, 0) for record in records)
    if think:
        print(
            f"error: {think} model replies carried a non-empty think block (must be 0);"
            " the run is recorded but is not a valid measurement",
            file=sys.stderr,
        )
        print("hint: serve with thinking off (--enable-thinking false)", file=sys.stderr)
        return EXIT_ENV
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
