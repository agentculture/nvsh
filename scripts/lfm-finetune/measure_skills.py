#!/usr/bin/env python3
"""Skill-routing measurement against the Jetson skills eval set (issue 39, t9).

A development-machine tool for the Tier 2 (LFM2.5) fine-tune's method
validation. It sends every eval from ``scripts/lfm-finetune/jetson_skills.py
build``'s ``test.jsonl`` to a *running* OpenAI-compatible chat endpoint with
that build's ``tools.json`` (one no-argument function tool per skill,
38 at the time this script was written) and ``tool_choice: "auto"``, then
scores which skill the model called against the eval's ``expected_skill``.

By default this script never launches or stops a model server: serve it the
same way nvsh's own Tier 2 launcher does -- vLLM with
``--tool-call-parser lfm2`` (see ``nvsh/tiers/runtime_docker.py`` and
``docs/tier2.md``) -- for example::

    docker run --rm --gpus all -p 127.0.0.1:8000:8000 \\
        vllm/vllm-openai@sha256:<pinned digest> \\
        --model <repo-or-path> --tool-call-parser lfm2 \\
        --gpu-memory-utilization 0.08 --max-model-len 4096

then point this script at it::

    NVSH_SKILLS_URL=http://127.0.0.1:8000/v1 \\
    NVSH_SKILLS_MODEL=<model id> \\
    python scripts/lfm-finetune/measure_skills.py \\
        --tools /tmp/jetson-skills-out/tools.json \\
        --test /tmp/jetson-skills-out/test.jsonl \\
        --manifest /tmp/jetson-skills-out/manifest.json \\
        --label stock

``--url`` / ``--model`` override the environment variables; neither is read
from a config file, and the endpoint is never launched, stopped or
configured by this script -- only called. The URL is restricted to
``127.0.0.1`` / ``localhost`` / ``::1`` (and must carry no userinfo/
credentials) so an eval prompt (which may carry whatever a Jetson operator
typed) is never sent anywhere but the box the operator is measuring.

Pass ``--launch`` to have this script serve the model itself, the same way
nvsh's daemon does: it reads ``[tiers.lfm]`` from the operator's nvsh config
(``nvsh.config.load``), overrides only ``model`` from ``--model``, builds the
runtime with :func:`nvsh.tiers.runtime_docker.build_runtime` (the same
launcher ``nvsh/tiers/manager.py`` uses for Tier 2), calls ``ensure()`` to
start it, measures against its base URL, and always stops it again in a
``finally``. As with ``measure.py``, ``--launch`` refuses to start while a
container named ``nvsh-tier2-<uid>`` is already running, and it never stops
one it did not start. Without ``--launch``, behaviour is unchanged: an
endpoint from the environment, localhost only.

Neither the endpoint URL nor the raw command line is ever written into the
results file: only the fact that a local endpoint was used, and the command
line with any ``--url`` value replaced by ``<local endpoint>``. Recorded
text is also passed through :func:`nvsh.redact.redact` before it is written.

A **tuned** run -- any ``--label`` other than ``stock``, or ``--tuned`` --
is refused unless ``--margin`` is given: the improvement the run is expected
to show, decided *before* the numbers exist (e.g. ``"+15 points overall"``).
That margin is written as the first section of the results file, ahead of
any number, so the claim cannot be quietly fitted to the result afterwards.

Scoring, per eval:

- ``correct``    -- exactly one tool call, naming the expected skill's tool.
- ``wrong_skill``-- exactly one tool call, naming a different skill's tool.
- ``no_call``    -- no tool call at all.
- ``several_calls`` -- more than one tool call.

Reported columns: overall correct/total, skill-named-in-prompt correct/total
(``names_skill: true`` in the eval -- these test copying a name out of the
prompt, not routing), not-named correct/total, a per-repo split (``device``,
``bsp``), and warm latency median/p95 across all calls.

Results are written to ``docs/benchmarks/<YYYY-MM-DD>-skills-<label>.md``
(override with ``--out``; tests write to a temp path) with the command line,
date, endpoint model id and ``--model-revision``, and, when ``--manifest``
points at a ``jetson_skills.py build`` manifest, the commit of each NVIDIA
repo it lists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess  # nosec B404 - fixed argv lists, never a shell
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))  # runnable from any directory

from nvsh import config as nvsh_config  # noqa: E402
from nvsh.platform._model import Platform  # noqa: E402
from nvsh.redact import redact  # noqa: E402
from nvsh.tiers.runtime import Runtime, RuntimeUnavailable  # noqa: E402
from nvsh.tiers.runtime_docker import build_runtime, container_name  # noqa: E402

#: Environment variables the endpoint is read from; ``--url`` / ``--model``
#: override them. No config file is read.
URL_ENV_VAR = "NVSH_SKILLS_URL"
MODEL_ENV_VAR = "NVSH_SKILLS_MODEL"

#: A loopback address only -- never a real default endpoint.
DEFAULT_URL = "http://127.0.0.1:8000/v1"

_HOST_ACCEPT: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

DEFAULT_TIMEOUT = 30.0

EXIT_OK = 0
EXIT_USER = 1
EXIT_ENV = 2

#: Never the real endpoint: this is what stands in for it in anything
#: written to disk (the results file, the recorded command line).
LOCAL_ENDPOINT_LABEL = "<local endpoint>"

STOCK_LABEL = "stock"

OUTCOME_CORRECT = "correct"
OUTCOME_NO_CALL = "no_call"
OUTCOME_WRONG_SKILL = "wrong_skill"
OUTCOME_SEVERAL_CALLS = "several_calls"
OUTCOMES = (OUTCOME_CORRECT, OUTCOME_WRONG_SKILL, OUTCOME_NO_CALL, OUTCOME_SEVERAL_CALLS)


# ---------------------------------------------------------------------------
# localhost enforcement
# ---------------------------------------------------------------------------


def require_localhost(base_url: str) -> None:
    """Raise ``ValueError`` unless *base_url* is ``http://`` to this machine.

    Also refuses any URL carrying userinfo (a username and/or password):
    such a URL is a credential leak waiting to happen the moment it is
    logged, echoed on a command line, or written to a results file.
    """
    parsed = urlsplit(base_url)
    if parsed.scheme != "http":
        raise ValueError(f"only http:// scheme accepted (got {parsed.scheme!r})")
    if parsed.username or parsed.password:
        raise ValueError("URL must not contain credentials (userinfo)")
    if parsed.hostname not in _HOST_ACCEPT:
        raise ValueError(f"host must be 127.0.0.1, ::1 or localhost (got {parsed.hostname!r})")


# ---------------------------------------------------------------------------
# inputs: tools.json + test.jsonl (from jetson_skills.py build)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolRecord:
    skill: str
    repo: str
    tool: dict[str, Any]

    @property
    def name(self) -> str:
        return self.tool["function"]["name"]


def load_tools(path: Path) -> list[ToolRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of {{skill, repo, tool}} records")
    return [ToolRecord(skill=r["skill"], repo=r["repo"], tool=r["tool"]) for r in raw]


def load_evals(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# endpoint call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallResult:
    tool_names: tuple[str, ...]
    elapsed: float  # seconds


def call_endpoint(
    base_url: str,
    model: str,
    tools: list[dict[str, Any]],
    text: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> CallResult:
    """One ``POST {base_url}/chat/completions`` with *tools*, ``tool_choice: "auto"``,
    ``temperature: 0``. stdlib ``urllib`` only. Raises ``RuntimeError`` on any
    transport or protocol failure."""
    require_localhost(base_url)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"request to {url} failed: {exc}") from exc
    elapsed = time.monotonic() - start
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(f"{url}: reply is not valid JSON: {exc}") from exc
    return CallResult(tool_names=tuple(_tool_names_from_payload(payload)), elapsed=elapsed)


def _tool_names_from_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return []
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def classify(tool_names: tuple[str, ...], expected_tool_name: str) -> str:
    """One of the four outcomes in :data:`OUTCOMES` for a single eval."""
    if len(tool_names) == 0:
        return OUTCOME_NO_CALL
    if len(tool_names) > 1:
        return OUTCOME_SEVERAL_CALLS
    return OUTCOME_CORRECT if tool_names[0] == expected_tool_name else OUTCOME_WRONG_SKILL


@dataclass(frozen=True)
class EvalResult:
    id: str
    repo: str
    skill: str
    expected_skill: str
    names_skill: bool
    outcome: str
    elapsed: float  # seconds


def run_measurement(
    base_url: str,
    model: str,
    tools: list[ToolRecord],
    evals: list[dict[str, Any]],
    timeout: float = DEFAULT_TIMEOUT,
) -> list[EvalResult]:
    """Calls the endpoint once per eval, in order, and scores each reply.

    Raises ``ValueError`` up front if any eval names an ``expected_skill``
    absent from *tools* -- a data-integrity problem, not a model failure.
    """
    skill_to_tool_name = {t.skill: t.name for t in tools}
    for record in evals:
        expected = record.get("expected_skill")
        if expected not in skill_to_tool_name:
            raise ValueError(
                f"eval {record.get('id')!r} expects skill {expected!r}, " "which is not in --tools"
            )
    tool_schemas = [t.tool for t in tools]
    results: list[EvalResult] = []
    for record in evals:
        expected_skill = record["expected_skill"]
        expected_tool_name = skill_to_tool_name[expected_skill]
        call = call_endpoint(base_url, model, tool_schemas, record["text"], timeout=timeout)
        outcome = classify(call.tool_names, expected_tool_name)
        results.append(
            EvalResult(
                id=str(record.get("id", "")),
                repo=str(record.get("repo", "")),
                skill=str(record.get("skill", "")),
                expected_skill=expected_skill,
                names_skill=bool(record.get("names_skill", False)),
                outcome=outcome,
                elapsed=call.elapsed,
            )
        )
    return results


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bucket:
    correct: int
    total: int

    @property
    def pct(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, int]:
        return {"correct": self.correct, "total": self.total}


def _bucket(results: list[EvalResult], keep) -> Bucket:
    subset = [r for r in results if keep(r)]
    correct = sum(1 for r in subset if r.outcome == OUTCOME_CORRECT)
    return Bucket(correct=correct, total=len(subset))


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


@dataclass(frozen=True)
class Aggregate:
    overall: Bucket
    named_in_prompt: Bucket
    not_named: Bucket
    per_repo: dict[str, Bucket]
    outcome_counts: dict[str, int]
    latency_median_ms: float
    latency_p95_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.as_dict(),
            "named_in_prompt": self.named_in_prompt.as_dict(),
            "not_named": self.not_named.as_dict(),
            "per_repo": {repo: bucket.as_dict() for repo, bucket in self.per_repo.items()},
            "outcome_counts": dict(self.outcome_counts),
            "latency_median_ms": self.latency_median_ms,
            "latency_p95_ms": self.latency_p95_ms,
        }


def aggregate(results: list[EvalResult]) -> Aggregate:
    per_repo = {
        repo: _bucket(results, lambda r, repo=repo: r.repo == repo)
        for repo in sorted({r.repo for r in results})
    }
    latencies_ms = sorted(r.elapsed * 1000.0 for r in results)
    return Aggregate(
        overall=_bucket(results, lambda _r: True),
        named_in_prompt=_bucket(results, lambda r: r.names_skill),
        not_named=_bucket(results, lambda r: not r.names_skill),
        per_repo=per_repo,
        outcome_counts=dict(Counter(r.outcome for r in results)),
        latency_median_ms=statistics.median(latencies_ms) if latencies_ms else 0.0,
        latency_p95_ms=_percentile(latencies_ms, 0.95),
    )


# ---------------------------------------------------------------------------
# manifest provenance (optional)
# ---------------------------------------------------------------------------


def load_manifest_provenance(path: Path | None) -> list[dict[str, str]]:
    """``[{"repo", "url", "commit"}, ...]`` from a ``jetson_skills.py build``
    manifest.json, or ``[]`` when *path* is None."""
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    repositories = data.get("repositories", [])
    return [
        {"repo": r["repo"], "url": r["url"], "commit": r["commit"]}
        for r in repositories
        if isinstance(r, dict)
    ]


# ---------------------------------------------------------------------------
# results file
# ---------------------------------------------------------------------------


def is_tuned_run(label: str, tuned_flag: bool) -> bool:
    return tuned_flag or label != STOCK_LABEL


def _fmt_bucket(bucket: Bucket, label: str) -> str:
    return f"| {label} | {bucket.correct} of {bucket.total} ({bucket.pct:.0f}%) |"


def redact_command_line(argv: list[str]) -> str:
    """The command line as run, with any ``--url`` value replaced.

    Never write the endpoint URL (which may carry credentials in its
    userinfo) into a committed or long-lived file. ``--url VALUE`` and
    ``--url=VALUE`` are both stripped.
    """
    redacted: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            redacted.append(LOCAL_ENDPOINT_LABEL)
            skip_next = False
            continue
        if token == "--url":
            redacted.append(token)
            skip_next = True
            continue
        if token.startswith("--url="):
            redacted.append(f"--url={LOCAL_ENDPOINT_LABEL}")
            continue
        redacted.append(token)
    line = " ".join(["measure_skills.py", *redacted])
    # Reports are committed: write the home directory as $HOME, never
    # /home/<user>/, which is not portable.
    home = str(Path.home())
    return line.replace(home + "/", "$HOME/") if home not in ("", "/") else line


def render_results(
    *,
    label: str,
    model: str,
    model_revision: str | None,
    command_line: str,
    when: str,
    margin: str | None,
    manifest_provenance: list[dict[str, str]],
    aggregated: Aggregate,
) -> str:
    lines: list[str] = [f"# Jetson skill-routing measurement -- {label}, {when}", ""]

    # The margin comes first, ahead of any number, so a tuned run's claim is
    # on record before the results below can be read.
    if margin is not None:
        lines += ["## Margin claimed before scoring", "", margin, ""]

    lines += [
        "## Run",
        "",
        f"- command: `{command_line}`",
        f"- date: {when}",
        f"- endpoint: `{LOCAL_ENDPOINT_LABEL}` (a local endpoint was used; " "never recorded)",
        f"- model: `{model}`",
    ]
    if model_revision:
        lines.append(f"- model revision: `{model_revision}`")
    for prov in manifest_provenance:
        lines.append(f"- {prov['repo']}: <{prov['url']}> at commit `{prov['commit']}`")
    lines.append("")

    lines += [
        "## Results",
        "",
        "| Split | Correct / total |",
        "|---|---|",
        _fmt_bucket(aggregated.overall, "overall"),
        _fmt_bucket(aggregated.named_in_prompt, "skill named in prompt"),
        _fmt_bucket(aggregated.not_named, "skill not named"),
    ]
    for repo, bucket in aggregated.per_repo.items():
        lines.append(_fmt_bucket(bucket, f"repo: {repo}"))
    lines += [
        "",
        "| Outcome | Count |",
        "|---|---|",
    ]
    for outcome in OUTCOMES:
        lines.append(f"| {outcome} | {aggregated.outcome_counts.get(outcome, 0)} |")
    lines += [
        "",
        "| Latency | ms |",
        "|---|---|",
        f"| median | {aggregated.latency_median_ms:.0f} |",
        f"| p95 | {aggregated.latency_p95_ms:.0f} |",
        "",
    ]
    return "\n".join(lines)


def default_out_path(label: str, when: str) -> Path:
    return _REPO_ROOT / "docs" / "benchmarks" / f"{when}-skills-{label}.md"


# ---------------------------------------------------------------------------
# --launch: serve the model through nvsh's own Tier 2 launcher
# ---------------------------------------------------------------------------

_GUARD_TIMEOUT = 30.0


def default_run_docker(argv: list[str], timeout: float) -> tuple[int, str]:  # pragma: no cover
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


def default_detect_platform() -> Platform:  # pragma: no cover - reads the real machine
    from nvsh import platform as platform_mod

    return platform_mod.detect()


def default_build_runtime(
    lfm_settings: Mapping[str, object],
    platform: Platform,
    *,
    floor_check: Callable[[], object],
) -> Runtime:
    """The runtime ``nvsh.tiers.manager.TierManager._lfm`` builds for Tier 2."""
    return build_runtime(lfm_settings, platform, floor_check=floor_check)


@dataclass
class LaunchSeams:
    """Everything ``--launch`` touches on the machine, injectable for tests."""

    run_docker: Callable[[list[str], float], tuple[int, str]] = default_run_docker
    detect_platform: Callable[[], Platform] = default_detect_platform
    load_config: Callable[[Path | None], object] = nvsh_config.load
    build_runtime: Callable[..., Runtime] = default_build_runtime
    uid: Callable[[], int] = os.getuid


def guard_container_not_running(seams: LaunchSeams) -> str | None:
    """``None`` when it is safe to launch; otherwise an error message.

    Refuses (never stops) while a container named ``nvsh-tier2-<uid>`` is
    already running -- the same guard ``measure.py`` uses.
    """
    uid = int(seams.uid())
    name = container_name(uid)
    code, output = seams.run_docker(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"], _GUARD_TIMEOUT
    )
    if code != 0:
        return f"cannot check whether {name} is running: docker ps exited {code}"
    if name in output.split():
        return f"container {name} is already running; refusing to start (it was not stopped)"
    return None


def build_lfm_settings(cfg: object, model: str) -> dict[str, object]:
    """``[tiers.lfm]`` from *cfg* with only ``model`` overridden from ``--model``."""
    tiers = dict(getattr(cfg, "tiers", {}) or {})
    lfm = tiers.get("lfm")
    settings = dict(lfm) if isinstance(lfm, Mapping) else {}
    settings["model"] = model
    return settings


def memory_floor_mb(cfg: object) -> int:
    tiers = dict(getattr(cfg, "tiers", {}) or {})
    return int(tiers.get("memory_floor_mb", 1024))  # type: ignore[arg-type]


def launch_runtime(model: str, config_path: Path | None, seams: LaunchSeams) -> Runtime:
    """Build (never starts) the Tier 2 runtime ``--launch`` will call ``ensure()`` on."""
    from nvsh.tiers.memfloor import check_floor

    cfg = seams.load_config(config_path)
    lfm_settings = build_lfm_settings(cfg, model)
    floor_mb = memory_floor_mb(cfg)

    def floor_check():
        return check_floor(floor_mb)

    platform = seams.detect_platform()
    return seams.build_runtime(lfm_settings, platform, floor_check=floor_check)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure_skills.py",
        description=(
            "Score an OpenAI-compatible chat endpoint's skill routing against "
            "the Jetson skills eval set (scripts/lfm-finetune/jetson_skills.py build)."
        ),
        epilog=(
            "This script never launches or stops a model server. Serve the model "
            "the same way nvsh's own Tier 2 launcher does -- vLLM with "
            "--tool-call-parser lfm2 (see nvsh/tiers/runtime_docker.py and "
            f"docs/tier2.md) -- then set {URL_ENV_VAR} / {MODEL_ENV_VAR} (or pass "
            "--url / --model) to point this script at it. The endpoint must be "
            "127.0.0.1, ::1 or localhost."
        ),
    )
    parser.add_argument(
        "--tools", required=True, type=Path, help="tools.json from jetson_skills.py build"
    )
    parser.add_argument(
        "--test", required=True, type=Path, help="test.jsonl from jetson_skills.py build"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get(URL_ENV_VAR, DEFAULT_URL),
        help=f"endpoint base URL ({URL_ENV_VAR}; default {DEFAULT_URL}; localhost only)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(MODEL_ENV_VAR),
        help=f"model id ({MODEL_ENV_VAR}); required",
    )
    parser.add_argument(
        "--label",
        default=STOCK_LABEL,
        help=f"free-text run label, e.g. {STOCK_LABEL} or tuned (default: {STOCK_LABEL})",
    )
    parser.add_argument(
        "--tuned",
        action="store_true",
        help="treat this run as tuned even if --label is left at its default",
    )
    parser.add_argument(
        "--margin",
        default=None,
        help=(
            "the improvement this run is expected to show, decided before scoring "
            "(e.g. '+15 points overall'); required for any tuned run"
        ),
    )
    parser.add_argument(
        "--model-revision", default=None, help="recorded verbatim in the results file"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest.json from jetson_skills.py build, for the source repos' commits",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results file path (default docs/benchmarks/<date>-skills-<label>.md)",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--limit", type=int, default=None, help="only score the first N evals (for a quick check)"
    )
    parser.add_argument(
        "--json", action="store_true", help="also print the aggregated result as JSON to stdout"
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help=(
            "serve --model through nvsh's own Tier 2 launcher ([tiers.lfm] settings, "
            "same build_runtime as nvsh/tiers/manager.py) instead of calling --url; "
            "refuses to start while nvsh-tier2-<uid> is already running, and always "
            "stops what it started"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="nvsh config.toml to read [tiers.lfm] from with --launch (default: XDG path)",
    )
    return parser


def main(argv: list[str] | None = None, *, launch_seams: LaunchSeams | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    launch_seams = launch_seams or LaunchSeams()

    if not args.model:
        parser.error(f"--model is required ({MODEL_ENV_VAR} is not set)")

    tuned = is_tuned_run(args.label, args.tuned)
    if tuned and not args.margin:
        parser.error(
            "--margin is required before a tuned run (any --label other than "
            f"'{STOCK_LABEL}', or --tuned) can be scored"
        )

    if not args.launch:
        try:
            require_localhost(args.url)
        except ValueError as exc:
            parser.error(str(exc))

    tools = load_tools(args.tools)
    evals = load_evals(args.test)
    if args.limit is not None:
        evals = evals[: args.limit]

    runtime: Runtime | None = None
    if args.launch:
        problem = guard_container_not_running(launch_seams)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return EXIT_ENV
        config_path = Path(args.config) if args.config else None
        try:
            runtime = launch_runtime(args.model, config_path, launch_seams)
        except (OSError, ValueError) as exc:
            print(f"error: cannot load nvsh config: {exc}", file=sys.stderr)
            return EXIT_USER

    try:
        if runtime is not None:
            try:
                base_url = runtime.ensure()
            except RuntimeUnavailable as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_ENV
            try:
                require_localhost(base_url)
            except ValueError as exc:
                print(
                    f"error: Tier 2 runtime returned an unusable endpoint: {exc}", file=sys.stderr
                )
                return EXIT_ENV
        else:
            base_url = args.url

        try:
            results = run_measurement(base_url, args.model, tools, evals, timeout=args.timeout)
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USER
    finally:
        if runtime is not None:
            runtime.stop()

    aggregated = aggregate(results)
    when = date.today().isoformat()
    out_path = args.out if args.out is not None else default_out_path(args.label, when)
    provenance = load_manifest_provenance(args.manifest)
    raw_argv = argv if argv is not None else sys.argv[1:]
    command_line = redact_command_line(raw_argv)

    rendered = render_results(
        label=args.label,
        model=args.model,
        model_revision=args.model_revision,
        command_line=command_line,
        when=when,
        margin=args.margin,
        manifest_provenance=provenance,
        aggregated=aggregated,
    )
    rendered = redact(rendered.encode("utf-8")).decode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"wrote {out_path}")

    if args.json:
        payload = aggregated.as_dict()
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(payload, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
