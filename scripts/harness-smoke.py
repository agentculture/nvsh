#!/usr/bin/env python3
"""Per-harness smoke check — prove all four harness configs are correct.

All four harnesses (Claude Code, Pi/associate, colleague, Qwen Code) are live
for interactive use over the *same* clone at all times, independent of what
``culture.yaml`` declares as the mesh resident. So all four configs must be
correct at all times, and nothing else in this repo checks more than the one
config the declared backend uses. That is what this script exists for: it
exercises **all four**, and fails when **any one** of them is broken.

The invocations come from ``docs/harness-invocations.yaml`` — the single source
of truth — and are run ``binary`` + ``args`` **verbatim**, never re-typed here.

Three stages, selected with ``--stage``:

``config`` (offline, always runnable — the load-bearing stage)
    Per harness: the config file(s) that harness resolves are present and
    non-empty, the harness's *skills discovery* wiring points at the one
    canonical ``.claude/skills`` tree, and every resolved prompt path is
    registered in the vendored backend-fingerprint registry. Plus one global
    check that no bare ``AGENTS.md`` has appeared (this template deliberately
    ships none — it would shadow the colleague/Pi prompt cascade).

``live`` (needs the four harness binaries on PATH)
    Runs each harness's documented invocation verbatim in the repo and asserts
    the answer proves that harness resolved this clone's config. Not available
    on a stock CI runner — it reports **SKIPPED**, never a pass.

``toolchain`` (needs ``steward`` and ``guild``, or ``uvx`` to fetch them)
    - ``steward doctor --scope self`` over this repo, which must report no
      findings beyond the known-accepted, pre-existing portability finding in
      the vendored ``recall``/``remember`` skills (reported as WAIVED, loudly,
      never silently folded into a pass).
    - ``guild create --harness X --json`` dry-runs for all four harnesses,
      checking each names the prompt file it says it writes.

Exit status is 0 unless some check FAILED, 1 on a failure or a bad argument.
Exit 2 is reserved for environment/system failures and is never used for user
input. A skipped check is never counted as a pass: it is printed as SKIP,
listed again under "NOT VERIFIED", and ``--require <stage>`` turns a skip in
that stage into a failure.

Output contract (the repo's, shared with the CLI): **results on stdout**
(the status table, the counts summary, or pure JSON under ``--json``);
**diagnostics on stderr** (the FAILED / NOT VERIFIED / WAIVED expansions and
the ``--require`` error). Redirecting stdout therefore captures outcomes
without diagnostics mixed in, and ``--json`` stdout stays parseable even when
checks fail.

Usage:
    harness-smoke.py [--repo DIR] [--stage config|live|toolchain|all]
                     [--require STAGE] [--json] [--timeout SECONDS]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess  # nosec B404 - runs fixed, repo-local tool invocations
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

INVOCATIONS_REL = Path("docs") / "harness-invocations.yaml"
FINGERPRINTS_REL = (
    Path(".claude") / "skills" / "agent-config" / "data" / "backend-fingerprints.yaml"
)
CANONICAL_SKILLS_REL = Path(".claude") / "skills"

#: The four harnesses this template supports, in report order. Keys must match
#: ``harnesses:`` in docs/harness-invocations.yaml exactly — a mismatch in
#: either direction is a hard failure, so a harness can never be silently
#: dropped from the smoke check by editing only one of the two files.
HARNESSES = ("claude", "pi", "qwen", "colleague")

#: harness -> the Culture backend it rides. Four harnesses occupy only three
#: backend names: Qwen Code rides ``acp`` and associate/Pi rides ``colleague``.
#: A harness's config files must be registered under ITS OWN backend in the
#: fingerprint registry, not merely somewhere in it.
HARNESS_BACKEND = {
    "claude": "claude",
    "pi": "colleague",
    "qwen": "acp",
    "colleague": "colleague",
}

#: harness -> (``guild create --harness`` name, the prompt file guild seeds).
#: The seeded file is additionally asserted to be one of that harness's
#: ``resolves`` entries in the YAML, so this table cannot drift off the source.
GUILD_HARNESS = {
    "claude": ("claude", "CLAUDE.md"),
    "pi": ("associate", ".pi/SYSTEM.md"),
    "qwen": ("qwen", "QWEN.md"),
    "colleague": ("colleague", "AGENTS.colleague.md"),
}

#: steward doctor findings we knowingly carry. These are the five pre-existing
#: ``~/.eidetic/memory`` references in two byte-verbatim vendored skills; they
#: are upstream's to fix (never patch a verbatim vendored copy), so they are
#: WAIVED — reported every run, never silently dropped. Any finding outside
#: this allowlist fails the check.
STEWARD_WAIVED_PORTABILITY_PATHS = frozenset(
    {
        ".claude/skills/recall/SKILL.md",
        ".claude/skills/remember/SKILL.md",
    }
)

PASS, FAIL, SKIP, WAIVED = "PASS", "FAIL", "SKIP", "WAIVED"


class Result:
    """One check outcome."""

    def __init__(self, stage: str, name: str, status: str, detail: str) -> None:
        self.stage = stage
        self.name = name
        self.status = status
        self.detail = detail

    @property
    def label(self) -> str:
        return f"{self.stage}/{self.name}"

    def as_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "check": self.name,
            "status": self.status,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def load_invocations(repo: Path) -> dict[str, Any]:
    """Read the single source of truth for the four harness invocations."""
    path = repo / INVOCATIONS_REL
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    harnesses = data.get("harnesses") or {}
    if set(harnesses) != set(HARNESSES):
        raise SystemExit(
            f"error: {INVOCATIONS_REL} lists harnesses {sorted(harnesses)}, "
            f"but this check knows {sorted(HARNESSES)}. Teach the check about "
            "the change rather than letting a harness go unchecked."
        )
    return harnesses


def registry_prompts_by_backend(repo: Path) -> dict[str, tuple[str, ...]]:
    """``backend -> accepted prompt paths`` from the vendored registry.

    Returns an empty mapping when the registry is absent or unreadable — the
    ``fingerprint-registry`` config check reports that, and every harness then
    fails on "not registered" rather than the run dying with a traceback.
    """
    path = repo / FINGERPRINTS_REL
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    mapping: dict[str, tuple[str, ...]] = {}
    for backend, entry in (data.get("backends") or {}).items():
        prompt = (entry or {}).get("prompt")
        if isinstance(prompt, str):
            mapping[str(backend)] = (prompt,)
        elif isinstance(prompt, list):
            mapping[str(backend)] = tuple(str(p) for p in prompt)
    return mapping


# --------------------------------------------------------------------------
# stage: config
# --------------------------------------------------------------------------


def _check_relative_skills_symlink(repo: Path, rel: str) -> list[str]:
    """A harness skills dir must be a *relative* symlink onto the one kit."""
    problems: list[str] = []
    link = repo / rel
    if not link.is_symlink():
        problems.append(f"{rel} must be a symlink onto {CANONICAL_SKILLS_REL}")
        return problems
    target = os.readlink(link)
    if os.path.isabs(target):
        problems.append(f"{rel} points at an absolute target {target!r}; must be relative")
    canonical = repo / CANONICAL_SKILLS_REL
    if not link.exists() or link.resolve() != canonical.resolve():
        problems.append(f"{rel} does not resolve to {CANONICAL_SKILLS_REL}")
    return problems


def _check_canonical_skill_tree(repo: Path) -> list[str]:
    tree = repo / CANONICAL_SKILLS_REL
    if not tree.is_dir():
        return [f"{CANONICAL_SKILLS_REL} is missing"]
    if not any(tree.glob("*/SKILL.md")):
        return [f"{CANONICAL_SKILLS_REL} holds no <name>/SKILL.md"]
    return []


#: harness -> extra, harness-specific skills-discovery wiring checks.
#:
#: colleague deliberately checks only that `.colleague/skills` is wired, NOT
#: that colleague *loads* those skills: it currently does not, and that is an
#: upstream blocker (agentculture/colleague#494 — configdir.collect_files skips
#: directories, and layers._within rejects symlinks escaping .colleague/). What
#: colleague does resolve today, and what the live stage asserts, is
#: AGENTS.colleague.md.
SKILLS_WIRING: dict[str, Callable[[Path], list[str]]] = {
    "claude": _check_canonical_skill_tree,
    "pi": lambda repo: _check_relative_skills_symlink(repo, ".pi/skills"),
    "qwen": lambda repo: _check_relative_skills_symlink(repo, ".qwen/skills"),
    "colleague": lambda repo: _check_relative_skills_symlink(repo, ".colleague/skills"),
}


def stage_config(repo: Path, harnesses: dict[str, Any]) -> list[Result]:
    results: list[Result] = []
    by_backend = registry_prompts_by_backend(repo)

    for name in HARNESSES:
        spec = harnesses[name]
        resolves = [str(p) for p in (spec.get("resolves") or [])]
        problems: list[str] = []

        if not resolves:
            problems.append("the YAML lists no `resolves` paths for this harness")
        for rel in resolves:
            path = repo / rel
            if not path.is_file():
                problems.append(f"{rel} is missing")
            elif not path.read_text(encoding="utf-8").strip():
                problems.append(f"{rel} is empty")
            backend = HARNESS_BACKEND[name]
            if rel not in by_backend.get(backend, ()):
                problems.append(
                    f"{rel} is not registered under backend '{backend}' in {FINGERPRINTS_REL}"
                )

        problems.extend(SKILLS_WIRING[name](repo))

        if problems:
            results.append(Result("config", name, FAIL, "; ".join(problems)))
        else:
            results.append(
                Result(
                    "config",
                    name,
                    PASS,
                    f"resolves {', '.join(resolves)}; skills discovery wired",
                )
            )

    if by_backend:
        total = sum(len(v) for v in by_backend.values())
        results.append(
            Result(
                "config",
                "fingerprint-registry",
                PASS,
                f"{FINGERPRINTS_REL} registers {total} prompt path(s) "
                f"across {len(by_backend)} backend(s)",
            )
        )
    else:
        results.append(
            Result(
                "config",
                "fingerprint-registry",
                FAIL,
                f"{FINGERPRINTS_REL} is missing, unreadable, or registers no prompt paths",
            )
        )

    bare = repo / "AGENTS.md"
    if bare.exists():
        results.append(
            Result(
                "config",
                "no-bare-agents-md",
                FAIL,
                "AGENTS.md exists; this template ships none on purpose — it would "
                "shadow AGENTS.override.md / AGENTS.colleague.md in the Pi and "
                "colleague prompt cascades",
            )
        )
    else:
        results.append(
            Result("config", "no-bare-agents-md", PASS, "no bare AGENTS.md, as intended")
        )

    return results


# --------------------------------------------------------------------------
# stage: live
# --------------------------------------------------------------------------


def _expectation(spec: dict[str, Any]) -> tuple[re.Pattern[str], str]:
    """Derive what the harness's own documented probe must answer.

    The three model probes are yes/no questions whose affirmative answer is
    only reachable from this clone's config; ``colleague agents list`` instead
    prints the file it resolved. Both expectations are derived from the YAML
    entry rather than hard-coded per harness.

    The yes/no probes ask for *exactly one word*, so the pattern is anchored
    to the whole (stripped) answer rather than merely searched for: a reply
    of "no, because ..." contains no bare ``yes``, but a diagnostic line or a
    hedged sentence easily could, and a substring search would read that as a
    pass.
    """
    args = " ".join(str(a) for a in spec.get("args") or [])
    if "yes or no" in args.lower():
        return re.compile(r"^yes[.!]?$", re.IGNORECASE), "exactly 'yes'"
    resolved = [Path(str(p)).name for p in spec.get("resolves") or []]
    pattern = "|".join(re.escape(n) for n in resolved) or r"(?!x)x"
    return re.compile(pattern), f"one of {', '.join(resolved)}"


def _answer_lines(stdout: str) -> list[str]:
    """Non-empty stdout lines, stripped — the harness's actual answer.

    Only stdout is matched against the expectation. stderr is kept for
    failure diagnostics but must never satisfy the assertion: a harness that
    exits 0 while answering "no" on stdout would otherwise pass whenever any
    warning on stderr happened to contain the expected word or filename.
    """
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def stage_live(repo: Path, harnesses: dict[str, Any], timeout: int) -> list[Result]:
    results: list[Result] = []
    for name in HARNESSES:
        spec = harnesses[name]
        binary = str(spec["binary"])
        argv = [binary] + [str(a) for a in spec.get("args") or []]

        if shutil.which(binary) is None:
            results.append(
                Result(
                    "live",
                    name,
                    SKIP,
                    f"{binary!r} is not on PATH — this harness's real invocation was "
                    "NOT run (stock CI runners ship none of the four agent CLIs)",
                )
            )
            continue

        before = _git_status(repo)
        try:
            proc = subprocess.run(  # nosec B603 - argv is read verbatim from the tracked YAML
                argv,
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            results.append(Result("live", name, FAIL, f"{binary} timed out after {timeout}s"))
            continue

        pattern, described = _expectation(spec)
        diagnostic = f"stdout={proc.stdout.strip()[:300]!r} stderr={proc.stderr.strip()[:300]!r}"
        answered = any(pattern.search(line) for line in _answer_lines(proc.stdout))
        if proc.returncode != 0:
            results.append(
                Result("live", name, FAIL, f"{binary} exited {proc.returncode}: {diagnostic}")
            )
        elif not answered:
            results.append(
                Result(
                    "live",
                    name,
                    FAIL,
                    f"{binary} did not answer on stdout with {described}: {diagnostic}",
                )
            )
        elif spec.get("read_only") and _git_status(repo) != before:
            results.append(
                Result("live", name, FAIL, f"{binary} is documented read_only but dirtied the tree")
            )
        else:
            results.append(
                Result("live", name, PASS, f"{' '.join(argv[:3])}… answered with {described}")
            )
    return results


def _git_status(repo: Path) -> str:
    try:
        proc = subprocess.run(  # nosec B603,B607 - fixed git invocation
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return proc.stdout


# --------------------------------------------------------------------------
# stage: toolchain
# --------------------------------------------------------------------------


def _tool_argv(binary: str, dist: str) -> list[str] | None:
    """Prefer an installed CLI; otherwise let uvx fetch the published dist."""
    if shutil.which(binary):
        return [binary]
    if shutil.which("uvx"):
        return ["uvx", "--from", dist, binary]
    if shutil.which("uv"):
        return ["uv", "tool", "run", "--from", dist, binary]
    return None


def _run(argv: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 - argv is assembled from fixed literals above
        argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


_FINDING_PATH = re.compile(r"(?:^|\s)([\w./-]+\.(?:md|yaml|yml|toml|json)):\d+")


def check_steward_doctor(repo: Path, timeout: int) -> Result:
    base = _tool_argv("steward", "steward-cli")
    if base is None:
        return Result(
            "toolchain",
            "steward-doctor",
            SKIP,
            "neither `steward` nor `uvx`/`uv` is available — `steward doctor "
            "--scope self` was NOT run",
        )
    argv = base + ["doctor", "--scope", "self", str(repo), "--json", "--no-write-reports"]
    try:
        proc = _run(argv, repo, timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Result("toolchain", "steward-doctor", SKIP, f"could not run steward doctor: {exc}")
    if not proc.stdout.strip():
        if "PosixPath' and 'list'" in proc.stderr:
            return Result(
                "toolchain",
                "steward-doctor",
                SKIP,
                "the resolved steward predates list-valued `prompt` entries in "
                "backend-fingerprints.yaml (it reads the registry of whatever "
                "checkout it is run from, i.e. this one) and crashed — check NOT "
                "run. It needs a steward release carrying `_agents._prompt_list`; "
                "re-run once one is published.",
            )
        return Result(
            "toolchain",
            "steward-doctor",
            SKIP,
            f"steward doctor unavailable (exit {proc.returncode}): {proc.stderr.strip()[:300]}",
        )
    try:
        findings = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return Result(
            "toolchain",
            "steward-doctor",
            FAIL,
            f"steward doctor emitted unparseable JSON: {proc.stdout.strip()[:300]}",
        )

    # A nonzero exit is expected when steward reports findings, and those are
    # judged below. A nonzero exit with an *empty* findings array is a failed
    # run wearing success-shaped JSON — it must never reach the PASS branch.
    if proc.returncode != 0 and not findings:
        return Result(
            "toolchain",
            "steward-doctor",
            FAIL,
            f"steward doctor exited {proc.returncode} but reported no findings; "
            f"the run failed rather than passing: {proc.stderr.strip()[:300]}",
        )

    unexpected: list[str] = []
    waived: list[str] = []
    for finding in findings:
        check = str(finding.get("check", "?"))
        message = str(finding.get("message", ""))
        paths = set(_FINDING_PATH.findall(message))
        if check == "portability" and paths and paths <= STEWARD_WAIVED_PORTABILITY_PATHS:
            waived.append(f"{check} ({', '.join(sorted(paths))})")
        else:
            unexpected.append(f"{check}: {message.strip()[:300]}")

    if unexpected:
        return Result("toolchain", "steward-doctor", FAIL, " | ".join(unexpected))
    if waived:
        return Result(
            "toolchain",
            "steward-doctor",
            WAIVED,
            "0 unexpected findings while carrying all four harness prompt files; "
            f"{len(waived)} known-accepted finding(s) waived: {'; '.join(waived)} "
            "(pre-existing, byte-verbatim vendored skills — upstream's to fix)",
        )
    return Result(
        "toolchain",
        "steward-doctor",
        PASS,
        "0 findings while carrying all four harness prompt files",
    )


def check_guild_create(repo: Path, harnesses: dict[str, Any], timeout: int) -> list[Result]:
    base = _tool_argv("guild", "guild-cli")
    if base is None:
        return [
            Result(
                "toolchain",
                "guild-create",
                SKIP,
                "neither `guild` nor `uvx`/`uv` is available — the four `guild "
                "create` dry-runs were NOT run",
            )
        ]

    results: list[Result] = []
    for name in HARNESSES:
        guild_name, expected = GUILD_HARNESS[name]
        resolves = [str(p) for p in harnesses[name].get("resolves") or []]
        if expected not in resolves:
            results.append(
                Result(
                    "toolchain",
                    f"guild-create/{name}",
                    FAIL,
                    f"{expected} is not among this harness's resolves {resolves} in "
                    f"{INVOCATIONS_REL}",
                )
            )
            continue

        argv = base + [
            "create",
            "--agent",
            "agentculture/harness-smoke-probe",
            "--desc",
            "dry-run probe for the culture-agent-template harness smoke check",
            "--harness",
            guild_name,
            "--json",
        ]
        try:
            proc = _run(argv, repo, timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append(
                Result("toolchain", f"guild-create/{name}", SKIP, f"could not run guild: {exc}")
            )
            continue
        if proc.returncode != 0:
            # A dry-run that exits nonzero did not succeed, whatever it wrote
            # to stdout: with no payload the tool is unavailable (SKIP, never
            # a pass), and with a payload it ran and failed (FAIL). Neither
            # may fall through to the plan-shape checks below.
            status = SKIP if not proc.stdout.strip() else FAIL
            detail = (
                f"guild create unavailable (exit {proc.returncode})"
                if status == SKIP
                else f"guild create exited {proc.returncode} for --harness {guild_name}"
            )
            results.append(
                Result(
                    "toolchain",
                    f"guild-create/{name}",
                    status,
                    f"{detail}: {proc.stderr.strip()[:300]}",
                )
            )
            continue
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            results.append(
                Result(
                    "toolchain",
                    f"guild-create/{name}",
                    FAIL,
                    f"guild create --json emitted unparseable output: {proc.stdout[:300]}",
                )
            )
            continue

        if payload.get("applied"):
            results.append(
                Result(
                    "toolchain",
                    f"guild-create/{name}",
                    FAIL,
                    "guild create reported applied=true for a dry-run",
                )
            )
            continue
        named = payload.get("prompt_file")
        steps = " ".join(str(s) for s in (payload.get("plan") or {}).get("steps") or [])
        if named != expected:
            results.append(
                Result(
                    "toolchain",
                    f"guild-create/{name}",
                    FAIL,
                    f"--harness {guild_name} named prompt_file {named!r}, expected {expected!r}",
                )
            )
        elif expected not in steps:
            results.append(
                Result(
                    "toolchain",
                    f"guild-create/{name}",
                    FAIL,
                    f"--harness {guild_name} declares prompt_file {expected!r} but no "
                    f"planned step writes it: {steps[:300]}",
                )
            )
        else:
            results.append(
                Result(
                    "toolchain",
                    f"guild-create/{name}",
                    PASS,
                    f"--harness {guild_name} names and writes {expected}",
                )
            )
    return results


def stage_toolchain(repo: Path, harnesses: dict[str, Any], timeout: int) -> list[Result]:
    return [check_steward_doctor(repo, timeout)] + check_guild_create(repo, harnesses, timeout)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

STAGES = ("config", "live", "toolchain")


def run(repo: Path, stages: list[str], timeout: int) -> list[Result]:
    harnesses = load_invocations(repo)
    results: list[Result] = []
    if "config" in stages:
        results.extend(stage_config(repo, harnesses))
    if "live" in stages:
        results.extend(stage_live(repo, harnesses, timeout))
    if "toolchain" in stages:
        results.extend(stage_toolchain(repo, harnesses, timeout))
    return results


def report(results: list[Result], repo: Path, required: set[str]) -> int:
    """Print the human-readable report; return the process exit code.

    Results go to **stdout**, diagnostics to **stderr** — the repo's CLI
    output contract. The per-check status table and the counts summary are the
    result; the NOT-VERIFIED / WAIVED expansions and the --require error are
    explanations of what went wrong, so redirecting stdout to a file leaves a
    clean record of outcomes and still shows the operator why.
    """
    lines = [f"harness smoke check — repo: {repo}"]
    for res in results:
        lines.append(f"[{res.status:<6}] {res.label:<28} {res.detail}")

    failed = [r for r in results if r.status == FAIL]
    skipped = [r for r in results if r.status == SKIP]
    waived = [r for r in results if r.status == WAIVED]
    passed = [r for r in results if r.status == PASS]

    lines.append("")
    lines.append(
        f"summary: {len(passed)} passed, {len(waived)} waived, "
        f"{len(skipped)} skipped, {len(failed)} failed"
    )

    diagnostics: list[str] = []
    if failed:
        diagnostics.append("FAILED:")
        for res in failed:
            diagnostics.append(f"  - {res.label}: {res.detail}")
    if skipped:
        diagnostics.append("NOT VERIFIED (a skipped check is not a pass):")
        for res in skipped:
            diagnostics.append(f"  - {res.label}: {res.detail}")
    if waived:
        diagnostics.append("WAIVED (known-accepted, reported every run):")
        for res in waived:
            diagnostics.append(f"  - {res.label}: {res.detail}")

    exit_code = 1 if failed else 0
    unmet = sorted({r.stage for r in skipped} & required)
    if unmet:
        diagnostics.append(f"ERROR: --require named stage(s) that skipped: {', '.join(unmet)}")
        exit_code = 1

    print("\n".join(lines))
    if diagnostics:
        print("\n".join(diagnostics), file=sys.stderr)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=str(REPO_ROOT_DEFAULT), help="Repo root to check.")
    parser.add_argument(
        "--stage",
        default="config",
        help="Comma-separated stages to run: config, live, toolchain, or all. Default: config.",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Stage whose checks must not skip (repeatable). Turns a SKIP into a failure.",
    )
    parser.add_argument("--json", action="store_true", help="Emit results as JSON.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-command timeout in seconds.")
    args = parser.parse_args(argv)

    requested = [s.strip() for s in args.stage.split(",") if s.strip()]
    stages = list(STAGES) if "all" in requested else requested
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        # Deliberately NOT parser.error(), which exits 2. This repo's CLI
        # contract reserves 2 for environment/system failures; a bad --stage
        # value is user input, which is exit 1.
        print(
            f"error: unknown stage(s): {', '.join(unknown)}; "
            f"choose from {', '.join(STAGES)}, all",
            file=sys.stderr,
        )
        return 1

    repo = Path(args.repo).resolve()
    results = run(repo, stages, args.timeout)

    if args.json:
        failed = [r for r in results if r.status == FAIL]
        skipped_stages = {r.stage for r in results if r.status == SKIP}
        unmet = sorted(skipped_stages & set(args.require))
        print(
            json.dumps(
                {
                    "repo": str(repo),
                    "stages": stages,
                    "results": [r.as_dict() for r in results],
                    "failed": len(failed),
                    "unmet_requirements": unmet,
                },
                indent=2,
            )
        )
        return 1 if failed or unmet else 0

    return report(results, repo, set(args.require))


if __name__ == "__main__":
    sys.exit(main())
