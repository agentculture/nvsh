#!/usr/bin/env python3
"""Jetson skills validation set: tools, the 104-eval test set, and provenance.

A development-machine tool for the Tier 2 (LFM2.5) fine-tune (issue 39,
work item "method validation"). It reads two NVIDIA repos --
``jetson-device-skills`` and ``jetson-bsp-skills`` -- at pinned commits into
scratch space, builds one OpenAI-format function tool per skill from each
skill's ``SKILL.md`` frontmatter, and turns every one of NVIDIA's own eval
files into a routing test set that is NEVER trained on.

This script is NEVER imported by the nvsh package, and nothing it reads from
those two repos is copied into ``nvsh/``: the validation task lives entirely
under ``scripts/lfm-finetune/``, separate from nvsh's own use case (c40).

Usage::

    # resolve fresh HEAD shas (network, informational only -- the defaults
    # below were resolved this way on 2026-09-22 and are what ships):
    git ls-remote https://github.com/NVIDIA-AI-IOT/jetson-device-skills HEAD
    git ls-remote https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills HEAD

    # clone both repos at the pinned shas into --work-dir and build the set:
    python scripts/lfm-finetune/jetson_skills.py build \\
        --work-dir /tmp/jetson-skills --out-dir /tmp/jetson-skills-out

    # fail (non-zero exit) if a training file contains an eval prompt/question
    # or ground_truth, verbatim or as a near-duplicate:
    python scripts/lfm-finetune/jetson_skills.py scan \\
        --training train.jsonl --test /tmp/jetson-skills-out/test.jsonl

Both eval shapes are handled (h27, s30):

- **device** (``jetson-device-skills``): one ``evals/evals.json`` per skill is
  a dict ``{"skill_name": ..., "evals": [{"id", "prompt", "expected_output",
  "assertions", "expected_skill", "expected_script"}, ...]}``.
- **BSP** (``jetson-bsp-skills``): one ``evals/evals.json`` per skill is a
  list ``[{"id", "question", "expected_skill", "expected_script",
  "ground_truth", "expected_behavior"}, ...]``.

Both repos declare (README + LICENSE): documentation under CC-BY-4.0,
source code under Apache-2.0. Every ``SKILL.md`` in both repos additionally
declares ``license: "Apache-2.0"`` in its own frontmatter, so tools and evals
derived from a skill's ``SKILL.md``/``evals.json`` are recorded with that
skill's own declared licence (c40, h28).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# pinned repositories
# ---------------------------------------------------------------------------

#: Resolved 2026-09-22 with ``git ls-remote <url> HEAD``. Pass --device-sha /
#: --bsp-sha to build against a different commit; these are only defaults.
DEVICE_REPO_URL = "https://github.com/NVIDIA-AI-IOT/jetson-device-skills"
DEVICE_REPO_SHA = "20137897aef549cc2fa36a18c10e45da94967c3e"
BSP_REPO_URL = "https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills"
BSP_REPO_SHA = "fdfafef0416be1eb3852b68fc802a9752f33fb28"

#: Both repos' README + LICENSE say the same split.
DOCS_LICENCE = "CC-BY-4.0"
SOURCE_LICENCE = "Apache-2.0"

DEVICE_REPO = "device"
BSP_REPO = "bsp"

# ---------------------------------------------------------------------------
# frontmatter (a minimal parser for the three keys this script needs --
# not a general YAML parser)
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> dict[str, str]:
    """Pull ``name``, ``description`` and ``license`` out of a SKILL.md's
    YAML frontmatter.

    Handles both a plain scalar (``key: value``, optionally quoted) and a
    folded block scalar (``key: >-`` followed by indented continuation
    lines, folded into one line with single spaces) -- the two shapes used
    across both repos' SKILL.md files. Anything else in the frontmatter
    (version, metadata, tags, ...) is ignored; this script never needs it.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md has no opening frontmatter fence")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError("SKILL.md frontmatter is never closed") from None
    body = lines[1:end]

    result: dict[str, str] = {}
    key_re = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$")
    i = 0
    while i < len(body):
        line = body[i]
        match = key_re.match(line)
        if match is None:
            i += 1
            continue
        key, value = match.group(1), match.group(2).strip()
        if value in (">-", ">", "|-", "|"):
            # Folded/literal block scalar: consume indented continuation
            # lines and join them into one string (folded style -- good
            # enough for the free-text `description` fields this is for).
            parts: list[str] = []
            j = i + 1
            while j < len(body) and (body[j].startswith((" ", "\t")) or not body[j].strip()):
                stripped = body[j].strip()
                if stripped:
                    parts.append(stripped)
                j += 1
            result[key] = " ".join(parts)
            i = j
            continue
        result[key] = value.strip('"').strip("'")
        i += 1
    return result


# ---------------------------------------------------------------------------
# tool names
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"[^a-zA-Z0-9_]+")


def to_tool_name(skill_name: str) -> str:
    """A valid OpenAI function-tool name for *skill_name* (e.g.
    ``jetson-diagnostic`` -> ``jetson_diagnostic``)."""
    name = _NAME_RE.sub("_", skill_name).strip("_")
    if not name:
        raise ValueError(f"skill name {skill_name!r} yields an empty tool name")
    if name[0].isdigit():
        name = f"skill_{name}"
    return name


# ---------------------------------------------------------------------------
# skill discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skill:
    repo: str  # DEVICE_REPO or BSP_REPO
    name: str  # directory / frontmatter name, e.g. "jetson-diagnostic"
    skill_md: Path
    evals_json: Path | None
    frontmatter: dict[str, str]

    @property
    def licence(self) -> str:
        return self.frontmatter.get("license", SOURCE_LICENCE)

    @property
    def description(self) -> str:
        return self.frontmatter.get("description", "")


def discover_skills(repo_root: Path, repo: str) -> list[Skill]:
    """Every ``skills/<name>/SKILL.md`` under *repo_root*, sorted by name."""
    skills_dir = repo_root / "skills"
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"{skills_dir} does not exist -- not a checked-out skills repo")
    found: list[Skill] = []
    for skill_dir in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue
        frontmatter = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = frontmatter.get("name", skill_dir.name)
        evals_json = skill_dir / "evals" / "evals.json"
        found.append(
            Skill(
                repo=repo,
                name=name,
                skill_md=skill_md,
                evals_json=evals_json if evals_json.is_file() else None,
                frontmatter=frontmatter,
            )
        )
    return found


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def build_tool(skill: Skill) -> dict[str, Any]:
    """The OpenAI function-tool schema for one skill: no parameters."""
    return {
        "type": "function",
        "function": {
            "name": to_tool_name(skill.name),
            "description": skill.description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def build_tools(skills: Iterable[Skill]) -> list[dict[str, Any]]:
    """One ``{"skill", "repo", "tool"}`` record per skill (the mapping from
    the original skill name to its derived tool name is on ``tool.function.name``)."""
    return [
        {"skill": skill.name, "repo": skill.repo, "tool": build_tool(skill)} for skill in skills
    ]


# ---------------------------------------------------------------------------
# "names the skill outright"
# ---------------------------------------------------------------------------


def _skill_name_variants(skill_name: str) -> list[str]:
    hyphen = skill_name
    underscore = skill_name.replace("-", "_")
    variants = {hyphen, underscore}
    variants |= {f"/{v}" for v in list(variants)}
    return sorted(variants)


def names_skill(text: str, skill_name: str) -> bool:
    """True if *text* names *skill_name* outright (with or without a leading
    slash, hyphen or underscore form) -- h27's "tests copying, not routing"."""
    lowered = text.lower()
    return any(variant.lower() in lowered for variant in _skill_name_variants(skill_name))


# ---------------------------------------------------------------------------
# eval test set (both shapes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRecord:
    id: str
    repo: str
    skill: str
    text: str
    expected_skill: str
    ground_truth: str | None
    names_skill: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo": self.repo,
            "skill": self.skill,
            "text": self.text,
            "expected_skill": self.expected_skill,
            "ground_truth": self.ground_truth,
            "names_skill": self.names_skill,
        }


def _load_evals_json(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Returns (raw eval dicts, top-level skill_name if the device shape has one)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # device shape: {"skill_name": ..., "evals": [...]}
        return list(data.get("evals", [])), data.get("skill_name")
    if isinstance(data, list):
        # BSP shape: a bare list of eval dicts
        return data, None
    raise ValueError(f"{path}: evals.json is neither a dict nor a list")


def load_skill_evals(skill: Skill) -> list[EvalRecord]:
    """Every eval for one skill, normalized to :class:`EvalRecord`."""
    if skill.evals_json is None:
        return []
    raw_evals, top_skill_name = _load_evals_json(skill.evals_json)
    records: list[EvalRecord] = []
    for raw in raw_evals:
        text = raw.get("prompt", raw.get("question", ""))
        expected_skill = raw.get("expected_skill") or top_skill_name or skill.name
        ground_truth = raw.get("ground_truth") or raw.get("expected_output")
        records.append(
            EvalRecord(
                id=raw["id"],
                repo=skill.repo,
                skill=skill.name,
                text=text,
                expected_skill=expected_skill,
                ground_truth=ground_truth,
                names_skill=names_skill(text, expected_skill),
            )
        )
    return records


def build_test_set(skills: Iterable[Skill]) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for skill in skills:
        records.extend(load_skill_evals(skill))
    return records


# ---------------------------------------------------------------------------
# manifest + README
# ---------------------------------------------------------------------------


@dataclass
class RepoProvenance:
    repo: str
    url: str
    commit: str


def build_manifest(
    skills: Iterable[Skill], evals: Iterable[EvalRecord], provenance: list[RepoProvenance]
) -> dict[str, Any]:
    """Maps every derived record (one per tool, one per eval) to its source
    repository, file, commit, licence and whether it was transformed."""
    by_repo = {p.repo: p for p in provenance}
    records: list[dict[str, Any]] = []
    skills_by_key = {(s.repo, s.name): s for s in skills}
    for skill in skills:
        prov = by_repo[skill.repo]
        records.append(
            {
                "kind": "tool",
                "repo": skill.repo,
                "repo_url": prov.url,
                "commit": prov.commit,
                "file": f"skills/{skill.name}/SKILL.md",
                "licence": skill.licence,
                "transformed": True,  # name/description reshaped into an OpenAI tool schema
            }
        )
    for ev in evals:
        skill = skills_by_key.get((ev.repo, ev.skill))
        prov = by_repo[ev.repo]
        licence = skill.licence if skill is not None else SOURCE_LICENCE
        records.append(
            {
                "kind": "eval",
                "id": ev.id,
                "repo": ev.repo,
                "repo_url": prov.url,
                "commit": prov.commit,
                "file": f"skills/{ev.skill}/evals/evals.json",
                "licence": licence,
                "transformed": True,  # reshaped into this script's unified eval schema
            }
        )
    return {
        "repositories": [
            {
                "repo": p.repo,
                "url": p.url,
                "commit": p.commit,
                "licences": {"documentation": DOCS_LICENCE, "source": SOURCE_LICENCE},
            }
            for p in provenance
        ],
        "records": records,
    }


def render_readme(
    skills: list[Skill], evals: list[EvalRecord], provenance: list[RepoProvenance]
) -> str:
    by_repo: dict[str, list[Skill]] = {}
    for skill in skills:
        by_repo.setdefault(skill.repo, []).append(skill)
    named = sum(1 for ev in evals if ev.names_skill)
    lines = [
        "# Jetson skills validation set",
        "",
        "A routing test set (never trained on) for the Tier 2 (LFM2.5) fine-tune's",
        "method validation (issue 39): one tool per NVIDIA Jetson agent skill, and",
        "NVIDIA's own evals as the held-out test set.",
        "",
        "## Attribution",
        "",
    ]
    for p in provenance:
        n = len(by_repo.get(p.repo, []))
        lines.append(f"- **{p.repo}**: [{p.url}]({p.url}) at commit `{p.commit}` ({n} skills)")
    lines += [
        "",
        "Both repositories are dual-licensed (their own README + LICENSE):",
        f"documentation under {DOCS_LICENCE}, source code under {SOURCE_LICENCE}. Every",
        f'`SKILL.md` in both repos additionally declares `license: "{SOURCE_LICENCE}"` in',
        "its own frontmatter; this dataset records that per-skill licence on every",
        "derived tool and eval record in `manifest.json`.",
        "",
        "Nothing from these repositories is copied into `nvsh/`: this dataset lives",
        "entirely under `scripts/lfm-finetune/`, separate from nvsh's own use case.",
        "",
        "## Contents",
        "",
        f"- `tools.json` -- {len(skills)} OpenAI function tools, one per skill, built",
        "  from each skill's `SKILL.md` frontmatter (`name`, `description`); no",
        "  parameters.",
        f"- `test.jsonl` -- exactly the {len(evals)} evals shipped by the two repos,",
        f"  one JSON object per line. {named} eval(s) name their expected skill",
        "  outright (`names_skill: true`) and test copying rather than routing.",
        "- `manifest.json` -- every derived tool and eval record mapped to its",
        "  repository, commit, source file, licence and whether it was transformed.",
        "- `README.md` -- this file.",
        "",
        "## Contamination scan",
        "",
        "`jetson_skills.py scan --training <file> --test test.jsonl` fails",
        "(non-zero exit) if any eval `text` or `ground_truth` appears in the",
        "training file, exactly or as a near-duplicate. See the script's module",
        "docstring / `--help` for the exact rule (normalized text containment, plus",
        "a token-shingle Jaccard threshold).",
        "",
        "## Generated",
        "",
        "This README and the files beside it are generated by",
        "`scripts/lfm-finetune/jetson_skills.py build`. Do not hand-edit them;",
        "re-run the script instead.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# contamination scan
# ---------------------------------------------------------------------------

#: The near-duplicate rule: normalize (lowercase, strip punctuation, collapse
#: whitespace), then either containment of the normalized *token sequence* in
#: each other (exact / near-exact, word-boundary-aware so "ok" can't match
#: inside "book"), or a token-shingle Jaccard similarity at or above this
#: threshold on shingles of this length. Texts shorter than this many tokens
#: are exempt from the exact-containment check (a short, generic phrase like
#: "ok" is not evidence of contamination).
NEAR_DUP_SHINGLE_SIZE = 5
NEAR_DUP_JACCARD_THRESHOLD = 0.8
MIN_TOKENS_FOR_EXACT_MATCH = 4
#: A light paraphrase keeps most of the eval's words in a different order, which
#: the 5-token shingles miss. Word-set Jaccard at or above this is flagged too.
#: Rewording that changes most words (a Jaccard near 0.3) is not caught by any
#: lexical check; that limit is documented in docs/lfm-finetune.md.
PARAPHRASE_WORD_JACCARD = 0.6

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    lowered = text.lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", stripped).strip()


def _contains_token_sequence(needle_tokens: list[str], haystack_tokens: list[str]) -> bool:
    """True if *needle_tokens* appears as a contiguous run inside
    *haystack_tokens* (or vice versa) -- word-boundary-aware, unlike a plain
    substring test on the joined strings."""
    if not needle_tokens or not haystack_tokens:
        return False
    shorter, longer = (
        (needle_tokens, haystack_tokens)
        if len(needle_tokens) <= len(haystack_tokens)
        else (haystack_tokens, needle_tokens)
    )
    n = len(shorter)
    for i in range(len(longer) - n + 1):
        if longer[i : i + n] == shorter:
            return True
    return False


def _shingles(text: str, size: int = NEAR_DUP_SHINGLE_SIZE) -> frozenset[tuple[str, ...]]:
    tokens = text.split()
    if len(tokens) < size:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass
class Contamination:
    eval_id: str
    field: str  # "text" or "ground_truth"
    reason: str  # "exact", "near-duplicate (jaccard=0.NN)" or "paraphrase (...)"
    training_excerpt: str


def _iter_strings(value: Any) -> Iterable[str]:
    """Every string found anywhere in a (possibly nested) JSON value."""
    if isinstance(value, str):
        if value:
            yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _iter_strings(v)


def load_training_texts(path: Path) -> list[str]:
    """Every string in every JSON object of a JSONL training file."""
    texts: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            texts.extend(_iter_strings(json.loads(line)))
    return texts


def _find_contamination(
    eval_id: str,
    field: str,
    text: str,
    training_texts: list[str],
    training_normalized: list[str],
    threshold: float,
) -> Contamination | None:
    norm = normalize_text(text)
    if not norm:
        return None
    eval_tokens = norm.split()
    eval_shingles = _shingles(norm)
    for raw, norm_t in zip(training_texts, training_normalized):
        if not norm_t:
            continue
        train_tokens = norm_t.split()
        # Both sides must be long enough: a one-word training string ("it")
        # contained in an eval is not evidence the eval was copied.
        if min(
            len(eval_tokens), len(train_tokens)
        ) >= MIN_TOKENS_FOR_EXACT_MATCH and _contains_token_sequence(eval_tokens, train_tokens):
            return Contamination(eval_id, field, "exact", raw[:200])
        if len(train_tokens) < MIN_TOKENS_FOR_EXACT_MATCH:
            continue
        # Compare eval-sized windows, so a near-copy buried in unrelated
        # text is not diluted by it.
        for window in _windows(train_tokens, len(eval_tokens)):
            sim = _jaccard(eval_shingles, _shingles(" ".join(window)))
            if sim >= threshold:
                return Contamination(
                    eval_id, field, f"near-duplicate (jaccard={sim:.2f})", raw[:200]
                )
            words = _jaccard(frozenset(eval_tokens), frozenset(window))
            if len(eval_tokens) >= MIN_TOKENS_FOR_EXACT_MATCH and words >= PARAPHRASE_WORD_JACCARD:
                return Contamination(
                    eval_id, field, f"paraphrase (word jaccard={words:.2f})", raw[:200]
                )
    return None


def _windows(tokens: list[str], size: int) -> Iterable[list[str]]:
    """Every run of *tokens* the length of the eval, give or take a quarter.

    A training string no longer than that is compared whole.
    """
    slack = max(1, size // 4)
    if len(tokens) <= size + slack:
        yield tokens
        return
    for length in range(max(1, size - slack), size + slack + 1):
        for start in range(len(tokens) - length + 1):
            yield tokens[start : start + length]


def scan_contamination(
    evals: Iterable[EvalRecord],
    training_texts: list[str],
    threshold: float = NEAR_DUP_JACCARD_THRESHOLD,
) -> list[Contamination]:
    """Every eval prompt/question or ground_truth that appears -- exactly or
    as a near-duplicate -- in *training_texts*. Empty means clean."""
    training_normalized = [normalize_text(t) for t in training_texts]
    found: list[Contamination] = []
    for ev in evals:
        hit = _find_contamination(
            ev.id, "text", ev.text, training_texts, training_normalized, threshold
        )
        if hit is not None:
            found.append(hit)
        if ev.ground_truth:
            hit = _find_contamination(
                ev.id,
                "ground_truth",
                ev.ground_truth,
                training_texts,
                training_normalized,
                threshold,
            )
            if hit is not None:
                found.append(hit)
    return found


# ---------------------------------------------------------------------------
# build orchestration
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    skills: list[Skill]
    evals: list[EvalRecord]
    tools: list[dict[str, Any]]
    manifest: dict[str, Any]
    readme: str

    @property
    def counts(self) -> dict[str, int]:
        return {
            "skills": len(self.skills),
            "evals": len(self.evals),
            "named_outright": sum(1 for e in self.evals if e.names_skill),
        }


def build_from_checkout(
    device_root: Path,
    bsp_root: Path,
    device_sha: str,
    bsp_sha: str,
    device_url: str = DEVICE_REPO_URL,
    bsp_url: str = BSP_REPO_URL,
) -> BuildResult:
    """The whole build, given two already-checked-out repo roots. No network,
    no writes -- callers decide whether/where to write the outputs. This is
    the function the tests call directly against small fixture checkouts."""
    device_skills = discover_skills(device_root, DEVICE_REPO)
    bsp_skills = discover_skills(bsp_root, BSP_REPO)
    skills = device_skills + bsp_skills

    evals = build_test_set(skills)
    tools = build_tools(skills)
    provenance = [
        RepoProvenance(DEVICE_REPO, device_url, device_sha),
        RepoProvenance(BSP_REPO, bsp_url, bsp_sha),
    ]
    manifest = build_manifest(skills, evals, provenance)
    readme = render_readme(skills, evals, provenance)
    return BuildResult(skills=skills, evals=evals, tools=tools, manifest=manifest, readme=readme)


def write_outputs(result: BuildResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tools.json").write_text(
        json.dumps(result.tools, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with open(out_dir / "test.jsonl", "w", encoding="utf-8") as handle:
        for ev in result.evals:
            handle.write(json.dumps(ev.as_dict(), sort_keys=True) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(result.readme, encoding="utf-8")


# ---------------------------------------------------------------------------
# fetch (network; never used by tests)
# ---------------------------------------------------------------------------


def fetch_repo(url: str, sha: str, dest: Path) -> None:
    """``git clone`` *url* into *dest* and check out *sha*.

    An existing, non-empty *dest* is reused only if it is a clean checkout of
    exactly *sha*; anything else raises, so the manifest never records a
    commit other than the files it was built from.
    """
    if dest.exists() and any(dest.iterdir()):
        verify_checkout(dest, sha)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", sha], check=True)


def verify_checkout(dest: Path, sha: str) -> None:
    """Raise ``ValueError`` unless *dest* is a clean git checkout at *sha*."""
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if head.returncode != 0:
        raise ValueError(
            f"{dest} exists but is not a git checkout; remove it or pick another --work-dir"
        )
    if head.stdout.strip() != sha:
        raise ValueError(f"{dest} is at {head.stdout.strip()}, not the pinned {sha}")
    dirty = subprocess.run(
        ["git", "-C", str(dest), "status", "--porcelain"], capture_output=True, text=True
    )
    if dirty.stdout.strip():
        raise ValueError(f"{dest} has local changes; the manifest would not describe it")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    device_root = work_dir / DEVICE_REPO
    bsp_root = work_dir / BSP_REPO
    if not args.skip_fetch:
        try:
            fetch_repo(args.device_url, args.device_sha, device_root)
            fetch_repo(args.bsp_url, args.bsp_sha, bsp_root)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    result = build_from_checkout(
        device_root, bsp_root, args.device_sha, args.bsp_sha, args.device_url, args.bsp_url
    )

    if args.training:
        training_texts = load_training_texts(Path(args.training))
        hits = scan_contamination(result.evals, training_texts, args.threshold)
        if hits:
            print(
                f"contamination scan failed: {len(hits)} eval field(s) found in "
                f"{args.training}",
                file=sys.stderr,
            )
            for hit in hits:
                print(
                    f"  {hit.eval_id} [{hit.field}]: {hit.reason}: {hit.training_excerpt!r}",
                    file=sys.stderr,
                )
            return 1

    write_outputs(result, Path(args.out_dir))
    counts = result.counts
    print(
        f"skills={counts['skills']} evals={counts['evals']} "
        f"named_outright={counts['named_outright']}"
    )
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    evals: list[EvalRecord] = []
    with open(args.test, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            evals.append(
                EvalRecord(
                    id=raw["id"],
                    repo=raw["repo"],
                    skill=raw["skill"],
                    text=raw["text"],
                    expected_skill=raw["expected_skill"],
                    ground_truth=raw.get("ground_truth"),
                    names_skill=raw.get("names_skill", False),
                )
            )
    training_texts = load_training_texts(Path(args.training))
    hits = scan_contamination(evals, training_texts, args.threshold)
    if hits:
        print(
            f"contamination scan failed: {len(hits)} eval field(s) found in {args.training}",
            file=sys.stderr,
        )
        for hit in hits:
            print(
                f"  {hit.eval_id} [{hit.field}]: {hit.reason}: {hit.training_excerpt!r}",
                file=sys.stderr,
            )
        return 1
    print(f"clean: {len(evals)} eval(s) checked against {len(training_texts)} training string(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="fetch both repos and build tools/test set/manifest")
    build_p.add_argument("--work-dir", required=True, help="scratch dir to clone the repos into")
    build_p.add_argument("--out-dir", required=True, help="where to write tools.json etc.")
    build_p.add_argument("--device-url", default=DEVICE_REPO_URL)
    build_p.add_argument("--device-sha", default=DEVICE_REPO_SHA)
    build_p.add_argument("--bsp-url", default=BSP_REPO_URL)
    build_p.add_argument("--bsp-sha", default=BSP_REPO_SHA)
    build_p.add_argument(
        "--skip-fetch",
        action="store_true",
        help="assume --work-dir/device and --work-dir/bsp are already checked out",
    )
    build_p.add_argument(
        "--training",
        help="optional training JSONL; if given, the build fails if it contaminates the test set",
    )
    build_p.add_argument("--threshold", type=float, default=NEAR_DUP_JACCARD_THRESHOLD)
    build_p.set_defaults(func=_cmd_build)

    scan_p = sub.add_parser("scan", help="check a training file against a built test set")
    scan_p.add_argument("--training", required=True)
    scan_p.add_argument("--test", required=True, help="test.jsonl from a previous build")
    scan_p.add_argument("--threshold", type=float, default=NEAR_DUP_JACCARD_THRESHOLD)
    scan_p.set_defaults(func=_cmd_scan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
