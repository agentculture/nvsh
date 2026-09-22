"""Stage a merged checkpoint in a Hugging Face cache so Tier 2 can serve it (issue 39).

nvsh's launcher hands vLLM a repository id and, with ``[tiers.lfm]
hf_offline = true``, serves it from ``hf_cache_dir`` without asking the Hub.
This script puts a local merged checkpoint there under a repository id --
the layout ``hf download`` would leave: ``hub/models--<owner>--<name>/
snapshots/<revision>/`` plus ``refs/main`` -- so a candidate can be measured
through the real launcher before (or instead of) being pushed anywhere.

Before copying it checks that the checkpoint's chat template is byte-identical
to the base model's (spec c33): training, serving and the stock baseline must
render tools and tool calls with the same template. The revision is the first
40 hex digits of a SHA-256 over the checkpoint's files, so the same weights
always get the same revision and ``measure.py --revision`` names exactly them.

    python scripts/lfm-finetune/stage_cache.py --merged runs/r1/merged \
        --repo jetson-ai-lab/lfm2.5-350m-nvsh-triage --cache <hf_cache_dir> \
        --base-snapshot <cache>/hub/models--LiquidAI--LFM2.5-350M/snapshots/<commit>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

#: Where a tokenizer keeps its chat template: a separate file in newer
#: transformers releases, inside tokenizer_config.json in older ones.
TEMPLATE_FILE = "chat_template.jinja"
TOKENIZER_CONFIG = "tokenizer_config.json"


def chat_template(directory: Path) -> str:
    """The chat template a model directory ships, or raise if it has none."""
    template_file = directory / TEMPLATE_FILE
    if template_file.is_file():
        return template_file.read_text(encoding="utf-8")
    config = directory / TOKENIZER_CONFIG
    if config.is_file():
        template = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
        if isinstance(template, str) and template:
            return template
    raise ValueError(f"{directory} ships no chat template")


def revision_of(directory: Path) -> str:
    """A stable 40-hex revision for the files in *directory* (names and bytes)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()[:40]


def repo_dir(cache: Path, repo: str) -> Path:
    owner, _, name = repo.partition("/")
    if not owner or not name or "/" in name:
        raise ValueError(f"repository id must be owner/name, got {repo!r}")
    return cache / "hub" / f"models--{owner}--{name}"


def stage(merged: Path, repo: str, cache: Path, base_snapshot: Path) -> str:
    """Copy *merged* into the cache as *repo*; return the revision. Refuses a changed template."""
    if chat_template(merged) != chat_template(base_snapshot):
        raise ValueError(
            "the checkpoint's chat template differs from the base model's; restore the base"
            " template before merging, or discard this run (spec c33)"
        )
    revision = revision_of(merged)
    target = repo_dir(cache, repo)
    snapshot = target / "snapshots" / revision
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(merged, snapshot)
    (target / "refs").mkdir(parents=True, exist_ok=True)
    (target / "refs" / "main").write_text(revision, encoding="utf-8")
    return revision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged", required=True, type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cache", required=True, type=Path, help="[tiers.lfm] hf_cache_dir")
    parser.add_argument("--base-snapshot", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        revision = stage(args.merged, args.repo, args.cache, args.base_snapshot)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"staged {args.repo} at revision {revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
