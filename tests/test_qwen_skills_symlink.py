"""`.qwen/skills` must be a relative symlink onto the canonical skill tree.

Qwen Code discovers project skills at `.qwen/skills/` (see
`docs/skills.md` in an installed Qwen Code checkout). Rather than fork or
duplicate every skill under a Qwen-specific tree, this template follows the
sensibo-cli precedent (`docs/skill-sources.md`): `.qwen/skills` is a
*relative* symlink to `../.claude/skills`, so it survives a fresh clone on
another machine (git stores a symlink as its target text, mode 120000) and
`guild create`'s identifier-rename transform leaves symlinks untouched
(it walks real files only).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QWEN_SKILLS = REPO_ROOT / ".qwen" / "skills"
CLAUDE_SKILLS = REPO_ROOT / ".claude" / "skills"


def test_qwen_skills_is_a_symlink() -> None:
    assert QWEN_SKILLS.is_symlink(), ".qwen/skills must be a symlink, not a real directory"


def test_qwen_skills_symlink_target_is_relative() -> None:
    target = os.readlink(QWEN_SKILLS)
    assert not os.path.isabs(
        target
    ), f".qwen/skills must point at a relative target, got {target!r}"
    assert target == "../.claude/skills"


def test_qwen_skills_resolves_to_claude_skills() -> None:
    assert QWEN_SKILLS.resolve() == CLAUDE_SKILLS.resolve()


def test_qwen_skills_lists_the_same_skills_as_claude_skills() -> None:
    canonical = sorted(p.name for p in CLAUDE_SKILLS.iterdir() if p.is_dir())
    via_symlink = sorted(p.name for p in QWEN_SKILLS.iterdir() if p.is_dir())
    assert via_symlink == canonical
    assert via_symlink, "expected at least one vendored skill"


def test_no_skill_scripts_are_forked_under_qwen() -> None:
    """No script under .claude/skills/*/scripts/ should be duplicated per harness."""
    qwen_dir = REPO_ROOT / ".qwen"
    # Everything under .qwen/ must resolve back through the single symlink;
    # there must be no independent "skills" copy anywhere else under .qwen/.
    real_dirs = [
        p for p in qwen_dir.rglob("*") if p.is_dir() and not p.is_symlink() and p.name == "scripts"
    ]
    assert real_dirs == [], f"found forked scripts/ directories under .qwen/: {real_dirs}"
