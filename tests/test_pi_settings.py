"""Tests for the ``.pi/skills`` symlink that wires Pi to the vendored kit.

Pi discovers project skills from conventional directories — ``.pi/skills/`` is
one of them. A ``skills`` array in ``.pi/settings.json`` does NOT do this: that
key belongs to a *package manifest* (``package.json``'s ``pi`` block), not to
``settings.json``.

This was established by syscall instrumentation, not by asking the model. In a
fresh clone, ``strace -e trace=openat`` showed:

* ``.pi/settings.json`` present, no symlink  -> **0** ``SKILL.md`` files opened
* ``pi --skill .claude/skills``              -> **19** opened
* ``.pi/skills`` symlink, no flag            -> **19** opened

See deviations ``d2`` and ``d4``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PI_SKILLS = REPO_ROOT / ".pi" / "skills"


def test_pi_skills_is_a_symlink() -> None:
    assert PI_SKILLS.is_symlink(), ".pi/skills must be a symlink, not a copied tree"


def test_pi_skills_target_is_relative() -> None:
    """An absolute target would break on every other machine."""
    target = PI_SKILLS.readlink()
    assert not target.is_absolute()
    assert str(target) == "../.claude/skills"


def test_pi_skills_resolves_to_the_vendored_tree() -> None:
    assert PI_SKILLS.resolve() == (REPO_ROOT / ".claude" / "skills").resolve()
    assert (PI_SKILLS / "cicd" / "SKILL.md").is_file()


def test_pi_skills_is_tracked_as_a_git_symlink() -> None:
    """Mode 120000 is what survives a clone as a link rather than a copy."""
    out = subprocess.run(
        ["git", "ls-files", "-s", ".pi/skills"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.startswith("120000"), out


def test_no_inert_pi_settings_json() -> None:
    """Its `skills` key is a package-manifest key and does nothing here."""
    assert not (REPO_ROOT / ".pi" / "settings.json").exists()
    assert not (REPO_ROOT / ".pi" / "settings.json.example").exists()


def test_no_forked_skill_scripts_under_pi() -> None:
    """The symlink is the only thing under .pi/ that reaches the kit."""
    real = [p for p in (REPO_ROOT / ".pi").rglob("scripts") if not p.is_symlink()]
    assert real == []
