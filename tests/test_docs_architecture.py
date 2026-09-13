"""t1: docs/architecture.md and the steward portability rule (h12, h23, h24).

Acceptance criteria for t1 (plan covers c14, h12, c31, h24, c30, h23):

- ``docs/architecture.md`` exists, opens with the before-state citing
  ``README.md:10-14`` and the retired PTY-wrapper wording "as of version
  0.9.1", and records the hook-over-wrap decision with its reasons and the
  parked login-shell mode.
- No tracked doc (``*.md``) contains a ``~/`` path (the steward-portability
  rule from this agent's memory: harness-smoke fails on any ``~/.dotfile``
  path in committed nvsh docs). ``$XDG_*`` variables are fine; a literal
  ``~/`` is not.
- ``docs/platforms.md`` exists with a heading for each of the three target
  machines, as a skeleton for a later task to fill in.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = REPO_ROOT / "docs" / "architecture.md"
PLATFORMS = REPO_ROOT / "docs" / "platforms.md"

TILDE_PATH_RE = re.compile(r"~/[\w./-]")

# The docs this task (t1) owns and keeps ~/-free, per the steward portability
# rule. docs/specs/ and docs/plans/ are devague's own historical record (not
# ours to rewrite) and .claude/skills/ is vendored verbatim from guildmaster
# (already tracked and WAIVED by steward doctor / harness-smoke), so neither
# is in scope for this check.
OWNED_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "AGENTS.override.md",
    REPO_ROOT / "AGENTS.colleague.md",
    REPO_ROOT / "QWEN.md",
    REPO_ROOT / ".pi" / "SYSTEM.md",
    ARCHITECTURE,
    PLATFORMS,
)


def test_architecture_doc_exists() -> None:
    assert ARCHITECTURE.is_file(), "docs/architecture.md must exist (t1)"


def test_architecture_doc_cites_before_state() -> None:
    text = ARCHITECTURE.read_text(encoding="utf-8")
    assert "README.md:10-14" in text
    assert "0.9.1" in text


def test_architecture_doc_records_hook_over_wrap_decision() -> None:
    text = ARCHITECTURE.read_text(encoding="utf-8").lower()
    assert "hook" in text
    assert "wrap" in text
    # The reasoning, not just the word: success-path cost is the load-bearing
    # reason the spec gives for choosing hook over wrap.
    assert "prompt_command" in text or "prompt command" in text


def test_architecture_doc_records_parked_login_shell_mode() -> None:
    text = ARCHITECTURE.read_text(encoding="utf-8").lower()
    assert "park" in text
    assert "login shell" in text or "login-shell" in text


def test_platforms_doc_skeleton_exists() -> None:
    assert PLATFORMS.is_file(), "docs/platforms.md must exist (t1 skeleton for t4)"
    text = PLATFORMS.read_text(encoding="utf-8")
    for heading in ("DGX Spark", "Jetson AGX Thor", "Jetson AGX Orin"):
        assert heading in text, f"docs/platforms.md must have a {heading} heading"


def test_no_tilde_home_path_in_owned_docs() -> None:
    """Steward portability rule: no ``~/`` path in the docs this task owns."""
    offenders: dict[str, list[str]] = {}
    for path in OWNED_DOCS:
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if TILDE_PATH_RE.search(line):
                offenders.setdefault(str(path.relative_to(REPO_ROOT)), []).append(
                    f"{lineno}: {line.strip()}"
                )
    assert not offenders, f"~/ paths found in owned docs: {offenders}"
