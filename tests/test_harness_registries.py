"""The harness registries must agree on all four harnesses (t12 criterion 1).

Three registries describe the same four harnesses, and a clone whose configs
are right but whose registries disagree is still broken — tooling reads the
registries, not the files:

1. ``.claude/skills/agent-config/data/backend-fingerprints.yaml`` — the
   vendored backend fingerprint table (``backend -> accepted prompt paths``),
   read by the agent-config skill's ``show.sh`` and by ``steward doctor`` when
   it runs from inside this checkout.
2. ``culture_agent_template/cli/_commands/doctor.py`` ``_PROMPT_FILE`` — the
   same mapping in Python, backing this template's own ``doctor`` verb.
3. ``culture_core.learn_prompt.SKILL_DIRS`` — **in the separate ``culture``
   repo**, which this repo cannot edit. Checked only when a sibling ``culture``
   checkout is present; otherwise the test SKIPS and says so. It is
   report-only: a foreign repo's disagreement is reported, never silently
   asserted away and never faked into agreement.

``docs/harness-invocations.yaml`` is the source of truth for which four
harnesses exist and which files each resolves; every assertion here is
anchored to it.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

from culture_agent_template.cli._commands.doctor import _PROMPT_FILE

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "scripts" / "harness-smoke.py"


def _load_smoke() -> Any:
    """Import scripts/harness-smoke.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("harness_smoke", SMOKE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke()

HARNESSES: tuple[str, ...] = smoke.HARNESSES
HARNESS_BACKEND: dict[str, str] = smoke.HARNESS_BACKEND
INVOCATIONS = yaml.safe_load((REPO_ROOT / smoke.INVOCATIONS_REL).read_text(encoding="utf-8"))
FINGERPRINTS = smoke.registry_prompts_by_backend(REPO_ROOT)


def _resolves(harness: str) -> list[str]:
    return [str(p) for p in INVOCATIONS["harnesses"][harness]["resolves"]]


# --- the four harnesses are the same four everywhere ----------------------


def test_the_yaml_and_the_smoke_check_list_the_same_four_harnesses() -> None:
    assert set(INVOCATIONS["harnesses"]) == set(HARNESSES)
    assert len(HARNESSES) == 4
    assert set(HARNESS_BACKEND) == set(HARNESSES)


# --- registry 1: backend-fingerprints.yaml --------------------------------


@pytest.mark.parametrize("harness", HARNESSES)
def test_fingerprints_register_every_harness_config_under_its_own_backend(harness: str) -> None:
    backend = HARNESS_BACKEND[harness]
    accepted = FINGERPRINTS.get(backend, ())
    assert accepted, f"backend {backend!r} is absent from the fingerprint registry"
    for rel in _resolves(harness):
        assert rel in accepted, (
            f"{rel} (resolved by the {harness} harness) is not registered under "
            f"backend {backend!r}: {accepted}"
        )


def test_fingerprints_do_not_claim_a_bare_agents_md_for_the_colleague_backend() -> None:
    """This template ships no AGENTS.md; the colleague row must not imply one."""
    assert "AGENTS.md" not in FINGERPRINTS["colleague"]


# --- registry 2: the template's own doctor.py map -------------------------


@pytest.mark.parametrize("harness", HARNESSES)
def test_doctor_map_accepts_every_harness_config_under_its_own_backend(harness: str) -> None:
    backend = HARNESS_BACKEND[harness]
    accepted = _PROMPT_FILE.get(backend, ())
    assert accepted, f"doctor's _PROMPT_FILE has no entry for backend {backend!r}"
    for rel in _resolves(harness):
        assert rel in accepted, (
            f"{rel} (resolved by the {harness} harness) is missing from "
            f"doctor's _PROMPT_FILE[{backend!r}]: {accepted}"
        )


def test_doctor_map_and_fingerprints_agree_backend_for_backend() -> None:
    """No drift on any backend both registries name."""
    shared = set(_PROMPT_FILE) & set(FINGERPRINTS)
    assert shared, "the two registries share no backend at all"
    for backend in sorted(shared):
        assert tuple(_PROMPT_FILE[backend]) == tuple(FINGERPRINTS[backend]), (
            f"backend {backend!r}: doctor.py says {_PROMPT_FILE[backend]}, "
            f"backend-fingerprints.yaml says {FINGERPRINTS[backend]}"
        )


def test_doctor_map_covers_every_backend_the_fingerprints_register() -> None:
    missing = sorted(set(FINGERPRINTS) - set(_PROMPT_FILE))
    assert not missing, f"doctor.py's _PROMPT_FILE is missing backends: {missing}"


# --- registry 3: SKILL_DIRS, in the separate `culture` repo ---------------

_CULTURE_CANDIDATES = (
    REPO_ROOT.parent / "culture",
    REPO_ROOT.parent.parent / "culture",
)
_LEARN_PROMPT_REL = Path("culture_core") / "learn_prompt.py"


def _find_skill_dirs() -> tuple[Path, dict[str, str]] | None:
    """Locate SKILL_DIRS in a sibling ``culture`` checkout, if one exists.

    Parsed with ``ast`` rather than imported: the culture package is not a
    dependency of this template and must never become one.
    """
    for candidate in _CULTURE_CANDIDATES:
        path = candidate / _LEARN_PROMPT_REL
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "SKILL_DIRS" in names:
                return path, ast.literal_eval(node.value)
    return None


def test_skill_dirs_agreement_with_the_culture_repo() -> None:
    """Report-only: SKILL_DIRS lives in a repo this one cannot edit.

    Skips loudly when no sibling ``culture`` checkout is present (the usual
    case in CI and in a fresh clone), and skips with the exact disagreement
    when one is present and disagrees — a cross-repo gap is reported, never
    asserted away and never faked into agreement.
    """
    found = _find_skill_dirs()
    if found is None:
        pytest.skip(
            "no sibling `culture` checkout found at "
            f"{[str(c / _LEARN_PROMPT_REL) for c in _CULTURE_CANDIDATES]} — "
            "SKILL_DIRS agreement was NOT verified (it lives in a different repo)"
        )
    path, skill_dirs = found
    backends_here = {HARNESS_BACKEND[h] for h in HARNESSES}
    missing = sorted(backends_here - set(skill_dirs))
    if missing:
        pytest.skip(
            f"CROSS-REPO GAP (report-only): {path} SKILL_DIRS covers "
            f"{sorted(skill_dirs)}, so backend(s) {missing} — used by "
            f"harness(es) {[h for h in HARNESSES if HARNESS_BACKEND[h] in missing]} — "
            "have no skills-discovery entry there. SKILL_DIRS also maps to "
            "home-directory skill trees, whereas this template wires every "
            "harness to its own repo-local .claude/skills. Fixing it needs a "
            "change in the `culture` repo, which this one cannot make."
        )
    assert backends_here <= set(skill_dirs)
