"""``doctor`` must require the prompt the DECLARED backend actually reads.

Four harnesses occupy only three Culture backend names in this template, so
several prompt files are *recognized* under one backend: ``colleague`` covers
``AGENTS.colleague.md`` (the resident prompt) plus Pi's ``AGENTS.override.md``
and ``.pi/SYSTEM.md``, and ``acp`` covers ``AGENTS.md`` plus Qwen Code's
``QWEN.md``.

Recognition is not health. The Culture daemon reads exactly one of those files
for a resident on a given backend; the others belong to interactively
available harnesses it never loads. Treating them as interchangeable let a
clone with no resident prompt at all report healthy — these tests pin the
distinction shut.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nvsh.cli._commands import doctor as doctor_mod


def _diagnose_in(root: Path, backend: str, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run ``_diagnose()`` against a throwaway tree declaring ``backend``."""
    cfg = root / "culture.yaml"
    cfg.write_text(f"agents:\n- suffix: nvsh\n  backend: {backend}\n", encoding="utf-8")
    monkeypatch.setattr(doctor_mod, "find_culture_yaml", lambda: cfg)
    monkeypatch.setattr(
        doctor_mod,
        "read_agent_fields",
        lambda: {"nick": "nvsh", "backend": backend, "model": "unknown"},
    )
    (root / ".claude" / "skills" / "cicd").mkdir(parents=True)
    (root / ".claude" / "skills" / "cicd" / "SKILL.md").write_text("x", encoding="utf-8")
    return doctor_mod._diagnose()


def _check(report: dict, check_id: str) -> dict:
    return next(c for c in report["checks"] if c["id"] == check_id)


# --- the resident prompt is required, alternates never substitute ---------


def test_colleague_is_unhealthy_when_only_pi_harness_files_are_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact hole: Pi's files satisfied a colleague declaration."""
    (tmp_path / "AGENTS.override.md").write_text("pi context", encoding="utf-8")
    (tmp_path / ".pi").mkdir()
    (tmp_path / ".pi" / "SYSTEM.md").write_text("pi system prompt", encoding="utf-8")

    report = _diagnose_in(tmp_path, "colleague", monkeypatch)

    assert report["healthy"] is False
    prompt = _check(report, "prompt_file_present")
    assert prompt["passed"] is False
    assert "AGENTS.colleague.md" in prompt["message"]
    # ...and the harness files are reported, not counted as the resident.
    harness = _check(report, "harness_prompts")
    assert harness["severity"] == "info"
    assert "AGENTS.override.md" in harness["message"]
    assert ".pi/SYSTEM.md" in harness["message"]


def test_acp_is_unhealthy_when_only_qwen_md_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QWEN.md is Qwen Code's file; the acp resident reads AGENTS.md."""
    (tmp_path / "QWEN.md").write_text("qwen guidance", encoding="utf-8")

    report = _diagnose_in(tmp_path, "acp", monkeypatch)

    assert report["healthy"] is False
    assert _check(report, "prompt_file_present")["passed"] is False
    assert "QWEN.md" in _check(report, "harness_prompts")["message"]


def test_colleague_is_healthy_with_its_own_resident_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.colleague.md").write_text("colleague prompt", encoding="utf-8")

    report = _diagnose_in(tmp_path, "colleague", monkeypatch)

    assert report["healthy"] is True
    assert _check(report, "prompt_file_present")["passed"] is True
    assert "none" in _check(report, "harness_prompts")["message"]


def test_claude_is_unhealthy_when_claude_md_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No other harness file rescues the backend this template declares."""
    (tmp_path / "QWEN.md").write_text("qwen guidance", encoding="utf-8")
    (tmp_path / "AGENTS.colleague.md").write_text("colleague prompt", encoding="utf-8")

    report = _diagnose_in(tmp_path, "claude", monkeypatch)

    assert report["healthy"] is False
    assert _check(report, "prompt_file_present")["passed"] is False


def test_an_unknown_backend_still_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _diagnose_in(tmp_path, "notabackend", monkeypatch)
    assert report["healthy"] is False
    assert _check(report, "backend_consistency")["passed"] is False


# --- the two tables stay coherent with each other -------------------------


def test_every_resident_prompt_is_recognized_under_its_own_backend() -> None:
    for backend, resident in doctor_mod._RESIDENT_PROMPT.items():
        assert resident in doctor_mod._PROMPT_FILE[backend], (
            f"backend {backend!r} reads {resident} but the recognition table "
            f"does not list it: {doctor_mod._PROMPT_FILE[backend]}"
        )


def test_the_two_tables_cover_the_same_backends() -> None:
    assert set(doctor_mod._RESIDENT_PROMPT) == set(doctor_mod._PROMPT_FILE)


def test_this_repo_declares_claude_and_ships_its_resident_prompt() -> None:
    """The template's own declaration and prompt file agree on disk."""
    repo = Path(__file__).resolve().parents[1]
    assert "backend: claude" in (repo / "culture.yaml").read_text(encoding="utf-8")
    assert (repo / doctor_mod._RESIDENT_PROMPT["claude"]).is_file()
