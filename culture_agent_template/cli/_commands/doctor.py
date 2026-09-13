"""``culture-agent-template doctor`` — check the agent-identity invariants.

Mirrors the two invariants ``steward doctor`` verifies for a mesh agent:

* **prompt-file-present** — the repo declares an agent in ``culture.yaml`` and
  has the matching prompt file on disk;
* **backend-consistency** — the declared ``backend`` is one this template
  knows, and the *resident* prompt file that backend actually reads is on disk
  (``claude`` → ``CLAUDE.md``, ``colleague`` → ``AGENTS.colleague.md``,
  ``acp``/``codex``/``copilot`` → ``AGENTS.md``, ``gemini`` → ``GEMINI.md``).

Additionally a **harness-prompts** info check reports which *other* recognized
harness prompt files are present. Those files (``AGENTS.override.md``,
``.pi/SYSTEM.md``, ``QWEN.md``) belong to interactively available harnesses
that ride the same backend name; they are never accepted as substitutes for
the resident prompt, because the Culture daemon does not read them.

Plus a **skills-present** check (the vendored ``.claude/skills/`` kit). Read-only.

Reports the rubric-shaped contract
``{healthy, checks: [{id, passed, severity, message, remediation}]}`` so the
agent-first rubric's bundle 7 passes. When run from a wheel install (no
``culture.yaml`` alongside the package), it reports a single info check and
exits 0 — there is nothing to diagnose.
"""

from __future__ import annotations

import argparse

from culture_agent_template.cli._commands.whoami import find_culture_yaml, read_agent_fields
from culture_agent_template.cli._output import emit_result

#: Shared by every AGENTS.md-reading backend (codex, acp, copilot).
_AGENTS_MD = "AGENTS.md"

# backend → every prompt file RECOGNIZED under that backend name.
#
# Values are tuples because four harnesses occupy only three backend names in
# this template: Qwen Code runs on ``acp`` (QWEN.md), and associate/Pi runs on
# ``colleague`` (AGENTS.override.md as context plus .pi/SYSTEM.md as the system
# prompt). This is a *recognition* table — it answers "does some harness on
# this backend read this file?", which is what tooling needs when it walks a
# checkout and attributes files to backends.
#
# It is deliberately NOT the health check. See ``_RESIDENT_PROMPT`` below.
#
# This mirrors ``backends[*].prompt`` in
# ``.claude/skills/agent-config/data/backend-fingerprints.yaml``;
# ``tests/test_harness_registries.py`` asserts the two never drift apart.
_PROMPT_FILE = {
    "claude": ("CLAUDE.md",),
    "colleague": ("AGENTS.colleague.md", "AGENTS.override.md", ".pi/SYSTEM.md"),
    "acp": (_AGENTS_MD, "QWEN.md"),
    "codex": (_AGENTS_MD,),
    "copilot": (_AGENTS_MD,),
    "gemini": ("GEMINI.md",),
}

# backend → the ONE prompt file the Culture daemon reads for a resident on
# that backend. This is what ``prompt_file_present`` requires.
#
# The distinction from ``_PROMPT_FILE`` is load-bearing. Several files are
# recognized under ``colleague`` and ``acp`` because *interactive harnesses*
# ride those backend names — Pi reads ``AGENTS.override.md`` + ``.pi/SYSTEM.md``
# and Qwen Code reads ``QWEN.md`` — but the mesh resident reads neither. A
# clone declaring ``backend: colleague`` with only Pi's files on disk has no
# resident prompt at all, and treating the harness files as interchangeable
# alternatives let exactly that pass as healthy.
_RESIDENT_PROMPT = {
    "claude": "CLAUDE.md",
    "colleague": "AGENTS.colleague.md",
    "acp": _AGENTS_MD,
    "codex": _AGENTS_MD,
    "copilot": _AGENTS_MD,
    "gemini": "GEMINI.md",
}


def _diagnose() -> dict[str, object]:
    cfg = find_culture_yaml()
    if cfg is None:
        check = {
            "id": "source_checkout",
            "passed": True,
            "severity": "info",
            "message": "no culture.yaml found alongside the package; identity checks skipped",
            "remediation": "",
        }
        return {"healthy": True, "checks": [check]}

    root = cfg.parent
    fields = read_agent_fields()
    backend = fields["backend"]
    checks: list[dict[str, object]] = []

    # 1. backend-consistency: the RESIDENT prompt file for the declared
    #    backend exists. Other harness prompt files recognized under the same
    #    backend name are reported separately (check 2) and never substituted
    #    here — the mesh daemon does not read them.
    resident = _RESIDENT_PROMPT.get(backend)
    if resident is None:
        checks.append(
            {
                "id": "backend_consistency",
                "passed": False,
                "severity": "error",
                "message": f"unknown backend '{backend}' in culture.yaml",
                "remediation": f"set backend to one of: {', '.join(sorted(_RESIDENT_PROMPT))}",
            }
        )
    else:
        present = (root / resident).is_file()
        checks.append(
            {
                "id": "prompt_file_present",
                "passed": present,
                "severity": "error",
                "message": (
                    f"backend '{backend}' reads {resident} as its resident prompt — "
                    + ("present" if present else "missing")
                ),
                "remediation": "" if present else f"create {resident} at the repo root",
            }
        )

        # 2. harness-prompts: report the other recognized prompt files on
        #    disk for this backend. Informational — an interactively available
        #    harness is not a health requirement, and its absence is not a
        #    failure — but it must never stand in for the resident prompt.
        others = [
            name
            for name in _PROMPT_FILE.get(backend, ())
            if name != resident and (root / name).is_file()
        ]
        checks.append(
            {
                "id": "harness_prompts",
                "passed": True,
                "severity": "info",
                "message": (
                    "other harness prompt files on this backend: "
                    + (", ".join(others) if others else "none")
                    + " (not substitutes for the resident prompt)"
                ),
                "remediation": "",
            }
        )

    # 3. skills-present: the vendored skill kit is on disk.
    skills_dir = root / ".claude" / "skills"
    has_skills = skills_dir.is_dir() and any(skills_dir.iterdir())
    checks.append(
        {
            "id": "skills_present",
            "passed": has_skills,
            "severity": "warning",
            "message": (
                ".claude/skills/ vendored" if has_skills else ".claude/skills/ missing or empty"
            ),
            "remediation": (
                "" if has_skills else "vendor the skill kit (see docs/skill-sources.md)"
            ),
        }
    )

    healthy = all(c["passed"] for c in checks)
    return {"healthy": healthy, "checks": checks}


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _diagnose()
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(report, json_mode=True)
    else:
        status = "healthy" if report["healthy"] else "unhealthy"
        lines = [f"culture-agent-template doctor: {status}", ""]
        for check in report["checks"]:
            mark = "ok" if check["passed"] else "FAIL"
            lines.append(f"[{mark}] {check['id']}: {check['message']}")
            if not check["passed"] and check["remediation"]:
                lines.append(f"  hint: {check['remediation']}")
        emit_result("\n".join(lines), json_mode=False)
    return 0 if report["healthy"] else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "doctor",
        help="Check the agent-identity invariants (prompt-file-present, backend-consistency).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_doctor)
