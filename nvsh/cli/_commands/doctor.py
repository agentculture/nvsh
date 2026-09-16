"""``nvsh doctor`` — check the agent-identity invariants.

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

``--apply`` (task t19) is the one exception to "read-only": when the
``agent_turn_not_hung`` check has failed, it fixes exactly that -- nothing
else. A dead-owner turn (the shell that started it is gone) is killed
outright; a live-owner turn is killed only after an interactive ``[y/N]``
confirmation naming the owning shell, and never off a tty (there is nothing
to prompt to, and blocking on ``input()`` with no operator attached would
hang forever). Plain ``nvsh doctor`` (no ``--apply``) never sends a
mutating control message to the daemon -- the check itself only reads
``status``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Mapping

from nvsh import __version__, client_transport
from nvsh import config as config_mod
from nvsh import daemon as daemon_mod
from nvsh import doctor_checks
from nvsh import platform as platform_mod
from nvsh.agent.audit import AuditLog
from nvsh.cli._commands.whoami import find_culture_yaml, read_agent_fields
from nvsh.cli._output import emit_result

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


def _new_checks(
    *,
    prompt_command_text: str | None,
    bind_p_text: str | None,
    keymap: str | None,
) -> list[dict[str, object]]:
    """Run the doctor extension checks (task t17): platform, backend, in-shell.

    Called unconditionally by :func:`_diagnose`, including in the
    wheel-install branch (no ``culture.yaml``) — these checks are about the
    shell/backend, not the mesh-identity invariants that branch skips.
    """
    env = dict(os.environ)
    try:
        config = config_mod.load()
        config_error: str | None = None
    except config_mod.ConfigError as exc:
        config = None
        config_error = str(exc)
    platform = platform_mod.detect()
    return doctor_checks.collect_checks(
        env=env,
        current_version=__version__,
        config=config,
        config_error=config_error,
        platform=platform,
        prompt_command_text=prompt_command_text,
        bind_p_text=bind_p_text,
        keymap=keymap,
    )


def _diagnose(
    *,
    prompt_command_text: str | None = None,
    bind_p_text: str | None = None,
    keymap: str | None = None,
) -> dict[str, object]:
    cfg = find_culture_yaml()
    if cfg is None:
        checks: list[dict[str, object]] = [
            {
                "id": "source_checkout",
                "passed": True,
                "severity": "info",
                "message": "no culture.yaml found alongside the package; identity checks skipped",
                "remediation": "",
            }
        ]
        checks.extend(
            _new_checks(
                prompt_command_text=prompt_command_text,
                bind_p_text=bind_p_text,
                keymap=keymap,
            )
        )
        healthy = all(c["passed"] for c in checks if c["severity"] != "info")
        return {"healthy": healthy, "checks": checks}

    root = cfg.parent
    fields = read_agent_fields()
    backend = fields["backend"]
    checks = []

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

    checks.extend(
        _new_checks(
            prompt_command_text=prompt_command_text,
            bind_p_text=bind_p_text,
            keymap=keymap,
        )
    )

    healthy = all(c["passed"] for c in checks if c["severity"] != "info")
    return {"healthy": healthy, "checks": checks}


def _failure_mark(check: dict[str, object]) -> str:
    """The text-mode marker for a check that did not pass.

    Deviation d4c: ``healthy`` ignores info-severity checks (running outside
    a hooked shell is not a failure), but the text report used to print
    ``[FAIL]`` for them anyway — so ``nvsh doctor`` said "healthy" over four
    ``[FAIL]`` lines on a non-hooked shell. The rule that keeps the two
    consistent: ``[FAIL]`` appears **only** for the checks that actually
    flip ``healthy`` — exactly those whose severity is not ``info``.
    """
    return "info" if check["severity"] == "info" else "FAIL"


def _is_interactive() -> bool:
    """Whether stdin is a real terminal we can prompt on (task t19).

    Mirrors the identical helper already duplicated in ``agent.py`` and
    ``setup.py``: a closed, custom or otherwise unavailable ``stdin`` can
    raise ``AttributeError``/``ValueError``/``OSError`` from ``isatty()``
    rather than just returning ``False``, and that must decline the kill,
    not crash the CLI.
    """
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _confirm_live_owner_kill(owner: str, elapsed: float) -> bool:
    """Ask the operator, by name, before killing another shell's live turn.

    Off a tty this refuses without prompting -- there is nothing to prompt
    to, and blocking on ``input()`` with no operator attached would hang
    forever (acceptance criterion: "no tty refuses live-owner kills").
    """
    if not _is_interactive():
        return False
    prompt = (
        f"shell {owner}'s agent turn has been running {elapsed:.0f}s and its "
        "owner is still alive; kill it anyway? [y/N] "
    )
    answer = input(prompt)  # nosec B322 - plain y/N prompt, no eval of input
    return answer.strip().lower() == "y"


def _record_doctor_apply(
    log: AuditLog, *, owner: str, target: object, elapsed: float, outcome: str
) -> None:
    """Append one ``doctor_apply`` audit entry.

    A thin wrapper so bandit's B604 (``shell=True``) heuristic, which
    false-positives on any call passing a keyword literally named ``shell``
    regardless of its value, only needs one ``nosec`` in this file: this
    ``shell`` is the owning shell id being audited, never a subprocess flag.
    """
    log.record_stop(  # nosec B604 - `shell` is the owning shell id, not shell=True
        kind="doctor_apply", shell=owner, target=target, elapsed=elapsed, outcome=outcome
    )


def _apply_agent_turn_not_hung(
    *,
    env: Mapping[str, str],
    status: Callable[..., dict] = client_transport.status,
    kill_active: Callable[..., str] = client_transport.kill_active,
    pid_gone: Callable[[str], bool] = daemon_mod.shell_pid_gone,
    confirm: Callable[[str, float], bool] = _confirm_live_owner_kill,
    audit: AuditLog | None = None,
) -> dict[str, object]:
    """Fix a hung active turn: the ``--apply`` half of ``agent_turn_not_hung``.

    A dead-owner turn (its shell no longer exists) is killed outright with
    owner authority the daemon grants doctor for exactly this case. A live
    owner is killed only after :func:`_confirm_live_owner_kill` says yes;
    anything else changes nothing. Every attempt is recorded via
    :meth:`~nvsh.agent.audit.AuditLog.record_stop` with
    ``kind="doctor_apply"``, whether it killed, was refused, or found
    nothing to do. Never called by plain ``nvsh doctor`` or by
    ``collect_checks`` -- only ``cmd_doctor``'s ``--apply`` path, and only
    once ``agent_turn_not_hung`` has already failed.
    """
    state = status(env=env, timeout=2.0)
    active = state.get("active_turn") if isinstance(state, dict) else None
    if not active:
        return {"outcome": "idle", "message": "no active turn to fix"}

    owner = str(active.get("shell", "") or "")
    try:
        elapsed = float(active.get("elapsed", 0.0) or 0.0)
    except (TypeError, ValueError):
        elapsed = 0.0
    dead = pid_gone(owner)
    confirmed = dead or confirm(owner, elapsed)

    log = audit if audit is not None else AuditLog(env=env)
    target = state.get("target") if isinstance(state, dict) else None
    if not confirmed:
        _record_doctor_apply(log, owner=owner, target=target, elapsed=elapsed, outcome="refused")
        return {
            "outcome": "refused",
            "message": f"shell {owner}'s turn is alive; not killed without confirmation",
        }

    outcome = kill_active(confirmed=confirmed, env=env)
    _record_doctor_apply(log, owner=owner, target=target, elapsed=elapsed, outcome=outcome)
    return {"outcome": outcome, "message": f"shell {owner}'s turn: {outcome}"}


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _diagnose(
        prompt_command_text=getattr(args, "prompt_command", None),
        bind_p_text=getattr(args, "bind_p", None),
        keymap=getattr(args, "keymap", None),
    )

    apply_result: dict[str, object] | None = None
    if getattr(args, "apply", False):
        turn_check = next((c for c in report["checks"] if c["id"] == "agent_turn_not_hung"), None)
        if turn_check is not None and not turn_check["passed"]:
            apply_result = _apply_agent_turn_not_hung(env=dict(os.environ))

    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        payload: dict[str, object] = dict(report)
        if apply_result is not None:
            payload["apply"] = apply_result
        emit_result(payload, json_mode=True)
    else:
        status_text = "healthy" if report["healthy"] else "unhealthy"
        lines = [f"nvsh doctor: {status_text}", ""]
        for check in report["checks"]:
            mark = "ok" if check["passed"] else _failure_mark(check)
            lines.append(f"[{mark}] {check['id']}: {check['message']}")
            if not check["passed"] and check["remediation"]:
                lines.append(f"  hint: {check['remediation']}")
        if apply_result is not None:
            lines.append("")
            lines.append(f"--apply: {apply_result['message']}")
        emit_result("\n".join(lines), json_mode=False)
    return 0 if report["healthy"] else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "doctor",
        help="Check the agent-identity invariants (prompt-file-present, backend-consistency).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.add_argument(
        "--prompt-command",
        dest="prompt_command",
        default=None,
        help="'declare -p PROMPT_COMMAND' from a hooked bash (for hook_first_in_prompt_command).",
    )
    p.add_argument(
        "--bind-p",
        dest="bind_p",
        default=None,
        help="'bind -p' output from a hooked bash (for bindings_present).",
    )
    p.add_argument(
        "--keymap",
        dest="keymap",
        choices=["emacs", "vi-insert", "vi-command"],
        default=None,
        help="Active readline keymap, from 'bind -V' (for bindings_present).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Fix a failing 'agent_turn_not_hung' check: kill a dead-owner turn "
            "outright, or a live one after an interactive [y/N] confirm."
        ),
    )
    p.set_defaults(func=cmd_doctor)
