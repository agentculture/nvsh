"""Markdown catalog for ``nvsh explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple
and ``("nvsh",)`` both resolve to the root entry.

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# nvsh

An agent-first shell for NVIDIA Jetson, DGX Spark and RTX Spark. It runs your
commands like a normal shell; when a command fails, it hands the error and
device context to an agent (shell -> agent) to diagnose and propose a fix, which
you confirm before anything runs.

Early scaffold: the agent-first verbs below exist today; the shell itself is
planned (GitHub issues #1 and #2). nvsh is also an AgentCulture mesh agent
(`culture.yaml` + `CLAUDE.md`).

## Verbs

- `nvsh whoami` — identity probe from `culture.yaml`.
- `nvsh learn` — structured self-teaching prompt.
- `nvsh explain <path>` — markdown docs for any noun/verb.
- `nvsh overview` — descriptive snapshot of the agent.
- `nvsh doctor` — check the agent-identity invariants.
- `nvsh cli overview` — describe the CLI surface.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error
- `3+` reserved

## See also

- `nvsh explain whoami`
- `nvsh explain doctor`
"""

_WHOAMI = """\
# nvsh whoami

Reports the agent's identity from `culture.yaml`: nick (`suffix`), backend,
served model, and the package version. Read-only.

## Usage

    nvsh whoami
    nvsh whoami --json
"""

_LEARN = """\
# nvsh learn

Prints a structured self-teaching prompt covering purpose, command map,
exit-code policy, `--json` support, and the `explain` pointer.

## Usage

    nvsh learn
    nvsh learn --json
"""

_EXPLAIN = """\
# nvsh explain <path>

Prints markdown documentation for any noun/verb path. Unlike `--help` (terse,
positional), `explain` is global and addressable by path.

## Usage

    nvsh explain nvsh
    nvsh explain whoami
    nvsh explain --json <path>
"""

_OVERVIEW = """\
# nvsh overview

Read-only descriptive snapshot of the agent: identity (from `culture.yaml`), the
verb surface, and the sibling-pattern artifacts the template carries. Accepts an
ignored `target` so a stray path never hard-fails.

## Usage

    nvsh overview
    nvsh overview --json
"""

_DOCTOR = """\
# nvsh doctor

Checks the agent-identity invariants `steward doctor` verifies:
prompt-file-present and backend-consistency (`claude` → `CLAUDE.md`), plus a
skills-present check. Exits 1 when unhealthy.

prompt-file-present requires the *resident* prompt the declared backend
actually reads. Other harness prompt files recognized under the same backend
name (`AGENTS.override.md`, `.pi/SYSTEM.md`, `QWEN.md`) belong to
interactively available harnesses the mesh daemon never loads; they are
reported by the informational harness-prompts check and never substituted.

## Usage

    nvsh doctor
    nvsh doctor --json
"""

_CLI = """\
# nvsh cli

Noun group for CLI-surface introspection. `cli overview` describes the CLI
itself (distinct from the global `overview`, which describes the agent).

## Usage

    nvsh cli overview
    nvsh cli overview --json
"""


ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("nvsh",): _ROOT,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
    ("cli",): _CLI,
    ("cli", "overview"): _CLI,
}
