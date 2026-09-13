# QWEN.md

This file provides guidance to Qwen Code when working with code in this
repository. Qwen Code's context loader reads exactly `QWEN.md` and `AGENTS.md`
in a directory. This repo deliberately ships only `QWEN.md` and no
`AGENTS.md` (each harness gets its own file; see "Prompt files by harness"
below), so this file is the only project guidance a Qwen Code session gets.

## What this project is

`nvsh` is an **agent-first shell for NVIDIA Jetson (AGX Orin, Thor), DGX
Spark and RTX Spark**. It runs commands like a normal shell. When a command
fails, it hands the error and device context to an agent (shell → agent),
which diagnoses the failure and proposes a fix. The human confirms before
anything runs. *A shell first, an agent second.*

The spec lives in GitHub issues **#1** (build brief) and **#2** (interactive
self-healing shell). Read them before designing a feature. The operator's
goal is to use nvsh as their **default login shell** (`chsh`) on their Spark
and Jetson machines.

nvsh is not a new POSIX shell (no bash parser or job-control rewrite). It is
not an autonomous agent that runs commands on its own. It is not a
device-management CLI either: it calls `jetson-cli` / `dgx-spark-cli` /
`rtx-spark-cli` when they are installed.

It is an AgentCulture mesh agent, a sibling to
[`guildmaster`](https://github.com/agentculture/guildmaster) (skills
supplier), [`steward`](https://github.com/agentculture/steward) (alignment,
`steward doctor`), and [`teken`](https://github.com/agentculture/teken) (the
afi-cli scaffolder this CLI is cited from).

## Current state

This is still the scaffold. No shell features exist yet, and only the
agent-first verbs are implemented (see "The CLI"). If you describe shell
behavior below as though it exists, mark it `(planned)`.

## Design constraints (planned work)

`CLAUDE.md` has the full write-up. The essentials:

- **Architecture:** a thin PTY wrapper around a real `bash`, not a
  reimplementation. The hook-vs-wrap decision goes in
  `docs/architecture.md` before building.
- **Login-shell safety:** if nvsh fails in any way, it falls back to `exec`
  of the real shell. Non-interactive invocations (`-c`, no TTY, `scp`,
  `rsync`, `ssh host cmd`) pass straight through with no extra stdout
  output. Login semantics (`-nvsh` / `-l`) are preserved. Commands that
  succeed get no added latency, and the runtime package has no third-party
  dependencies.
- **Trigger rules** are table-tested. `130`, `141`, `grep`/`diff` exiting
  `1`, `false`, and `test` are not errors. Automatic calls are rate-limited,
  and manual invocation (`nvsh ask`, `Ctrl+G`) always works.
- **Propose, don't run:** agent-suggested commands never run without
  confirmation.
- **Pluggable, offline-first backends** sit behind one adapter, with a
  fixture backend for tests. Nemotron is only the initial model. Config
  lives under `$XDG_CONFIG_HOME/nvsh/`.
- **Device context with redaction always on.** Record the source of each
  detected value in `docs/platforms.md`. Support `--show-context`, and give
  the redactor its own tests.

## Prompt files by harness

This repo's root carries one prompt file per agent harness, each read by
exactly one of them. There is no shared base file for them to inherit from:

- **Claude Code** → [`CLAUDE.md`](CLAUDE.md), the fullest write-up. Read it
  first if you are new to the repo.
- **Pi / associate** → [`AGENTS.override.md`](AGENTS.override.md) for context,
  plus [`.pi/SYSTEM.md`](.pi/SYSTEM.md) for its system prompt.
- **colleague** → [`AGENTS.colleague.md`](AGENTS.colleague.md).
- **Qwen Code** → this file.

When project facts change, update all four so they don't drift.

## Identity

Declared in `culture.yaml`:

```yaml
agents:
- suffix: nvsh
  backend: claude
```

`backend: claude` makes `CLAUDE.md` the *mesh resident* prompt file. The
mesh runtime reads that file, not this one. A Qwen Code session in this repo
is a separate, local tool session: it reads `QWEN.md` whatever
`culture.yaml` declares, and running Qwen Code here neither requires nor
changes that declaration.

## The CLI

The CLI is cited (cite-don't-import) from teken's `python-cli` reference, so
the runtime package has **no third-party dependencies**. `teken` is a dev
dependency only. The agent-first verbs:

- `nvsh whoami` reports identity from `culture.yaml`.
- `nvsh learn` prints a structured self-teaching prompt.
- `nvsh explain <path>` prints markdown docs for any noun/verb.
- `nvsh overview` gives a descriptive snapshot of the agent.
- `nvsh doctor` runs health checks (today the agent-identity invariants;
  planned: platform detection and agent-backend reachability).
- `nvsh cli overview` describes the CLI surface itself.

Conventions:

- Every command supports `--json`.
- Results go to stdout, and errors and diagnostics go to stderr; the two are
  never mixed.
- Handlers raise `CliError` with a remediation hint.
- Exit codes: `0` success, `1` user error, `2` environment error, `3+`
  reserved.
- Every new verb needs an `explain` catalog entry.
- CI enforces the agent-first rubric with `teken cli doctor . --strict`.

## Conventions

- **Every PR bumps the version**, even docs, config or CI changes. Use the
  `version-bump` skill; the `version-check` CI job blocks merge otherwise.
- **Tests:** `uv run pytest -n auto`. For a single test, use
  `uv run pytest tests/test_cli.py::test_whoami_json`.
- **Lint:** black, isort, flake8 (line length 100), bandit, markdownlint,
  and `scripts/scan-secrets.py`.
- **Deploy:** pushing to `main` publishes to PyPI through Trusted Publishing
  (`.github/workflows/publish.yml`) when `pyproject.toml` or `nvsh/**`
  changes. Same-repo PRs touching those paths publish a TestPyPI dev build;
  docs-only and fork PRs don't.
- **Skills:** `.claude/skills/` vendors the guildmaster skill kit verbatim
  (cite-don't-import). Don't edit vendored scripts; re-sync from upstream
  (`docs/skill-sources.md`).

## Layout

```text
nvsh/                     agent-first CLI (cited from teken's python-cli reference)
  cli/                    parser, error/output contract, _commands/ (verbs)
  explain/                markdown catalog for `explain`
tests/                    pytest smoke + introspection + harness tests
scripts/                  harness-smoke.py, scan-secrets.py (CI gates)
.claude/skills/           vendored guildmaster skill kit (cite-don't-import)
docs/                     harness selection/contract docs, skill provenance
culture.yaml              mesh identity (suffix + backend)
.github/workflows/        tests + deploy (PyPI Trusted Publishing)
```

This file describes the repository **as it exists on disk today**. When you
edit it, keep claims grounded in what is checked in. If a section drifts
ahead of reality, mark it `(planned)`.
