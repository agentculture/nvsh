# Colleague Resident — `nvsh`

You are a colleague session working in this repo. You are reading this file
because colleague's prompt cascade resolves it here, not because
`culture.yaml` selected you. That declaration says `backend: claude`, so
`CLAUDE.md` is the *mesh resident* prompt. colleague is still fully usable
interactively over the same clone, and this file is what it loads when you
run it.

Your job is to help with scoped tasks delegated by the operator or peer
agents, using the colleague tool-loop (`read_file` / `write_file` /
`edit_file` / `list_dir` / `run_command` / `finish`).

## The prompt cascade (and what this repo actually ships)

colleague concatenates up to three files, in order, as its prompt cascade:

1. `AGENTS.md`, a shared base, if present.
2. `AGENTS.colleague.md`, this file.
3. `AGENTS.colleague.<sanitized-model>.md`, a model-specific override, if
   present.

**This repo ships only layer 2.** There is deliberately no `AGENTS.md` at the
root: a shared base across the four harness files was proposed and rejected,
and each harness gets its own unrelated file. There is also no per-model
override. So the cascade for colleague here starts and ends at this file. If
you add either of the other files later, update this section.

## What this project is

`nvsh` is an **agent-first shell for NVIDIA Jetson (AGX Orin, Thor), DGX
Spark and RTX Spark**. It runs commands like a normal shell. When a command
fails, it hands the error and device context to an agent (shell → agent),
which diagnoses the failure and proposes a fix. nvsh **hooks into the
operator's existing bash** — one marked block that `nvsh setup` inserts
into the rc file adds a function to the `PROMPT_COMMAND` array — rather
than wrapping bash in a pty; see
[`docs/architecture.md`](docs/architecture.md) for the decision. A
default-login-shell (`chsh`) mode is a parked possible follow-up, not this
scope. The spec is in GitHub issues #1 and #2.

**Current state:** still mostly the AgentCulture scaffold, with the hook
design now converged (`CLAUDE.md`, `docs/architecture.md`) but its
implementation still in progress. Only the agent-first verbs (`whoami`,
`learn`, `explain`, `overview`, `doctor`, `cli overview`) exist on disk
today; the hook and shell verbs are not built yet.

`CLAUDE.md` is written for a Claude Code session working *on* the repo. It is
not your runtime prompt, but it is the fullest write-up of the planned design
and the repo's conventions. Read it before any design or implementation
task.

## Rules that apply to any change you make

- **Hook safety first.** Any failure in nvsh's own hook must degrade and
  return control to the prompt, never block it (`NVSH_DISABLE=1` disables
  the hook outright). The hook only installs into interactive shells, so
  non-interactive invocations (`-c`, no TTY, `scp`, `rsync`) never source it
  and get nothing written to stdout. Commands that succeed get no added
  latency. Don't add third-party runtime dependencies.
- **Propose, don't run.** Never make nvsh run agent-suggested commands
  without human confirmation.
- **Trigger rules and the redactor need table-driven tests.** Agent backends
  sit behind an adapter with a fixture backend for tests.
- **CLI contract:** every verb supports `--json`. Results go to stdout and
  errors to stderr. Handlers raise `CliError` with a remediation hint. Every
  verb needs an `explain` catalog entry, and `teken cli doctor . --strict`
  must pass.

## How you work

- Prefer small, reversible steps, and hand off with `finish` when done.
- Verify with `uv run pytest -n auto` and the linters (black, isort, flake8,
  line length 100) before you finish.
- Follow the operator's instructions and any skills loaded from
  `.colleague/skills/` when present.
- The vendored skills under `.claude/skills/` are cited **verbatim** from
  guildmaster. Don't reformat or edit their scripts; a fix belongs upstream
  (see `docs/skill-sources.md` for the re-sync procedure).
- Every PR bumps the version (`version-bump` skill). CI's `version-check` job
  blocks merge otherwise.
