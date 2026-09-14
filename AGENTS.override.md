# AGENTS.override.md

This file is the **context layer** for the Pi harness (the `pi` CLI, and the
`associate` non-coding harness modelled on it) when it runs inside this repo.
Pi's CONTEXT loader concatenates `AGENTS.md` or `CLAUDE.md` from its
user-level config directory (see Pi's own docs), from each parent directory,
and from the working directory. An `AGENTS.override.md` in a directory
replaces that directory's `AGENTS.md`/`CLAUDE.md` entry outright instead of
adding to it. That is why this repo ships this file instead of an
`AGENTS.md`: Pi must **not** inherit `CLAUDE.md` (the Claude Code guidance
file). The two harnesses read the same repository very differently, and
`CLAUDE.md` assumes a coding session with full repo-write authority that Pi's
non-coding lane does not have.

The identity and behavioral bounds for that lane (who Pi is here, what it may
and may not do) live one layer up, in Pi's **system prompt** file,
[`.pi/SYSTEM.md`](.pi/SYSTEM.md). That file replaces Pi's default
coding-assistant system prompt entirely. This file is project *context* only:
what the repo is and how it is laid out, not who is reading it.

## What this project is

`nvsh` is an **agent-first shell for NVIDIA Jetson (AGX Orin, Thor), DGX
Spark and RTX Spark**. It runs commands like a normal shell. When a command
fails, it hands the error and device context to an agent (shell → agent),
which diagnoses the failure and proposes a fix that the human confirms.
nvsh **hooks into the operator's existing bash** — one marked block that
`nvsh setup` inserts into the operator's rc file adds a function to the
`PROMPT_COMMAND` array — rather than wrapping bash in a pty; see
[`docs/architecture.md`](docs/architecture.md) for the decision and its
reasons. A default-login-shell (`chsh`) mode is a parked possible follow-up,
not this scope.

The spec is in GitHub issues **#1** (build brief) and **#2** (interactive
self-healing shell). **Current state:** the hook-and-agent-on-error design
is implemented, not just converged — the hook installer (`nvsh setup`/
`nvsh uninstall`/`nvsh on`/`nvsh off`), the trigger table, redaction,
platform detection, the pluggable `NvshAgent` backends, the session daemon,
the failure panel, slash commands and the approval store are all on disk
alongside the original agent-first CLI verbs. Still open: the
default-login-shell (`chsh`) mode is parked, an auto-apply mode (running a
fix without confirmation) is out of scope for v1, and machine-level undo
beyond the approve/execute/verify loop is tracked as issue #7. When
summarizing, don't describe those still-open items as implemented.

nvsh now registers eight harness adapters (`nvsh/agent/registry.py`'s
`ADAPTERS`): `pi`, `qwen` (ACP, plan mode by default), `qwen-p`
(stream-json print-mode, read-only fallback), `claude`, `codex`, `agy`
(stream-json, always read-only for commands), `kiro` (ACP) and
`openai-compat`. `[aliases]` in `$XDG_CONFIG_HOME/nvsh/config.toml` maps a
short name to a `backend[/model[/effort]]` target, with `default` reserved
for a bare `nvsh --agent default`; `@target` at the prompt (`@name` or
`@backend/model/effort`) marks one request for that harness only. Where a
harness has no client-side approval channel it runs read-only rather than
being auto-approved: the spec is explicit that "nvsh never edits, creates
or overrides a harness's own settings or trust files (agy/claude
settings.json, codex config.toml, kiro trust settings, qwen settings): it
only passes launch flags and protocol-level policy, and reports what it
finds," and that an operator opts a harness into its own agent-side
approval only with `[agents.<name>] approval = "harness"`. Everything that
leaves the process is redacted first (`nvsh/redact.py`), and a spawned
harness's environment has `CLAUDECODE`/`CLAUDE_CODE_*` stripped
(`nvsh/agent/_env.py`).

It is an AgentCulture mesh agent, a sibling to
[`guildmaster`](https://github.com/agentculture/guildmaster) (the skills
supplier), [`steward`](https://github.com/agentculture/steward) (alignment),
and [`teken`](https://github.com/agentculture/teken) (the CLI scaffolder this
package is cited from). At the device level it relates to `jetson-cli`,
`dgx-spark-cli` and `rtx-spark-cli`, which nvsh calls and does not duplicate.

## Four harnesses, four files, no shared base

This repo's root carries one prompt file per harness, each read by exactly
one of them. There is deliberately no shared `AGENTS.md` base for them to
cascade from:

- **Claude Code** reads [`CLAUDE.md`](CLAUDE.md).
- **Pi / associate** reads this file (`AGENTS.override.md`) for context, plus
  [`.pi/SYSTEM.md`](.pi/SYSTEM.md) for its system prompt.
- **colleague** reads [`AGENTS.colleague.md`](AGENTS.colleague.md) (the start
  of colleague's own cascade; see that file).
- **Qwen Code** reads [`QWEN.md`](QWEN.md).

If you are a human reading this, `CLAUDE.md` is the fullest write-up of the
repo's conventions and the shell design, so read it first. The other
three files exist so that no non-Claude harness silently inherits
Claude-specific instructions it can't act on the same way.

## Identity

Declared in `culture.yaml`:

```yaml
agents:
- suffix: nvsh
  backend: claude
```

The *mesh* resident runs on `backend: claude`, so `CLAUDE.md` is the live
resident prompt. A Pi session working in this repo is a **local tool
session**, not the mesh resident. It reads this file and `.pi/SYSTEM.md`
whatever `culture.yaml` declares, and running `pi` here neither requires nor
changes that declaration.

## Layout (what you can read/find/summarize here)

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

`docs/architecture.md` (the hook-vs-wrap decision) and `docs/platforms.md`
(where each detected device value comes from) now exist, and so do the
detectors and the hook installer they describe.

## Conventions worth knowing before you answer a question about this repo

- The vendored skills under `.claude/skills/` are cited **verbatim** from
  guildmaster. Never propose editing their scripts; the fix belongs upstream
  (`docs/skill-sources.md` has the re-sync procedure).
- Every PR bumps the version (`version-bump` skill). CI's `version-check` job
  blocks merge otherwise.
- The runtime package has no third-party dependencies. Keeping it that way
  is intentional: nvsh's Python entrypoint runs on every qualifying shell
  failure and must not break when a venv or wheel breaks.
- Agent-suggested commands must never run without human confirmation. That
  is a core design rule, so flag any proposal that violates it.
- This file describes the repo **as it exists on disk today**. If you are
  asked to update it, keep claims grounded in what is checked in.
