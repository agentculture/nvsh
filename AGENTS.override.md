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
which diagnoses the failure and proposes a fix that the human confirms. The
operator's goal is to use nvsh as their **default login shell** on their
Spark and Jetson machines.

The spec is in GitHub issues **#1** (build brief) and **#2** (interactive
self-healing shell). **Current state:** the repo is still the
AgentCulture agent scaffold. Only the agent-first CLI verbs exist, and no
shell features have been built yet. When summarizing, don't describe planned
shell behavior as implemented.

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
repo's conventions and the planned shell design, so read it first. The other
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

Planned (not on disk yet): `docs/architecture.md` (the hook-vs-wrap
decision) and `docs/platforms.md` (where each detected device value comes
from).

## Conventions worth knowing before you answer a question about this repo

- The vendored skills under `.claude/skills/` are cited **verbatim** from
  guildmaster. Never propose editing their scripts; the fix belongs upstream
  (`docs/skill-sources.md` has the re-sync procedure).
- Every PR bumps the version (`version-bump` skill). CI's `version-check` job
  blocks merge otherwise.
- The runtime package has no third-party dependencies. Keeping it that way
  is intentional: a login shell has to start fast and must not break.
- Agent-suggested commands must never run without human confirmation. That
  is a core design rule, so flag any proposal that violates it.
- This file describes the repo **as it exists on disk today**. If you are
  asked to update it, keep claims grounded in what is checked in.
