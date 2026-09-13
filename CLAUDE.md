# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`culture-agent-template` is a **clonable template for AgentCulture mesh agents**.
It is a working, minimal example of the sibling pattern every Culture agent
follows: an agent-first CLI, a mesh identity, the canonical skill kit, and a
buildable/deployable package baseline. Clone it, rename the package, edit
`culture.yaml`, and you have a new agent that `steward doctor` recognizes.

It is a sibling to [`guildmaster`](https://github.com/agentculture/guildmaster)
(the **skills supplier**), [`steward`](https://github.com/agentculture/steward)
(**alignment** — `steward doctor`, the sibling-pattern baseline), and
[`teken`](https://github.com/agentculture/teken) (the **afi-cli** "Agent First
Interface" scaffolder this CLI is cited from) within the Organic Development
framework.

## Prompt files by harness

This repo's root carries one prompt file per agent harness, each read by
exactly one of them — there is deliberately no shared base file (e.g. a
common `AGENTS.md`) for them to inherit from:

- **Claude Code** → this file, `CLAUDE.md`.
- **Pi / associate** → [`AGENTS.override.md`](AGENTS.override.md) for
  project context (Pi loads `AGENTS.override.md` in a directory *instead of*
  that directory's `AGENTS.md`/`CLAUDE.md`, so it does not inherit this file),
  plus [`.pi/SYSTEM.md`](.pi/SYSTEM.md), which replaces Pi's default
  coding-assistant system prompt with `associate`'s non-coding
  read/find/summarize identity.
- **colleague** → [`AGENTS.colleague.md`](AGENTS.colleague.md) — the first
  (and, in this repo, only) layer of colleague's prompt cascade; see that
  file for what the cascade is and why it stops there.
- **Qwen Code** → [`QWEN.md`](QWEN.md) (Qwen Code's context loader reads
  `QWEN.md` and `AGENTS.md`; there is no `AGENTS.md` here, so `QWEN.md` is its
  sole source of guidance).

## Identity

Declared in `culture.yaml`:

```yaml
agents:
- suffix: culture-agent-template
  backend: claude
```

This repo ships four harness prompt files (see
[Prompt files by harness](#prompt-files-by-harness) above), but `culture.yaml`
only ever selects **one** of them — and that selection is not the same thing
as "which harness can I use right now." There are **two separate selections**
over one clone, and this template exists to keep them from being conflated:

1. **The interactive harness** — chosen by which binary you run. `cd` into
   this clone and run `claude`, `pi`, `colleague`, or `qwen`; whichever you
   invoke reads its own prompt file (`CLAUDE.md`, `AGENTS.override.md` +
   `.pi/SYSTEM.md`, `AGENTS.colleague.md`, or `QWEN.md`). All four are live
   simultaneously against the same working tree — there is no switch to flip
   and no file to edit to "pick" one, and any harness can be force-chosen by
   a human or by automation (see `docs/automation-contract.md`) regardless of
   what `culture.yaml` says. Forcing a harness this way is invocation-level
   only — flags to that one process — and must never rewrite `culture.yaml`
   or any other tracked file, since a shared checkout may have other callers
   running concurrently.
2. **The mesh resident** — the single backend `culture.yaml` declares, which
   is what the Culture daemon starts and what `steward doctor` checks.
   `guild harness use <name>` (guildmaster's write verb) changes only this
   selection; it does not touch which binaries you can run interactively.

`backend: claude` fixes the mesh-resident prompt file to **this file**,
`CLAUDE.md` — the Culture daemon starts the `claude` backend for this agent,
and the CLI's own `culture-agent-template doctor` (plus `steward doctor`)
checks the two invariants that follow from the declaration:
**prompt-file-present** (an agent is declared and the matching prompt file is
on disk) and **backend-consistency** (`claude` ↔ `CLAUDE.md`). The other three
prompt files still exist and are still fully usable interactively — declaring
`backend: claude` does not remove or disable `AGENTS.override.md` +
`.pi/SYSTEM.md`, `AGENTS.colleague.md`, or `QWEN.md`; it only decides which one
the mesh daemon reads for this clone.

See [`docs/harness-selection.md`](docs/harness-selection.md) for the full
writeup of these two selections, who reads the harness config, and why this
template does not retrofit already-provisioned siblings.

## Cloning this template (re-initialization)

When you start a new agent from this template:

1. Rename the package directory `culture_agent_template/` → `<your_module>/` and
   replace `culture_agent_template` (module) / `culture-agent-template` (CLI and
   dist name) throughout `pyproject.toml`, the package, `tests/`,
   `sonar-project.properties`, and `README.md`. The name is hard-coded in ~100
   places — including every CLI command file under the package and `_ISSUES_URL`
   in `culture_agent_template/cli/__init__.py` — so list every occurrence first
   rather than renaming by hand (`git grep` is portable and skips `.git` /
   untracked `__pycache__`):

   ```bash
   git grep -nF -e 'culture-agent-template' -e 'culture_agent_template'
   ```

2. Set your `suffix` (and `backend`) in `culture.yaml`. `whoami` and `doctor`
   then reflect the new identity with no further code change.
3. Rewrite **this file** to describe your agent, and run `/init` to regenerate
   guidance grounded in your actual repo.
4. Re-vendor the skill kit you need from guildmaster (see
   `docs/skill-sources.md`) — keep only the skills your agent uses.

## The CLI

The CLI is cited (cite-don't-import) from teken's `python-cli` reference
(`teken cli cite`), so the runtime package has **no third-party dependencies**;
`teken` (a.k.a. `afi-cli`) is a dev dependency only. Agent-first verbs:

- `culture-agent-template whoami` — identity from `culture.yaml`.
- `culture-agent-template learn` — structured self-teaching prompt.
- `culture-agent-template explain <path>` — markdown docs for any noun/verb.
- `culture-agent-template overview` — descriptive snapshot of the agent.
- `culture-agent-template doctor` — check the agent-identity invariants.
- `culture-agent-template cli overview` — describe the CLI surface itself.

Conventions: every command supports `--json`; results go to stdout, errors and
diagnostics to stderr (never mixed); exit codes are `0` success, `1` user error,
`2` environment error, `3+` reserved. The agent-first rubric is enforced in CI by
`teken cli doctor . --strict`.

## Skills

`.claude/skills/` vendors the **canonical guildmaster skill kit** (cite-don't-import).
Provenance and the re-sync procedure live in `docs/skill-sources.md`. Three skills
(`think`, `spec-to-plan`, `assign-to-workforce`) originate in `devague`
(re-broadcast via guildmaster). The `ask-colleague` skill (the first-party front
door to the `colleague` CLI, formerly `outsource`) originates in `colleague` and
is vendored directly from it — guildmaster's re-broadcast still carries the old
`outsource` name, so this is a tracked local divergence (see
`docs/skill-sources.md`). Tooling prerequisites: **`devex`** on PATH (the `cicd`
skill delegates the PR lifecycle to `devex pr`) and **`agtag`** on PATH (the
`communicate` skill wraps `agtag issue`); **`colleague`** on PATH is an *optional*
prerequisite the `ask-colleague` skill needs only when invoked (it exits with an
install hint if absent).

## Conventions

- **Reach for `ask-colleague` reflexively.** Treat the `ask-colleague` skill (the
  `colleague` CLI) as the teammate at the next desk, not a last resort — its value
  is a *second, independent mind* (a different backend/model), not a stronger one.
  Lean on it to explore, review, write, or grade work per the skill's own rubric:
  before presenting or opening a PR on a non-trivial committed diff, run `review`
  for a diverse second opinion; for a fresh read of an unfamiliar area whose answer
  is independent of your current context, run `explore`. Both are **read-only**
  (isolated in a throwaway worktree, zero side effects), so the reflex is always
  safe. The side-effecting `write --apply` / `write --pr` still needs the user's
  go-ahead. Colleague's output is a second opinion to verify and own, never
  authority. `colleague` on PATH is an optional prerequisite (the skill exits with
  an install hint if absent).
- **Every PR bumps the version** — even docs/config/CI. Use the `version-bump`
  skill; the `version-check` CI job blocks merge otherwise.
- **PRs** go through the `cicd` skill (`devex pr` + SonarCloud gating). Sign
  online posts as `- culture-agent-template (Claude)` — the `cicd` / `communicate`
  scripts resolve the nick from `culture.yaml` automatically.
- **Tests**: `uv run pytest -n auto`. **Lint**: black, isort, flake8 (line length
  100), bandit, markdownlint.
- **Deploy**: pushing to `main` publishes to PyPI via Trusted Publishing
  (`.github/workflows/publish.yml`); PRs do a TestPyPI dry-run. Configure the
  `pypi` / `testpypi` GitHub environments and a PyPI Trusted Publisher before the
  publish job can succeed.
- The vendored `.claude/skills/` are cited verbatim — do not reformat or edit
  their scripts; re-sync from guildmaster instead (see `docs/skill-sources.md`).

## Layout

```text
culture_agent_template/   agent-first CLI (cited from teken's python-cli reference)
  cli/                    parser, error/output contract, _commands/ (verbs)
  explain/                markdown catalog for `explain`
tests/                    pytest smoke + introspection tests
.claude/skills/           vendored guildmaster skill kit (cite-don't-import)
docs/skill-sources.md     skill provenance ledger
culture.yaml              mesh identity (suffix + backend)
.github/workflows/        tests + deploy (PyPI Trusted Publishing)
```

This file describes the repository **as it exists on disk today**. When you edit,
keep claims grounded in checked-in reality; if a section drifts ahead of reality,
mark it `(planned)` or move it under a `## Roadmap` heading.

## Conventions and workflow

**Git worktrees you create live in `../.worktrees.culture-agent-template/<name>/`.**
Every worktree you add by hand — workforce fan-out lanes, scratch checkouts,
anything you will still care about in an hour — goes in that one repo-named
directory beside the checkout, one subfolder per worktree. Never a shared
sibling folder, never anywhere else:

```bash
git worktree add ../.worktrees.culture-agent-template/<name> -b <branch>
```

Do **not** use a shared `../worktrees/` directory. This workspace holds many
sibling projects, and a generic shared
folder accumulates orphaned trees from several repos at once with nothing
indicating who owns which — someone clearing stale trees cannot tell yours from
junk, and a `rm -rf` on the shared folder takes your lane with it. The
repo-named, dot-prefixed folder makes ownership unambiguous and keeps the
sweep-up safe. An unowned worktree is a stale worktree — so make sure the name
says who the owner is.

Use a branch prefix scoped to the work (`retire/t2`, not `agent/t2`): plain
`agent/*` names collide with leftovers from earlier fan-outs and `git worktree
add -b` fails on an existing branch.

**Following the vendored `assign-to-workforce` skill:** its fan-out example
uses both the shared `../worktrees/` path and `agent/<task-id>` branch names —
the two things above say not to. That skill is cited verbatim and must not be
edited, so override *both* when you follow it: put the worktree in this repo's
directory and scope the branch prefix to the work.

**Exception — tool-managed throwaways.** `ask-colleague`'s `explore` / `review`
/ `write`-preview verbs create their own detached worktree under
`${TMPDIR:-/tmp}` (`.claude/skills/ask-colleague/scripts/ask-colleague.sh`) and
delete it on an EXIT trap. That is vendored code this repo does not edit, the
tree never outlives the command, and nobody has to guess who owns it — so it is
outside this rule, not a violation of it. The rule targets worktrees that
*persist* and therefore need an owner. Expect `git worktree list` to show one of
these while such a command is in flight.

Remove a worktree you are done with using `git worktree remove <path>`, which
deletes the directory and its bookkeeping together. `git worktree prune` only
clears stale metadata for worktrees whose directories are *already* gone — it
never deletes a directory, so it tidies the list but does not finish the job.
Never `rm -rf` a worktree directory you did not create.

When cloning this template, the directory name follows the new repo name — the
`git grep` rename sweep in [Cloning this template](#cloning-this-template-re-initialization)
already catches it.

**Memory discipline — recall before, remember after.** This repo keeps its
eidetic memory **in-repo and public**: records resolve to
`<repo-root>/.eidetic/memory` — committed, and shared with the team and mesh
peers (the `claude` and `colleague` backends both read the same
`culture-agent-template` scope), so memory travels with the repo, not a private
home-dir store. Make it a per-task habit:

- **`/recall` before you start.** Search the store for the area you're about
  to touch — prior decisions, gotchas, "have we done this before?" — so you
  build on what's already known instead of re-deriving it. Do this before
  non-trivial tasks, not just when asked.
- **`/remember` when something worth keeping surfaces.** A non-obvious
  decision and its rationale, a constraint, a fix and *why* it was needed, a
  gotcha that cost time, a fact the next session would otherwise re-learn.
  Capture it as it happens, not at the end when it's faded.

A plain `/remember` lands the note in `./.eidetic/memory` in this repo — no
flag needed (the wrappers here default to `--visibility public`; in-repo
routing needs `eidetic >= 0.10.0`, older CLIs keep records in `$HOME`). Keep
something out of the committed store only by passing `--visibility private`
(routes to `$HOME/.eidetic/memory`, never committed); `/recall` reads both
stores and merges. Don't store what the repo already records (code structure,
git history, what's already in this file or `CHANGELOG.md`) — store what you'd
have to re-derive. These are the `recall`/`remember` skills (`.claude/skills/`),
backed by the `eidetic` store.
