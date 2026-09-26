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

**Current state:** the hook design (`CLAUDE.md`, `docs/architecture.md`) is
implemented, not just converged: the hook installer (`nvsh setup`/
`nvsh uninstall`/`nvsh on`/`nvsh off`), the trigger table, redaction,
platform detection, the pluggable `NvshAgent` backends, the session daemon,
the failure panel, slash commands and the approval store are all on disk
alongside the original agent-first verbs (`whoami`, `learn`, `explain`,
`overview`, `doctor`, `cli overview`). Still open: the login-shell (`chsh`)
mode is parked, auto-apply (running a fix without confirmation) is out of
scope for v1, and machine-level undo beyond the approve/execute/verify loop
is tracked as issue #7 — don't describe those as implemented.

Stopping the agent asks first: while it works, the first Ctrl+C or lone
Esc pauses the panel with a choice, `nvsh: paused -- [t] steer  [s] stop
[Esc] keep going` (`[t] stop & correct` where the harness cannot steer
mid-turn — only pi and codex can). `[Esc]`, or 30 s of silence, keeps
going and leaves the turn's exit status unaffected; `[s]` stops exactly as
before — the panel shows `stopping… press again to kill` and a further
press kills the harness's process tree (`NvshAgent.force_stop()`, daemon
`kill` control); `[t]` delivers the correction into the running turn or
cancels and resends it as the next request. A session without a terminal
stops at once, as before. A new request from a shell whose own turn is
still running (or whose owner shell is gone) gets a busy prompt — steer,
replace or exit; declining exits 3 (`EXIT_DECLINED`, visible in `--json`
and the audit log, never as the prompt's `$?`); `nvsh overview` shows the
active turn; `nvsh doctor --apply` clears a hung one. See
`docs/shell-integration.md` "Stopping the agent".

nvsh now registers eleven harness adapters (nine harnesses plus the
in-process `needle` and `lfm` tier adapters) (pi, qwen over ACP, qwen-p as a
read-only stream-json fallback, claude, codex, agy — always read-only for
commands, kiro over ACP, openai-compat, and demo — a scripted fixture
replayed through the real daemon and panel, used for the README recording,
excluded from setup's probe, and refused as a persisted default;
`nvsh/agent/registry.py`).
`[aliases]` in `$XDG_CONFIG_HOME/nvsh/config.toml` maps a short name to a
`backend[/model[/effort]]` target, `default` reserved for a bare
`nvsh --agent default`; `nvsh setup` probes `PATH` and picks whichever
harness is already installed as that default, rather than hard-wiring pi.
`@target` at the prompt marks one request for that harness only. Where a
harness has no approval channel it runs read-only:
"nvsh never edits, creates or overrides a harness's own settings or trust
files (agy/claude settings.json, codex config.toml, kiro trust settings,
qwen settings): it only passes launch flags and protocol-level policy, and
reports what it finds." Everything leaving the process is redacted first
(`nvsh/redact.py`), and a spawned harness's environment has
`CLAUDECODE`/`CLAUDE_CODE_*` stripped (`nvsh/agent/_env.py`).

**Local tiers (nvsh 0.16.0, opt-in, off by default).** With `[tiers] enabled = true` a request typed at the prompt is first offered to Needle3 (Tier 1, a 121M local model run in its own child process by the daemon) which picks one typed operation from `nvsh/ops/table.py`; nvsh grounds the arguments, renders the command from the table and shows it as an ordinary proposal naming the operation it understood — approval, `sudo` and destructive-command handling are unchanged, a failed command never goes to Tier 1, and `@target` bypasses the tiers. `needle` is a tenth registered adapter (`@needle`, in-process, excluded from `setup`'s probe and refused as a persisted default like `demo`). `nvsh tiers stats|export|prefetch|bench` and three `doctor` checks cover it. `lfm` is an eleventh (`@lfm`, in-process, same exclusions): the explicit Tier-2-only counterpart to `needle`, usable once `[tiers.lfm] model` is set — with no model configured it explains that in one line rather than building a broken tier. `nvsh/tiers/manager.py`'s `TierManager._build` wires the same `LfmTier` into the daemon's automatic Tier 2 slot when no `tier2_factory` is injected and the flavor is configured, so a FAILURE request reaches it directly rather than Tier 1 (which never sees a FAILURE at all). Tier 2 (`nvsh/tiers/lfm.py`, container launcher `nvsh/tiers/runtime_docker.py`, the only `docker run` in nvsh, arguments from config and detection only) runs read-only table operations on its own while it inspects, never a mutating one, and ends in propose, explain or escalate; stock LFM2.5 models measured on the DGX Spark almost never use propose or escalate, so read `docs/tier2.md` before describing Tier 2 as useful. A LoRA-tuned Needle3 works in JAX but cannot ship until the upstream export fault (cactus-compute/needle#134) is fixed. Measured accuracy of stock Needle3 is about half of the target, so read `docs/tiers-improving-accuracy.md` and `docs/needle-finetune.md` before describing the tiers as good; never add code that switches on a specific operation name — the table is the only place operations are named.

**Evaluation gate (issue 64, in progress, development only).** `evals/` holds a DeepEval-based release gate for the Tool-Jev checkpoints (`a3-heal.q4_k_m`, `scorer-r3b.q4_k_m`): it replays the candidates' saved per-case outputs, runs reference models (OpenAI, Anthropic, OpenRouter, build.nvidia.com, local) on the same issue-53 test cases, and reports model-only and model+harness rows separately. It lives in its own uv dependency group (`uv sync --group evals`), outside the root `tests/` testpaths, and nothing under `nvsh/` imports it; run its tests with `uv run pytest -c evals/pytest.ini --rootdir=. -q`. The runner (`uv run --group evals python -m evals.tool_jev run|continue|status|smoke|drive`) and the autonomous docker compose driver (`evals/docker/`, with Discord progress alerts) are built and have been exercised live; `evals/README.md` is the operator guide. The first full gate run is still in progress, so no gate result is published yet: read `docs/deepeval-gate.md` before describing it as producing results.

`CLAUDE.md` is written for a Claude Code session working *on* the repo. It is
not your runtime prompt, but it is the fullest write-up of the shell design
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
