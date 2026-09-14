# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What nvsh is

`nvsh` is a **shell for NVIDIA Jetson (AGX Orin, Thor), DGX Spark and RTX
Spark that calls an agent when a command fails** (shell → agent). A human
works at the prompt as usual. The agent comes in only when something breaks:
it diagnoses the failure, proposes a fix, and after the human approves it
applies the fix, retries the command and checks the result. *"A shell first,
an agent second."*

The spec lives in two GitHub issues. Read them before designing a feature
(`gh issue view 1`, `gh issue view 2`):

- **#1: Build brief** (guildmaster). Covers the hook-vs-wrap decision,
  trigger rules, "propose, don't run", offline-first pluggable backends,
  device context with required redaction, and the first-milestone checklist.
- **#2: Interactive self-healing shell.** Covers the interactive UX (inline
  diagnosis panels, `Ctrl+G` to call the agent, slash commands), deterministic
  operator tools under the model, long-context machine awareness, and
  known-good state (`remember-good` / `diff-good` / `restore-good`). The
  PTY-wrapper delivery this issue originally sketched is retired — see
  [`docs/architecture.md`](docs/architecture.md) — but its inline-panel UX,
  deterministic tools and activation rules carry forward onto the hook.
  Nemotron ("associate", served by Pi) is the first model, but it must not
  be hard-wired.

**Architecture: nvsh HOOKS into the operator's existing bash, it does not
wrap it.** `nvsh setup` inserts one marked block into the operator's rc file
right after the distro's interactive guard, which appends nvsh's function to
the `PROMPT_COMMAND` array; bash itself still parses, does job control,
completion, aliases and rc files exactly as before. See
[`docs/architecture.md`](docs/architecture.md) for the decision and its
reasons, and [Hook constraints](#hook-constraints) for what that means in
practice. **The login-shell (`chsh`) goal is parked, not dropped** — a
wrapper is a possible later phase on top of the same agent and tools layer,
but is not this scope; see docs/architecture.md's "Parked: login-shell mode".

nvsh is **not** a new POSIX shell (don't rewrite a bash parser or job
control). It is not an autonomous agent that runs commands on its own
(`shell-cli` / `callsmith` do that). It is not a device-management CLI
either: it *calls* `jetson-cli` / `dgx-spark-cli` / `rtx-spark-cli` when they
are installed and still works when they are not.

## Current state

The bash-hook-and-agent-on-error design (this file, `docs/architecture.md`,
`docs/platforms.md`) is **implemented** as of nvsh 0.9.2, not just
converged. The hook installer (`nvsh setup`/`nvsh uninstall`, `nvsh on`/`off`,
`nvsh/rcfile.py`, `nvsh/shell/hook.bash`, `nvsh/shell/readline.bash`), the
trigger table (`nvsh/triggers.py`), redaction (`nvsh/redact.py`), platform
detection (`nvsh/platform/`), output capture (`nvsh/capture.py`), the
pluggable `NvshAgent` backends (`nvsh/agent/`: `base`, `fake`, `pi`,
`openai_compat`, `claude`, `codex`, `qwen` (ACP and the `qwen-p`
stream-json print-mode fallback), `agy`, `acp` (the generic ACP client
behind the `kiro` and `qwen` registry entries), `registry`, `loop`,
`audit`, `playbooks`, and the Pi extension at
`nvsh/agent/pi_ext/approval.ts`), the per-user session daemon
(`nvsh/daemon.py`), the failure client and panel (`nvsh/client.py`,
`nvsh/panel.py`), slash-command routing (`nvsh/slash.py`; `nvsh slash`,
`nvsh complete`), the approval store (`nvsh/approvals.py`, `nvsh approve`)
and the helper-tool installers (`nvsh/installers.py`) are all on disk,
alongside the original agent-first verbs (`whoami`, `learn`, `explain`,
`overview`, `doctor`, `cli overview`). `doctor` now also runs the extended
rubric in `nvsh/doctor_checks.py`: platform detection, agent
configured/reachable, and, from a hooked shell, hook sourced/first-in-
`PROMPT_COMMAND`, bindings, capture and daemon status. Before describing any
of this as implemented in a future change, confirm the file still exists
and the verb still runs (`uv run --frozen nvsh --help`,
`uv run --frozen nvsh doctor --json`) rather than assuming this paragraph
stays accurate forever.

nvsh registers eight harness adapters in `nvsh/agent/registry.py`'s
`ADAPTERS` table: `pi` (rpc), `qwen` (acp, `qwen --acp`, plan mode by
default), `qwen-p` (stream-json print-mode, read-only fallback for when
ACP is unavailable), `claude` (stream-json, `claude -p --output-format
stream-json --input-format stream-json --permission-prompt-tool stdio`),
`codex` (app-server, falling back to `exec`), `agy` (stream-json, always
read-only — see below), `kiro` (acp, `kiro-cli acp`) and `openai-compat`
(http). `[aliases]` in `$XDG_CONFIG_HOME/nvsh/config.toml` is a flat TOML
table mapping a short name to a `backend[/model[/effort]]` target, with
`default` reserved for the bare `nvsh --agent default` (or no `--agent` at
all) case (`Config.resolve_target`, `nvsh/config.py`); `nvsh agent use
<name>` and `nvsh setup` write `[aliases].default`, and `nvsh agent list
--json` reports every adapter with its `installed`/`path`/`hosted`/
`capabilities` state, default first. `nvsh setup` probes `PATH` for every
adapter and defaults to whichever harness is already installed (prompting
the operator when several are, tool-calling adapters listed first), rather
than hard-wiring pi. At the prompt, `@target` (`@name` for
a registered alias or adapter, `@backend/model/effort` for a literal) marks
one request for that harness only, rewritten to `/ask --agent <target>`; an
ad-hoc target runs one-shot, the default target rides the daemon's warm
session. See [`docs/shell-integration.md`](docs/shell-integration.md) for
the full `@target` grammar and [`docs/daemon.md`](docs/daemon.md) for how
the target travels on the wire.

Approval channels differ per harness, and where none exists the harness
runs read-only rather than getting nvsh's own auto-approve switches passed
to it. The spec (`docs/specs/2026-09-14-first-class-multi-harness-with-aliases.md`)
states the boundary this way: "nvsh never edits, creates or overrides a
harness's own settings or trust files (agy/claude settings.json, codex
config.toml, kiro trust settings, qwen settings): it only passes launch
flags and protocol-level policy, and reports what it finds." Qwen over ACP
never sends `session/request_permission` (verified against qwen 0.23.3), so
it ships in plan mode with `tool_calling=False` by default; the spec's opt-out
reads: "an operator may opt a harness into its own agent-side approval with
`[agents.<name>] approval = "harness"`, which is recorded in capabilities and
the audit log." `agy` headless auto-denies any tool needing the `command`
permission, so it is always registered `tool_calling=False` for commands
regardless of `approval`. Everything that leaves the process — the prompt
composer's output, `--show-context`, log lines — is redacted first
(`nvsh/redact.py`), and every subprocess-backed adapter's child environment
has `CLAUDECODE` and the whole `CLAUDE_CODE_*` family dropped
(`nvsh/agent/_env.py`) so a spawned harness never believes it is nested
inside the Claude Code session that may be driving nvsh's own development.

What is still genuinely open, so don't describe it as implemented: the
default-login-shell (`chsh`) mode stays parked, not built — see
`docs/architecture.md`'s "Parked: login-shell mode"; an auto-apply mode
that runs an agent-suggested fix without operator confirmation is out of
scope for v1 (nvsh always proposes, the operator always approves); and
machine-level undo beyond the current approve/execute/verify loop (rolling
back changes a *fix* made to the machine, not just retrying the original
command) is tracked separately as issue #7, not this branch. The package,
CLI, import and PyPI names are all already `nvsh`, so no rename is needed.
Some code comments and test docstrings still say "this template"; that
wording is internal and harmless.

## Commands

```bash
uv sync                                     # install (dev group included)
uv run pytest -n auto                       # full suite (parallel)
uv run pytest tests/test_cli.py::test_whoami_json -v   # single test
uv run pytest -n auto --cov=nvsh --cov-report=term     # with coverage (fail_under = 60)

# Lint, exactly as CI runs it
uv run black --check nvsh tests
uv run isort --check-only nvsh tests
uv run flake8 nvsh tests
uv run bandit -c pyproject.toml -r nvsh
npm install -g markdownlint-cli2@0.21.0     # not installed by uv sync; CI pins this version
markdownlint-cli2 "**/*.md" "#node_modules" "#.local" "#.claude/skills" "#.teken"
python3 scripts/scan-secrets.py             # committed secrets / non-localhost endpoints
uv run teken cli doctor . --strict          # agent-first rubric gate
uv run python scripts/harness-smoke.py --stage config --require config

uv run nvsh doctor --json                   # run the CLI from the checkout
```

Python ≥ 3.12, line length 100 (black/isort/flake8 agree). The runtime
package has **no third-party dependencies** (`dependencies = []`). Keep it
that way unless there is a strong reason: nvsh's Python entrypoint runs on
every qualifying failure of an interactive shell and must not break when a
venv or wheel breaks. `termios`, `select`, `subprocess`, `tomllib` and
`json` are all in the stdlib.

## CLI architecture (the contract new verbs must keep)

- `nvsh/cli/__init__.py` holds `_build_parser()`, which imports each module
  in `nvsh/cli/_commands/` and calls its `register(sub)`. A new verb or noun
  group gets its own module and a `register` call there. `_dispatch()` runs
  `args.func(args)`.
- **Errors:** handlers raise `CliError(code, message, remediation)` from
  `nvsh/cli/_errors.py`. They never call `sys.exit` or print tracebacks.
  `_dispatch` wraps any unexpected exception into a `CliError`.
  `_CliArgumentParser.error()` sends argparse errors through the same path.
  Exit codes: `0` ok, `1` user error, `2` environment error, `3+` reserved.
  Text-mode errors must include a `hint:` line.
- **Output:** results go to stdout through `emit_result`, and diagnostics and
  errors go to stderr through `emit_diagnostic` / `emit_error`
  (`nvsh/cli/_output.py`). The two streams never mix. **Every verb supports
  `--json`.** `main()` checks raw argv for `--json` before parsing, so even
  parse errors come out as JSON.
- `nvsh/explain/catalog.py` maps command-path tuples to markdown. Every new
  verb needs an entry, or the `teken cli doctor --strict` rubric and the
  introspection tests will fail.
- `doctor` returns the rubric shape
  `{healthy, checks: [{id, passed, severity, message, remediation}]}`. It
  currently checks the mesh-identity invariants (`prompt_file_present`,
  `harness_prompts`, `skills_present`). Issue #1 adds platform detection and
  "agent backend configured + reachable" checks. Add those as more checks
  in the same shape.
- `whoami.find_culture_yaml()` walks up from the module to find this repo's
  `culture.yaml`. When nvsh runs from a wheel install there is no
  `culture.yaml`, and `doctor` reports a single info check. Shell and device
  checks must still work in that case, because that is how nvsh runs when
  hooked into an operator's rc file on a machine without this checkout.

## Design decisions the issues force

**Hook, not wrap.** Issue #1 recommends a bash/zsh hook (`trap ERR` /
`PROMPT_COMMAND` / `precmd`); issue #2 originally asked for a PTY-backed
interactive shell delivered as a default-login-shell wrapper. The converged
decision (`docs/architecture.md`) is the hook: `nvsh setup` appends one
function to the operator's `PROMPT_COMMAND` array (first position, so
`PIPESTATUS` survives; when `bash-preexec` takes first position by its own
design the hook reads `BP_PIPESTATUS` instead — see below) via one marked
block inserted into
`$HOME/.bashrc` right after the distro's interactive guard, and `nvsh uninstall`
removes it. Bash itself still parses, does job control, completion, aliases
and rc files exactly as before; nothing wraps it, and there is no pty to
stay transparent to. Issue #2's UX (inline panels, `Ctrl+G`, deterministic
operator tools, long-context awareness) carries forward onto the hook. See
[`docs/architecture.md`](docs/architecture.md) for the full decision, the
measured reasons (success-path latency, `PROMPT_COMMAND` ordering, Ghostty
composition, output capture without a pty) and the parked login-shell mode.

### Hook constraints

These hold for the hook, replacing the login-shell/PTY-wrapper constraints
this section used to carry (as of nvsh 0.9.1; see `docs/architecture.md`'s
"Before" section for that prior wording):

- **Never lock the operator out.** A hook function that errors must never
  block the prompt: hook functions never use `set -e` and end with
  `|| return 0`, so a bug in nvsh degrades the panel, it does not lock the
  terminal. `NVSH_DISABLE=1` makes the sourced hook file a no-op outright.
- **Non-interactive invocations are untouched by construction.** The hook
  only installs into an *interactive* shell's `PROMPT_COMMAND` array, placed
  after the distro's own interactive guard in the rc file. `-c <cmd>`,
  non-TTY stdin, `scp`/`sftp`/`rsync`/`ssh host cmd` and other non-interactive
  invocations never source the hook at all, so there is nothing to pass
  through and nothing extra on stdout to break them.
- **No added latency on the success path.** The hook is a pure-bash function:
  it reads `$?`/`PIPESTATUS`, applies the trigger pre-filter in bash, and
  execs a Python process only when a failure qualifies. A successful command
  pays one bash function call and nothing else — no fork, no import, no
  model call, no network I/O.
- **Installation is idempotent and reversible.** `nvsh setup` inserts one
  marked block into `$HOME/.bashrc`, keeping a timestamped backup; running it
  twice leaves the rc unchanged. `nvsh uninstall` restores the rc from that
  backup, removes the hook file, runtime sockets and logs, and stops any
  daemon. There is no `/etc/shells` entry and no `chsh` involved.
- **Headless over SSH.** Jetsons are often reached only over SSH and are
  often air-gapped. The panel must render in a plain terminal with no
  mouse or GUI, with no dependency on terminfo (hard-coded SGR sequences,
  guarded by `NO_COLOR`/`TERM=dumb`/non-tty checks, never `tput`).

### Behavior rules (both issues agree)

- **Trigger rules are a first-class, table-tested module.** `130` (Ctrl-C),
  `141` (SIGPIPE), `grep`/`diff` exiting `1`, `false`, `test`/`[ ]`, and
  commands inside scripts that handle their own errors are *not* errors.
  Pipelines only report the last stage's status unless `pipefail` /
  `PIPESTATUS` is used. Interactive or long-running programs (`vim`, `htop`,
  `jtop`, long builds) never auto-trigger mid-run. Automatic calls are
  rate-limited. Pattern triggers for tools that print an error and still
  exit 0 are opt-in. Manual invocation (`nvsh ask`, `Ctrl+G`) always works.
- **Propose, don't run.** nvsh never runs an agent-suggested command without
  the operator's confirmation. Fixes should go through deterministic, allowed
  operator tools (`inspect-*`, `manage-*`, `rollback`, `verify`), not
  free-form shell. Any future auto-apply mode refuses `sudo` and destructive
  commands and logs what it ran.
- **Pluggable, offline-first agent backend.** Put backends behind one
  adapter interface: local model (vLLM / llama.cpp on the same box),
  Culture mesh, or hosted API. Include a **fixture backend for tests**.
  Nemotron is only the initial default. Config lives under
  `$XDG_CONFIG_HOME/nvsh/`, never a hard-coded path.
- **Device context, with redaction always on.** Jetson (`/etc/nv_tegra_release`,
  L4T/JetPack), DGX Spark (GB10, DGX OS), RTX: CUDA / cuDNN / TensorRT /
  driver versions, arch, **unified memory** on Jetson and GB10 (CUDA OOM
  means something different there than on discrete VRAM), `nvpmodel`, disk
  space, and container runtime. Report what was detected and how; don't
  guess. Record the source of each value in `docs/platforms.md`, since the
  issues say detection paths have not been checked on current releases.
  Redact tokens (`HF_TOKEN=`, `--api-key`, `Authorization:`, `.env`) before
  anything leaves the process. Support `--show-context`. The redactor needs
  its own tests.

## Repo conventions (inherited from the AgentCulture template)

- **Every PR bumps the version** (use the `version-bump` skill: it edits
  `pyproject.toml` and prepends to `CHANGELOG.md`). CI's `version-check`
  blocks the merge otherwise, even for docs-only PRs. `nvsh.__version__` is
  read from package metadata, so `pyproject.toml` is the only version source.
- **Publishing:** a push to `main` that touches `pyproject.toml` or `nvsh/**`
  publishes to PyPI through Trusted Publishing. A PR from a branch in this
  repo (not a fork) that touches those same paths publishes a `.devN` build
  to TestPyPI; docs-only PRs and fork PRs don't. The package is public, so
  keep `main` green.
- **PR workflow:** use the `cicd` skill (built on `devex pr`, with SonarCloud
  gating). PR replies are signed automatically as `- nvsh (Claude)`.
- **`.claude/skills/` is vendored verbatim** from guildmaster
  (cite-don't-import). Don't edit skill scripts here; fixes go upstream.
  Provenance and the re-sync steps are in `docs/skill-sources.md`.
  `.qwen/skills`, `.colleague/skills` and `.pi/skills` are symlinks to it.
- **Four harness prompt files, no shared base:** `CLAUDE.md` (Claude Code),
  `AGENTS.override.md` + `.pi/SYSTEM.md` (Pi/associate),
  `AGENTS.colleague.md` (colleague), and `QWEN.md` (Qwen Code). There is
  deliberately **no `AGENTS.md`**. `scripts/harness-smoke.py` fails CI if any
  of the four is broken, and `docs/harness-invocations.yaml` holds the probe
  prompts (the Claude probe asks whether `CLAUDE.md` describes the project as
  `nvsh`). When project facts change in this file, update the other three so
  they don't drift.
- `culture.yaml` (`suffix: nvsh`, `backend: claude`) is the mesh identity.
  `CLAUDE.md` is the resident prompt the Culture daemon reads. No code
  rewrites `backend`.
- `tests/test_harness_registries.py` checks `doctor`'s `_PROMPT_FILE` table
  against `.claude/skills/agent-config/data/backend-fingerprints.yaml`. Its
  one skip is a known cross-repo gap in `culture`, not a failure.
