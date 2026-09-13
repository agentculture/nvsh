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
- **#2: Interactive self-healing shell.** Covers the PTY-backed interactive
  UX (inline diagnosis panels, `Ctrl+G` to call the agent), deterministic
  operator tools under the model, long-context machine awareness, and
  known-good state (`remember-good` / `diff-good` / `restore-good`).
  Nemotron 3.5 Lightning is the first model, but it must not be hard-wired.

**Operator goal: nvsh will be the operator's default login shell (`chsh`) on
their Spark and Jetson machines.** This goal settles questions the issues
leave open, and it overrides any issue text that conflicts with it (see
[Login-shell constraints](#login-shell-constraints)).

nvsh is **not** a new POSIX shell (don't rewrite a bash parser or job
control). It is not an autonomous agent that runs commands on its own
(`shell-cli` / `callsmith` do that). It is not a device-management CLI
either: it *calls* `jetson-cli` / `dgx-spark-cli` / `rtx-spark-cli` when they
are installed and still works when they are not.

## Current state

The repo is still the **culture-agent-template scaffold** renamed to `nvsh`.
No shell features exist yet. `nvsh/` only holds the agent-first verbs
(`whoami`, `learn`, `explain`, `overview`, `doctor`, `cli overview`). The
harness prompt files and the CLI's own descriptions (`learn`, `explain`,
`--help`) already describe nvsh and mark the shell as planned. Keep them
that way: don't describe planned behavior as implemented. The package, CLI,
import and PyPI names are all already `nvsh`, so no rename is needed. Some
code comments and test docstrings still say "this template"; that wording is
internal and harmless.

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
that way unless there is a strong reason: a login shell has to start fast
and must not break when a venv or wheel breaks. PTY handling, `termios`,
`select`, `subprocess` and `json` are all in the stdlib.

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
  checks must still work in that case, because that is how nvsh runs as a
  login shell.

## Design decisions the issues force

**Hook vs wrap.** Issue #1 recommends a bash/zsh hook (`trap ERR` /
`PROMPT_COMMAND` / `precmd`). Issue #2 asks for a PTY-backed interactive
shell. As a *default login shell*, nvsh has to be the executable that `login`
/ `sshd` start, which points to #2's model: a thin PTY wrapper around a real
`bash` (or the user's configured inner shell). The inner shell still parses,
does job control, completion and aliases, and loads rc files. The wrapper
watches exit status and output. Hook-style integration (`nvsh init bash`)
stays useful when nvsh is *not* the login shell. Record the decision and the
reasons in `docs/architecture.md` before building (milestone item 1 in
issue #1).

### Login-shell constraints

These hold whenever nvsh is set with `chsh`:

- **Never lock the operator out.** Any failure in nvsh (import error,
  corrupt config, backend down, PTY setup failure) must fall back to `exec`
  of the real shell. Treat this as a tested invariant, not a best effort.
  Keep a way to skip the wrapper entirely, such as an env var or a sentinel
  file.
- **Non-interactive invocations pass straight through.** When nvsh gets
  `-c <cmd>`, stdin is not a TTY, or a remote command arrives, it `exec`s the
  inner shell with the same argv and adds nothing: no banner, no stdout
  output, no PTY. `scp`, `sftp`, `rsync`, `ssh host cmd`, `git` over ssh, VS
  Code Remote and Ansible all depend on this, and extra bytes on stdout
  break them.
- **Login semantics.** Support being started as `-nvsh` (argv[0] starts with
  `-`) and `-l`. Pass login-ness on to the inner shell so `/etc/profile` and
  the user's profile still load.
- **No added latency on the success path.** Import nothing heavy at startup,
  make no model call and no network I/O before the prompt, and do nothing
  extra for a command that succeeds.
- **Installation.** `chsh` needs an absolute path listed in `/etc/shells`
  (on this DGX Spark it currently lists only sh/bash/dash/rbash/screen/tmux).
  A `uv tool install` puts the entry point in the user's tool bin directory, so the
  install/uninstall story (e.g. `nvsh install-shell` / `nvsh uninstall`) has
  to cover `/etc/shells`, the `chsh` itself, and a documented recovery path.
  Jetson images may ship an older system Python than 3.12, so don't rely on
  the system interpreter.
- **Headless over SSH.** Jetsons are often reached only over SSH and are
  often air-gapped. The UI must work in a plain terminal with no
  mouse or GUI.

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
