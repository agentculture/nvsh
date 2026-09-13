# nvsh

**An agent-first shell for NVIDIA Jetson AGX Orin/Thor and DGX Spark**,
usable locally in [Ghostty](https://ghostty.org) or over `ssh`. It hooks
into your existing interactive `bash`: your commands run exactly as they do
today, and when one fails, nvsh hands the error and device context to an
agent (shell → agent), which diagnoses the problem and proposes a fix.

> Works like a shell. Helps when things break. Fixes when you let it.

**Status: hook implemented.** The bash hook (`nvsh setup`/`uninstall`/`on`/`off`),
the per-user session daemon, the trigger table, redaction, platform
detection, output capture, the pluggable `NvshAgent` backends (fake, Pi,
OpenAI-compatible, Claude, Codex, Qwen), the failure panel, slash commands
and the approve/execute/verify loop are all on disk and covered by tests —
see [`docs/architecture.md`](docs/architecture.md) for the design and
[`docs/verification.md`](docs/verification.md) for what has been checked on
real hardware. Still open: the default-login-shell (`chsh`) mode stays
parked (see "Parked: login-shell mode" in `docs/architecture.md`), an
auto-apply mode (running a fix without confirmation) is out of scope for
v1, and machine-level undo beyond the current approve/execute/verify loop
is tracked as a follow-up
([#7](https://github.com/agentculture/nvsh/issues/7)). The design is
tracked in [#1](https://github.com/agentculture/nvsh/issues/1) (build
brief) and [#2](https://github.com/agentculture/nvsh/issues/2) (interactive
self-healing shell).

## Goal

nvsh **hooks into your existing bash** rather than replacing or wrapping it:

- **A shell first.** `nvsh setup` inserts one marked block into your rc file
  that adds a function to bash's `PROMPT_COMMAND` array. Bash itself keeps
  parsing, doing job control, completion, aliases and rc files exactly as
  it always has. A command that succeeds gets no added latency and no model
  call — see [`docs/architecture.md`](docs/architecture.md) for the decision
  and why a hook was chosen over a PTY wrapper.
- **An agent second.** The agent is called only on a real failure (non-zero
  exit, traceback, CUDA OOM, container or service failure, missing binary)
  or when you ask for it (`nvsh ask`, `Ctrl+G`, slash commands like
  `/doctor`). Exit codes that aren't errors, such as Ctrl-C, SIGPIPE, or
  `grep` finding nothing, don't trigger it, and automatic calls are
  rate-limited.
- **Propose, don't run.** You get a diagnosis and a proposed fix, then
  accept, edit or reject it. Nothing the agent suggests runs without your
  confirmation. After an approved fix, nvsh can retry the command and check
  that it worked.
- **NVIDIA-aware.** Platform detection (JetPack/L4T, DGX OS/GB10, RTX),
  CUDA / TensorRT / driver versions, unified memory, `nvpmodel` and the
  container runtime are attached to each diagnosis; see
  [`docs/platforms.md`](docs/platforms.md) for sources.
- **Offline by default, pluggable.** The agent backend sits behind an
  adapter: a local/LAN model first (Nemotron's "associate", via Pi, is the
  initial default), with the Culture mesh or a hosted API as options.
- **Reversible.** `nvsh uninstall` removes the marked block from your rc
  file (restoring it from a backup), the hook file, and any runtime state
  it created. `NVSH_DISABLE=1` and `nvsh off`/`nvsh on` are kill switches
  for the current shell.

nvsh is not a new POSIX shell. It is not an autonomous agent that runs
commands by itself, and it doesn't replace `jetson-cli` / `dgx-spark-cli`
(it calls them when they are installed). It is not built as a login shell
or PTY wrapper in this scope — that mode is a parked possible follow-up,
not a current goal (see `docs/architecture.md`).

## What leaves the machine

- **Nothing, by default.** nvsh's own hook makes no network call on a
  successful command. On a qualifying failure, only a bounded context
  slice — the command line, exit status, a capped (<=64 KB) slice of the
  command's own output, and the detected platform block — is sent to the
  configured agent backend, which defaults to a LAN-local model, not a
  public API.
- **Redaction runs before anything leaves the process.** Tokens shaped like
  `HF_TOKEN=`, `--api-key`, `Authorization:` headers, and `.env`-style
  assignments are scrubbed from the context before it is handed to the
  agent. `--show-context` prints exactly the bytes that would be sent, so
  you can check before you trust it.

## Quickstart (development)

```bash
uv sync
uv run pytest -n auto                 # run the test suite
uv run nvsh whoami                    # identity from culture.yaml
uv run nvsh doctor                    # health checks
uv run nvsh learn                     # self-teaching prompt (add --json)
uv run teken cli doctor . --strict    # the agent-first rubric gate CI runs
```

## CLI

| Verb | What it does |
|------|--------------|
| `whoami` | Report this agent's nick, version, backend, and model from `culture.yaml`. |
| `learn` | Print a structured self-teaching prompt. |
| `explain <path>` | Markdown docs for any noun/verb path. |
| `overview` | Read-only descriptive snapshot of the agent. |
| `doctor` | Health checks: agent-identity invariants, platform detection, agent backend configured/reachable, and (from a hooked shell) hook sourced, bindings, capture and daemon status. |
| `cli overview` | Describe the CLI surface itself. |
| `setup` | Render the bash hook files and insert the rc block. |
| `uninstall` | Remove the rc block, rendered files, sockets, logs and daemon. |
| `on` / `off` | Print the bash that rebinds / unbinds the hook in the current shell. |
| `agent` | List, choose, or install `NvshAgent` harness backends. |
| `approve` | Check or manage the approved-command pattern store. |
| `capture` | Show the last captured command output. |
| `context` | Show exactly the context that would be sent to the agent. |
| `daemon` | Run, inspect or stop the per-user session daemon. |
| `slash` | Dispatch one `/verb ...` line. |
| `complete` | Tab-completion candidates. |

Every command supports `--json`. Results go to stdout, and errors and
diagnostics go to stderr; the two are never mixed. Exit codes: `0` success,
`1` user error, `2` environment error, `3+` reserved.

**Verified on:** see [`docs/verification.md`](docs/verification.md) for the
devices and scenarios this has actually been exercised against.

## Repository layout

nvsh is an [AgentCulture](https://github.com/agentculture) mesh agent built
from the culture-agent-template:

- `culture.yaml` holds the mesh identity (`suffix: nvsh`, `backend: claude`).
- One prompt file per agent harness, with no shared base:
  `CLAUDE.md` for Claude Code, `AGENTS.override.md` + `.pi/SYSTEM.md` for
  Pi/associate, `AGENTS.colleague.md` for colleague, and `QWEN.md` for Qwen
  Code. There is deliberately no `AGENTS.md`. See
  [`docs/harness-selection.md`](docs/harness-selection.md) and
  [`docs/automation-contract.md`](docs/automation-contract.md).
- `.claude/skills/` holds the guildmaster skill kit, vendored
  cite-don't-import. See [`docs/skill-sources.md`](docs/skill-sources.md).
- [`docs/architecture.md`](docs/architecture.md) records the hook-vs-wrap
  decision, and [`docs/platforms.md`](docs/platforms.md) records where each
  detected device value comes from.
- CI covers pytest, lint, secret scanning, the agent-first rubric gate, a
  per-harness smoke check, and PyPI Trusted Publishing.

Every PR bumps the version. See [`CLAUDE.md`](CLAUDE.md) for the full
contributor conventions.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
