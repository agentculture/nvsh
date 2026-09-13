# nvsh

**An agent-first shell for NVIDIA Jetson, DGX Spark and RTX Spark.** It runs
your commands like a normal shell. When a command fails, it hands the error
and device context to an agent (shell → agent), which diagnoses the problem
and proposes a fix.

> Works like a shell. Helps when things break. Fixes when you let it.

**Status: early scaffold.** The agent-first CLI baseline (below) works. The
shell itself is not built yet. The design is tracked in
[#1](https://github.com/agentculture/nvsh/issues/1) (build brief) and
[#2](https://github.com/agentculture/nvsh/issues/2) (interactive self-healing
shell).

## Goal

nvsh is meant to be **usable as your default login shell** (`chsh`) on
Jetson and DGX Spark machines:

- **A shell first.** A thin PTY layer around your real `bash`, so parsing,
  job control, completion, aliases and rc files work as they always have. A
  command that succeeds gets no added latency and no model call.
- **An agent second.** The agent is called only on a real failure (non-zero
  exit, traceback, CUDA OOM, container or service failure, missing binary)
  or when you ask for it (`nvsh ask`, `Ctrl+G`). Exit codes that aren't
  errors, such as Ctrl-C, SIGPIPE, or `grep` finding nothing, don't trigger
  it, and automatic calls are rate-limited.
- **Propose, don't run.** You get a diagnosis and a proposed fix, then
  accept, edit or reject it. Nothing the agent suggests runs without your
  confirmation. After an approved fix, nvsh can retry the command and check
  that it worked.
- **NVIDIA-aware.** Platform detection (JetPack/L4T, DGX OS/GB10, RTX),
  CUDA / TensorRT / driver versions, unified memory, `nvpmodel` and the
  container runtime are attached to each diagnosis.
- **Offline by default, pluggable.** The agent backend sits behind an
  adapter: a local model on the same box first (Nemotron initially), with
  the Culture mesh or a hosted API as options.
- **Private by default.** Secrets are redacted before anything leaves the
  process, and `--show-context` shows exactly what would be sent.
- **Safe as a login shell.** `scp`, `rsync`, `ssh host cmd` and other
  non-interactive sessions pass straight through to the real shell. If nvsh
  itself fails, it falls back to that shell instead of locking you out.

nvsh is not a new POSIX shell. It is not an autonomous agent that runs
commands by itself, and it doesn't replace `jetson-cli` / `dgx-spark-cli`
(it calls them when they are installed).

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
| `doctor` | Health checks (today: agent-identity invariants; planned: platform + agent backend reachability). |
| `cli overview` | Describe the CLI surface itself. |

Every command supports `--json`. Results go to stdout, and errors and
diagnostics go to stderr; the two are never mixed. Exit codes: `0` success,
`1` user error, `2` environment error, `3+` reserved.

The shell verbs, including the interactive shell, `run`, `ask`,
`init bash|zsh`, install/uninstall as login shell, and known-good state, will
be added as the milestones in #1 and #2 land.

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
- CI covers pytest, lint, secret scanning, the agent-first rubric gate, a
  per-harness smoke check, and PyPI Trusted Publishing.

Every PR bumps the version. See [`CLAUDE.md`](CLAUDE.md) for the full
contributor conventions.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
