# nvsh

You already have an agent CLI — `claude`, `codex`, `qwen`, `kiro`, `agy`
or a local Nemotron. nvsh puts it behind your bash prompt on NVIDIA
hardware: commands run exactly as they do today, and when one fails, the
agent you already trust diagnoses it and proposes a fix.

> Works like a shell. Helps when things break. Fixes when you let it.

## Install

```bash
uv tool install nvsh        # or: pipx install nvsh
nvsh setup
```

`nvsh setup` inserts one marked block into your `.bashrc` (with a
timestamped backup) that appends a function to bash's `PROMPT_COMMAND`.
It never wraps bash and never becomes your login shell.

## Set up

| Command | What it does |
|---------|--------------|
| `nvsh setup` | Probes `PATH` for installed harnesses. One hit becomes the default; several hits prompt you once. |
| `nvsh setup --agent claude` | Picks Claude Code explicitly, no prompt. |
| `nvsh setup --agent codex` | Picks Codex explicitly, no prompt. |
| `nvsh agent list` | All nine adapters: `pi`, `qwen`, `qwen-p`, `claude`, `codex`, `agy`, `kiro`, `openai-compat`, `demo`. |
| `nvsh agent use <name>` | Change the default afterwards. |
| `nvsh uninstall` | Remove the rc block, hook files, sockets, logs and daemon. |
| `NVSH_DISABLE=1`, `nvsh off` / `nvsh on` | Kill switches for the current shell. |

## Work with it

A failing command opens an inline diagnosis panel. Nothing runs without
your approval — nvsh proposes, you accept, edit or reject.

| Command | What it does |
|---------|--------------|
| `Ctrl+G`, `/ask` | Call the agent on demand at the prompt. |
| `/fix`, `/doctor` | More slash commands at the prompt. |
| `@claude ...`, `@codex ...` | Send one request to a specific harness. |
| `nvsh doctor` | Health checks: platform, agent reachable, hook, capture, daemon. |
| `nvsh context --show` | Print exactly the bytes that would be sent. |

Exit codes that aren't errors — Ctrl-C (`130`), SIGPIPE (`141`), `grep`
finding nothing — never trigger it, and automatic calls are rate-limited.

## Safety first

Nothing leaves the machine except to the agent you trust. On a successful
command nvsh makes no network call at all. On a qualifying failure a
redacted, bounded context slice — the command line, its exit status, at
most 64 KB of its output, and the detected platform block — goes only to
the agent you chose, and nowhere else.

With `claude` or `codex`, that agent is a third-party hosted service: your
failure context leaves your network. With `pi`/Nemotron or `openai-compat`
pointed at localhost or your LAN, it does not.

Redaction runs first, before anything leaves the process: `HF_TOKEN=`,
`--api-key`, `Authorization:` headers and `.env`-style assignments are
scrubbed from the context. `nvsh context --show` prints the
post-redaction bytes so you can check before you trust it. nvsh never
edits a harness's own settings or trust files — it passes launch flags and
protocol-level policy, and reports what it finds.

## What nvsh never does

- Run an agent-suggested command without your confirmation.
- Add latency or a network call to a successful command.
- Wrap or replace bash — it hooks into the bash you already run.
- Replace `jetson-cli` / `dgx-spark-cli`; it calls them when installed.

Not yet: macOS and zsh are untested —
[#11](https://github.com/agentculture/nvsh/issues/11).

## What lands where

| Path | What |
|------|------|
| `$HOME/.bashrc` | One marked rc block, with a timestamped backup. |
| `$XDG_DATA_HOME/nvsh/shell/` | Rendered hook files. |
| `$XDG_CONFIG_HOME/nvsh/config.toml` | Backend, aliases, default target. |
| `$XDG_RUNTIME_DIR/nvsh` | Daemon socket and logs. |
| Audit log | Every proposal and every decision you made on it. |

More:

- [architecture](https://github.com/agentculture/nvsh/blob/main/docs/architecture.md)
  — hook over wrapper, and the parked login-shell mode.
- [platforms](https://github.com/agentculture/nvsh/blob/main/docs/platforms.md)
  — where each detected device value comes from.
- [shell integration](https://github.com/agentculture/nvsh/blob/main/docs/shell-integration.md)
  and [daemon](https://github.com/agentculture/nvsh/blob/main/docs/daemon.md)
  — the `@target` grammar and the warm session.
- [CLAUDE.md](https://github.com/agentculture/nvsh/blob/main/CLAUDE.md)
  — contributor conventions.

## License

Apache 2.0 — see
[LICENSE](https://github.com/agentculture/nvsh/blob/main/LICENSE).
