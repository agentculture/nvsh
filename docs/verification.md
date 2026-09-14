# Live verification: first-class multi-harness with aliases

Recorded 2026-09-14 for nvsh 0.10.0 (task t21 of
`docs/plans/2026-09-14-first-class-multi-harness-with-aliases.md`). Every
transcript below was produced by a real interactive, hooked bash driven
through a pty (a small stdlib driver that types one line, answers every
approval prompt with `n`, and strips SGR sequences); nothing an agent
proposed was ever applied, and `/nonexistent-dir-for-nvsh-test` still does
not exist on any machine afterwards. Home paths, e-mail addresses and keys
are redacted as `<home>` / `<redacted>`.

## Fleet and install

| Machine | Platform | nvsh | pi | claude | codex | qwen | agy | kiro-cli |
|---------|----------|------|----|--------|-------|------|-----|----------|
| spark | DGX Spark (GB10) | 0.10.0 | 0.85.1 | 2.1.270 | 0.147.0 | 0.23.3 | 1.2.2 | 2.21.4 |
| thor | Jetson AGX Thor | 0.10.0 | not installed | 2.1.267 | 0.147.0 (not logged in) | 0.22.0 | not installed | 2.21.4 |
| orin | Jetson AGX Orin | 0.10.0 | not installed | 2.1.260 | 0.147.0 | 0.23.0 | not installed | 2.21.4 |

Install on each machine: `uv build` on spark, then
`uv tool install --force nvsh-0.10.0-py3-none-any.whl` and `nvsh setup`
(idempotent; it wrote `[aliases].default` = the configured provider).
Config overlay on all three (the prior config was backed up first):

```toml
[agents.kiro]
model = "glm-5"          # operator decision: one of minimax-m2.5 | glm-5 | qwen3-coder-next

[agents.claude]
effort = "low"

[aliases]
default = "pi"           # spark; "openai-compat" (associate) on thor and orin
reviewer = "claude/sonnet/medium"
local = "kiro/glm-5"
fast = "claude/haiku/low"
```

## Baseline on `main`

`git grep -l -w THINKING main -- nvsh` and `git grep -l -w agy main -- nvsh`
both return nothing; `git grep -l -w effort main -- nvsh` matches four files
(approvals, capture, daemon, doctor_checks) where the word appears in prose
only. On this branch THINKING appears in 7 modules, agy in 5 and effort in 18.

## `nvsh agent list --json`

All three machines list eight adapters, each with its protocol path and
hosted flag, the resolved default first:

```text
spark: pi rpc local default | qwen acp local | qwen-p stream-json local |
       claude stream-json hosted | codex app-server hosted |
       agy stream-json hosted | kiro acp hosted | openai-compat http local
thor:  openai-compat http local default | pi MISSING | qwen acp | qwen-p |
       claude | codex | agy MISSING | kiro
orin:  same as thor
```

(The plan text said "six adapters"; eight is the delivered count, see
deviation d11.)

## spark

### Default target (pi/associate) on a failure, warm daemon session

```text
$ ls /nonexistent-dir-for-nvsh-test
ls: cannot access '/nonexistent-dir-for-nvsh-test': No such file or directory
nvsh: ls /nonexistent-dir-for-nvsh-test failed (exit 2), forwarding to pi/associate
pi/associate · rpc · warm
... waiting for the agent (1s)The user is asking me to diagnose a failure of the
command `ls /nonexistent-dir-for-nvsh-test` which returned exit code 2. ...
[dimmed thinking run continues for ~25 lines]
The `ls` command returned exit code 2 because the directory
`/nonexistent-dir-for-nvsh-test` does not exist — this is expected behavior.
**Fix:** Create the directory if it should exist, or use an existing path.
```

The thinking run is rendered dimmed (SGR 2) on the tty; the plain text above
is what remains after stripping the escape.

### `@reviewer` alias through the readline Enter binding

```text
$ @reviewer explain the last failure in one sentence
$  <home>/.local/share/uv/tools/nvsh/bin/nvsh slash '/ask --agent reviewer explain the last failure in one sentence'
nvsh: asking claude/sonnet: explain the last failure in one sentence
claude/sonnet/medium · stream-json · one-shot
... init
... requesting
`ls` failed because the target path `/nonexistent-dir-for-nvsh-test` does not exist in the filesystem.
```

The same line typed with a bare line feed (which bypasses the readline
binding) still reaches the harness through the hook's second route: bash
reports `@reviewer: command not found`, the hook recognises the mark and asks
the aliased harness, but then the failure context is the mark line itself.

### Literal `@backend/model/effort`, one-shot

```text
$ @claude/sonnet/medium say OK
nvsh: asking claude/sonnet: say OK
claude/sonnet/medium · stream-json · one-shot
... init
... requesting
OK
```

`@fast` (claude/haiku/low) answered the same way with two `thinking_tokens`
status lines before `OK`.

### `@local` (kiro/glm-5 over ACP)

```text
$ @local why would ls /nonexistent-dir fail, one sentence
nvsh: asking kiro/glm-5: why would ls /nonexistent-dir fail, one sentence
kiro/glm-5 · acp · one-shot
... waiting for the agent (16s)ls /nonexistent-dir fails because the directory does not exist. ...
```

kiro streams no thought chunks (its ACP has none), as documented.

### `@codex/gpt-5.6-sol/low` (app-server)

Codex ran its own read-only tools (`ls -la <home>/`, `ls -la <home>/.codex/`,
shown as `... running: …` / `... tool bash finished`), then proposed a
command that reached nvsh's fix panel with stages and the
`[Enter] run [s/S] +session [u/U] +user [e] why [d] details [t] tell [Esc] ignore`
prompt; the driver ignored it and nothing ran.

### `@qwen` (ACP, plan mode), `@qwen-p` (print mode), `@agy`

```text
$ @qwen say OK
qwen · acp · one-shot
... waiting for the agent (6s)The user is setting up context for me as an
"on-failure assistant" on their NVIDIA DGX Spark machine. ... [thought chunk]
OK

$ @qwen-p say OK
qwen-p · stream-json · one-shot
... init
The user wants me to say OK.
OK.   [the print-mode process took ~26 s more to exit]

$ @agy say OK
agy · stream-json · one-shot
... waiting for the agent (15s)OK
```

## thor (Jetson AGX Thor)

```text
$ ls /nonexistent-dir-for-nvsh-test
nvsh: ls /nonexistent-dir-for-nvsh-test failed (exit 2), forwarding to openai-compat/associate
openai-compat/associate · http · warm
**Diagnosis:** The path `/nonexistent-dir-for-nvsh-test` does not exist ...
**Proposed fix:** Create the directory.

$ @reviewer say OK
claude/sonnet/medium · stream-json · one-shot
... init
... requesting
OK

$ @local say OK
kiro/glm-5 · acp · one-shot
Command `@local` does not exist. ... running: type @local 2>&1 || true
[fix panel with stages; ignored]

$ @qwen say OK
qwen · acp · one-shot
... [thought chunk] ... OK

$ @codex say OK
codex · app-server · one-shot
nvsh: Reconnecting... 2/5
```

codex on thor is **not verified**: `codex login status` prints
`Not logged in` and there is no auth file, so the app-server turn never
started. pi and agy are not installed on thor and were not exercised.

## orin (Jetson AGX Orin)

```text
$ ls /nonexistent-dir-for-nvsh-test
nvsh: ls /nonexistent-dir-for-nvsh-test failed (exit 2), forwarding to openai-compat/associate
openai-compat/associate · http · warm
mkdir -p /nonexistent-dir-for-nvsh-test        [proposal; not run]

$ @reviewer say OK          -> claude/sonnet/medium · stream-json · one-shot ... OK
$ @local say OK             -> kiro/glm-5 · acp · one-shot ... OK
$ @qwen say OK              -> qwen · acp · one-shot ... [thought chunk] ... OK
$ @codex say OK             -> codex · app-server · one-shot ... OK
```

pi and agy are not installed on orin and were not exercised.

## Summary

| Harness | spark | thor | orin |
|---------|-------|------|------|
| pi (rpc, default) | verified, thinking shown | not installed | not installed |
| claude (stream-json) | verified (alias + literal) | verified | verified |
| codex (app-server) | verified (tools + proposal) | **not verified: not logged in** | verified |
| qwen (acp) | verified, thought chunks | verified | verified |
| qwen-p (stream-json) | verified | not exercised | not exercised |
| agy (stream-json) | verified | not installed | not installed |
| kiro (acp, glm-5) | verified | verified, proposal | verified |
| openai-compat (http, default on Jetsons) | not exercised | verified | verified |

Every `@target` ran one-shot with its header line; the default target ran
through the warm daemon session on all three machines.

## Setup works with any installed agent (0.11.0)

Spec: `docs/specs/2026-09-14-setup-works-with-any-installed-agent.md`.
Both runs below use a scratch `$HOME`/`$XDG_*` and a `PATH` that carries
only `claude` (plus `node` and `python3`) so the box's own pi/codex/qwen
installs cannot leak in. Machine: spark (DGX Spark, GB10).

### Before, on `main` (0.10.1) -- the challenge-pass probe

```text
$ nvsh setup --no-install --json --rc $HOME/.bashrc
agent: {"name": "openai-compat",
        "reason": "pi not on PATH; using openai-compat (no base_url configured, defaults apply)",
        "key_hint": "no bearer resolved: put the gateway's key in $XDG_CONFIG_HOME/nvsh/api_key (mode 0600), ..."}
installs: ['pi']
$ nvsh setup --no-install --json --rc $HOME/.bashrc      # second run, same PATH
agent: openai-compat | [aliases].default = 'openai-compat' kept
```

while `nvsh agent list --json` on the same box reported every one of the
eight adapters `installed: true`. Setup steered a claude-only operator to
an OpenAI-compatible key it did not need, offered to install pi, and kept
that wrong pick on every later run.

### After, on this branch

```text
$ nvsh setup --no-install --json --rc $HOME/.bashrc      # nothing on PATH
agent: openai-compat | pi not on PATH and node missing; using openai-compat (...)
key_hint: yes | installs: ['node', 'pi']                  # bootstrap offers, deviation d1
$ nvsh setup --no-install --json --rc $HOME/.bashrc      # claude appears
agent: claude | claude is the only agent harness on PATH
key_hint: None | installs: [] | daemon_stopped: True
$ grep -A1 aliases $XDG_CONFIG_HOME/nvsh/config.toml
[aliases]
default = "claude"
$ nvsh setup --no-install --json --rc $HOME/.bashrc --agent codex/gpt-5/high
{"code": 2, "message": "codex is not installed (forced backend 'codex')",
 "remediation": "install codex, or drop --agent to let nvsh choose"}
```

Text mode, `nvsh setup --agent claude`:

```text
agent: claude (forced via --agent 'claude')
default target: claude
daemon stopped: True
claude is hosted: on a failure the redacted command, output and device context leave this machine
agent reachable: True (claude reachable (version 2.1.270; auth state not verified (no safe non-model probe)))
```

### End to end: a failure reaches claude with no pi and no key prompt

A fresh interactive bash sourcing the rc block setup just wrote (tmux,
same restricted `PATH`). The first failing pipeline was a `nvsh doctor
--json | python3 ...` line in which bash's `nvsh` shell function called
`command nvsh`, which is not on that `PATH` (plan risk r3):

```text
nvsh: nvsh doctor --json | python3 -c "..." failed (exit 1), forwarding to claude
claude · stream-json · warm
... init
... requesting
... thinking_tokens
`nvsh` is not on PATH (bash printed "Command 'nvsh' not found"), so the pipeline's
stdout was empty and the JSON parse failed downstream. ...
... running: ls -l $HOME/git/nvsh/.venv/bin/nvsh 2>&1; echo "PATH=$PATH"
... tool Bash finished
**Proposed fix** (you declined the uv retry, so pick one of these to run yourself):
    uv run --frozen nvsh doctor --json | python3 -c "..."
or, to avoid uv, call the venv entrypoint directly:
    .venv/bin/nvsh doctor --json | python3 -c "..."
If you want a bare `nvsh` to work in this shell, `source .venv/bin/activate` or
prepend the venv's bin directory to `PATH`. No sudo and no package install is needed.
```

No pi was installed or offered, no key was asked for, the hosted line
was printed at setup, and the `uv` install claude proposed was held at the
approval gate ("you declined the uv retry"). A second failure inside the
same 30-second window was rate-limited as designed:

```text
$ ls /nonexistent-dir-for-nvsh-test
ls: cannot access '/nonexistent-dir-for-nvsh-test': No such file or directory
nvsh: held back (auto calls limited to 1 per 30s window); /fix or Ctrl+G asks now
```

Not exercised here: the several-harnesses prompt on a real tty (covered
by `tests/test_cli_setup.py` with an injected prompt), thor and orin
(unchanged hook, same wheel), and macOS/zsh (issue #11).
