"""Markdown catalog for ``nvsh explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple
and ``("nvsh",)`` both resolve to the root entry.

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# nvsh

An agent-first shell for NVIDIA Jetson, DGX Spark and RTX Spark. It runs your
commands like a normal shell; when a command fails, it hands the error and
device context to an agent (shell -> agent) to diagnose and propose a fix, which
you confirm before anything runs.

Early scaffold: the agent-first verbs below exist today; the shell itself is
planned (GitHub issues #1 and #2). nvsh is also an AgentCulture mesh agent
(`culture.yaml` + `CLAUDE.md`).

## Verbs

- `nvsh whoami` — identity probe from `culture.yaml`.
- `nvsh learn` — structured self-teaching prompt.
- `nvsh explain <path>` — markdown docs for any noun/verb.
- `nvsh overview` — descriptive snapshot of the agent.
- `nvsh doctor` — check the agent-identity invariants.
- `nvsh cli overview` — describe the CLI surface.
- `nvsh approve check <cmd>` — check whether a command is already approved.
- `nvsh capture --show` — print the last captured-output slice for this shell.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error
- `3+` reserved

## See also

- `nvsh explain whoami`
- `nvsh explain doctor`
"""

_WHOAMI = """\
# nvsh whoami

Reports the agent's identity from `culture.yaml`: nick (`suffix`), backend,
served model, and the package version. Read-only.

## Usage

    nvsh whoami
    nvsh whoami --json
"""

_LEARN = """\
# nvsh learn

Prints a structured self-teaching prompt covering purpose, command map,
exit-code policy, `--json` support, and the `explain` pointer.

## Usage

    nvsh learn
    nvsh learn --json
"""

_EXPLAIN = """\
# nvsh explain <path>

Prints markdown documentation for any noun/verb path. Unlike `--help` (terse,
positional), `explain` is global and addressable by path.

## Usage

    nvsh explain nvsh
    nvsh explain whoami
    nvsh explain --json <path>
"""

_OVERVIEW = """\
# nvsh overview

Read-only descriptive snapshot of the agent: identity (from `culture.yaml`), the
verb surface, and the sibling-pattern artifacts the template carries. Accepts an
ignored `target` so a stray path never hard-fails.

## Usage

    nvsh overview
    nvsh overview --json
"""

_DOCTOR = """\
# nvsh doctor

Checks the agent-identity invariants `steward doctor` verifies:
prompt-file-present and backend-consistency (`claude` → `CLAUDE.md`), plus a
skills-present check. Exits 1 when unhealthy.

prompt-file-present requires the *resident* prompt the declared backend
actually reads. Other harness prompt files recognized under the same backend
name (`AGENTS.override.md`, `.pi/SYSTEM.md`, `QWEN.md`) belong to
interactively available harnesses the mesh daemon never loads; they are
reported by the informational harness-prompts check and never substituted.

## Usage

    nvsh doctor
    nvsh doctor --json
"""

_CLI = """\
# nvsh cli

Noun group for CLI-surface introspection. `cli overview` describes the CLI
itself (distinct from the global `overview`, which describes the agent).

## Usage

    nvsh cli overview
    nvsh cli overview --json
"""

_APPROVE = """\
# nvsh approve

Checks or manages the approved-command pattern store (`nvsh.approvals`).
Backs the "propose, don't run" contract: an agent-proposed fix is never
executed without operator approval, and this is the single shared decision
point other components (the bash hook, the daemon client) call into.

Patterns are `fnmatch` globs matched against the *full* command line, after
whitespace normalization — not just the program name. `add()` refuses
`sudo *`, `rm *`, a bare `*`, and any pattern starting with `sudo` or `rm`.

Two scopes:

- `user` — persisted to `$XDG_CONFIG_HOME/nvsh/approved.toml` (mode 0600).
- `session` — held in memory only for the current process; never written to
  disk and gone once the process exits.

## Usage

    nvsh approve check "nvidia-smi -q"
    nvsh approve check "nvidia-smi -q" --json
    nvsh approve add "docker logs *"
    nvsh approve add "docker logs *" --session
    nvsh approve list --json
    nvsh approve remove "docker logs *"
"""

_APPROVE_CHECK = """\
# nvsh approve check <cmd>

Returns the approval decision for a command line: `{decision, pattern}` where
`decision` is `user`, `session`, or `ask`, and `pattern` is the matching glob
(or `null`/absent when nothing matched). User patterns are checked before
session patterns.

## Usage

    nvsh approve check "docker ps -a"
    nvsh approve check "docker ps -a" --json
"""

_APPROVE_ADD = """\
# nvsh approve add <pattern>

Approves an `fnmatch` glob pattern, matched against the full command line.
Persists to `user_patterns` by default; pass `--session` to hold it in memory
only for the current process.

Refused outright (raises a user error, nothing is written): `sudo *`, `rm *`,
a bare `*`, and any pattern starting with `sudo` or `rm`.

## Usage

    nvsh approve add "docker logs *"
    nvsh approve add "docker logs *" --session
"""

_APPROVE_LIST = """\
# nvsh approve list

Lists both pattern lists: `{user: [...], session: [...]}`. `user` is loaded
from `approved.toml`; `session` reflects only the current process (always
empty in a freshly started process).

## Usage

    nvsh approve list
    nvsh approve list --json
"""

_APPROVE_REMOVE = """\
# nvsh approve remove <pattern>

Removes a pattern from both the persisted `user_patterns` and the in-memory
`session_patterns` lists, if present. Idempotent.

## Usage

    nvsh approve remove "docker logs *"
"""

_CAPTURE = """\
# nvsh capture

Prints the last captured-output slice for the current shell session
(`nvsh.capture.last_slice`), backing `--show-context`.

Each interactive session the bash hook installs runs under a per-session
typescript (`script -qfc "$BASH" "$log"`), or, inside tmux, under
`tmux pipe-pane -o` writing to the same log path. Ghostty's OSC 133 `C`
(command start) / `D` (command end) markers let nvsh slice out exactly the
last command's real output without ever re-running it. The slice is capped
at 64 KB (head + tail with a truncation marker), has escape sequences
stripped, invalid UTF-8 replaced, and is redacted (`nvsh.redact`) before it
is ever printed or sent to an agent. **The session log's own path never
appears in the output.**

`status` is one of `ok`, `partial` (the command was still running, or
`script(1)` was killed mid-command), `truncated` (the region exceeded the
64 KB cap), or `no capture` (no session log, or no OSC 133 markers found).

## Usage

    nvsh capture --show
    nvsh capture --show --json
    nvsh capture --show --pid 12345

## See also

- `nvsh explain doctor`
_AGENT = """\
# nvsh agent

Lists, chooses, and installs `NvshAgent` harness backends (`nvsh.agent.registry`):
`pi`, `qwen`, `claude`, `codex`, and the stdlib `openai-compat` fallback that
needs no binary on PATH. `nvsh setup` calls the same `choose()` logic to pick
a backend automatically — the configured provider if it is on PATH, else
`openai-compat`, with a reason.

## Usage

    nvsh agent list
    nvsh agent list --json
    nvsh agent use claude
    nvsh agent install pi
"""

_AGENT_LIST = """\
# nvsh agent list

Reports every registered backend with its installed status (from PATH) and
whether it is the currently configured provider:
`{adapters: [{name, installed, binary, description, configured}]}`.

## Usage

    nvsh agent list
    nvsh agent list --json
"""

_AGENT_USE = """\
# nvsh agent use <name>

Sets `[agent] provider` in `config.toml` to `<name>` (one of `pi`, `qwen`,
`claude`, `codex`, `openai-compat`), preserving every other table already on
disk. Refuses an unknown name with a user error.

## Usage

    nvsh agent use claude
    nvsh agent use openai-compat --json
"""

_AGENT_INSTALL = """\
# nvsh agent install <name>

Prints the install command for `<name>` (only `pi` today:
`npm install -g @earendil-works/pi-coding-agent`). Runs it only with `--yes`
or an interactive `y` confirmation, and only when `npm` is on PATH — it never
runs anything on its own.

## Usage

    nvsh agent install pi
    nvsh agent install pi --yes
"""


ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("nvsh",): _ROOT,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
    ("cli",): _CLI,
    ("cli", "overview"): _CLI,
    ("approve",): _APPROVE,
    ("approve", "check"): _APPROVE_CHECK,
    ("approve", "add"): _APPROVE_ADD,
    ("approve", "list"): _APPROVE_LIST,
    ("approve", "remove"): _APPROVE_REMOVE,
    ("capture",): _CAPTURE,
    ("capture", "show"): _CAPTURE,
    ("agent",): _AGENT,
    ("agent", "list"): _AGENT_LIST,
    ("agent", "use"): _AGENT_USE,
    ("agent", "install"): _AGENT_INSTALL,
}
