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
skills-present check, plus the shell/backend checks below. Exits 1 when
unhealthy — but an info-severity failed check (see below) never makes
`healthy` false on its own.

prompt-file-present requires the *resident* prompt the declared backend
actually reads. Other harness prompt files recognized under the same backend
name (`AGENTS.override.md`, `.pi/SYSTEM.md`, `QWEN.md`) belong to
interactively available harnesses the mesh daemon never loads; they are
reported by the informational harness-prompts check and never substituted.

## Shell/backend checks (`nvsh.doctor_checks`)

These run unconditionally, including from a wheel install with no
`culture.yaml` (the prompt-file checks above are skipped there, these are
not):

- `platform_detected` — `nvsh.platform.detect()` found a non-generic kind.
- `agent_configured` — `config.toml` loaded and `[agent] provider` names a
  known adapter.
- `agent_reachable` — probes the configured backend's `/models` endpoint
  (3s timeout, bearer read from `api_key_env` at call time, never printed).
  Distinguishes `pi-missing` (pi not on PATH), `endpoint-unreachable`
  (refused/timeout), `endpoint-401` (bad/missing key) and "endpoint
  unknown" (nothing configured), each with its own remediation.
- `hook_sourced`, `hook_first_in_prompt_command`, `bindings_present` — read
  state a hooked bash passes on the command line (see "the `/doctor`
  invocation" in `docs/shell-integration.md`): `NVSH_HOOK_VERSION`,
  `--prompt-command` (`declare -p PROMPT_COMMAND` output) and `--bind-p`
  (`bind -p` output) plus `--keymap`. Outside a hooked shell that state is
  absent, so these report `passed=false, severity=info` with the
  remediation "run /doctor from a hooked shell" instead of failing health.
- `capture_active` — `NVSH_LOG`/`TMUX` from the environment; reports
  `script: <path>` or `tmux: <path>`, or a warning when capture is off.
- `daemon_status` — `nvsh.daemon.is_running()`; always `severity=info`,
  since "not running" is the normal idle state.
- `terminfo_present` — `infocmp $TERM` (or a `TERMINFO`/`TERMINFO_DIRS`/
  `~/.terminfo`/system search); missing gives the
  `infocmp -x <TERM> | ssh <host> -- tic -x -` remediation, run from the
  machine that has the terminfo entry.

## Usage

    nvsh doctor
    nvsh doctor --json
    nvsh doctor --json --prompt-command "$(declare -p PROMPT_COMMAND)" \\
        --bind-p "$(bind -p)" --keymap "$(bind -V | grep keymap | awk '{print $2}')"
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
    nvsh approve audit --tool bash --command "nvidia-smi -L" --decision user
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

_APPROVE_AUDIT = """\
# nvsh approve audit --tool <tool> --command <cmd> --decision <decision>

Appends one `{tool, command}` decision to the audit log
(`$XDG_STATE_HOME/nvsh/audit.jsonl`, mode 0600). This is what
`nvsh/agent/pi_ext/approval.ts` — the pi approval extension — calls after
every branch (an existing user/session match, "once", "session", "user", or
a deny/block), so the extension stays a pure forwarder: all policy is
Python's (`nvsh.approvals`), the extension never decides anything itself.

`--decision` is one of `user`, `session`, `ask`, `once`, `deny`, `block`.

## Usage

    nvsh approve audit --tool bash --command "nvidia-smi -L" --decision user
    nvsh approve audit --tool bash --command "apt install foo" --decision ask --json
"""

_CONTEXT = """\
# nvsh context

Prints **exactly** the bytes nvsh would send to the agent for the last
recorded failure — the prompt text `nvsh.agent.pi.build_prompt` builds from
the request and the assembled `AgentContext`. Nothing is summarised, and
nothing is added on the way to the backend.

The context has four parts:

- the failed command and its exit code;
- the **platform block** from `nvsh.platform.detect()`: every value nvsh
  looked for, present or absent, each with the file or command it came from
  (`render_block()`), so a wrong fact is traceable to its source;
- the **cwd**;
- the **output slice** for that command from `nvsh.capture.last_slice`,
  already bounded (64 KB), escape-stripped and redacted. The session log's
  own path never appears.

Redaction runs before anything leaves the process, so `--show` is also the
honest way to check what a redaction rule did: `--json` lists the rules that
fired in `redaction_rules`.

With no recorded failure the verb still prints the platform block, so the
machine's detected facts can be inspected at any time.

## Usage

    nvsh context --show
    nvsh context --show --json

## See also

- `nvsh explain capture` — where the output slice comes from.
- `nvsh explain approve` — the "propose, don't run" side of the same flow.
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
"""

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

_DAEMON = """\
# nvsh daemon

The per-user **session daemon** (`nvsh.daemon`). It owns the warm agent
processes and one conversation per shell, and listens on
`$XDG_RUNTIME_DIR/nvsh/daemon.sock` (directory `0700`, socket `0600`).

It is started **lazily** by the hook client on the first qualifying failure —
never at shell start — and it stops when the last shell unregisters (bash's
`EXIT` trap) or after an idle timeout. A stale socket left by a crashed
daemon is reaped at the next start. If the daemon or the backend is missing,
the client falls back to a one-shot adapter run, so a failure is never left
without a diagnosis.

Each shell (keyed by bash's `$$`) owns its own conversation. With
`sessions.max = 1` (the default) one agent process serves every shell: the
active conversation is put to sleep (`switch_session` / `new_session`) and
the caller's is resumed, so contexts from different terminals never mix.
Raising `sessions.max` allows that many concurrent agent processes.

The daemon never writes to a terminal; it logs to
`$XDG_STATE_HOME/nvsh/daemon.log` (`0600`).

## Usage

    nvsh daemon status --json
    nvsh daemon run --foreground
    nvsh daemon stop
    nvsh daemon unregister --shell $$

## See also

- `nvsh explain agent`
- `docs/daemon.md` — the wire protocol
"""

_DAEMON_RUN = """\
# nvsh daemon run

Serves the session daemon. With `--foreground` it serves in this process
(what the tests and `systemd`-style supervision use); without it, the daemon
is spawned detached and the pid is reported. `--idle-timeout <seconds>` sets
how long the daemon may sit idle before it closes its agents and exits.

Refuses to start when another daemon is already listening on the socket
(exit code 2); a socket left behind by a crashed daemon is reaped instead.

## Usage

    nvsh daemon run --foreground
    nvsh daemon run --idle-timeout 300 --json
"""

_DAEMON_STATUS = """\
# nvsh daemon status

Reports `{running, pid, socket, shells, agents, conversations, backend,
backend_reason, fallback_notice, idle_timeout}`. When no daemon is
listening it reports `{"running": false, "socket": "..."}` and still exits
`0` — "not running" is a state, not an error.

`backend` is what the daemon actually chose (`nvsh.agent.registry.choose`);
`fallback_notice` is set when that is not the configured provider, e.g.
`pi unavailable: pi not on PATH and node missing`.

## Usage

    nvsh daemon status
    nvsh daemon status --json
"""

_DAEMON_STOP = """\
# nvsh daemon stop

Asks a running daemon to close its agent processes, unlink its socket and
exit. Idempotent: with no daemon running it reports `{"stopped": false}` and
exits `0`.

## Usage

    nvsh daemon stop
    nvsh daemon stop --json
"""

_DAEMON_UNREGISTER = """\
# nvsh daemon unregister --shell <pid>

Tells the daemon that a shell session has exited. The bash integration's
`EXIT` trap calls this; when the **last** registered shell unregisters, the
daemon closes its agents and stops, so closing the last terminal leaves no
nvsh or agent process behind.

Never starts a daemon: with none running it is a no-op that exits `0`.

## Usage

    nvsh daemon unregister --shell $$
    nvsh daemon unregister --shell 12345 --json
"""

_SETUP = """\
# nvsh setup

Installs the bash hook: renders `nvsh/shell/hook.bash` and `readline.bash`
(via `importlib.resources`, so this works from a wheel install) into
`$XDG_DATA_HOME/nvsh/shell/`, each version-stamped, and inserts one small
marked block into the rc file (default the user's bash rc file, override
with `--rc`)
immediately after the distro's interactive guard (`nvsh.rcfile`). A
timestamped backup of the rc is written before any change, and a second run
is idempotent: an unchanged rc after the first run makes `setup` write
nothing at all. Also picks and reports the agent backend
(`nvsh.agent.registry.choose()`), and prints the pi-install offer text when
`pi` is missing and `npm` is on PATH — it never runs `npm` itself.

## Usage

    nvsh setup
    nvsh setup --rc /path/to/bashrc --json
"""

_UNINSTALL = """\
# nvsh uninstall

Reverses `nvsh setup`: removes the marked rc block (restoring the newest
timestamped backup instead, if the block was hand-edited since `setup` wrote
it), deletes the rendered `$XDG_DATA_HOME/nvsh/shell/*.bash` files, removes
`$XDG_RUNTIME_DIR/nvsh/*.log`, `*.notice` and `daemon.sock`, and stops a
running daemon if a (parallel-task) `nvsh.daemon` module is present —
detected with `importlib.util.find_spec`, never imported directly. A no-op
(exit 0) when nothing was installed.

## Usage

    nvsh uninstall
    nvsh uninstall --rc /path/to/bashrc --json
"""

_OFF = """\
# nvsh off

Prints the bash that unbinds the hook and the readline layer in the
*current* shell and sets `NVSH_DISABLE=1`, meant for
`eval "$(nvsh off)"`. The marked rc block's `nvsh()` shell function makes
plain `nvsh off` typed at the prompt do exactly this, via `--shell` (which
prints the raw bash with no JSON/text wrapper). Without `--shell`, prints the
same snippet inside a `{action, eval}` payload instead of running it.

## Usage

    nvsh off --shell    # meant for: eval "$(nvsh off)"
    nvsh off --json
"""

_ON = """\
# nvsh on

The reverse of `nvsh off`: prints `unset NVSH_DISABLE` plus the `source`
lines for the rendered `hook.bash` / `readline.bash`, meant for
`eval "$(nvsh on)"`. Same `--shell` / `--json` shape as `nvsh off`.

## Usage

    nvsh on --shell    # meant for: eval "$(nvsh on)"
    nvsh on --json
"""

_HOOK = """\
# nvsh hook

Internal: the bash hook (`__nvsh_hook` in `hook.bash`) calls this on every
qualifying failure, never the operator directly. Builds a
`nvsh.triggers.TriggerEvent` from its flags and calls `decide()`; on
`"skip"` it exits 0 silently. On `"ask"` it hands off to the failure client
(`nvsh.client.handle_failure`, task t13) via a lazy, `ImportError`-guarded
import — until that lands, it prints a one-line placeholder instead. Also
compares the `NVSH_HOOK_VERSION` the rc block exported against this
package's own version and prints a one-line refresh notice the first time
they differ in a given shell session (state file under
`$XDG_RUNTIME_DIR/nvsh/<shell-pid>.notice`).

## Usage

    nvsh hook --exit 2 --pipestatus "2" --line "ls /nope" --cwd "$PWD" --log ""
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
    ("approve", "audit"): _APPROVE_AUDIT,
    ("capture",): _CAPTURE,
    ("capture", "show"): _CAPTURE,
    ("context",): _CONTEXT,
    ("context", "show"): _CONTEXT,
    ("agent",): _AGENT,
    ("agent", "list"): _AGENT_LIST,
    ("agent", "use"): _AGENT_USE,
    ("agent", "install"): _AGENT_INSTALL,
    ("daemon",): _DAEMON,
    ("daemon", "run"): _DAEMON_RUN,
    ("daemon", "status"): _DAEMON_STATUS,
    ("daemon", "stop"): _DAEMON_STOP,
    ("daemon", "unregister"): _DAEMON_UNREGISTER,
    ("setup",): _SETUP,
    ("uninstall",): _UNINSTALL,
    ("off",): _OFF,
    ("on",): _ON,
    ("hook",): _HOOK,
}
