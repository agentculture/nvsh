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
- `agent_turn_not_hung` — reads the daemon's `active_turn` via a read-only
  `status` control message (never mutating, never autostarts the daemon).
  `severity=info` and passes when there is no active turn, or the owner
  shell is alive and elapsed time is within the daemon's own turn cap.
  `severity=warning` and fails when the owner shell's pid is gone, or
  elapsed exceeds that cap — either way the remediation is
  `nvsh doctor --apply`.
- `terminfo_present` — `infocmp $TERM` (or a `TERMINFO`/`TERMINFO_DIRS`/
  `~/.terminfo`/system search); missing gives the
  `infocmp -x <TERM> | ssh <host> -- tic -x -` remediation, run from the
  machine that has the terminfo entry.

## `--apply`

The one exception to "read-only": when `agent_turn_not_hung` has failed,
`--apply` fixes exactly that turn, nothing else. A dead-owner turn (its
shell no longer exists) is killed with owner authority the daemon grants
doctor for this case alone. A live-owner turn is killed only after an
interactive `[y/N]` prompt naming the owning shell; off a tty it refuses
without prompting and changes nothing. Every attempt — killed, refused, or
nothing to do — is recorded to the audit log as `kind="doctor_apply"`.

## Usage

    nvsh doctor
    nvsh doctor --json
    nvsh doctor --apply
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

Patterns are `fnmatch` globs matched against one *stage* of the command
line, after whitespace normalization — not just the program name. `add()`
refuses `sudo *`, `rm *`, a bare `*`, and any pattern starting with `sudo`
or `rm`.

## Stages

A command line is split on `|`, `&&`, `||`, `&`, `;` and newlines, honouring
quotes — `ssh orin "ps | head"` is ONE stage whose first argument is
`orin`. Approving stores one pattern per stage, and a command is approved
only when **every** stage matches a pattern, so a broad `ls *` never
authorizes `ls | sudo tee /etc/x`. A line with a subshell or a command
substitution (`(...)`, `$(...)`, backticks) is opaque and is never
auto-approved at all.

## Scopes

Two lifetimes, each in two forms:

- `user` — persisted to `$XDG_CONFIG_HOME/nvsh/approved.toml` (mode 0600),
  so it survives logout and reboot.
- `session` — persisted to `$XDG_RUNTIME_DIR/nvsh/session-approvals.toml`
  (mode 0600), the directory the system creates at login and removes at
  logout. It is therefore in force for every later nvsh process in this
  login session, and gone at the next one. It never reaches
  `approved.toml`.
- `user-specific` / `session-specific` — the same two lifetimes, but the
  pattern keeps the command's *first argument*: `ssh orin *` rather than
  `ssh *`. With no second word to keep, they fall back to the plain form.

The panel's proposal keys write into the same store: `[s]`/`[S]` run the
command and approve it for this session, `[u]`/`[U]` run it and persist the
approval for this user; the uppercase key of each pair stores the specific
form. All four are refused for a command that escalates privilege, for an
opaque line, and for any pattern `add` itself refuses.

On a line with more than one stage the panel numbers the stages and then
asks which of them the approval covers (`stages [all,1,2]: `); only those
stages are stored, and the command still runs once whatever was picked.
`--stages` is that same choice from the CLI.

## Usage

    nvsh approve check "nvidia-smi -q"
    nvsh approve check "nvidia-smi -q" --json
    nvsh approve add "docker logs *"
    nvsh approve add "docker logs *" --session
    nvsh approve add "ssh orin uptime" --scope user-specific
    nvsh approve add "ls /etc | grep -i net" --scope user --stages 2
    nvsh approve list --json
    nvsh approve remove "docker logs *"
    nvsh approve audit --tool bash --command "nvidia-smi -L" --decision user
"""

_APPROVE_CHECK = """\
# nvsh approve check <cmd>

Returns the approval decision for a command line:
`{decision, pattern, stage}` where `decision` is `user`, `session`, or
`ask`, `pattern` is the matching glob (the stage patterns joined by ` | `
for a pipeline, or `null` when nothing matched), and `stage` names the first
stage that is *not* approved when the decision is `ask`. User patterns are
checked before session patterns, and every stage of the line has to match
something for the answer to be anything but `ask`.

## Usage

    nvsh approve check "docker ps -a"
    nvsh approve check "docker ps -a" --json
    nvsh approve check "ps -eo pid | head -n 20" --json
"""

_APPROVE_ADD = """\
# nvsh approve add <pattern>

Approves an `fnmatch` glob pattern, matched against one stage of a command
line. Persists to `user_patterns` by default; pass `--session` to write it
to this login session's runtime-dir store instead.

With `--scope session|session-specific|user|user-specific` the argument is a
*command line*, not a pattern: nvsh splits it into stages and derives one
pattern per stage (the `-specific` scopes keep each stage's first argument).
This is the single writer the pi approval extension calls, so the widening
rules live in exactly one place.

`--stages` narrows a `--scope` approval to some of those stages: `all` (the
default), a single number, or a comma- or space-separated list, numbered
from 1 in the order the panel shows them. It is the operator's answer to the
panel's `stages [all,1,2]: ` prompt, which the pi extension forwards here
after splitting it off the `<scope>:<stages>` value pi's `ctx.ui.select`
carries back. A number past the end of the line, or `--stages` without
`--scope`, is a user error and nothing is written.

Refused outright (raises a user error, nothing is written): `sudo *`, `rm *`,
a bare `*`, any pattern starting with `sudo` or `rm`, and — under `--scope`
— any command line with a privileged stage or a command substitution.

## Usage

    nvsh approve add "docker logs *"
    nvsh approve add "docker logs *" --session
    nvsh approve add "ssh orin uptime" --scope user-specific
    nvsh approve add "ps -eo pid | head -n 20" --scope user
    nvsh approve add "ps -eo pid | head -n 20" --scope user --stages 2
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
`pi`, `qwen`, `qwen-p`, `claude`, `codex`, `agy`, `kiro`, and the stdlib
`openai-compat` fallback that needs no binary on PATH. `claude`, `codex`,
`agy` and `kiro` talk to a hosted (non-local) service. `nvsh setup` calls the
same `choose()` logic to pick a backend automatically — the configured
provider if it is on PATH, else `openai-compat`, with a reason — and writes
its pick to `[aliases].default`.

## Usage

    nvsh agent list
    nvsh agent list --json
    nvsh agent use claude
    nvsh agent install pi
"""

_AGENT_LIST = """\
# nvsh agent list

Reports every registered backend: installed status (from PATH), its wire
`path` (`rpc`, `stream-json`, `app-server`, `acp`, or `http`), whether it is
`hosted` (a non-local service — `claude`, `codex`, `agy`, `kiro`), its
self-reported `capabilities` (`null` when constructing the adapter raised),
and two markers — `default` (this is what the resolved `[aliases].default`
alias, or the legacy `[agent] provider` when no alias table is set, currently
names) and `configured` (the legacy `[agent] provider` match, kept for
backward compatibility). The row matching `default` sorts first:
`{adapters: [{name, installed, binary, path, hosted, capabilities, default,
configured, description}]}`. Text mode tags each hosted and/or default row
inline.

## Usage

    nvsh agent list
    nvsh agent list --json
"""

_AGENT_USE = """\
# nvsh agent use <name>

Sets `[agent] provider` AND `[aliases].default` in `config.toml` to `<name>`
(one of `pi`, `qwen`, `qwen-p`, `claude`, `codex`, `agy`, `kiro`,
`openai-compat`), preserving every other table already on disk. Refuses an
unknown name with a user error.

## Usage

    nvsh agent use claude
    nvsh agent use openai-compat --json
"""

_AGENT_INSTALL = """\
# nvsh agent install <name>

Prints the install step `nvsh.installers.harness_install_step` computes for
`<name>` (any of the eight `nvsh agent` names): `pi` keeps its own command
(`npm install -g @earendil-works/pi-coding-agent`, when `npm` is on PATH);
`claude`, `codex`, `qwen` and `qwen-p` get `npm install -g <package>` for
their npm package, also gated on `npm` being on PATH; `agy`, `kiro` and
`openai-compat` have no known installer, so the step prints
"no known installer for `<name>`" and nothing is run. Executes the step only
with `--yes` or an interactive `y` confirmation — never a non-executable
step even then — and writes an audit-log row either way.

## Usage

    nvsh agent install pi
    nvsh agent install claude --yes
    nvsh agent install agy       # prints "no known installer"
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
nothing at all.

## Picking the agent backend

Without `--agent`, setup probes PATH for every registered harness
(`nvsh.agent.registry.probe`): nothing installed falls back to
`openai-compat` (with the gateway-key hint and the `node`/`pi` bootstrap
offers); exactly one harness installed becomes `[aliases].default` silently;
several installed means it asks once on a real terminal — a numbered list,
tool-calling adapters first, non-tool-calling rows labelled `read-only /
plan mode` — and `--yes` never answers that pick (it only answers the
install-tool prompts below); off a terminal, or with `--json`, the first row
of the probe wins without asking. `--agent <target>` (an alias, a bare
adapter name such as `claude`, or `backend[/model[/effort]]`) skips the
probe entirely and fails with exit 2 naming the missing binary when that
target is not installed. An operator's own existing `[aliases].default` is
kept as long as its backend is still installed; it is only re-probed when it
is `openai-compat` with no `base_url` configured (i.e. nvsh chose it, not
the operator). The pick is written to `[aliases].default` (without touching
`[agent] provider`, so a fallback pick never silently overwrites the
operator's own configured provider), and a running daemon is stopped
afterward so the new default takes effect on the next failure.

A hosted pick (`claude`, `codex`, `agy`, `kiro`) prints the disclosure line
"`<name>` is hosted: on a failure the redacted command, output and device
context leave this machine". Setup also probes whether the picked backend
is reachable right now and reports `agent reachable: <bool> (<message>)`.
On macOS, or when `$SHELL` ends in `zsh`, it prints
"warning: nvsh is not tested on macOS/zsh yet (see issue #11)" — the
platform and hook still install, this is disclosure, not a refusal.

## Installing missing helper tools

Detects any of nvsh's helper tools that are missing (`pi`, `node` as pi's
prerequisite, `uv`, `tmux` — see `nvsh.installers`), scoped to the picked
backend's own prerequisites once a backend is chosen (the unscoped list
only applies on a bare machine, to help bootstrap a harness at all), and
prints each one's purpose and exact install command. Nothing installs
without explicit confirmation: interactively it asks `install <tool>? [y/N]`
once per tool; `--yes` answers yes to all of them; `--no-install` lists the
offers and installs nothing; `--json` is non-interactive by construction and
only lists offers unless `--yes` is also given. A tool with no known
installer for this machine (missing package manager) or whose only known
installer is a curl-pipe-sh (`uv`, when neither `snap` nor a package
manager applies) is only ever printed, never executed — not even with
`--yes`. After any installs run, the agent backend is re-picked (unless
`--agent` forced it), so a freshly installed harness is reported
immediately.

## Usage

    nvsh setup
    nvsh setup --rc /path/to/bashrc --json
    nvsh setup --agent claude          # skip the probe, force this target
    nvsh setup --agent codex/gpt-5/high
    nvsh setup --yes                   # install every missing helper tool
    nvsh setup --no-install            # list missing tools and commands only
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


_SLASH = """\
# nvsh slash <line>

Dispatches one operator-typed `/verb ...` line, `/`-prefixed, against the
registry in `nvsh.slash` (task t14). This is what the readline layer's Enter
macro rewrites a recognized slash line into
(`nvsh/shell/readline.bash`, see `docs/shell-integration.md`); `Ctrl+G`
dispatches `nvsh slash "/ask"` with the half-typed line exported as
`NVSH_DRAFT`.

The registry — not this CLI wrapper — decides what is visible: a command
with a `platforms` restriction (e.g. `/power`, `/clocks`, Jetson-only) is
reported as unknown outside those platforms. `--platform` overrides the
detected kind (mostly for tests); by default it comes from
`nvsh.platform.detect().kind`.

## Usage

    nvsh slash "/ask why is memory high?"
    nvsh slash "/doctor" --json

## See also

- `nvsh explain complete`
- `nvsh explain ask`, `nvsh explain doctor`, `nvsh explain undo`, ...
"""

_COMPLETE = """\
# nvsh complete

Prints Tab-completion candidates for the slash-command surface, backing
`nvsh/shell/readline.bash`'s `complete -I` (initial word) and `complete -F`
(per-command arguments) registrations. The bash layer holds no command list
of its own — everything comes from this verb.

With no words after `--`, prints the full palette (`/`-prefixed values,
platform-filtered). With `-- <command> <partial>`, prints that command's own
argument candidates from its registered completion provider (e.g. `/doctor`
offers `--json --strict`; `/agent` offers `list use <adapter names>`;
`/approve` offers `list add remove --session`). The bash side prefix-filters
the returned values itself, so nvsh may return the full candidate set.

## Usage

    nvsh complete --json
    nvsh complete --json -- /doctor --st
"""

_ASK = """\
# nvsh slash /ask <text>

A free-form question with the machine's context attached — the same entry
point `Ctrl+G` uses, with the operator's half-typed line passed along as
`NVSH_DRAFT` instead of as the question itself. Routed with request kind
`slash` (a direct `nvsh ask`/`Ctrl+G` call uses `explicit`), so the two are
distinguishable on the wire without duplicating `nvsh.client.ask`.

`--agent <name>` (or `--agent=<name>`) makes one named harness answer this
one request, one-shot, instead of the configured one; an unavailable harness
is a single `nvsh: @<name> is not available: ...` line and never a silent
fallback. This is what the `@name` mark at the prompt is rewritten to, and
`? <text>` is rewritten to a plain `/ask <text>` (deviation d23; see
docs/shell-integration.md).

## Usage

    nvsh slash "/ask why is memory high?"
    nvsh slash "/ask --agent qwen why is memory high?"
"""

_FIX = """\
# nvsh slash /fix

Asks for the smallest fix for the last recorded failure
(`nvsh.client.fix`). Nothing runs without the operator's confirmation.

## Usage

    nvsh slash "/fix"
"""

_EXPLAIN_SLASH = """\
# nvsh slash /explain

Asks what the last recorded failure means on this machine
(`nvsh.client.explain`).

## Usage

    nvsh slash "/explain"
"""

_RETRY_SLASH = """\
# nvsh slash /retry

Re-runs the last failed command, but only after the operator confirms the
exact command shown (`nvsh.client.retry`), then reports pass/fail with
`nvsh.client.verify`.

## Usage

    nvsh slash "/retry"
"""

_STEER_SLASH = """\
# nvsh slash /steer <text>

Tells the agent what to do instead, in the conversation it is already
having (`nvsh.client.steer`). The same move the proposal prompt's `[t]`
key makes, from the shell prompt instead of from a panel.

If this shell has a turn running, the text is injected into it mid-turn —
for pi that is an rpc `prompt` carrying `streamingBehavior: "steer"` (see
`docs/pi-rpc.md`), delivered after the current assistant turn finishes its
tool calls — and the command returns at once. Otherwise the text becomes
the next request in the same conversation, with the last recorded failure
as its context, and the answer streams into the panel.

Nothing is ever run by steering: the agent may propose again, and that
proposal goes through the same panel as any other.

## Usage

    nvsh slash "/steer just run free -h"
"""

_CONTEXT_SLASH = """\
# nvsh slash /context [--show|--json]

Prints exactly the bytes that would be sent to the agent for the last
recorded failure (`nvsh.client.context_show`). See `nvsh explain context`.

## Usage

    nvsh slash "/context"
    nvsh slash "/context --json"
"""

_AGENT_SLASH = """\
# nvsh slash /agent [list|use <name>]

Lists or chooses the configured `NvshAgent` harness backend, in-shell. With
no argument, behaves like `list`. See `nvsh explain agent`.

## Usage

    nvsh slash "/agent list"
    nvsh slash "/agent use pi"
"""

_HELP_SLASH = """\
# nvsh slash /help

Prints the slash-command registry — every command visible on this
platform, its aliases and its one-line description — straight from
`nvsh.slash.REGISTRY`. There is no separate hard-coded list to drift from it.

## Usage

    nvsh slash "/help"
"""

_UNDO = """\
# nvsh slash /undo

Drops the last agent turn and its pending proposal from the conversation.
**Never runs anything on the machine and never reverts a change a proposal
already made** — the spec's `/undo` decision scopes this to nvsh's own view
of the conversation; reverting machine changes is tracked as a separate,
later GitHub issue.

With a daemon running, sends the `undo` control message and the daemon's
`Conversation.undo()` drops its last transcript entry and clears
`pending_proposal` — the backend's own on-disk session (e.g. pi's) is left
untouched, since there is no RPC to selectively erase one turn there; only
nvsh's view of it, and what it would re-show, changes. With no daemon
running, clears any `pending_proposal` recorded in
`$XDG_STATE_HOME/nvsh/last-failure.json` instead.

## Usage

    nvsh slash "/undo"
"""

_APPROVE_SLASH = """\
# nvsh slash /approve [list|add <pattern> [--session]|remove <pattern>]

The in-shell form of `nvsh approve` (see `nvsh explain approve`), over the
same `nvsh.approvals.Approvals` store. With no argument, behaves like
`list`.

## Usage

    nvsh slash "/approve list"
    nvsh slash "/approve add 'kubectl get *'"
    nvsh slash "/approve add 'apt install *' --session"
    nvsh slash "/approve remove 'kubectl get *'"
"""

_DOCTOR_SLASH = """\
# nvsh slash /doctor [--json]

Runs `nvsh doctor`'s checks in-shell, including the three that only make
sense inside a hooked bash: `hook_sourced`, `hook_first_in_prompt_command`
and `bindings_present`. Those need state only bash itself can see, so
`nvsh/shell/readline.bash`'s Enter macro exports `NVSH_PROMPT_COMMAND`
(`declare -p PROMPT_COMMAND`), `NVSH_BIND_P` (`bind -p`) and `NVSH_KEYMAP`
(from `bind -V`) right before every slash dispatch, and this handler reads
them from the environment. Run `nvsh doctor` directly (not through
`nvsh slash`) and those three checks report `passed=false, severity=info`
with "run /doctor from a hooked shell" instead, since that state cannot
exist outside one. See `docs/shell-integration.md`.

## Usage

    nvsh slash "/doctor"
    nvsh slash "/doctor --json"
"""

_POWER = """\
# nvsh slash /power

Jetson-only stub (`platforms={"jetson"}`): hidden in `nvsh complete` and
`/help` on DGX Spark, RTX Spark and any undetected platform, and prints
"not implemented in this milestone" when run. Exists to prove the
`SlashCommand.platforms` filtering works end to end, not as a real power
control yet.

## Usage

    nvsh slash "/power"
"""

_CLOCKS = """\
# nvsh slash /clocks

Jetson-only stub, the `/clocks` twin of `/power` — see `nvsh explain power`.

## Usage

    nvsh slash "/clocks"
"""

_TIERS = """\
# nvsh tiers

Inspects, exports and prefetches the local response tiers (Needle3 tier 1,
LFM2.5 tier 2, in front of the full agent tier). Read-only over
`$XDG_STATE_HOME/nvsh/tiers.jsonl` (`nvsh.tiers.records.TierRecords`) and the
pinned engine/weights/image cache (`nvsh.tiers.fetch`). `nvsh.tiers` is
imported lazily inside each handler, never at CLI startup.

## Usage

    nvsh tiers stats
    nvsh tiers export ./tiers-bundle.json
    nvsh tiers prefetch --yes

## See also

- `nvsh explain tiers stats`
- `nvsh explain tiers export`
- `nvsh explain tiers prefetch`
- `nvsh explain tiers bench`
"""

_TIERS_STATS = """\
# nvsh tiers stats

Aggregates `TierRecords.read_all()` (`nvsh.tiers.stats.compute_stats`) into
per-tier counts, latency p50/p95 (nearest-rank, deterministic, empty-safe),
an escalation/decline-reason histogram (from each record's
`decline_reason`), and operator approve/decline rates (from
`operator_decision`). Tier groups come from whichever `tier` values appear in
the records — nothing here hard-codes a tier list. Also reports `dropped`:
writes `TierRecords` counted as lost (disk full, a lock that would not
open), from the live `TierRecords.dropped` counter.

## Usage

    nvsh tiers stats
    nvsh tiers stats --json
"""

_TIERS_EXPORT = """\
# nvsh tiers export <file>

Writes a redacted bundle — the nvsh version, a platform-kind summary, and
every record (re-redacted on the way out through `nvsh.redact.redact`,
even though records are already redacted at write time) — to a **local file
only**, mode `0600`. Refuses any target that looks like a URL
(`scheme://...`) or an scp-style remote (`host:path`); refuses to overwrite
an existing file without `--force`. Opens no socket.

## Usage

    nvsh tiers export ./tiers-bundle.json
    nvsh tiers export ./tiers-bundle.json --force --json
"""

_TIERS_PREFETCH = """\
# nvsh tiers prefetch

Shows what `nvsh.tiers.fetch.plan_prefetch` says would be fetched — engine,
weights, and any pinned container images — with sizes, and asks before
downloading anything. Off a terminal, or under `--json` without `--yes`,
it refuses outright: a `CliError` whose remediation names `--yes`, never a
silent download. On an interactive terminal without `--yes` it prompts once
per missing item. `--yes` downloads without asking. Tests inject
`nvsh.tiers.fetch.prefetch`; nothing here ever reaches the real network in a
test run.

## Usage

    nvsh tiers prefetch
    nvsh tiers prefetch --yes
    nvsh tiers prefetch --json
"""

_TIERS_BENCH = """\
# nvsh tiers bench

Runs the committed benchmark corpus (`nvsh/tiers/corpus/dev.json` or
`held-out.json`) through a real `nvsh.tiers.router.TierRouter`
(`nvsh.tiers.bench.bench`) — the same router that answers a live request —
and reports accuracy, argument accuracy, a false-mutating-pick count,
escalation precision/recall, cold/warm latency, idle/peak/reserved memory,
image size, and a pass/miss line per spec-c20 success-signal target. A
target that was not measured (no memory reading supplied, too few corpus
items) prints "not measured", never "pass".

`--split held-out` reports "held-out: 0 entries (operator has not added
any)" until an operator adds entries there — held-out phrasings must not
be authored alongside the dev-split operation descriptions in the same
sitting (assumption c38), so the file ships empty on purpose.

`--tier fixture` (the default) uses `nvsh.tiers.bench.UnavailableTier`,
which declines every request, so the verb runs end to end with no model
installed. `--tier needle` imports `nvsh.tiers.needle` lazily and, until
that module ships, fails with a `CliError` naming `--tier fixture` as the
remediation. Records written during a run go to a throwaway
`TierRecords` in a temp directory, never `$XDG_STATE_HOME/nvsh/tiers.jsonl`.

## Usage

    nvsh tiers bench
    nvsh tiers bench --split held-out
    nvsh tiers bench --tier needle --out results.json
    nvsh tiers bench --json
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
    ("slash",): _SLASH,
    ("complete",): _COMPLETE,
    ("slash", "ask"): _ASK,
    ("slash", "fix"): _FIX,
    ("slash", "explain"): _EXPLAIN_SLASH,
    ("slash", "retry"): _RETRY_SLASH,
    ("slash", "steer"): _STEER_SLASH,
    ("slash", "context"): _CONTEXT_SLASH,
    ("slash", "agent"): _AGENT_SLASH,
    ("slash", "help"): _HELP_SLASH,
    ("slash", "undo"): _UNDO,
    ("slash", "approve"): _APPROVE_SLASH,
    ("slash", "doctor"): _DOCTOR_SLASH,
    ("slash", "power"): _POWER,
    ("slash", "clocks"): _CLOCKS,
    ("tiers",): _TIERS,
    ("tiers", "stats"): _TIERS_STATS,
    ("tiers", "export"): _TIERS_EXPORT,
    ("tiers", "prefetch"): _TIERS_PREFETCH,
    ("tiers", "bench"): _TIERS_BENCH,
}
