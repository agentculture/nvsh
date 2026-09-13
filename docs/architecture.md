# Architecture: hook, not wrap

This is the milestone-1 decision record issue #1 asks for: how nvsh attaches
itself to the operator's shell, and why. It is written from the converged
spec, [`docs/specs/2026-09-13-nvsh-bash-hook-agent-on-error.md`](specs/2026-09-13-nvsh-bash-hook-agent-on-error.md),
and the scope-exploration probes that back it (`s4`-`s6`, `s15`-`s19`).

## Before

As of nvsh 0.9.1, this repository is a scaffold with no shell behaviour:
`README.md:10-14` states plainly that "the shell itself is not built yet."
The four harness prompt files at that version — `CLAUDE.md`, `QWEN.md`,
`AGENTS.override.md`/`.pi/SYSTEM.md` and `AGENTS.colleague.md` — all record
the same planned design: a **PTY wrapper**. In that model nvsh would be the
executable `login`/`sshd` start (installed with `chsh`), spawning a real
`bash` inside a pseudo-terminal it owns, watching that inner shell's exit
status and output, and falling back to `exec`ing the real shell on any
internal failure so the operator is never locked out. `CLAUDE.md`'s
"Login-shell constraints" section (as of 0.9.1) spelled out the rules that
model would have needed: never lock the operator out, pass non-interactive
invocations straight through, preserve login semantics (`-nvsh`, `-l`),
add no latency on the success path, and install itself into `/etc/shells`.

Nobody had built that PTY layer yet, and building it is a materially harder
problem than the one below: a PTY wrapper has to reimplement enough of a
terminal to stay transparent to `tmux`, `vim`, job control, `Ctrl-Z`,
window-resize (`SIGWINCH`) forwarding, and every tool that checks `isatty()`
on its own stdout — none of which a hook has to touch, because the hook
never sits between the operator and the real shell.

## After

nvsh now **hooks** into the operator's existing interactive bash rather than
wrapping it. `nvsh setup` inserts one clearly marked block into the
operator's `$HOME/.bashrc` immediately after the distro's interactive guard
(`If not running interactively, don't do anything` — line 40 on the Spark,
29 on Thor, 28 on Orin) rather than at the end of the file, so the rest of
the rc (`nvm`, `cargo`, `kiro-cli`, `bash-completion`, ...) is never sourced
twice. `nvsh uninstall` removes that block, restoring the backup it kept.
Bash itself is untouched: it still parses, does job control, completion,
aliases, and loads rc files exactly as it always has. Nothing sits between
the operator's keystrokes and bash's own line editing except the specific,
narrow bindings described below — there is no pseudo-terminal, no `exec`
into a wrapper process, and no PTY-transparency problem to solve.

## The decision: hook over wrap, with reasons

- **The login-shell goal is achievable without a wrapper.** The operator's
  actual want — "an NVIDIA engineer appears when something breaks" — needs
  nvsh to see the exit status of a failed command and offer help, not to own
  the terminal session. A `PROMPT_COMMAND` hook sees exactly what a wrapper
  would see (`$?`, and `PIPESTATUS` when ordering is respected — see below)
  without owning the pty, forwarding window resizes, or reimplementing job
  control.
- **Zero cost on the success path is easier to prove for a hook than a
  wrap.** Measured on the Spark (`s5`): `bash -ic true` costs ~0.24 s,
  `uv run nvsh --help` costs ~0.043 s, and `pi --version` (node startup)
  costs ~0.35 s. A hook is a pure-bash function appended to the
  `PROMPT_COMMAND` array; it reads `$?`/`PIPESTATUS`, applies the cheap
  trigger pre-filter in bash, and only execs a Python process when a
  failure actually qualifies. A successful command therefore pays one bash
  function call and nothing else — no fork, no Python import, no model
  call. A PTY wrapper pays its own startup cost on every single shell
  launch, before the first prompt is even drawn, which is the exact
  latency issue #1's milestone-1 checklist rules out.
- **`PROMPT_COMMAND` ordering keeps `PIPESTATUS` intact, and composes with
  Ghostty.** Probed directly on the Spark (`s19`, spec requirement "Hook
  ordering and status capture"): a hook placed *after* Ghostty's own
  `__ghostty_hook` entry saw `PIPESTATUS` already clobbered, while a
  first-position hook saw `false | true` as `1 0` and
  `true | false | true` as `0 1 0` correctly, and still respected the
  operator's own `set -o pipefail`. So nvsh's hook inserts itself as the
  **first** element of the `PROMPT_COMMAND` array, before Ghostty's own
  hook is appended (Ghostty's shell integration
  (`/usr/share/ghostty/shell-integration/bash/ghostty.bash:295-308`)
  already appends array-aware and idempotently, which is the same
  append-don't-clobber idiom nvsh's installer follows). A wrapper would not
  have this ordering problem, but it would also not get Ghostty's own OSC
  133 prompt markers for free — see the output-capture point next.
- **First position is not always ours, and that is fine: under
  `bash-preexec` the hook reads `BP_PIPESTATUS`** (deviation `d2`, measured
  on the Spark against a copy of the operator's real rc). nvsh is first
  when no other prompt manager is present. But `bash-preexec.sh` — which
  Ghostty's own integration sources on bash < 5.3, and which kiro-cli /
  fig / amazon-q load on any bash (on the Spark it is kiro-cli's copy,
  `shell/bashrc.pre.bash`, that wins first position) — rewrites `PROMPT_COMMAND` on its first prompt by
  design: its `__bp_install` puts `__bp_precmd_invoke_cmd` first and folds
  whatever was there before into a single newline-joined first element
  behind it, so the array becomes
  `([0]=$'__bp_precmd_invoke_cmd\n__nvsh_hook\n…' …)`. No installer
  ordering can win that race, because the rewrite happens after every rc
  file has been sourced. What survives it: `$?` is correct, because
  bash-preexec restores it for each folded command via
  `__bp_set_ret_value`; `PIPESTATUS` is not, because that `return` leaves
  it with exactly one element (`false | true` arrived as `0` rather than
  `1 0`). bash-preexec keeps the real per-stage statuses in its own global
  copy, `BP_PIPESTATUS`
  (`/usr/share/ghostty/shell-integration/bash/bash-preexec.sh:63-67,148-174`),
  so `__nvsh_hook` substitutes that copy when bash-preexec is loaded, its
  entry is the first thing in `PROMPT_COMMAND` (which proves the copy is
  this prompt's, not the previous one's) and the captured `PIPESTATUS` has
  the tell-tale single element. `nvsh doctor`'s
  `hook_first_in_prompt_command` check accepts both layouts and names the
  one it saw; it still fails when `__nvsh_hook` appears more than once.
- **Output capture reuses Ghostty's own markers instead of re-inventing
  a scrollback reader.** A hook cannot see a failed command's output the
  way a PTY wrapper could (a wrapper sits in the data path; a hook only
  sees the exit status at the next prompt). v1 closes that gap without a
  wrapper: each interactive session runs under a per-session typescript
  (`script -qfc "$BASH" log`, or `tmux pipe-pane -o` inside tmux), and the
  hook slices that log between the last Ghostty OSC 133 `C` (command
  start) and `D` (command end) marker to recover exactly the failed
  command's real output — verified experimentally (`s17`): slicing the
  typescript or the pipe-pane log between `C` and `D` yielded exactly the
  failed command's stdout+stderr with no re-run. The log lives under
  `$XDG_RUNTIME_DIR/nvsh/` (tmpfs), mode `0600`, and only the last <=64 KB
  slice — redacted first — is ever read; the log itself is never sent
  whole and never leaves the machine.
- **Composing with an existing terminal integration was the deciding
  probe.** Ghostty's bash integration is already active on the operator's
  Spark and reads/writes `PROMPT_COMMAND` and OSC 133 the same way nvsh
  now does (`s16`). A wrapper would have had to either bypass Ghostty's own
  shell integration (losing jump-to-prompt and command selection) or
  reimplement it inside the wrapper; the hook instead composes with it by
  construction, because both are `PROMPT_COMMAND` array entries in the same
  bash process.

## What the hook actually does

- **Trigger rules** (`nvsh/triggers.py`, table-tested) decide, from `$?`,
  `PIPESTATUS`, and session state, whether a failure qualifies for an
  automatic call. Exit `130` (Ctrl-C), `141` (SIGPIPE), `grep`/`diff`
  exiting `1`, `false`, `test`/`[ ]`, and commands a script has already
  handled are never treated as errors; interactive or long-running programs
  never auto-trigger mid-run; automatic calls are rate-limited.
- **Slash commands** (`/ask /fix /explain /retry /doctor /context /agent
  /help`) and Ctrl+G reach the same agent on demand, independent of the
  trigger rules, by binding Enter to a readline macro that inspects
  `READLINE_LINE` before the line runs (`s18`).
- **The agent is pluggable**, behind a small `NvshAgent` contract (request
  kinds, streamed events, cancel, capabilities), with `PiAgent` (driving
  `pi --mode rpc`) as the default and a stdlib OpenAI-compatible adapter as
  the fallback when `pi` is not on `PATH` — as it is not on either Jetson
  today.
- **One session daemon per user** owns the warm agent process, started
  lazily on the first qualifying failure, never at shell start; each shell
  session gets its own conversation, put to sleep when another shell's
  failure needs the daemon's attention and resumed when that shell comes
  back.
- **Nothing an agent proposes runs without the operator's confirmation.**
  Proposals — including `sudo` commands — are shown verbatim and never
  pre-typed into the prompt; there is no auto-apply mode in v1.
- **Kill switches stay simple:** `NVSH_DISABLE=1` makes the sourced hook a
  no-op, `nvsh off`/`nvsh on` unbind and rebind the hook in the current
  shell, and `nvsh uninstall` restores `$HOME/.bashrc` from its backup and
  removes the hook file, runtime sockets, and logs.

## The failure client and the panel

`nvsh hook` is a thin verb: it classifies the event and, on `ask`, hands off
to `nvsh/client.py`, which owns everything up to the operator's prompt
coming back.

- **The failure is recorded first**, as 0600 JSON under
  `$XDG_STATE_HOME/nvsh/last-failure.json`, before the agent is contacted at
  all — so `/fix` still finds it when the backend is down or the operator
  presses Ctrl+C.
- **The rate limit lives on disk.** `nvsh hook` is a fresh process on every
  failure, so an in-memory `RateState` is always empty; the persisted state
  in `$XDG_STATE_HOME/nvsh/rate.json` is what actually holds the window.
- **One prompt composer, shared by every backend**
  (`nvsh/agent/prompt.py`). `nvsh context --show` prints exactly the bytes
  that composer produces for the recorded failure — command, exit code, the
  platform block with every value's source, cwd, and the redacted output
  slice — so what the operator is shown is what the model is sent, for `pi`,
  the subprocess harnesses and the OpenAI-compatible adapter alike.
- **The panel never consults a terminal capability database.** Colour is a
  handful of literal SGR constants, switched off entirely under `NO_COLOR`,
  `TERM=dumb`, or a non-tty, so an ssh in from Ghostty to a machine with no
  matching terminfo entry renders plain readable text instead of failing.
  The first `text_delta` is written and flushed as it arrives.
- **Proposals are decided by one keypress**: Enter runs, `e` explains, `d`
  shows details, Esc ignores. A read-only `inspect` proposal that already
  matches the operator's approval patterns is run by the client itself
  (10 s timeout) and its redacted output fed back as one follow-up prompt;
  anything privileged (`sudo`, `doas`, `pkexec`) is never auto-run, whatever
  the patterns say.
- **Ctrl+C while streaming** cancels the agent run, restores termios, prints
  one line and returns 130 — with the failure still recorded for `/fix`.
- `NVSH_NO_DAEMON=1` keeps the client one-shot (no background process),
  which is also how the air-gapped test reaches a localhost endpoint with
  every non-loopback socket refused.

## What this retires

This decision retires the PTY-wrapper design recorded in `CLAUDE.md`,
`QWEN.md`, `AGENTS.override.md`, `AGENTS.colleague.md` and `.pi/SYSTEM.md`
as of nvsh 0.9.1, and specifically the "Login-shell constraints" section of
`CLAUDE.md` at that version — replaced in the current `CLAUDE.md` by a
"Hook constraints" section covering the same concerns (no lockout, success
path is free of latency, non-interactive shells are untouched by
construction) in hook terms.

## Parked: login-shell mode

nvsh is **not** built as a login shell or PTY wrapper in this scope: no
`chsh`, no `/etc/shells` entry, no `install-shell` verb, no `argv[0]`
`-nvsh` handling. The operator keeps `/bin/bash` as their login shell on
the Spark, Thor and Orin. The login-shell/PTY-wrapper mode is **parked as a
follow-up**, not dropped: it would sit on top of the same `NvshAgent` and
deterministic-tools layer this hook uses, and is judged materially harder
to get right (full terminal transparency, `/etc/shells` installation and
recovery, and the "never lock the operator out" invariant as a tested
property of a process that owns the pty) than revisiting it is worth before
the hook itself is proven.
