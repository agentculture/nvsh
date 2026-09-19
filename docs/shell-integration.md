# Shell integration: the bash readline layer

`nvsh/shell/readline.bash` is the readline half of the hook design recorded
in [architecture.md](architecture.md). It is a pure-bash file sourced into an
interactive bash (by `nvsh setup`'s block in the operator's `.bashrc`, or by
hand). It owns three keys — Enter, Tab and `Ctrl+G` — and nothing else: it
does not install a `PROMPT_COMMAND` entry, does not trap `ERR`, and makes no
agent call of its own.

The file ships in the wheel: `pyproject.toml`'s
`[tool.hatch.build.targets.wheel]` lists `artifacts = ["nvsh/shell/*.bash"]`
so the `.bash` file travels with `nvsh/shell/__init__.py`.

## The bash-side contract with the Python CLI

The bash layer never contains the slash-command list. Everything it knows
about commands and their arguments comes from two `nvsh` calls (implemented
by plan task t14; `tests/fakes/nvsh` stands in for them today).

### `nvsh complete --json`

Prints the slash-command palette — the first word of a slash line:

```json
{"items": [{"value": "/doctor", "description": "run the in-shell checks"}]}
```

- `items` is an array; each entry has a `value` (the literal completion
  candidate, `/`-prefixed for a command) and a human `description`.
- The bash side reads only `value`. Values are plain words with no JSON
  escapes, which is what lets the layer scan the payload with a bash regex
  instead of forking a JSON parser.
- A non-zero exit, missing binary or unparseable payload yields an empty
  list, and the layer degrades to plain bash.

### `nvsh complete --json -- <word>...`

Everything after `--` is the words of the line being completed: the command
first, then the partial current word. Argument candidates for `/doctor`:

```console
$ nvsh complete --json -- /doctor --st
{"items": [{"value": "--strict", "description": "fail on warnings"}]}
```

The bash side still prefix-filters the returned values itself, so `nvsh`
may return the full candidate set and ignore the partial word.

### `nvsh slash <line>`

The dispatch. It receives the operator's original line as **one** argument,
`/`-prefix included:

```bash
nvsh slash '/ask why is memory high?'
```

`Ctrl+G` dispatches `nvsh slash /ask` with the line the operator had typed
exported as `NVSH_DRAFT`, so the agent gets the unfinished command as
context without it being sent as a question.

#### Exit status

`nvsh slash` exits **0 whenever a registered, visible command actually ran**,
whatever that command reported — a `/doctor` that finds an unhealthy check
still exits 0, and the finding is in the panel (and in `--json`'s
`exit_code`, which stays the verb's own result). Only an *unhandled* line —
an unknown command, one hidden on this platform, or an empty one — is a user
error and exits 1.

The reason is the hook: the dispatch is a real command line at the
operator's prompt, so a non-zero status there is a failing command as far as
`__nvsh_hook` is concerned, and nvsh would answer its own diagnostic with a
full agent turn. Two independent guards stop that, and each works alone:

- `__nvsh_enter` and `__nvsh_ctrl_g` set the shell variable
  `__NVSH_SLASH_DISPATCH=1` immediately before the macro's `accept-line`, so
  the very next prompt is known to be nvsh's own. `__nvsh_hook` consumes the
  flag (a plain assignment — no fork, and one-shot, so the failure *after* a
  slash command still triggers normally) and returns before the trigger
  pre-filter. Nothing is passed through the environment and nothing is
  written to disk.
- `nvsh slash`'s exit status above, which also covers an operator who types
  `nvsh slash "/doctor"` by hand (leading space or not).

The verb's *own* result — including the `declined` status below — is still
reported, just not as the process's exit status: it is `exit_code` in
`nvsh slash --json`, and it is written to the audit log.

| Result | Code | Where it shows |
| --- | --- | --- |
| ran | 0 | `nvsh slash` exit status; `--json` `exit_code` |
| unhandled line | 1 | `nvsh slash` exit status |
| stopped: `[s]` at the choice prompt, or Ctrl+C at another nvsh prompt | 130 | `--json` `exit_code`; audit `cancel` / `force_kill` |
| declined: Esc/ignore at a proposal, or exit at the busy prompt | 3 (`EXIT_DECLINED`) | `--json` `exit_code`; audit `declined` |

Choosing `[t]` steer at the choice prompt and letting the turn finish
normally, or dismissing the prompt with `[Esc]` or its 30 s timeout, both
leave the exit status untouched — 0 for a normal completion, never 130.
See "Stopping the agent" below.

`nvsh hook` returns the same 0/3/130, and `hook.bash` discards it with
`|| return 0`, so the operator's `$?` and `PIPESTATUS` for the failing
command are never changed (`tests/test_hook_bash.py`).

## What each key does

### Enter

`C-m` is bound to the macro `"\C-x\C-n\C-j"`: a `bind -x` callback on the
otherwise unused `\C-x\C-n`, then `\C-j` (`accept-line`). A `bind -x`
function cannot itself call `accept-line`, so the macro chain is required
(spec scope entry `s18`).

`__nvsh_enter` looks at `READLINE_LINE`:

- a line not starting with `/`, `?` or `@` returns immediately — an ordinary
  command costs one bash function call and **no fork**;
- a first word that is not `/name` (so `/tmp/x`, or a bare path with a
  second slash) passes through unchanged;
- a first word that `nvsh complete --json` lists is rewritten to
  `" nvsh slash '<line>'"` — note the single leading space — and the
  original line is pushed with `history -s`, so `history` shows what the
  operator typed and not the dispatch.
- a line carrying one of the marks below (`? …`, `@target …`) is rewritten
  the same way, to `/ask …`.

`/notacmd` therefore still runs as bash and still fails as bash.
`command_not_found_handle` does **not** fire for `/doctor`, which is why
this rebind exists at all.

#### HISTCONTROL

The leading space only keeps the dispatch line out of history when
`HISTCONTROL` contains `ignorespace`. On load the layer appends
`ignorespace` to `HISTCONTROL` if neither `ignorespace` nor `ignoreboth` is
already there, and never replaces an existing value
(`erasedups` becomes `erasedups:ignorespace`). If you unset `HISTCONTROL`
later in your rc, re-source the file or your slash dispatches will show up
in `history`.

### Talking to the agent from the prompt

Five ways in, all landing on the same request (deviation d23):

| You type | What happens |
|----------|--------------|
| `/ask <text>` | The slash verb. `--agent <target>` (or `--agent=<target>`) makes one named harness — or a `backend/model[/effort]` literal — answer this request. |
| `? <text>` | A request to the **default** agent. |
| `@<target> <text>` | A request answered by **that target**, for this request only — a plain harness name (`@pi`, `@qwen`, `@claude`, `@codex`, `@openai-compat`, an alias, or `default`) or a `backend/model[/effort]` literal (`@claude/sonnet/medium why is the gpu slow`, `@claude/sonnet …`). See "The `@target` grammar" below. |
| `Ctrl+G` | Asks with the half-typed line as the draft, not as the question. |
| a plain sentence (`what are the memory levels?`) | A *guess*: bash reports `command not found` and nvsh's `prose_request` heuristic recognises the shape (deviation d20). |

`?` and `@target` are **explicit**, exactly like `Ctrl+G`: they are never held
back by the automatic-call rate limiter and never consume its window. The
plain-sentence route is a guess, so it stays rate-limited.

Whichever route a request takes, and whichever of the eight registered
adapters (`nvsh agent list --json`) answers a resolved target, the same
redaction boundary applies before anything leaves the process: `nvsh/redact.py`
runs on the prompt composer's output, `--show-context` prints exactly
those redacted bytes, and every subprocess-backed adapter's child
environment has `CLAUDECODE`/`CLAUDE_CODE_*` stripped
(`nvsh/agent/_env.py`) — see `CLAUDE.md`'s "Device context, with redaction
always on" for the full rule.

A harness that is not installed or configured produces one line and nothing
else — no panel, no fallback to the default:

```console
$ @qwen why is memory high?
nvsh: @qwen is not available: 'qwen' is not on PATH
```

#### When a mark counts

The rules are implemented twice — in `__nvsh_mark_line` (bash, the preferred
route) and in `nvsh.triggers.parse_mark` / `_is_agent_name` (Python, the hook
fallback) — and they are deliberately identical, table-tested against each
other by `tests/test_readline_bash.py`'s import of
`tests/test_triggers_prose.py`'s `_TARGET_POSITIVES` / `_TARGET_NEGATIVES`
(task t7's bash/python sync test):

- `?` — the character after the `?` is a space or an ASCII letter, **and**
  the line either contains a space or ends in `?`, **and** something is left
  once the mark and surrounding blanks are stripped. So `? what is the ram
  level`, `?whats the cuda version?` and `? ram` are requests, while `?`,
  a lone `?` with a trailing blank, `?*.txt`, `?1x`, `?.bashrc` and a bare `?foo` are left to bash as the
  globs they look like.

##### The `@target` grammar (task t6)

- `@target` — `target` is followed by whitespace and a non-empty question,
  and is one of:
  - a **plain name** — a letter, then letters, digits, `_` or `-` — that is
    a *registered* alias (including `default`) or adapter, e.g. `@pi`,
    `@reviewer`, `@default`; or
  - a **`backend/model[/effort]` literal** — 1 to 3 non-empty
    `/`-separated segments, the first character an ASCII letter — whose
    *first* segment is a *registered adapter* name (never merely an alias),
    e.g. `@claude/sonnet/medium why is the gpu slow` or
    `@claude/sonnet why is the gpu slow`.

  `@`, `@pi` alone, `@foo.bar hello`, `@notaharness what is up` (an
  unregistered plain name), `@notaharness/model what is up` (an unregistered
  backend segment), `@claude//` and `@claude/sonnet/` (an empty segment),
  `@claude/sonnet/medium/extra` (too many segments), `@/claude/sonnet` (a
  `/` cannot lead) are all ordinary commands, and an address in argument
  position (`mail a@b.c`, `mail a@b.c/d`) never starts the line, so it is
  never a mark.

The bash side gets the harness list the same way it gets the command list,
and holds no list of its own:

- a plain `@name` is checked against the `@name` entries of the
  `nvsh complete --json` palette (`nvsh.slash.agent_mark_items` — the union
  of configured aliases and registered adapters);
- a slashed `@backend/…` target is checked against
  `nvsh complete --json -- /ask --agent`'s answer — the *adapter-only* list
  `nvsh.slash._complete_ask` returns for `/ask --agent <TAB>` — so an alias
  name is never accepted as a backend segment, on either side.

No name — alias, adapter or otherwise — is hard-coded in the `.bash` file.

No Tab completion is offered for either shape of `@target` — bash completes
a first word starting with `@` as a *hostname* before any programmable
completer is consulted (see "Tab" below), so both `@name` and
`@backend/model[/effort]` marks are Enter-only.

#### The two routes

- **Readline (preferred).** `__nvsh_enter` rewrites `? what is the ram level`
  to `" nvsh slash '/ask what is the ram level'"`, `@pi how much ram` to
  `" nvsh slash '/ask --agent pi how much ram'"`, and
  `@claude/sonnet/medium why is the gpu slow` to
  `" nvsh slash '/ask --agent claude/sonnet/medium why is the gpu slow'"`,
  pushing the original line with `history -s`. Bash never runs the mark, so
  there is no `?: command not found` on screen at all.
- **Hook fallback.** In a shell where only `hook.bash` is sourced, bash does
  run the line and reports 127; `nvsh hook` then classifies it with
  `prose_request`, which returns the same question and target string, skips
  the rate limit because the request is explicit, and answers it.

A `@target` request always runs **one-shot** rather than through the warm
daemon: the daemon holds a session for the *configured* harness, so asking
it would quietly answer from the default backend instead.

### Tab

- `complete -I -F __nvsh_complete_initial` claims the **initial-word** slot,
  which bash-completion leaves free — its `complete -D` loader is untouched.
  For a first word matching `/*` with no second `/`, the palette is merged
  with `compgen -f` path matches, so `/` + Tab lists `/doctor` next to
  `/etc`, `/do` + Tab completes `/doctor` and `/tm` + Tab still completes
  `/tmp/`. Any other first word falls through via
  `compopt -o bashdefault -o default`, so `ec` + Tab still lists `echo`.
- `complete -F __nvsh_complete_args <command>` is registered for each
  command at load time — one `nvsh complete --json` call per interactive
  shell — because readline consults the per-command completion for a
  non-initial word and never calls back into the `-I` function.

The command list lives in a bash variable only for the duration of one call
and is cleared afterwards; there is no static list in the file.

Tab does **not** complete the `@target` marks — neither a plain `@name` nor
a slashed `@backend/model[/effort]` target (task t6/t7): bash completes a
first word starting with `@` as a *hostname* before any programmable
completer is consulted (checked on bash 5.2), so `@` + Tab stays bash's own
behaviour and the marks are Enter-only.

### Ctrl+G

`C-g` is bound to the macro `"\C-x\C-g\C-j"`, the same shape as Enter: the
`bind -x` callback `__nvsh_ctrl_g` on `\C-x\C-g`, then `accept-line`.

The callback never talks to the agent itself. A `bind -x` function runs
while readline owns the tty, so a panel streamed from inside one is mangled
at best — and when its output is redirected away (as it was through nvsh
0.9.2) `Ctrl+G` prints its marker and silently discards the answer. Instead
the callback:

- prints the one-line `⚡ nvsh (Ctrl+G): asking the agent` marker,
- stashes the typed line in the shell variable `__NVSH_DRAFT` (a stash, not
  an inlined argument, keeps the visible line short and needs no quoting),
- pushes that line with `history -s`, so the half-typed command is one `Up`
  away after the panel,
- sets `__NVSH_SLASH_DISPATCH=1` (see "Exit status" above), and
- rewrites `READLINE_LINE` to the hidden
  `" NVSH_DRAFT=$__NVSH_DRAFT nvsh slash \"/ask\""`.

The macro's `accept-line` then runs that as an ordinary command line, so the
answer streams on the tty exactly as a typed `/ask` does, with job control
and `Ctrl+C` working normally. The leading space keeps it out of history.

## Keymaps

Every binding is registered three times, with `bind -m emacs`,
`bind -m vi-insert` and `bind -m vi-command`. A plain `bind -x` does not
fire after `set -o vi`; a vi-insert keymap binding does.

## Kill switches

- `NVSH_DISABLE=1` in the environment makes sourcing the file a complete
  no-op: no bindings, no `complete` registration, no `HISTCONTROL` change.
- `__nvsh_readline_unbind` (behind `nvsh off`) restores `C-m` to
  `accept-line`, removes both dispatch sequences (`\C-x\C-n`, `\C-x\C-g`)
  and `C-g` in all three keymaps, and removes the `-I` and per-command
  completions.
- Sourcing twice is a no-op, guarded by `__NVSH_READLINE_LOADED`.

## Degrade, never lock

The file never sets `-e`, every function returns 0, and the Enter macro
runs `accept-line` whether or not the callback succeeded — so a broken or
erroring `__nvsh_enter` costs you the slash routing, not your terminal.
`tests/test_readline_bash.py` asserts that with a deliberately erroring
hook installed.

## Tests

`tests/test_readline_bash.py` drives a real `bash --norc --noprofile -i` on
a pty and feeds CR (`\r`) keystrokes: a terminal sends CR for Enter and that
is what `\C-m` binds, so feeding LF would bypass the whole layer. Tab tests
set `show-all-if-ambiguous` so a single Tab prints the candidate list. The
suite skips when no pty is available.

## The inline panel (`nvsh/panel.py`)

Everything the operator sees after a qualifying failure is drawn by
`nvsh/panel.py`. It consults no capability database and shells out to
nothing: colour and cursor control are a handful of literal SGR/CSI
constants, switched off entirely under `NO_COLOR`, `TERM=dumb`/empty, or
when stdout is not a tty — so a headless ssh from Ghostty into a machine
with no `xterm-ghostty` terminfo entry renders exactly the same text,
minus the escapes.

Three things make a slow turn legible (deviation d13, after an operator on
the DGX Spark reported "I don't see proper indications things run"):

- **Waiting indicator.** The real model routinely takes tens of seconds
  before its first token. When no event has arrived for more than a second,
  a background ticker paints one dim line, `... waiting for the agent (12s)`,
  updated once a second with an honest elapsed count. On a tty with styling
  on it is repainted in place with a literal `\r` plus `ESC [ 2 K`
  (erase-to-end-of-line) — never `tput`, never an animated spinner.
  Everywhere else (pipe, `NO_COLOR`, `TERM=dumb`) the panel prints the plain
  line `... waiting for the agent` **once** and never repaints, so a
  captured log stays one line per fact. The line is always erased before the
  next real event reaches the screen, and the ticker is paused for as long
  as the panel is handling an event — a proposal prompt is never painted
  over. One lock guards every write, so the ticker can never interleave
  with the agent's text, and `Ctrl+C` or `Esc` still opens the choice
  prompt within a second while waiting (see "Stopping the agent" below).
- **Keypress acknowledgement.** The instant a key is pressed at a proposal,
  the panel prints `nvsh: running ...` (Enter),
  `nvsh: running; 'ssh orin *' approved for this session` (`s`/`S`),
  `nvsh: running; 'ssh orin *' approved for this user` (`u`/`U`) — each ack
  names the pattern it just stored — `nvsh: explaining ...`
  (`e`), `nvsh: details ...` (`d`) or `nvsh: ignored` (Esc or anything
  else) — before anything else happens. Whatever follows can take seconds,
  and the operator must never wonder whether the key registered. This ack
  is the *first* line; the client's own outcome lines (`nvsh: not run`,
  `nvsh: <cmd> -> exit N`, the dim `... running <cmd>` of an auto-approved
  inspection) still follow it and are deliberately worded differently.
- **Fallback notice.** A `status` event that announces the daemon was not
  usable — text starting with the word `one-shot`, or containing `daemon did not start`,
  `daemon refused` or `daemon connection lost`, all produced by
  `nvsh/client_transport.py` — is rendered as a visible (bold, not dim)
  `nvsh: falling back - <original text>` line, because it explains the
  slower turn that follows. Every other status stays a dim `... text` line,
  and a status with empty text prints nothing at all.

### Local tiers

With `[tiers] enabled = true` in `config.toml`, the client offers each
request to the daemon's resident local tiers *before* the full agent
(`nvsh/client.py`, in front of the one place that sends to an agent).
`[tiers] enabled = false` — the default — is off in the strongest sense: no
tier module is imported, no extra socket round trip is made, and the turn is
byte for byte what it was before the tiers existed. A request that names a
harness (`@claude ...`, `/ask --agent qwen ...`) goes past both tiers
untouched, and so does the follow-up turn of a conversation the full agent is
already having.

**The header names who answered.** The panel's first line is the same slot
that names the target:

| What happened | Header |
|---|---|
| A tier answered | `needle`, or `lfm` for Tier 2 |
| No tier answered | `needle -> claude/opus · stream-json · warm` |
| The tiers were not consulted | `claude/opus · stream-json · warm` |

A tier that could not be loaded (no model, below the memory floor, no
container runtime) says so in one dim status line, once, and never as an
error: the request is on its way to the full agent either way. What the
tiers already inspected travels to that agent as a short, bounded
`local inspection results:` block appended to the context, so the work is
not repeated.

**Nothing about approval changes.** A tier's proposal is rendered through the
same panel and the same approve/execute/verify path an agent's proposal
takes — the same keys, the same scope patterns, the same refusal of `sudo`
and destructive commands. Tier 1 never executes anything. Tier 2, once you
have configured it, runs **read-only** operations from the table on its own
while it looks into a request (memory, disk, a unit's status or logs); it
never runs a mutating one, and anything it wants changed comes to you as a
proposal. See [`tier2.md`](tier2.md).

**Declining sends it on, but only if you say so.** After you decline what a
tier proposed, nvsh asks once, `nvsh: send the same request to the full
agent? [y/N]`. `y` sends the *same* request to the full agent (the header
then reads `needle -> …`); `n`, Esc, Ctrl+C and silence all end the turn as
declined (exit 3). Off a terminal — not a tty, `TERM=dumb`, no stdin — the
question is never asked and never assumed: the turn ends as declined rather
than escalating silently.

Whatever you decide is reported back to the daemon as `approved`/`declined`
for that route, which is how the tier measurement log learns what its
proposals were worth. A report that cannot be delivered costs a measurement
and nothing else.

### Stopping the agent (stop-choice-prompt)

While the agent works — thinking, streaming text, or the waiting ticker —
the first `Ctrl+C` or a lone `Esc` no longer cancels the turn at once. It
pauses the panel and shows a one-line choice:

```text
nvsh: paused -- [t] steer  [s] stop  [Esc] keep going
```

Where the harness cannot steer mid-turn — every adapter except `pi` and
`codex` — the first key reads `[t] stop & correct` instead. Nothing is
sent to the harness until the operator picks:

- **`[t]` steer / stop & correct:** types one line. On `pi` and `codex` the
  line is delivered into the running turn and nothing is cancelled. On
  every other harness the running turn is cancelled first, and the typed
  line is sent as the next request with the original request folded in, so
  the follow-up is self-contained.
- **`[s]` stop:** does exactly what the first press did before this change:
  1. **First press (of the stop path):** nvsh asks the harness to stop
     through its own channel (pi `abort`, codex `turn/interrupt`, ACP
     `session/cancel`, a signal for agy; claude, qwen-p and openai-compat
     stop at once). The panel stays up and prints `stopping… press again
     to kill` within a second, and keeps rendering until the turn really
     ends.
  2. **Second press:** nvsh kills the harness's whole process tree — its
     tool subprocesses too — and the prompt comes back. The next request
     starts a fresh harness session (the panel says `new session`); the
     harness's in-memory conversation is gone. There is no automatic kill
     timer: only `$NVSH_TURN_TIMEOUT` bounds a turn on its own. A `Ctrl+C`
     typed while another nvsh prompt (the choice prompt itself, a
     proposal, or the busy prompt) is already open goes straight to this
     stop path.
- **`[Esc]` keep going, or no key for 30 s (±1 s):** dismisses the prompt.
  Nothing is sent to the harness, rendering resumes, and the turn's exit
  status and `result.interrupted` are unaffected — a dismissed accidental
  press is never reported as exit 130.

Both the daemon path (`client_transport.cancel`, then `kill`) and the
one-shot path (the in-process adapter's `cancel()`, then `force_stop()`)
work this way. After a stop in a one-shot run, whatever tool processes the
harness left behind are killed once the turn is over (deviation d8).

At the correction line (the `nvsh>` prompt) an empty line or `Ctrl+C` means never
mind: nothing is sent, nothing is cancelled and the turn keeps going. The
typed line is redacted (`nvsh/redact.py`) before it goes anywhere, and the
audit log keeps only its length (`correction_chars`), never the text.

If `pi` or `codex` cannot take the correction after all — codex fell back
to `exec` mode, or the turn has no id yet — nvsh says `nvsh: could not
steer the running turn` and asks once whether to stop and correct instead.
`y` does that; any other key discards the text, which is written to the
audit log as `steer` / `discarded`, and the turn keeps going. The text is
never dropped silently.

After a stop & correct the first turn is cancelled politely and nvsh waits
for it to end before sending the follow-up; a harness that ignores the
cancel can still be killed with a further press, and then no follow-up is
sent. If the turn had already finished while the prompt was open, the
correction simply becomes the next request and `[s]` stops nothing.

Exit status: 0 after keep going or after a delivered steer whose turn ends
normally; the follow-up turn's own status after a stop & correct; 130 is
reserved for `[s]` stop on a running turn and for a kill (including a kill
during stop & correct). `[s]` on a turn that had already finished exits 0.

Every outcome writes one `event: "stop"` audit line: `keep_going` (with
`reason` `key` or `timeout`), `cancel`, `force_kill`, or `steer` with
outcome `delivered`, `queued` or `discarded`. Lines written from this
prompt carry `origin: "stop_prompt"`; a stop & correct is a `steer` /
`queued` line followed by a `cancel` line with that origin.

Notes:

- An `Esc` counts only when it arrives alone: arrow and function keys start
  with the same byte, so nvsh waits 50 ms and drains a whole escape sequence
  instead of reading it as `Esc` (`nvsh/keys.py`).
- **Typeahead is dropped.** While the panel streams, stdin is held in cbreak
  mode to see `Esc`; anything else typed is discarded, never executed, and
  two presses arriving back-to-back (a key-repeated `Esc`) leave the choice
  prompt open rather than silently answering it.
- The terminal is restored on every exit path, including `SIGHUP` (ssh drop)
  and `SIGTERM`. `Esc` watching is off when stdin is not a tty, under
  `TERM=dumb`, or with `NVSH_DISABLE` set.
- A press while an approved command is running is acted on when that command
  returns; the command itself still receives `SIGINT` from the terminal
  (deviation d2).
- A `Ctrl+C` that lands while the panel is already shutting down (the turn
  is over, typical after a double press on a harness whose cancel ends the
  turn at once) is recorded as an interrupt and nothing more: no traceback,
  the terminal is restored, and nothing is sent to the harness.
- On a hung-up terminal, end of input at a proposal means **ignore**, never
  approve (deviation d1); end of input at the choice prompt, or at the
  correction line after `[t]`, means the same thing — keep going, nothing
  sent, nothing cancelled.
- **A session without a terminal stops at once.** Where the choice prompt
  cannot be shown or answered — stdin or stdout is not a tty, `TERM=dumb`,
  or `--json` — the first `Ctrl+C` stops immediately exactly as it did
  before this change: no prompt text reaches stdout or stderr, so scripts
  and piped sessions keep their current behavior.

#### Amendment to reliable-agent-stop (2026-09-17)

This section supersedes claims c1, c4, c5, c17, c20, c21, c22 and c34 of
the reliable-agent-stop spec
(`docs/specs/2026-09-16-reliable-agent-stop.md`), which described the
first press as an immediate polite cancel, with no choice offered and the
`stopping… press again to kill` line appearing within 1 s. The 1 s promise
now applies to the choice prompt appearing, and the 3 s kill promise is
measured from the press that follows `[s]` stop.
`.devague/frames/reliable-agent-stop.json` is not edited by this change;
the amendment is recorded here instead.

The operator asked for this change after reproducing, on nvsh 0.13.1 on
thor on 2026-09-17, that the first Esc or Ctrl+C cancelled a running turn
at once with no choice, and that the turn in question had already taken
about 70 s of model time — so an accidental press threw away real work.
The stated reason for the change: "clearer for the user and avoid
accidental clicks."

### The busy prompt

A new request from a shell whose own turn is still running — or from any
shell when the shell that started the turn no longer exists (an ssh drop,
then a reconnect) — does not queue silently. The panel asks:

```text
nvsh: busy -- 3422579 still running (42s)
[t] steer [r] replace [Esc] exit
```

- `[t] steer` sends the new request into the running turn. It is offered only
  for harnesses with a mid-turn channel (pi, codex). If nothing happens for
  10 s after steering, the prompt comes back with replace/exit only.
- `[r] replace` kills the running turn and runs the new request.
- `[Esc] exit` (or any other key) leaves the running turn alone and returns
  to the prompt with the declined status (3).

A live *other* shell's turn is never touched: requests from other shells
wait in the queue with the usual `waiting for the agent` notice. If the
prompt is not answered within 60 s the request queues as before. `nvsh
overview` shows the active turn (shell, target, elapsed) and the queue, and
`nvsh doctor --apply` clears a hung turn (see `/doctor` below). Every stop,
steer, replace, exit and decline is written to the audit log as an
`event: "stop"` line with `kind`, `shell`, `target`, `elapsed` and
`outcome`.

### Approving from the proposal keys (deviations d15, d24)

The legend under a proposal is one line, kept within 80 columns so it never
wraps on a bare ssh into a Jetson:

```text
[Enter] run [s/S] +session [u/U] +user [e] why [d] details [t] tell [Esc] ignore
```

`Enter` runs the command once and changes nothing. The scope keys run it
*and* stop nvsh asking about that class of command again — the operator
used to re-approve the same `docker logs …` on every failure:

| key | pattern stored | scope | where |
|-----|----------------|-------|-------|
| `s` | the exact command line | this login session | `$XDG_RUNTIME_DIR/nvsh/session-approvals.toml` (0600) |
| `S` | `<command> <first argument> *` | this login session | same file |
| `u` | `<first word> *` | this user, until removed | `$XDG_CONFIG_HOME/nvsh/approved.toml` (0600) |
| `U` | `<command> <first argument> *` | this user, until removed | same file |

**The specific form (`S`/`U`, deviation d24).** An operator on the Spark was
shown `ssh orin "ps -eo pid,rss,comm --sort=-rss | head -n 20"` and offered
`[u] allows 'ssh *' (any arguments)` — an approval for *every* ssh command
to *every* host, when all they wanted was that one box. The uppercase keys
keep the command's first argument: `ssh orin *`, `docker ps *`,
`git status *`. When the line has no second word there is nothing to be
specific about, and `S`/`U` behave exactly like `s`/`u`; the scope line says
so (`same as [u] (no second word)`) rather than quoting a pattern twice.

The scope line above the legend names both forms of both lifetimes, so no
key can be mistaken for a narrower one than it is:

```text
[s] this exact line  [S] 'ssh orin *'  (this session)
[u] 'ssh *'  [U] 'ssh orin *'  (persisted for you)
```

### Numbered stages, and picking which of them an approval covers (d26)

An operator on the Spark was shown `ls /srv/models | grep -i orin` and
told `[s] each stage exactly` — which never said what the stages *were*, and
offered no way to approve only the `grep` half. A line with more than one
stage now numbers them above the scope lines, and each scope family names
the pattern it would store *per stage*, in the same order:

```text
stages: 1 'ls /srv/models'  2 'grep -i orin'
[s] exact  [S] 'ls /srv/models *' | 'grep -i *'  (this session)
[u] 'ls *' | 'grep *'  [U] 'ls /srv/models *' | 'grep -i *'  (persisted for you)
```

Each line is clipped to 80 columns with a trailing `…`. A stage no pattern
may ever cover — a `sudo`/`rm` stage — reads `(not approvable)` instead of
being quoted as though a key could store it. An opaque line (a subshell, a
command substitution) is *one* stage by construction, so it keeps the d24
single-stage rendering and its whole-line refusal. A single-stage command is
unchanged: no `stages:` line, and `[s]` still reads `this exact line`.

After `s`, `S`, `u` or `U` on a multi-stage command the panel reads one
cooked line:

```text
stages [all,1,2]: 2
nvsh: running; 'grep -i *' approved for this session (stage 2 of 2)
```

`all`, an empty line, EOF and `Ctrl+C` all mean every stage — the pre-d26
behaviour — and so does a second unreadable answer after the one re-ask
(`nvsh: type 'all' or stage numbers 1-2`). A single number or a comma- or
space-separated list stores only those stages. The ack then names exactly
the patterns that were stored and which stages they came from, so it can
never over-report the approval; picking every stage keeps the pre-d26
wording. **The proposal still runs once whatever was picked**: the keypress
approved *this* execution, and the store write is only about future turns.

`nvsh approve add <cmd> --scope <choice> --stages 1,2` is the same choice
from the CLI, and it takes the same `all`/number-list spelling (both go
through one parser, `nvsh.approvals.parse_stages`). `--stages` without
`--scope` is a user error, and a stage number past the end of the line is
refused rather than silently dropped.

**One pattern per stage (deviation d24).** A command line is split into
stages on `|`, `&&`, `||`, `&`, `;` and newlines, honouring quotes —
`ssh orin "ps | head"` is ONE stage, because the pipe is inside the quoted
argument ssh carries to the far end. Approving at any scope stores one
pattern per stage (`ps *` *and* `head *`), and a later command is
auto-approved only when **every** one of its stages matches a stored
pattern. That is what stops a broad `ls *` from authorizing
`ls | sudo tee /etc/x`: the second stage is a separate stage, and a
privileged or destructive one can never be pre-approved at all — it makes
the whole line unapprovable for `s`/`S`/`u`/`U`, exactly as a bare
`sudo …` always has been. When there is more than one stage the scope line
lists the per-stage patterns (`[u] 'ps *' 'head *'`), clipped at 80
columns, numbered and shown stage by stage (see below), and `[s]` reads
`exact`.

A line carrying a subshell or a command substitution — `(cd /tmp && ls)`,
`echo $(id)`, a backtick — is **opaque**: what it really runs is not in its
own text, so nvsh treats it as one stage that no pattern may ever match and
refuses every scope key for it. `Enter` still runs it once, in front of the
operator who just read it.

`nvsh approve check` reports the stage that is holding a line back:

```console
$ nvsh approve check "nvidia-smi -q | grep -i fan" --json
{"decision": "ask", "pattern": null, "stage": "grep -i fan"}
```

**Where a session approval lives, and why.** "Session" used to mean one
Python process's memory — and since every writer of a session approval is a
throwaway process (`nvsh approve add <cmd> --session`, which the pi approval
extension shells out to; one `nvsh hook` run), the pattern died before
anything could match it and the operator was asked again on the very next
command. It is now written to `$XDG_RUNTIME_DIR/nvsh/session-approvals.toml`,
which is the lifetime the word promises: the runtime dir is per-user tmpfs
that the system creates at login and removes at logout, so a session
approval outlives the process that made it, is shared by the daemon and
every hook invocation of that login, and is gone at the next login — never
at reboot-surviving rest. With `XDG_RUNTIME_DIR` unset nvsh falls back to
`/run/user/<uid>` when that exists and otherwise to a per-uid directory
under the system temp dir; it never falls back to a persistent location, so
a missing environment variable can't quietly promote a session approval into
a permanent one. `nvsh approve list` and `/approve list` print both stores
under their scope headings (`user:` / `session:`), and
`nvsh approve remove <pattern>` drops a pattern from both.

**Two paths, one answer.** When the proposal carries a `request_id` it came
from a backend dialog (pi's approval extension), and that backend is blocked
waiting: nvsh forwards the operator's answer verbatim as
`{"value": <choice>}`, where `<choice>` is one of the six the extension
offers `ctx.ui.select` — in this order, which is part of the contract:
`once`, `session`, `session-specific`, `user`, `user-specific`, `deny`.

**How the stage pick travels (d26).** pi carries exactly one value string
back to the extension (pi 0.85.1 reduces an `extension_ui_response` to its
`value` — see `docs/pi-rpc.md`), and there is no second field to put a stage
list in. So a *partial* pick is encoded into the value itself as
`<scope>:<stages>` — `session-specific:1,2` — which the extension splits on
the first colon and forwards as
`nvsh approve add <cmd> --scope session-specific --stages 1,2`. A pick that
covers every stage stays the bare token, so nothing that reads these answers
had to change. The extension interprets the stage list no more than it
interprets a pattern: the CLI parses it, and the audit records the bare
scope. The
extension then does the store write and the run — nvsh must not run the
command a second time. It derives no pattern of its own: it shells out to
`nvsh approve add <command> --scope <choice>`, the single writer, which
splits the line into stages and applies the scope's form to each. Without a
`request_id` (the openai-compat adapter and the other non-dialog backends)
nvsh owns execution, so it writes the same patterns through
`nvsh.approvals` itself and then runs the command. Either way the decision
is audited as
`decision: once | session | session-specific | user | user-specific | deny`.

**The scope keys are refused for privileged and destructive commands.** A
command with a `sudo`/`doas`/`pkexec` stage, a stage whose pattern
`Approvals.add` refuses (`sudo …`, `rm …`, a bare `*`), or an opaque
subshell, cannot be pre-approved at any scope: the
panel prints one line —
`nvsh: cannot approve for this user: patterns starting with 'rm' are never
approved` — and asks again with `[Enter] run` still on the table. Approving
a *class* of privileged command is exactly the blanket authorization
"propose, don't run" exists to prevent; running one once, in front of the
operator who just read it, is not.

See `tests/test_panel.py`, `tests/test_client.py`, `tests/test_approvals.py`.

## The slash-command registry (`nvsh/slash.py`)

`nvsh complete` and `nvsh slash` (the two calls above) are thin CLI verbs
(`nvsh/cli/_commands/slash.py`) over one registry in `nvsh/slash.py`
(plan task t14): a `SlashCommand` per verb — name, aliases, description, a
completion provider for its own arguments, the handler, a safety
classification, and an optional `platforms` restriction. `/power` and
`/clocks` are registered with `platforms={"jetson"}`; both `nvsh complete`
and `/help` hide them everywhere else, and dispatching either one on a
non-Jetson platform reports "unknown slash command" exactly like a typo
would. The registry is the *only* place any of this is decided — bash never
re-implements it.

`--platform` overrides the detected kind on both verbs (mostly for tests);
by default it comes from `nvsh.platform.detect().kind`.

## `/doctor`

`/doctor` is one of the palette entries `nvsh complete --json` returns (see
above), routed through the same Enter-macro dispatch as every other slash
command — but its handler needs state only bash itself can see: the live
`PROMPT_COMMAND` array and the active readline bindings. `nvsh.doctor_checks`
(task t17) never shells out to read that state itself; bash hands it over as
three environment variables, exported by `__nvsh_enter` right before *every*
slash dispatch (not only `/doctor` — the cost is one `declare -p`/`bind -p`/
`bind -V` per slash line, which already forks `nvsh`, so there is nothing to
gain by special-casing the command name, and `readline.bash` must never carry
a command name in its code — see "No static command list" above):

```bash
export NVSH_PROMPT_COMMAND="$(declare -p PROMPT_COMMAND)"
export NVSH_BIND_P="$(bind -p)"
export NVSH_KEYMAP="$(__nvsh_keymap)"   # emacs / vi-insert / vi-command
```

**`bind -p` alone cannot show nvsh's bindings** (deviation d4a, measured on
spark/thor/orin and reproduced on a `bash --norc --noprofile -i` pty). `bind
-p` lists only key sequences bound to readline *functions*, so the `bind -x`
handlers (`\C-x\C-n` → `__nvsh_enter`, `\C-g` → `__nvsh_ctrl_g`) show up only
under `bind -X`, the Enter macro (`"\C-m": "\C-x\C-n\C-j"`) only under `bind
-s`, and binding that macro removes `\C-m` from `bind -p` entirely. The
payload `bindings_present` can actually verify therefore merges all three
dumps, separated by a marker line
(`nvsh.doctor_checks.BIND_SECTION_MARKER`):

```bash
export NVSH_BIND_P="$(bind -p; echo '# nvsh: bind -s/-X follow'; bind -s; bind -X)"
```

The `/doctor` slash handler (`nvsh.slash._handle_doctor`) reads these three
variables straight from its environment and passes them to
`nvsh.cli._commands.doctor.cmd_doctor`, exactly as the `nvsh doctor
--prompt-command/--bind-p/--keymap` CLI flags would:

- `NVSH_PROMPT_COMMAND` is `declare -p PROMPT_COMMAND`'s own output (array or
  string form); `hook_first_in_prompt_command` parses it to confirm
  `__nvsh_hook` is element `[0]`, exactly as `__nvsh_hook_install` (in
  `nvsh/shell/hook.bash`) placed it.
- `NVSH_BIND_P` is bind-dump output for whichever keymap is currently active
  (not `bind -m <keymap> -…` — `readline.bash` registers every binding across
  all three keymaps, so the active keymap's own dump is enough);
  `bindings_present` looks for the `\C-x\C-n` dispatch binding, the `\C-m`
  Enter macro and the `\C-g` binding described above, accepting both the bare
  form (`"\C-g": __nvsh_ctrl_g`) and the quoted form `bind -X` actually
  prints (`"\C-g": "__nvsh_ctrl_g"`). When the payload carries none of the
  three *and* no `bind -s`/`bind -X` marker — i.e. it is a `bind -p`-only
  export, which cannot list them in the first place — the check reports
  `passed=false, severity=info` ("cannot verify …") rather than a false
  `error`; only a payload that demonstrably includes the other dumps can
  fail this check.
- `NVSH_KEYMAP` is parsed from `bind -V`'s `keymap is set to` line by the
  small `__nvsh_keymap` helper, one of `emacs`, `vi-insert` or
  `vi-command`, and only labels the `bindings_present` message — it does not
  change which bindings are checked, since the dump already reflects the
  active keymap.

Run `nvsh doctor` (or `nvsh doctor --json`) directly, or `/doctor` from a
plain terminal, a script, or over `nvsh explain doctor` — anywhere none of
these three variables are set — and the in-shell checks (`hook_sourced`,
`hook_first_in_prompt_command`, `bindings_present`) report
`passed=false, severity=info` with the remediation "run /doctor from a
hooked shell" rather than failing `healthy`, since that state genuinely
cannot exist outside a hooked bash.

## `/undo`

`/undo` (`nvsh.slash._handle_undo`) drops the last agent turn and its
pending proposal from the conversation. It never runs anything on the
machine and never reverts a change a proposal already made — the converged
spec's `/undo` decision scopes it to nvsh's own view of the conversation;
reverting machine changes is tracked as a separate, later GitHub issue on
`agentculture/nvsh`. With a daemon running it sends the `undo` control
message (`nvsh.daemon.Conversation.undo()` drops the daemon's last
transcript entry and clears `pending_proposal` — the backend's own on-disk
session, e.g. pi's, is left untouched, since there is no RPC to selectively
erase one turn there); with no daemon running it clears any
`pending_proposal` recorded in `$XDG_STATE_HOME/nvsh/last-failure.json`
instead.

## Installation: `nvsh setup` and the rc block

`nvsh setup` (`nvsh/cli/_commands/setup.py`, rc editing logic in
`nvsh/rcfile.py`) is what actually wires `hook.bash` and `readline.bash`
into an operator's shell. It does two things:

1. **Renders** both files from package resources (`importlib.resources`, so
   this works from a wheel install with no `nvsh/shell` source directory on
   disk) into `$XDG_DATA_HOME/nvsh/shell/` (default the user's local share
   directory, `nvsh/shell/` under `.local/share`), each prefixed with a
   two-line stamp naming the nvsh version that rendered it
   (`nvsh/shell/render.py`).
2. **Inserts** one small marked block into the rc file (default the user's
   bash rc file; override with `--rc`), immediately after the distro's
   interactive guard (`# If not running interactively, don't do anything`
   plus its `case $- in ... esac`, or a single-line test on `$-`) — never at
   the end of the file, so the rest of the rc is never sourced twice. A
   timestamped backup (`<rc>.nvsh-backup-<YYYYmmdd-HHMMSS>`) is written
   before the first change.

The block itself stays under ten lines:

```bash
# >>> nvsh setup >>> sha256:<content hash>
export NVSH_HOOK_VERSION="<version>"
export NVSH_BIN="<absolute path to the nvsh entrypoint>"
[[ -n $NVSH_DISABLE ]] || { source "<data>/shell/hook.bash"; source "<data>/shell/readline.bash"; }
nvsh() { case $1 in on|off) eval "$(command nvsh "$@" --shell)";; *) command nvsh "$@";; esac; }
# <<< nvsh setup <<<
```

Every path in it is `$HOME`-relative or absolute — never a literal `~/` —
because it runs before any alias or function expansion that might redefine
`~`. `NVSH_BIN` is resolved once at `setup` time
(`render.resolve_nvsh_bin()`: `shutil.which("nvsh")` first, `sys.argv[0]`
resolved as a fallback) so a `uv tool install` works without a `PATH` edit.
The rc-defined `nvsh()` function is what makes typing `nvsh off` / `nvsh on`
at the prompt actually rebind readline in *that* shell: a subprocess cannot
rebind its parent's bindings, so the function re-invokes `command nvsh "$@"
--shell` (which prints raw bash instead of JSON/text) and `eval`s the
result — exactly what `eval "$(nvsh off)"` / `eval "$(nvsh on)"` do directly.

`nvsh setup` is idempotent: a second run against an rc that already has the
identical block writes nothing at all (same bytes in, same bytes out). The
opening marker carries a short content hash of the block's body, so
`nvsh uninstall` can tell whether the block was hand-edited since `setup`
wrote it — if not, it strips exactly the marked span (the exact inverse of
the insertion); if so, it restores the newest backup instead.

`nvsh hook` is the thin verb `__nvsh_hook` in `hook.bash` calls on a
qualifying failure (never invoked by the operator directly): it rebuilds a
`nvsh.triggers.TriggerEvent` from its flags, calls `decide()`, and on
`"skip"` exits 0 silently. On `"ask"` it hands off to the failure client
(`nvsh.client.handle_failure`, task t13) behind a lazy, `ImportError`-guarded
import; until that lands it prints a one-line placeholder instead. It also
compares the rc block's exported `NVSH_HOOK_VERSION` against the running
package's own version and prints a one-time-per-session refresh notice when
they differ (state file under `$XDG_RUNTIME_DIR/nvsh/<shell-pid>.notice`).

See `tests/test_rcfile.py`, `tests/test_shell_render.py`,
`tests/test_cli_setup.py` and `tests/test_setup_timing.py`.

## What setup can install

`nvsh setup` also detects the helper tools it depends on that are missing
from `PATH` and, only with explicit confirmation, installs them
(`nvsh/installers.py`). This exists because the fleet is uneven: the Jetson
AGX Orin ships with neither `uv` nor `tmux`, and the Jetson AGX Thor has
`npm` but no `pi`, so `setup` was the natural place to close that gap
instead of leaving it as a manual step per machine.

| tool | why nvsh wants it | command tried |
| --- | --- | --- |
| `node`/`npm` | prerequisite for installing `pi` via npm | `sudo apt-get install -y nodejs npm` |
| `pi` | the default agent backend nvsh's failure client talks to | `npm install -g @earendil-works/pi-coding-agent` |
| `uv` | Python package/dependency manager | `sudo snap install astral-uv --classic` when `snap` is present |
| `tmux` | terminal multiplexer nvsh's inline panel and daemon target | `sudo apt-get install -y tmux` |

Rules that hold regardless of platform:

- **Confirmation is never skipped.** Interactively, `setup` asks
  `install <tool>? [y/N]` once per tool. `--yes` answers yes to all of them
  in one run. `--no-install` lists every offer and installs nothing.
  `--json` is non-interactive by construction (there is no terminal to
  prompt on), so it only lists offers unless `--yes` is also given.
- **`sudo` is never pre-typed or stored.** A `sudo` command runs with
  `sudo` as `argv[0]` in list form — never through a shell string — and its
  output is left attached to the real terminal (no output capture) so the
  operator sees and answers the password prompt themselves. nvsh never
  reads, stores or types a password on the operator's behalf. The same
  no-`sudo`-typed rule already governs agent-proposed fixes; see
  [architecture.md](architecture.md)'s "Propose, don't run".
- **`uv`'s curl-pipe-sh installer is printed only, never executed.** When
  no package manager can install `uv` but `curl` is present, `setup` prints
  `curl -LsSf https://astral.sh/uv/install.sh | sh` as the command the
  operator can run by hand — it is never run automatically, not even with
  `--yes`, because a curl-pipe-sh has no place running unattended.
- **Every attempt is audited.** Whether a tool install actually ran, was
  declined, or was never executable at all, it is written to the audit log
  (`nvsh.agent.audit.AuditLog`, event `"install"`) alongside the proposed
  command and its outcome — the same trail an agent-proposed fix leaves.
- **A freshly installed `pi` is picked up immediately.** After any installs
  run, `setup` re-runs `nvsh.agent.registry.choose()`, so `pi` becomes the
  reported (and usable) agent backend in the same `setup` invocation that
  installed it.

See `tests/test_installers.py` for the exact planned command per fleet
machine and `tests/test_cli_setup.py` for the `setup` integration.
