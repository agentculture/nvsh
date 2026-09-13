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

## What each key does

### Enter

`C-m` is bound to the macro `"\C-x\C-n\C-j"`: a `bind -x` callback on the
otherwise unused `\C-x\C-n`, then `\C-j` (`accept-line`). A `bind -x`
function cannot itself call `accept-line`, so the macro chain is required
(spec scope entry `s18`).

`__nvsh_enter` looks at `READLINE_LINE`:

- a line not starting with `/` returns immediately — an ordinary command
  costs one bash function call and **no fork**;
- a first word that is not `/name` (so `/tmp/x`, or a bare path with a
  second slash) passes through unchanged;
- a first word that `nvsh complete --json` lists is rewritten to
  `" nvsh slash '<line>'"` — note the single leading space — and the
  original line is pushed with `history -s`, so `history` shows what the
  operator typed and not the dispatch.

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

### Ctrl+G

`bind -x '"\C-g": __nvsh_ctrl_g'` saves `READLINE_LINE` / `READLINE_POINT`,
prints a one-line panel marker, dispatches `/ask` with the draft, and
restores the line and point, so the half-typed command is still there
afterwards.

## Keymaps

Every binding is registered three times, with `bind -m emacs`,
`bind -m vi-insert` and `bind -m vi-command`. A plain `bind -x` does not
fire after `set -o vi`; a vi-insert keymap binding does.

## Kill switches

- `NVSH_DISABLE=1` in the environment makes sourcing the file a complete
  no-op: no bindings, no `complete` registration, no `HISTCONTROL` change.
- `__nvsh_readline_unbind` (behind `nvsh off`) restores `C-m` to
  `accept-line`, removes the dispatch sequence and `C-g` in all three
  keymaps, and removes the `-I` and per-command completions.
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

The `/doctor` slash handler (`nvsh.slash._handle_doctor`) reads these three
variables straight from its environment and passes them to
`nvsh.cli._commands.doctor.cmd_doctor`, exactly as the `nvsh doctor
--prompt-command/--bind-p/--keymap` CLI flags would:

- `NVSH_PROMPT_COMMAND` is `declare -p PROMPT_COMMAND`'s own output (array or
  string form); `hook_first_in_prompt_command` parses it to confirm
  `__nvsh_hook` is element `[0]`, exactly as `__nvsh_hook_install` (in
  `nvsh/shell/hook.bash`) placed it.
- `NVSH_BIND_P` is plain `bind -p` output for whichever keymap is currently
  active (not `bind -m <keymap> -p` — `readline.bash` registers every
  binding across all three keymaps, so the active keymap's own `bind -p` is
  enough); `bindings_present` looks for the `\C-x\C-n` dispatch binding, the
  `\C-m` Enter macro and the `\C-g` binding described above.
- `NVSH_KEYMAP` is parsed from `bind -V`'s `keymap is set to` line by the
  small `__nvsh_keymap` helper, one of `emacs`, `vi-insert` or
  `vi-command`, and only labels the `bindings_present` message — it does not
  change which bindings are checked, since `bind -p` already reflects the
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
   disk) into `$XDG_DATA_HOME/nvsh/shell/` (default
   `~/.local/share/nvsh/shell/`), each prefixed with a two-line stamp naming
   the nvsh version that rendered it (`nvsh/shell/render.py`).
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
