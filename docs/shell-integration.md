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
