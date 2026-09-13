# Three-machine verification — 2026-09-13

Executed on 2026-09-13 against the nvsh **0.9.2** hook as it is installed on
the operator's three machines (`nvsh setup` block present in each rc,
backups on disk). Every session ran on a real pty; every line below quotes
output actually observed, trimmed and with escape sequences stripped. Where
a line failed, it is recorded as a failure and raised as a deviation — it
was not fixed here (task t19 is a verification pass, not an implementation
pass).

**Safety note.** `nvsh setup` / `nvsh uninstall` were never run against a
real rc file. Each machine's setup/uninstall row was exercised with `HOME`
(and the `XDG_*` dirs) pointed at a throwaway temp directory holding a copy
of that machine's rc with the existing nvsh block stripped.

## Machines

| Machine | Platform detected | Shell context | Agent backend | Terminal |
|---------|-------------------|---------------|---------------|----------|
| spark | `dgx-spark` (GB10, CUDA 13.0.2, driver 580.126.09, unified memory) | local, Ghostty + bash-preexec + fig integrations | `pi` (nemotron/associate) | `TERM=xterm-ghostty` |
| thor | `jetson` (AGX Thor Dev Kit, L4T R38, CUDA 13.0.0, TensorRT 10.13.3.9) | ssh, with and without tmux | `openai-compat` against the LAN gateway | `TERM=xterm-256color` |
| orin | `jetson` (AGX Orin, L4T R39, driver 595.78) | ssh, no tmux available | `openai-compat` against the LAN gateway | `TERM=xterm-256color` |

The endpoint and bearer token for the `openai-compat` backend are
deliberately not written down here. On thor and orin the config is
`$XDG_CONFIG_HOME/nvsh/config.toml` with `base_url` plus
`api_key_env = "NVSH_API_KEY"`, and the key itself lives in a `0600` file
that the session exports with
`export NVSH_API_KEY=$(cat "$XDG_CONFIG_HOME/nvsh/api_key")`.

## DGX Spark — Ghostty, local

| # | Check | Result | Observed |
|---|-------|--------|----------|
| 1 | `nvsh setup` | PASS | `block inserted: True`, `backup: $TMPHOME/.bashrc.nvsh-backup-20260913-170606`, `agent: pi (pi is configured and on PATH)`. Re-run: `block inserted: False`, block count still `1`. |
| 2 | No visible change on success | PASS | `echo ok-success` → `ok-success` and nothing else; measured added prompt cost **0.055 ms** (below). |
| 3 | `ls /nope` panel with the real error slice | **FAIL** | The panel renders (`nvsh: ls /nope failed (exit 2)`) but the slice sent to the agent is the **previous** command's output. After `echo hello-success; ls /nope`, `nvsh context --show` reported `Output:` / `hello-success`, and the model answered "the output shows `hello-success` which is unusual". Deviation **v1**. |
| 4 | `/doctor` | PARTIAL | Dispatches correctly (rewritten to `nvsh slash '/doctor'`). Reports `nvsh doctor: unhealthy` with `hook_first_in_prompt_command` FAIL (superseded by deviation d2) and `bindings_present` FAIL, which is a false negative — the very Enter macro that dispatched `/doctor` is one of the bindings it reports missing. Deviations **v2**, **v3**, **v4**. |
| 5 | `Ctrl+G` | **FAIL** | Prints `⚡ nvsh (Ctrl+G): asking the agent`, restores the typed line correctly — and shows nothing else, ever. `readline.bash` runs `nvsh slash "/ask" >/dev/null 2>&1`, so the agent's answer is discarded. Running the same `/ask` by hand does stream a panel. Deviation **v5**. |
| 6 | slash + Tab palette | PASS | `/` + Tab Tab listed `agent approve ask context doctor explain fix help retry undo` merged with root-path completions. |
| 7 | vi mode (`set -o vi`) | PASS | After `set -o vi`: `/ret` + Tab → `/retry` (with the trailing space); `/help` + Enter dispatched to `nvsh slash '/help'` and printed the palette; `Ctrl+G` printed its banner and restored the line. |
| 8 | `nvsh uninstall` restores the rc | PASS | `block removed: True`, `removed files: 2`; `diff` against the pristine copy: identical. |

## Jetson AGX Thor — ssh, no tmux

| # | Check | Result | Observed |
|---|-------|--------|----------|
| 1 | `nvsh setup` | PASS (with a caveat) | `block inserted: True`, backup written, `agent: openai-compat (pi not on PATH; using openai-compat …)`; re-run `block inserted: False`, block count `1`. Caveat: `--yes` also **ran** `npm install -g @earendil-works/pi-coding-agent` (`ran: True (returncode: 243)`, i.e. it failed, nothing installed). Deviation **v6**. |
| 2 | No visible change on success | PASS | `echo ok-success` → `ok-success`, nothing else. |
| 3 | `ls /nope` panel with the real error slice | PASS | `ls: cannot access '/nope': No such file or directory` / `nvsh: ls /nope failed (exit 2)`, then a diagnosis quoting that exact line and the Thor platform block (`l4t_release: R38`, CUDA 13.0.0). |
| 4 | `/doctor` | PARTIAL | `nvsh doctor: unhealthy`; `hook_sourced`, `hook_first_in_prompt_command` (`__nvsh_hook is first in PROMPT_COMMAND`), `capture_active`, `daemon_status` and `agent_reachable` all `[ok]`; only `bindings_present` FAILs — the same false negative as spark, in a shell whose Enter macro had just dispatched `/doctor`. Exiting 1 then triggered a full agent diagnosis of the check. Deviations **v3**, **v4**. |
| 5 | `Ctrl+G` | **FAIL** | Same as spark: banner only, answer discarded (deviation **v5**). |
| 6 | slash + Tab palette | PASS | `/` + Tab Tab listed `agent approve ask clocks context doctor explain fix help power retry undo` merged with root-path completions. |
| 7 | vi mode | PASS | `/ret` + Tab → `/retry` (with the trailing space) in vi-insert. |
| 8 | `nvsh uninstall` restores the rc | PASS | `block removed: True`; `diff` against the pristine copy: identical. |

## Jetson AGX Thor — ssh, inside tmux

| # | Check | Result | Observed |
|---|-------|--------|----------|
| 1 | Hook active inside tmux | PASS | `tmux new-session -A -s nvshv`, then `declare -p PROMPT_COMMAND` → `declare -a PROMPT_COMMAND=([0]="__nvsh_hook")`. |
| 2 | No visible change on success | PASS | `echo ok-success` → `ok-success`. |
| 3 | `ls /nope` panel with the real error slice | PASS | `nvsh: ls /nope failed (exit 2)` inside the tmux pane, followed by a diagnosis that quotes the missing path and the Thor platform block. The status line kept redrawing behind the panel without disturbing it. |
| 4 | Capture path inside tmux | PASS (by construction) | The hook takes the `tmux pipe-pane` branch rather than re-`exec`ing `script(1)`; the panel above proves the slice reached the agent. |

## Jetson AGX Orin — ssh, no tmux

| # | Check | Result | Observed |
|---|-------|--------|----------|
| 1 | `nvsh setup` | PASS (with a caveat) | `block inserted: True`, re-run `block inserted: False`, block count `1`. Caveat: `--yes` attempted `sudo apt-get install …` three times, each rejected with `sudo: a password is required` — nothing was installed. Deviation **v6**. |
| 2 | No visible change on success | PASS | `echo ok-success` → `ok-success`. |
| 3 | `trtexec --version` panel with the real error slice | PASS | `bash: trtexec: command not found` / `nvsh: trtexec --version failed (exit 127)`, then a diagnosis naming `tensorrt_version: absent`, L4T R39 and driver 595.78 from the platform block. A later `nvsh context --show` confirmed the slice was the failing command's own output (`bash: xit: command not found` for `xit`). |
| 4 | `/doctor` | PARTIAL | `nvsh doctor: unhealthy`; `hook_sourced`, `hook_first_in_prompt_command`, `capture_active`, `daemon_status`, `agent_reachable` all `[ok]`; only `bindings_present` FAILs (same false negative). `/doctor` exiting 1 then triggered a full agent diagnosis of itself. Deviations **v3**, **v4**. |
| 5 | `Ctrl+G` | **FAIL** | Banner only (deviation **v5**). |
| 6 | slash + Tab palette | PASS | `/` + Tab Tab listed `agent approve ask clocks context doctor explain fix help power retry undo` — note the Jetson-only `clocks` and `power` entries. |
| 7 | vi mode | PASS | `set -o vi`, then `/ret` + Tab → `/retry` (with the trailing space). |
| 8 | `nvsh uninstall` restores the rc | PASS | `block removed: True`; `diff` against the pristine copy: identical. |
| 9 | openai-compat fallback produces the panel | PASS | `... one-shot openai-compat: openai-compat is configured and on PATH`, followed by the streamed diagnosis in row 3. |
| 10 | With tmux | **BLOCKED** | tmux is not installed on orin (`which tmux` → nothing, `which npm` → nothing). `nvsh setup` does offer to install it — `sudo apt-get install -y tmux` — which needs a password and therefore an operator present. Not installed during this pass. |

## Timing

Measured by `tests/test_timing.py` on spark, `uv run pytest tests/test_timing.py -s`:

| Number | Target | Measured on spark |
|--------|--------|-------------------|
| Added cost per prompt on the success path | < 5 ms | **0.055 ms** (baseline median 0.014 ms, hooked median 0.069 ms, n=201 successful commands on a pty) |
| Time to first agent text with the fake agent | < 2 s | **37.3 ms** median (min 35.3 ms, max 48.1 ms, n=5), measured end to end from spawning a cold interpreter |

Test output, verbatim:

```text
[t19] success-path overhead: baseline median 0.014 ms, hooked median 0.069 ms, delta 0.055 ms (target < 5.0 ms, n=201)
[t19] time to first agent text (fake agent, cold interpreter): median 37.3 ms, min 35.3 ms, max 48.1 ms (target < 2.0 s, n=5)
```

Both numbers are medians, both tests print their measurement unconditionally,
and both skip cleanly when there is no usable bash/pty (so a CI runner
without one reports a skip rather than a fabricated pass).

The real-agent latency is a different number entirely and is not part of
this budget: with `pi`/`associate` on spark, first panel text arrived tens of
seconds after the failure, and a full turn with tool calls took one to three
minutes — visible in the recordings under `docs/demos/`.

## `PROMPT_COMMAND` on the Spark

`declare -p PROMPT_COMMAND` in a Ghostty session, verbatim:

```text
declare -a PROMPT_COMMAND=([0]=$'__bp_precmd_invoke_cmd\n__nvsh_hook\n__fig_post_prompt\n__bp_interactive_mode' [1]=$'__bp_trap_string="$(trap -p DEBUG)"\ntrap - DEBUG\n__bp_install; __ghostty_hook' [2]="__bp_interactive_mode")
```

- **"both hooks present, once each": PASS.** `__nvsh_hook` appears exactly
  once and `__ghostty_hook` exactly once.
- **"nvsh first": FAIL — superseded by deviation d2.** bash-preexec has
  collapsed its own entry, nvsh's and fig's into a single newline-joined
  element `[0]`, and `__bp_precmd_invoke_cmd` runs ahead of `__nvsh_hook`
  inside it. nvsh is second, not first. This is exactly the ordering
  deviation d2 already describes.

For contrast, thor and orin (no bash-preexec, no terminal integration):

```text
declare -a PROMPT_COMMAND=([0]="__nvsh_hook")
```

## OSC 133 and jump-to-prompt

Taken from the raw pty bytes, matching `\e]133;<letter>`:

- **spark, Ghostty, four commands one of which rendered a panel:**
  `A B B C D  A B B C D  A B B C D  A B B C` — Ghostty's own
  `__ghostty_hook`/precmd keeps emitting the full `A` (prompt start),
  `B` (command start), `C` (output start) and `D;<exit>` (command end)
  sequence, and the group containing the failure panel still closes with its
  `D`. Ghostty's jump-to-prompt therefore still works across a panel: the
  panel's text is inside the failing command's `C`..`D` region, not between
  prompts. (The doubled `B` is Ghostty's own doing, present with or without
  nvsh.)
- **thor and orin, no terminal integration:** `D C D C D C …` — nvsh emits
  its own `133;C` from `PS0` and its own `133;D;<exit>` from the hook, and
  emits **no** `A`/`B`. A terminal relying on `A` to find prompts gets
  nothing to jump to on those machines. Recorded as an observation, not a
  failure: nothing in the spec promises `A`/`B`.

The ordering of these markers is also the root cause of the spark error-slice
failure (row 3, deviation v1): because Ghostty's `D` for the failing command
is emitted *after* `__nvsh_hook` runs, the session log has no closed
`C`..`D` region for that command yet, and the slicer falls back to the last
complete region — the previous command's.

## Flicker and duplicated prompts

Counted from the same raw pty captures, not from a screenshot:

- spark, 4 commands: exactly 4 `133;A` prompt-start markers and 4
  `\r\e[K` in-place line clears. One prompt per command, redrawn in place —
  what looks like a repeated prompt in a plain-text dump of the stream is
  Ghostty's `redraw=last;cl=line` rewriting the same line, not a second
  prompt.
- Across a panel (`ls /nope` → panel → next prompt) the count does not
  change: no extra prompt is emitted before, during or after the panel, and
  no partially drawn prompt appears in the byte stream.
- Nothing in any capture shows the prompt being drawn and then erased
  (the signature of flicker); the only erases are the single `\r\e[K` per
  command that bash itself performs.

## Recordings

Under [`demos/`](demos/README.md):

- `demos/spark-cuda-oom.cast` — a CUDA-out-of-memory-style `RuntimeError` on
  the Spark in Ghostty, diagnosed by `pi`/`associate`, ending on a proposed
  command the operator can decline.
- `demos/orin-missing-package.cast` — a missing TensorRT tool on the Orin
  over ssh, diagnosed through the `openai-compat` fallback.

Both are asciicast v2 and were scrubbed at record time; see
[`demos/README.md`](demos/README.md) for how to play them without asciinema.

## Deviations raised by this pass

None of these were fixed here.

| id | What | Where | Why it matters |
|----|------|-------|----------------|
| v1 | The error slice sent to the agent is the previous command's output when a terminal integration owns OSC 133 and emits `D` after nvsh's hook | spark (Ghostty) only; thor and orin are correct | The agent diagnoses the wrong output — observed first-hand, the model commented on the previous command's text |
| v2 | `doctor`'s `hook_first_in_prompt_command` reports the order as `['__bp_interactive_mode']` — it does not see inside bash-preexec's newline-joined element | spark | The check's message is misleading even where the underlying finding (d2) is real |
| v3 | `doctor`'s `bindings_present` is a false negative: it reports the Enter/Ctrl+G/dispatch bindings missing in a shell where they demonstrably work | spark, thor, orin | Makes `doctor` report `unhealthy` on a healthy shell, and then (v4) invites the agent to "fix" a non-problem |
| v4 | `/doctor` exits 1 on any failing check, which the hook treats as a qualifying failure and answers with a full agent turn | spark, thor, orin | A diagnostic command diagnosing itself; burns an agent turn and the rate-limit budget |
| v5 | `Ctrl+G` discards the agent's answer (`readline.bash` redirects `nvsh slash "/ask"` to `/dev/null`) | all three | The headline on-demand gesture from issue #2 produces a banner and nothing else |
| v6 | `nvsh setup --yes` *runs* installs rather than proposing them — `npm install -g …` on thor, `sudo apt-get install …` (three attempts) on orin | thor, orin | "Propose, don't run" and "any auto-apply mode refuses `sudo`" are the repo's own rules |
| v7 | The first failure after a cold daemon prints `... daemon connection lost: timed out` and falls back to one-shot | all three, reproducible | Every cold session pays a timeout before its first answer; see below |
| v8 | `doctor`'s `agent_reachable` line prints the backend's full endpoint URL | spark, thor, orin | A private endpoint ends up in terminal scrollback, screenshots and bug reports |
| v9 | With `pi`, `proposal.command` carries the rendered panel text (`nvsh: run this command?\ncommand: …`) rather than the bare command | spark | Pressing Enter on such a proposal would run the wrong thing; also why the panel looks nested |

### v7 in detail — the cold-daemon timeout

Reproduced deliberately: `nvsh daemon stop`, then one failing command.

```text
nvsh: python3 -c 'raise RuntimeError("CUDA out of memory. …")' failed (exit 1)
... agent_start
... daemon connection lost: timed out
... one-shot pi: pi is configured and on PATH
... agent_start
```

The same two lines appear on thor and orin on their first failure
(`... daemon connection lost: timed out` / `... one-shot openai-compat: …`),
so this is not specific to spark, to Ghostty, or to `pi`. The fallback works
— every panel in this pass still rendered — but the first answer on every
cold session is delayed by the timeout and runs outside the daemon.

## Re-verification after d2-d8 (2026-09-13)

A second pass over exactly the rows the pass above recorded as FAIL or
PARTIAL, plus the candidate deviations v1-v9, run against nvsh **0.9.2** as
refreshed on all three machines from the merged wheel (the installed shell
files under the per-user `share/nvsh/shell/` directory carry the
`BP_PIPESTATUS` read, the `__NVSH_SLASH_DISPATCH` flag and the `Ctrl+G`
macro). Every session below ran on a real pty — locally through
`bash --rcfile <copy of the operator's real rc> -i`, remotely through
`ssh -tt` into the machine's own login shell — and every quoted line was
observed, trimmed, with escape sequences stripped. Nothing was fixed here:
what still fails is recorded as a failure.

| # | Item (deviation) | Machine | Result |
|---|------------------|---------|--------|
| 1 | Pipeline `PIPESTATUS` under the real rc layout (d2) | spark | **PASS** |
| 2 | `ls /nope` error slice (d3) | spark | **PASS** |
| 3 | `/doctor`: no self-triggered turn, `hook_first_in_prompt_command`, `bindings_present`, no endpoint, consistent summary (d4, d5) | spark, thor, orin | **PASS** |
| 4 | `Ctrl+G` streams the answer, emacs and vi (d6) | spark, thor, orin | **PASS** (one empty-answer flake on spark, below) |
| 5 | Cold daemon: no `daemon connection lost: timed out` (d7) | thor, orin | **PASS** |
| 5 | Cold *and warm* daemon on spark with `pi` (d7) | spark | **FAIL** — new deviation **w1** |
| 6 | `pi` proposal carries the bare command (d8) | spark | **PASS** |
| 7 | `demos/spark-cuda-oom.cast` re-recorded | spark | **PASS** |

### 1 — pipeline `PIPESTATUS` on the Spark (d2): PASS

Driven on a pty with the operator's real rc (kiro-cli's bash-preexec pre-block
first, `TERM=xterm-ghostty`, `TERM_PROGRAM=ghostty`), `NVSH_AUTO=0` and
`NVSH_HOOK_DEBUG_FILE` set, so each prompt appends `<exit>\t<PIPESTATUS>`.
The debug file, verbatim (tabs shown as spaces), one line per prompt:

```text
0   0            # the first prompt
0   1 0          # false | true
0   0 1 0        # true | false | true
1   1 0          # set -o pipefail; false | true
0   0            # declare -p PROMPT_COMMAND
```

All three expectations hold: the per-stage statuses survive bash-preexec's
`return`, and `pipefail` still reports `1` for the pipeline itself.

`declare -p PROMPT_COMMAND` in the same session, unchanged from the pass
above (the d2 fix reads `BP_PIPESTATUS`; it does not reorder anything):

```text
declare -a PROMPT_COMMAND=([0]=$'__bp_precmd_invoke_cmd\n__nvsh_hook\n__fig_post_prompt\n__bp_interactive_mode' [1]=$'__bp_trap_string="$(trap -p DEBUG)"\ntrap - DEBUG\n__bp_install; __ghostty_hook' [2]="__bp_interactive_mode")
```

### 2 — the error slice on the Spark (d3, was v1): PASS

`echo hello-success`, then `ls /nope`, then `nvsh context --show`. The
`Output:` section now opens with the failing command's own stderr, not the
previous command's stdout:

```text
cwd: /tmp/…/scratchpad
Output:
ls: cannot access '/nope': No such file or directory
nvsh: ls /nope failed (exit 2)
…
```

The agent's answer confirms it from the other side — it diagnosed the
missing path (`The 'ls /nope' command fails with exit code 2 because the
path '/nope' doesn't exist on this system`) instead of commenting on
`hello-success`, which is what it did before d3.

### 3 — `/doctor` (d4, d5): PASS on all three

On spark, typed as `/doctor` and dispatched by the Enter macro to
`nvsh slash '/doctor'`:

```text
nvsh doctor: healthy

[ok] source_checkout: no culture.yaml found alongside the package; identity checks skipped
[ok] platform_detected: detected platform: dgx-spark
[ok] agent_configured: agent provider configured: pi
[ok] agent_reachable: endpoint reachable (base_url from pi models.json, bearer from pi models.json)
[ok] hook_sourced: hook sourced, version 0.9.2 matches installed nvsh
[ok] hook_first_in_prompt_command: __nvsh_hook runs directly after __bp_precmd_invoke_cmd in PROMPT_COMMAND (bash-preexec layout; $? is restored and PIPESTATUS is read from BP_PIPESTATUS)
[ok] bindings_present: nvsh readline bindings present (emacs)
[ok] capture_active: script: /run/user/1000/nvsh/3421065.log
[ok] daemon_status: daemon running (socket: /run/user/1000/nvsh/daemon.sock)
[ok] terminfo_present: terminfo present for TERM=xterm-ghostty
```

On thor and orin (no bash-preexec) the same run reports
`[ok] hook_first_in_prompt_command: __nvsh_hook is first in PROMPT_COMMAND`,
`[ok] bindings_present: nvsh readline bindings present (emacs)` and
`[ok] daemon_status: daemon not running (normal — the daemon starts on
demand)`, with the summary line `nvsh doctor: healthy`.

Point by point:

- **No self-triggered agent turn (d5).** Nothing follows the check list: the
  next command in each session (`echo marker-after-doctor` → `marker-after-doctor`)
  comes straight after, with no `... agent_start` line anywhere.
- **`bindings_present` (v3).** `[ok]` on all three, in shells whose Enter
  macro had just dispatched the command.
- **No endpoint in any line (v8).** `agent_reachable` names the *source* of
  the values — `base_url from pi models.json` on spark, `base_url from
  [agents.openai-compat], bearer from $NVSH_API_KEY` on thor and orin — and
  prints no URL, host or IP.
- **Consistent summary (v2/v4).** No `[FAIL]` line appears while the summary
  says healthy. The one failing run seen this pass was self-consistent the
  other way: driving thor from a Ghostty terminal over ssh, `[FAIL]
  terminfo_present: no terminfo entry for TERM=xterm-ghostty` came with the
  summary `nvsh doctor: unhealthy` — and still triggered no agent turn.
  Re-run with `TERM=xterm-256color` (what a Jetson session normally sees) it
  is healthy.

### 4 — `Ctrl+G` (d6, was v5): PASS

The banner is now followed by the answer on the tty, in both keymaps. On
thor (`openai-compat`), emacs and then after `set -o vi`:

```text
why is memory high
⚡ nvsh (Ctrl+G): asking the agent
… NVSH_DRAFT=$__NVSH_DRAFT … nvsh slash "/ask"
**Memory snapshot from your Jetson AGX Thor:**
- **Total RAM:** ~128.8 GB (`mem_total: 128790772 kB`)
- **Available for new processes:** ~50.0 GB (`mem_available: 50049864 kB`)
…
```

Orin's runs are the same shape against its own platform block (`~64 GB
(64324844 kB)` total, `~3.2 GB` available), in both keymaps. On spark with `pi` the stream is the event-by-event form —
`... turn_start`, `... message_start`, then the prose, e.g.
`Based on the system output, here are the current memory levels: … Total
memory: 127,601,228 kB … Available memory: 21,807,028 kB` — and it reaches
the terminal in both emacs and vi-insert. The redirect to `/dev/null` that
swallowed it before is gone.

One flake, recorded rather than smoothed over: of two vi-mode `Ctrl+G` runs
on spark, the first printed the banner and the rewritten dispatch line, ran
for 48 s, exited 0 and printed *nothing at all* — no status lines, no text.
The second, identical run streamed the full answer. Nothing in nvsh's own
output distinguishes the two, so this is logged as a `pi`-side empty turn,
not a reproducible nvsh defect; it is worth a second look if it recurs.

### 5 — cold daemon (d7, was v7): PASS on thor and orin, FAIL on spark

On **thor** and **orin** the deviation is fixed. With no daemon running
(`nvsh daemon stop`, then `nvsh daemon status --json` →
`{"running": false, …}`), the first failure produces no fallback line at
all — the daemon starts, accepts and answers:

| Machine | Failure | First agent text | Fallback line |
|---------|---------|------------------|---------------|
| thor | `ls /nope` (exit 2) | **24.0 s** after the panel's `failed` line | none |
| orin | `trtexec --version` (exit 127) | **24.1 s** after the panel's `failed` line | none |

On **spark** with the `pi` backend it still fails, and it is worse than v7
described: the timeout is not limited to a cold daemon. Three runs, two
cold (socket absent, verified with `nvsh daemon status --json` and a check
that nothing is listening on the socket path) and one warm (a daemon
already running, `"running": true`), all produced:

```text
nvsh: ls /nope failed (exit 2)
... agent_start
... daemon connection lost: timed out
... one-shot pi: pi is configured and on PATH
... agent_start
```

Measured from the pty stream, relative to the `failed (exit 2)` line:

| Run | Daemon | `daemon connection lost` | First agent text |
|-----|--------|--------------------------|------------------|
| 1 | cold | +120.4 s | +126.5 s |
| 2 | cold | +120.3 s | +125.8 s |
| 3 | **warm** | +120.1 s | +125.9 s |

The 120 s is `_DEFAULT_TIMEOUT`, the *stream* read timeout in
`nvsh/client_transport.py` — so the connect and accept wait d7 added does
succeed (the daemon logs `daemon listening …` and `new conversation for
shell <pid>` within a second of the failure), and it is the request itself
that is never answered. Supporting evidence on the daemon side: the
conversation's `session_path` (`pi-sessions/shell-<pid>.jsonl`) is never
created, while every `pi` session file that does exist on disk is a
one-shot's. The one-shot fallback then answers in about 6 s, so every agent
call on spark — failure panels and `Ctrl+G` alike — pays a flat two-minute
penalty and runs outside the daemon, losing the warm conversation.

Raised as deviation **w1** below.

### 6 — the `pi` proposal (d8, was v9): PASS

The panel now shows the command once, bare:

```text
+----------------------------------------------------------------------+
  ls / 2>&1
+----------------------------------------------------------------------+
(fix)
[Enter] run   [e] explain   [d] details   [Esc] ignore
nvsh: not run
```

The newest entries of the audit log (`<state dir>/nvsh/audit.jsonl`, read
only) carry the same bare string, with no rendered panel text:

```json
{"decision": null, "event": "proposal", "outcome": null, "proposal": {"command": "ls / 2>&1", "kind": "fix", "rationale": ""}, "ts": …}
{"decision": "ignore", "event": "decision", "outcome": null, "proposal": {"command": "ls / 2>&1", "kind": "fix", "rationale": ""}, "ts": …}
```

### 7 — recording

`demos/spark-cuda-oom.cast` was re-recorded with `scripts/record-cast.py`
against the same CUDA-OOM-style `RuntimeError` and the same scrub rules.
The new recording shows the d8 panel — the proposed `nvidia-smi` printed
once, no nested `nvsh: run this command?` text — and, honestly, also shows
the w1 two-minute `daemon connection lost: timed out` before the one-shot
answers. `demos/orin-missing-package.cast` is unchanged: it shows no
proposal panel and no daemon fallback, so nothing in it was fixed by
d2-d8.

### Deviation raised by this pass

| id | What | Where | Why it matters |
|----|------|-------|----------------|
| w1 | Every daemon-routed request on spark with the `pi` backend is accepted and then never answered; the client gives up after the 120 s stream timeout (`daemon connection lost: timed out`) and falls back to one-shot. Cold **and** warm daemon. The daemon-side `pi` session file is never created | spark (`pi`); thor and orin (`openai-compat`) are correct | Supersedes v7: the cost is not a cold-start tax but a flat two minutes on *every* agent call on the Spark, and the daemon's warm conversation is never used there |

Deviations v1-v6, v8 and v9 are all cleared by the runs above; v7 is
superseded by w1.
