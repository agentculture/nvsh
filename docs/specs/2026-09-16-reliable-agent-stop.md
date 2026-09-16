# reliable agent stop

> Stopping the agent is reliable: Ctrl+C and Esc stop a running agent on every backend, a second press kills a harness that ignores the stop, a new request from the same shell supersedes its own hung turn, a declined run exits with a distinct status, overview shows what the agent is doing and doctor fixes a hung agent

## Audience

- operators at an nvsh-hooked bash prompt on Jetson / DGX Spark / RTX, often over SSH, who called the agent and now want it to stop or take a different direction
  - instruction: check docs/shell-integration.md's stop/steer section is written for the prompt operator

## Before → After

- Before: Ctrl+C sends a polite cancel that pi, codex, acp and warm agy harnesses can ignore, the one-shot path never calls cancel() at all, Esc does nothing while the agent works, a second request from the same shell silently queues behind a hung turn for up to 300s, and a hung daemon turn can only be cleared with an OS-level kill
  - instruction: cite the s1-s10 scope entries
- After: a first Ctrl+C or Esc cancels the running turn on every backend and path, a second press kills an unresponsive harness, a new request from the same shell offers steer / replace / exit instead of silent queueing, overview shows the active turn, and doctor --apply clears a hung turn

## Requirements

- Ctrl+C reliably stops a running agent on every backend and both paths (daemon and one-shot); the operator reported Ctrl+C did not stop the agent earlier
  - honesty: an integration test reproduces the one-shot path: Ctrl+C mid-stream calls the in-process adapter's cancel() and no harness child survives the client
- every adapter's stop path ends the running turn: pi (abort), codex (turn/interrupt), acp (session/cancel) and warm agy (flag only, nothing sent) leave the harness running if it ignores cancel; the fix covers all adapters per the all-backends rule
  - honesty: each of pi, codex, acp, agy-warm, claude, qwen-p, agy-cold and openai-compat has a test where cancel() ends the running turn and the next run() works
- escalation is operator-driven: the first Ctrl+C/Esc sends the polite protocol cancel; a second press force-kills the harness process (terminate then kill) and the warm session is respawned on the next request; no automatic kill timer is added
  - honesty: a second press within the same stream calls a kill escalation (`escalate_close`: terminate, wait, kill) exactly once; a single press never kills
- Esc during agent work (thinking, text, waiting ticker) is identical to Ctrl+C: same cancel path, second press kills, exit 130; Esc keeps meaning 'ignore' at the proposal legend
  - honesty: a pty test sends a lone ESC during streaming and gets the same StreamResult(interrupted=True) and exit 130 as SIGINT
- a lone Esc is distinguished from multi-byte escape sequences (arrow/function keys) with a short read timeout, and the leftover sequence bytes are drained rather than leaked into the next read
  - honesty: a pty test sends an arrow-key sequence during streaming and at the proposal: it neither interrupts nor counts as Esc, and no stray bytes reach the next read
- a new request from the shell that owns a still-running turn opens a busy prompt that asks the operator: steer the running turn with the new request, replace the turn (stop it and run the new request), or exit back to the shell prompt leaving the turn alone
  - honesty: a daemon test with a running turn for shell A: a new request from A gets the busy prompt; steer delivers via agent.steer (or falls back as d16 describes), replace cancels then runs, exit leaves the turn running and returns 0 to the prompt
- when the operator declines the agent — choosing exit at the busy steer prompt, or Esc/ignore at a proposal — nvsh exits with one dedicated 'declined' exit code distinct from 0, 1, 2 and 130, recorded in the audit log; stops during work (Ctrl+C/Esc) stay 130
  - honesty: the declined path's exit code is distinct from 0, 1, 2 and 130, documented in docs/shell-integration.md, and written to the audit log
- nvsh overview shows what the agent is doing now: active turn (shell, target, elapsed) and the queue, from the daemon's existing status
  - honesty: nvsh overview --json includes the daemon's `active_turn` (shell, target, elapsed) and queued fields, and reports 'no daemon' cleanly when none runs
- nvsh doctor detects a hung agent turn (check in `doctor_checks` rubric) and, with --apply, fixes it by cancelling then killing the stuck harness so the next request respawns the session
  - honesty: doctor without --apply never mutates; doctor --apply on a hung turn cancels, then kills after the grace, and the check passes on re-run
- tests simulate a harness that ignores cancel for each adapter family (rpc, app-server, acp, stream-json warm) and assert the second press kills it and the next request respawns the session
  - honesty: the ignoring-harness fake exists for each adapter family and runs in the default pytest -n auto suite

## Honesty conditions

- every sub-claim (c2-c15) has a passing test named in the plan before the feature is called shipped
- a daemon test: shell B's request while A's turn runs still queues with the busy notice and never gets A's steer/replace prompt
- hook.bash still runs the client with '|| return 0' and a test asserts the operator's $? and PIPESTATUS for the failing command are unchanged after a declined run
- no timer-driven kill is added; grep shows `NVSH_TURN_TIMEOUT` remains the only automatic bound
- no stop path writes to any harness settings or trust file; only protocol messages and signals are used
- docs/shell-integration.md documents Ctrl+C/Esc stop, second-press kill, the busy prompt and the declined exit code for the prompt operator
- each before-state fact is cited by a recorded scope entry (s1-s10) with file:line
- each after-state behaviour maps to a requirement claim (c2-c12) with its own confirmed honesty condition
- the 1s / 3s thresholds are asserted by tests, not measured by hand

## Success signals

- with a fake harness that ignores cancel, on every adapter family and on both daemon and one-shot paths, the prompt returns within 1s of the first Ctrl+C/Esc and the harness process is gone within 3s of the second press
  - instruction: tests/`test_agent_stop.py` parametrised over pi-rpc, codex app-server, acp, agy warm, claude stream-json, openai-compat, one-shot and daemon

## Scope / boundaries

- supersede never cancels another shell's turn: `cancel_shell` stays scoped to the calling shell (daemon.py:852-856 fixed a cross-shell cancel bug); other shells keep queueing
- the operator's original $? for the failing command is never changed by the declined status; the hook stays '|| return 0' so nvsh can never lock the prompt (CLAUDE.md Hook constraints)
- nvsh never edits a harness's own settings to make cancel work; stops use launch flags, protocol messages and process signals only (docs/specs/2026-09-14-first-class-multi-harness-with-aliases.md)

## Non-goals

- no automatic grace-period kill timer is added to cancel; the existing `NVSH_TURN_TIMEOUT` 300s cap stays as the only automatic bound
- no auto-apply and no new autonomous behaviour: stopping, steering and doctor --apply never run an agent-suggested command

## Scope exploration

- `s1` — `nvsh/client.py + nvsh/client_transport.py one-shot path`: one-shot runs the agent in the client process (`client_transport.py`:264-313) but the cancel callback (client.py:1379) only sends a socket cancel to the daemon; the adapter's cancel() is never called, only KeyboardInterrupt breaks the read loop — contradicts panel.py:49-52 'aborting the daemon/one-shot run'
  - seeds: `c2`
- `s2` — `nvsh/agent/* cancel() implementations`: pi.py:837-845, codex.py:741-756, acp.py:883-907 send protocol cancels with no escalation; agy.py:422-433 warm mode is a no-op toward the child; claude/qwen-p/agy-cold already terminate->2s->kill via `_subprocess.py`:247-257; `escalate_close` (`_subprocess.py`:51-101) exists but is only used by close()
  - seeds: `c3`
- `s3` — `nvsh/panel.py stream() SIGINT guard`: stream() cancels once via a lock-free 'cancelled' list guard (panel.py:536, 571-577, 689-699); a second press is currently swallowed, so second-press-kills needs a new escalation hook on that guard
  - seeds: `c4`
- `s4` — `nvsh/panel.py stdin handling`: nothing reads stdin during the stream loop (panel.py:560-570) and the terminal is not in cbreak; `_read_key` reads one byte with no escape-sequence disambiguation (panel.py:984-1005, 1086-1107), so arrow keys read as Esc and leak remaining bytes
  - seeds: `c5`
- `s5` — `nvsh/daemon.py run lock and queue`: one global `_run_lock` serialises all turns (daemon.py:356-367, 934); a second request queues with 'waiting for the agent (busy with shell X)' every 15s (daemon.py:926-946); no preempt exists; a hung turn blocks all shells until `NVSH_TURN_TIMEOUT` 300s (daemon.py:67-70, 909-924)
  - seeds: `c7`
- `s6` — `nvsh/daemon.py cancel_shell`: cancel control messages never queue behind a turn (docs/daemon.md:110-113, tests/`test_daemon_busy.py`:353) and `cancel_shell` only touches the caller's own slots (daemon.py:850-864)
  - seeds: `c8`
- `s7` — `exit codes client->hook`: client returns only 0/1/130 (client.py:1468-1549, 1529, 1609, 1637); proposal IGNORE has no exit-code effect; hook.bash (~271) runs the client with '|| return 0', discarding its status; operator $? is captured first at hook.bash:187 and survives
  - seeds: `c9`
- `s8` — `nvsh daemon status`: daemon status already exposes `active_turn` and queued (daemon.py:978-980, docs/daemon.md:105-107, tests/`test_daemon_busy.py`:304); nothing operator-facing surfaces it in overview today; there is no nvsh status verb or /stop slash command (slash.py:441-524)
  - seeds: `c11`
- `s9` — `nvsh/doctor_checks.py rubric`: doctor returns the read-only {healthy, checks\[\]} rubric with remediation strings and already checks daemon status from a hooked shell (CLAUDE.md, nvsh/`doctor_checks.py`); it has no fix/mutating mode today
  - seeds: `c12`
- `s10` — `tests/ cancel coverage`: `test_agent_conformance.py`:115 `test_cancel_mid_stream` plus per-adapter cancel tests cover cooperative harnesses only; `escalate_close` is tested via close() (`test_agent_subprocess.py`:256,262); no test simulates a harness ignoring cancel
  - seeds: `c15`

## Decisions

- nvsh doctor detects a hung agent turn and fixes it only with --apply; plain doctor stays read-only and its remediation names 'nvsh doctor --apply'
- Esc during agent work is the same as Ctrl+C: same cancel path, same second-press kill, same exit status
- a new request from a shell whose own turn is still running asks the operator instead of silently queueing: steer the running turn with the new request, or exit from that prompt
- the 'declined' exit code covers exit at the busy steer prompt and Esc/ignore at a proposal; Ctrl+C/Esc during work stay exit 130
- the busy prompt for the operator's own running turn offers steer, replace and exit

## Hard questions

- does superseding need a confirm in the panel ('agent busy with this shell for 2m — replace?') or happen silently on any new request from the owning shell? (resolved: ask the user: allow steering the running turn with the new request, and from that prompt the operator can exit)
- which gestures count as 'declined' (Esc/Ctrl+C mid-stream today exit 130; Esc/ignore at the proposal exits 0), and which numeric code — does 130 stay for Ctrl+C or become 'declined' too? (resolved: Esc during work behaves exactly like Ctrl+C (same cancel path, same exit status))
- doctor is read-only today: does 'doctor fixes the hung agent' mean a new 'nvsh doctor --fix' that cancels/kills the stuck turn, or a failing check whose remediation names the fix command? (resolved: doctor fixes a hung agent only with --apply; without it, it reports the failing check and names the --apply remediation)

## Open parks

- [unknown_nonblocking] exact reproduction of the operator's earlier Ctrl+C failure (backend, daemon vs one-shot, panel state) is not yet known
