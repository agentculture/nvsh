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
- escalation is operator-driven: the first Ctrl+C/Esc sends the polite protocol cancel and the panel stays up showing 'stopping… press again to kill' until the turn ends; a second press force-kills the harness process tree and the warm session is respawned on the next request; no automatic kill timer is added
  - honesty: a second press within the same stream calls a kill escalation (`escalate_close`: terminate, wait, kill) exactly once; a single press never kills
- Esc during agent work (thinking, text, waiting ticker) is identical to Ctrl+C: same cancel path, second press kills, exit 130; Esc keeps meaning 'ignore' at the proposal legend
  - honesty: a pty test sends a lone ESC during streaming and gets the same StreamResult(interrupted=True) and exit 130 as SIGINT
- a lone Esc is distinguished from multi-byte escape sequences (arrow/function keys) with a short read timeout, and the leftover sequence bytes are drained rather than leaked into the next read
  - honesty: a pty test sends an arrow-key sequence during streaming and at the proposal: it neither interrupts nor counts as Esc, and no stray bytes reach the next read
- a new request from the shell that owns a still-running turn — or from any shell when the owning shell pid is gone — opens a busy prompt: steer the running turn with the new request, replace the turn (stop it and run the new request), or exit back to the prompt leaving the turn alone
  - honesty: a daemon test with a running turn for shell A: a new request from A gets the busy prompt; steer delivers via agent.steer (or falls back as d16 describes), replace cancels then runs, exit leaves the turn running and returns 0 to the prompt
- when the operator declines the agent — exit at the busy prompt, or Esc/ignore at a proposal — nvsh reports one dedicated 'declined' exit code (allocated from the reserved 3+ range, cli/`_errors.py`:20), distinct from 0, 1, 2 and 130, visible as --json `exit_code`, in the audit log, and as direct 'nvsh ask' exit status; slash dispatch still exits 0 (d5)
  - honesty: the declined path's exit code is distinct from 0, 1, 2 and 130, documented in docs/shell-integration.md, and written to the audit log
- nvsh overview shows what the agent is doing now: active turn (shell, target, elapsed) and the queue, from the daemon's existing status
  - honesty: nvsh overview --json includes the daemon's `active_turn` (shell, target, elapsed) and queued fields, and reports 'no daemon' cleanly when none runs
- nvsh doctor detects a hung agent turn and, with --apply, fixes it by cancelling then killing the stuck harness so the next request respawns the session; a dead-owner turn is fixed directly, a live other shell's turn only after an explicit confirm naming that shell
  - honesty: doctor without --apply never mutates; doctor --apply on a hung turn cancels, then kills after the grace, and the check passes on re-run
- tests simulate a harness that ignores cancel for each adapter family (rpc, app-server, acp, stream-json warm) and assert the second press kills it and the next request respawns the session
  - honesty: the ignoring-harness fake exists for each adapter family and runs in the default pytest -n auto suite
- a force-stop reaches the daemon as its own control message (not a repeat cancel): `_abort_turn` returns early once turn.aborted is set (daemon.py:883-886), and the run lock stays held while the thread is parked in a harness that ignores cancel, so the kill must close the slot's agent (`escalate_close`) and replace the slot so the lock is released
  - honesty: a daemon test with a harness that ignores cancel: the force-stop control message releases `_run_lock` within 3s and the next queued request runs on a fresh slot
- the force-stop kills the harness's whole process tree (its tool subprocesses too), not just the harness pid; harness children are spawned without `start_new_session` today (only the daemon uses it, daemon.py:241-253), so terminal SIGINT also hits one-shot harness children directly
  - honesty: a test harness that spawns a sleeping grandchild: after the force-stop neither the harness nor the grandchild pid is alive
- Esc detection does not depend on the event source yielding: stream() iterates a generator that can block for up to the 120s daemon read timeout, so the stdin watch runs alongside it (select over tty + event source, or a reader thread) and a pressed Esc cancels within 1s even when no event arrives
  - honesty: a pty test with an event source that yields nothing for 30s: Esc ends the stream within 1s
- every stop is audited: cancel (Ctrl+C/Esc), force-kill, steer, replace, busy-prompt exit, declined and doctor --apply each append an audit event with shell, target, turn elapsed and outcome; today audit records only proposal/decision/outcome/install (nvsh/agent/audit.py, loop.py:55-68, client.py:1071-1097)
  - honesty: after each stop kind, the audit log contains exactly one matching event with shell, target, elapsed and outcome
- the busy prompt tells the truth about steer: it is offered only when the adapter has a mid-turn channel (pi, codex steer; claude, acp, qwen, agy, openai-compat return False per base.py:322 and agy.py:419, `openai_compat.py`:172), and a steer accepted by a harness that stays silent is followed by the same stop/replace choice instead of an endless wait
  - honesty: for a non-steerable adapter the busy prompt omits steer; for a steerable harness that stays silent after steer, the prompt re-offers replace/exit

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
- a pty test kills the client with SIGHUP and SIGTERM mid-stream and the tty is back in canonical+echo mode; Esc detection is not armed when stdin is not a tty or TERM=dumb
- overview --json with no daemon running returns within 1s, starts no process, and the rubric's static sections are byte-identical to today
- a test kills each persistent adapter (pi, codex, acp, agy warm) mid-turn and the next request succeeds with a 'new session' status line

## Success signals

- with a fake harness that ignores cancel, on every adapter family and on both daemon and one-shot paths, a 'stopping… press again to kill' line appears within 1s of the first Ctrl+C/Esc, and the harness process tree is gone and the prompt returned within 3s of the second press
  - instruction: tests/`test_agent_stop.py` parametrised over pi-rpc, codex app-server, acp, agy warm, claude stream-json, openai-compat, one-shot and daemon

## Scope / boundaries

- replace/steer never touches a live other shell's turn: `cancel_shell` stays scoped to the calling shell (daemon.py:852-856); the only exceptions are a turn whose owning shell pid is gone (busy prompt applies) and doctor --apply after an explicit confirm naming the owning shell
- the operator's original $? for the failing command is never changed by the declined status; the hook stays '|| return 0' so nvsh can never lock the prompt (CLAUDE.md Hook constraints)
- nvsh never edits a harness's own settings to make cancel work; stops use launch flags, protocol messages and process signals only (docs/specs/2026-09-14-first-class-multi-harness-with-aliases.md)
- holding stdin in cbreak mode to see Esc must never leave the terminal without echo or canonical mode: attributes are restored on every exit path, including SIGTERM and SSH hangup (SIGHUP), and Esc detection is off when stdin is not a tty, TERM=dumb or `NVSH_DISABLE` is set
- overview's live agent section never autostarts the daemon and never blocks: it asks an existing socket with a short timeout and reports 'no daemon' otherwise; the static identity/verbs sections the teken rubric reads (cli/`_commands`/overview.py:36-53) stay unchanged

## Non-goals

- no automatic grace-period kill timer is added to cancel; the existing `NVSH_TURN_TIMEOUT` 300s cap stays as the only automatic bound
- no auto-apply and no new autonomous behaviour: stopping, steering and doctor --apply never run an agent-suggested command

## Assumptions

- after a force-kill the conversation may be lost: pi's in-memory session, codex's app-server thread and acp's session die with the process; the next request starts a fresh harness session carrying nvsh's own last-failure context, and the panel says so

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
- `s11` — `challenge pass / lifecycle lens: nvsh/panel.py _on_sigint + stream()`: first SIGINT calls cancel() once and raises KeyboardInterrupt, ending stream() immediately; there is no window in which a second press reaches nvsh, so c4's second-press kill has no landing point without changing when the prompt returns
  - seeds: `q4` (question, resolved)
- `s12` — `challenge pass / overlooked-actors lens: operator typing ahead during streaming`: today nothing reads stdin during stream() (panel.py:560-570), so typeahead reaches bash's readline after the panel; an Esc watcher in cbreak must read those bytes first and cannot give them back — decision routed to hard question q6 on c5 (seed q5 recorded in error; q5 is the c4/c22 contradiction)
  - seeds: `q5` (question, resolved)
- `s13` — `challenge pass / concurrency lens: nvsh/daemon.py _abort_turn + _run_lock`: `_abort_turn` is idempotent via turn.aborted (daemon.py:883-886) and only calls agent.cancel(); `_run_lock` is released only when the turn thread returns from agent.run(), so a hung harness keeps every shell queued
  - seeds: `c26`
- `s14` — `challenge pass / failure-mode lens: nvsh/agent/_subprocess.py spawn + escalate_close`: `start_new_session` appears only in daemon.py:253; adapter children share the client's process group in one-shot (terminal SIGINT reaches them) and `escalate_close` terminates a single pid, so tool grandchildren the agent started can be orphaned
  - seeds: `c27`
- `s15` — `challenge pass / operations lens: nvsh/panel.py _save_termios/_restore_termios`: termios is restored in stream()'s finally (panel.py:578-591) only for exits Python sees; SIGHUP/SIGTERM/SIGKILL of the client would leave cbreak/-echo in place once Esc detection holds the tty for the whole stream (hook constraint: never lock the operator out)
  - seeds: `c28`
- `s16` — `challenge pass / concurrency lens: nvsh/panel.py stream loop`: the for-event loop (panel.py:560-570) blocks inside the events generator (daemon read timeout 120s); only SIGINT wakes the main thread today, which is why Ctrl+C works and a key read inside the loop would not
  - seeds: `c29`
- `s17` — `challenge pass / adjacent-systems lens: nvsh/cli/_commands/slash.py + docs/shell-integration.md Exit status`: slash exits 0 whenever a registered command ran (d5) to stop `__nvsh_hook` diagnosing nvsh's own line; the operator-visible $? after /ask can never carry a declined code without reopening d5; cli/`_errors.py`:20 reserves 3+ for future categorisation, so the declined code must be allocated there
  - seeds: `q7` (question, resolved)
- `s18` — `challenge pass / observability lens: nvsh/agent/audit.py call sites`: audit events are proposal, decision, outcome and install only; no record exists of an interrupted turn, a timeout abort (daemon.py:909-924 logs to the daemon log only) or a steer, so a stop leaves no audit trail
  - seeds: `c30`
- `s19` — `challenge pass / overlooked-actors lens: SSH reconnect + nvsh/client.py _shell_pid`: shell identity is the shell pid; a reconnecting SSH operator, a new tmux pane or a subshell is a different shell, and the daemon's client-gone watch only calls cancel() (daemon.py:914-916), so the stuck turn of a dead shell blocks the reconnecting operator behind a plain queue notice
  - seeds: `q8` (question, resolved)
- `s20` — `challenge pass / failure-mode lens: steer path daemon.py:1088-1120 + client.steer`: daemon steer returns 'no running turn to steer' when no adapter accepts; client.steer then falls back to a new request (d16, client.py:1672-1675), which on a hung turn would queue behind the very turn it meant to steer
  - seeds: `c31`
- `s21` — `challenge pass / security+reversibility lens: doctor --apply vs daemon.py cancel_shell isolation`: `cancel_shell` is strictly per-shell after a cross-shell cancel bug (daemon.py:850-857); doctor is a separate process with no owning shell of the turn, so --apply needs its own authority rule; killing a harness mid tool call is not reversible
  - seeds: `q9` (question, resolved)
- `s22` — `challenge pass / adjacent-systems lens: nvsh/cli/_commands/overview.py`: overview today is static (Identity, Verbs, Sibling-pattern artifacts) and is read by the agent-first rubric (teken cli doctor --strict); adding live daemon state makes it environment-dependent
  - seeds: `c32`
- `s23` — `challenge pass / lifecycle lens: persistent_session adapters pi.py:870, codex.py:840, acp.py:929, agy.py:451`: these adapters keep one long-lived process; close() escalates and the daemon slot would need a fresh agent; whether each harness can resume the prior session after its process is killed was not probed
  - seeds: `c33`
- `s24` — `challenge pass / migration lens: nvsh/daemon.py version handshake`: clean pass: new control kinds (kill, busy prompt) ride the existing `VERSION_KEY` handshake (daemon.py:97-116, 1005-1010) which refuses a mismatched client, so an old warm daemon after an upgrade fails loudly rather than misreading a kill; residual: a mixed-version daemon still holding a hung turn needs 'nvsh daemon stop'

## Decisions

- nvsh doctor detects a hung agent turn and fixes it only with --apply; plain doctor stays read-only and its remediation names 'nvsh doctor --apply'
- Esc during agent work is the same as Ctrl+C: same cancel path, same second-press kill, same exit status
- a new request from a shell whose own turn is still running asks the operator instead of silently queueing: steer the running turn with the new request, or exit from that prompt
- the 'declined' exit code covers exit at the busy steer prompt and Esc/ignore at a proposal; Ctrl+C/Esc during work stay exit 130
- the busy prompt for the operator's own running turn offers steer, replace and exit
- panel stays up while stopping: first Ctrl+C/Esc prints 'stopping… press again to kill' within 1s and keeps the panel until the turn really ends; second press kills the harness process tree; then the prompt returns
- typeahead typed while the panel streams is consumed and discarded; only Esc and Ctrl+C act (kernel 6.2+ cannot push bytes back)
- d5 stands: the declined exit code is observable via --json `exit_code`, the audit log and direct 'nvsh ask', never as $? after a slash dispatch
- the busy prompt (steer/replace/exit) also appears for a turn whose owning shell pid is gone; doctor --apply kills a dead-owner turn freely, but a live other shell's turn only after an explicit confirm naming that shell

## Hard questions

- the first Ctrl+C raises KeyboardInterrupt and stream() returns the prompt at once (panel.py:689-699, 'Ctrl+C always returns the prompt' panel.py:49); once the prompt is back, a second Ctrl+C goes to readline, not nvsh — so where does the second press land? (a) after the first press the panel stays up showing 'stopping… Ctrl+C/Esc again to kill' until the turn ends, (b) the kill is reachable later via /stop or doctor --apply, (c) something else (resolved: panel stays up while stopping: first Ctrl+C/Esc prints 'stopping… press again to kill' within 1s and keeps the panel until the turn really ends; second press kills the harness process tree; then the prompt returns)
- contradiction with c22? (resolved: panel stays up while stopping: first Ctrl+C/Esc prints 'stopping… press again to kill' within 1s and keeps the panel until the turn really ends; second press kills the harness process tree; then the prompt returns — c22 amended to measure the stopping line, not prompt return)
- reading stdin during the stream consumes whatever the operator types ahead (the next command); TIOCSTI push-back is disabled on current kernels (6.2+), so bytes read cannot be returned to readline — drop typeahead, or only watch for Esc/Ctrl+C and accept that typeahead is lost while the agent streams? (resolved: accept: while the panel streams, keystrokes are consumed and discarded (never executed); only Esc and Ctrl+C act; documented)
- does superseding need a confirm in the panel ('agent busy with this shell for 2m — replace?') or happen silently on any new request from the owning shell? (resolved: ask the user: allow steering the running turn with the new request, and from that prompt the operator can exit)
- a shell is identified by its pid (`NVSH_SHELL_PID` or getppid, client.py:390-394); the common hung case over SSH is: connection drops, operator reconnects in a NEW shell, the old shell's turn is still stuck — the new shell is a different id, so it queues and never sees the busy prompt. Should the busy prompt also cover a turn whose owning shell is gone (dead pid / client-gone), or a turn owned by any shell of the same user? (resolved: the busy prompt (steer/replace/exit) also appears for a turn whose owning shell pid is gone; doctor --apply kills a dead-owner turn freely, but a live other shell's turn only after an explicit confirm naming that shell)
- which gestures count as 'declined' (Esc/Ctrl+C mid-stream today exit 130; Esc/ignore at the proposal exits 0), and which numeric code — does 130 stay for Ctrl+C or become 'declined' too? (resolved: Esc during work behaves exactly like Ctrl+C (same cancel path, same exit status))
- where is the declined exit code observable? nvsh slash deliberately exits 0 for any handled command (d5, cli/`_commands`/slash.py:41-50, docs/shell-integration.md 'Exit status') and hook.bash discards 'nvsh hook' status with '|| return 0' — so for Enter/Ctrl+G/@target/auto-trigger runs it would only show in 'nvsh slash --json' `exit_code`, the audit log, and direct 'nvsh ask'; is that enough, or should d5 change (risking nvsh diagnosing its own dispatch line)? (resolved: keep d5: declined code is visible as `exit_code` in --json, in the audit log, and as the exit status of direct 'nvsh ask'; $? after /ask stays 0)
- doctor is read-only today: does 'doctor fixes the hung agent' mean a new 'nvsh doctor --fix' that cancels/kills the stuck turn, or a failing check whose remediation names the fix command? (resolved: doctor fixes a hung agent only with --apply; without it, it reports the failing check and names the --apply remediation)
- doctor --apply runs from any shell of the user; c8 forbids cancelling another shell's turn — may doctor --apply kill a hung turn owned by another (possibly live) shell, only a turn whose owning shell is gone or past a threshold, or only after an interactive confirm naming the owning shell? (resolved: the busy prompt (steer/replace/exit) also appears for a turn whose owning shell pid is gone; doctor --apply kills a dead-owner turn freely, but a live other shell's turn only after an explicit confirm naming that shell)

## Open parks

- [unknown_nonblocking] exact reproduction of the operator's earlier Ctrl+C failure (backend, daemon vs one-shot, panel state) is not yet known
- [unknown_nonblocking] whether each persistent harness (pi, codex app-server, kiro/qwen ACP) can resume its conversation after a force-kill was not probed on the fleet
- [unknown_nonblocking] the operator's original Ctrl+C failure was never reproduced; it may be a path this pass did not examine (e.g. Ctrl+C during an approved command run via `_run_command`, or during the proposal key read)
