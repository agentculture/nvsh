# stop choice prompt

> nvsh asks before it stops the agent. While the agent is working, the first Ctrl+C or Esc no longer cancels the turn straight away: the panel pauses and offers a choice -- \[t\] steer, \[s\] stop, \[Esc\] keep going -- so an accidental press costs nothing and the operator can correct the agent instead of killing it. Stopping is still always two keys away, and a terminal that cannot show the prompt stops immediately as before.
> instruction: Implement in nvsh/panel.py (`_on_press` and a timed choice read), nvsh/client.py (stop-and-correct sequencing, capability lookup) and nvsh/agent/base.py + every adapter (Capabilities.steer); verify with pty tests in tests/`test_panel_stop.py` and tests/`test_client_stop.py`.

## Audience

- An operator at an interactive nvsh-hooked bash prompt on a Jetson, DGX Spark or RTX box, often over SSH, watching the agent work in the inline panel; secondarily the maintainers who read the audit log to see how a turn ended.
  - instruction: Confirm docs/shell-integration.md 'Stopping the agent' addresses the operator at the prompt.

## Before → After

- Before: Today the first Ctrl+C or Esc cancels the turn at once. An accidental press throws away a turn that may have cost a minute of model time (70 s observed on thor on 2026-09-17), and an operator who only wanted to redirect the agent has no way to do it from the panel: steer is offered only at a proposal, via /steer, or at the busy prompt.
  - instruction: Reproduce on 0.13.1: press Esc during a turn and observe 'stopping… press again to kill' with no choice.
- After: The first Ctrl+C or Esc pauses the panel and shows 'nvsh: paused -- \[t\] steer  \[s\] stop  \[Esc\] keep going'. Esc (or 30 s of silence) resumes as if nothing happened; s stops exactly as before, with a further press still killing; t takes one line and either steers the running turn (pi, codex) or stops and resends with the correction (every other harness).
  - instruction: Drive tests/`test_panel_stop.py` through each of the three keys and the timeout on a pty.

## Why it matters

- A stop key that acts instantly punishes a slip of the finger with a lost turn, and an operator who sees the agent heading the wrong way should be able to say so rather than kill it and start over. Asking first makes the stop deliberate and makes correcting the agent a first-class move.

## Requirements

- While the agent works on a terminal, the first Ctrl+C or lone Esc opens a one-line choice prompt -- \[t\] steer  \[s\] stop  \[Esc\] keep going -- instead of cancelling; nothing is sent to the harness until the operator picks. The change is confined to Panel.`_on_press` (nvsh/panel.py:705-717), the one place the first press is decided today.
  - honesty: Between the first press and the operator's key, neither cancel nor `force_stop` nor steer is called on the adapter -- asserted with a recording fake, not inferred from output.
- \[s\] stop does exactly what the first press does today: polite cancel through `_StopTarget`.cancel, the 'stopping… press again to kill' line, and a further Ctrl+C/Esc still force-kills the process tree. Stopping is never more than two keys away, and a Ctrl+C typed at the choice prompt itself means stop.
  - honesty: After \[s\], behaviour is byte-for-byte today's first press: same `STOPPING_TEXT`, same single cancel, and a further Ctrl+C or Esc calls `force_stop` exactly once; Ctrl+C typed at the choice prompt takes the \[s\] path.
- \[t\] steer reads one free-text line with the existing tell reader (Panel.`read_tell`, panel.py:1108-1133) and delivers it through the existing Responder.steer path (`client_transport.py`:101-121), the same move as the proposal prompt's \[t\] and /steer; no new daemon control message is added.
  - honesty: The text typed after \[t\] reaches the adapter's steer() through Responder.steer on both the daemon and the one-shot path with no change other than the redaction required by c29, and no new daemon control message exists.
- Whether a harness can take a correction mid-turn becomes a declared capability (Capabilities.steer in nvsh/agent/base.py), true only for pi and codex; the choice prompt and the daemon's busy prompt both read it, replacing daemon.`_overrides_steer` (daemon.py:454-456), which today reports openai-compat and agy as steerable because they override steer() only to return False.
  - honesty: Capabilities.steer is true for exactly pi and codex, false for every other registered adapter, and daemon.`_overrides_steer` is gone: the busy prompt no longer offers \[t\] steer on openai-compat or agy.
- \[Esc\] keep going dismisses the prompt, sends nothing to the harness and resumes rendering; the turn's exit status and result.interrupted are unaffected, so a dismissed accidental press is not reported as exit 130.
  - honesty: After keep going, result.interrupted is False, the turn's remaining events render in order with none lost, and the process exit status equals that of an uninterrupted turn.
- Where the prompt cannot be shown or answered -- stdin or stdout is not a tty, TERM=dumb, --json -- the first Ctrl+C stops immediately exactly as it does today, so scripts and piped sessions keep their current behaviour.
  - honesty: With stdin or stdout not a tty, TERM=dumb, or --json, the first SIGINT calls cancel at once and no prompt text is written to stdout or stderr.
- This amends the reliable-agent-stop spec: claims c1, c4, c5, c17, c20, c21, c22 and c34 (first press = immediate polite cancel, stopping line within 1 s) are superseded by the choice prompt; the 1 s promise now applies to the prompt appearing, and the 3 s kill promise is measured from the press that follows \[s\] stop. The new frame records the amendment; the old frame is not edited.
  - honesty: The exported spec names each superseded reliable-agent-stop claim id, and .devague/frames/reliable-agent-stop.json has no diff.
- Every written description of the first press is updated together: docs/shell-integration.md 'Stopping the agent' (368-397) and its exit-code table (94-95), docs/daemon.md control table (187-189), README.md:54, and the stop paragraph in all five harness prompt files (CLAUDE.md:122-128, AGENTS.override.md:46-52, .pi/SYSTEM.md:22-28, AGENTS.colleague.md:53-59, QWEN.md:49-55).
  - honesty: After the change, grep for 'press again to kill' finds it only as the post-\[s\] line, in every one of the listed files, and scripts/harness-smoke.py --stage config passes.
- Tests that pin today's first press are rewritten rather than deleted -- tests/`test_panel_stop.py` (about 5), tests/`test_client_stop.py` (about 4) and the parametrized `test_two_presses_stop_every_family_on_both_paths` in tests/`test_agent_stop.py` -- and the conformance suite gains its first steer tests, asserting pi and codex deliver mid-turn and every other adapter declares steer=False.
  - honesty: No first-press test is deleted without a replacement asserting the new behaviour for the same case, and the suite's pass count does not drop.
- The audit log records what the operator chose at the prompt. nvsh/agent/audit.py `STOP_KINDS` (19-29) gains `keep_going`; \[s\] keeps the existing cancel/`force_kill` kinds; a steer from the prompt reuses the existing steer kind with a field naming where it came from (stop prompt versus busy prompt), so existing audit readers keep working.
  - honesty: Each prompt outcome writes exactly one audit line; `keep_going` and timeout are distinguishable; a steer line names the stop prompt as its origin; existing audit tests pass unmodified.
- 'Stop and correct' sends the correction even though the turn was interrupted: today `handle_failure`, ask and the slash path return 130 before the follow-up request (nvsh/client.py:1739, 1826, 1904), which would drop the typed text. After a stop-and-correct the client waits for the cancelled turn to end, then streams the correction as the next request; only a plain \[s\] stop returns 130 without a follow-up.
  - instruction: Test in tests/`test_client_stop.py`: press, choose t on a Capabilities.steer=False fake, type a line; assert cancel was called once, a second request carrying the line was sent, and the exit status is the second turn's.
  - honesty: A correction typed after stop-and-correct is never dropped: it is sent as the next request on the daemon path and the one-shot path, and if the cancelled turn will not end the operator still gets the second-press kill rather than a hang.
- The stop-and-correct request is self-contained: nvsh composes it from the original request and the operator's correction (and says the previous attempt was stopped), so it does not depend on the harness remembering the cancelled turn. agy in warm mode respawns after cancel() (nvsh/agent/agy.py:493-515) and a one-shot run has no session at all.
  - instruction: Test: the follow-up AgentRequest.prompt contains both the original question and the correction text, on every adapter family, without reading harness session state.
  - honesty: The follow-up prompt contains the original request text and the correction on every adapter family, verified without any harness session state.
- The correction line typed at the stop prompt is passed through nvsh/redact.py before it leaves the process (steer, or the stop-and-correct follow-up), and the audit line records that a correction was sent and its length, never its text.
  - instruction: Test: type a line containing `HF_TOKEN`=abc123 at the prompt; assert the adapter's steer()/run() receives the redacted form and the audit line contains no token.
  - honesty: A token-shaped string typed as a correction never appears in what the adapter receives nor in audit.jsonl, on both the steer and the stop-and-correct path.
- Capabilities.steer says what the harness can do in principle; the runtime answer is steer()'s return value. When \[t\] was offered as 'steer' and steer() returns False (codex fell back to exec mode, no thread/turn id yet, the turn already ended), nvsh tells the operator it could not steer and asks once whether to stop and correct instead -- it never silently drops the text and never cancels without saying so.
  - instruction: Test with a fake whose Capabilities.steer is True but steer() returns False: the typed line is not lost, and cancel is called only after the operator agrees.
  - honesty: With a steer-capable adapter whose steer() returns False, the typed text is either delivered as a follow-up the operator agreed to or explicitly discarded by the operator; no third outcome exists.
- The client learns the active harness's Capabilities.steer without asking the daemon, the way registry.`_tool_calling` already reads `tool_calling`: by constructing the adapter class for the resolved target and reading .capabilities() (nvsh/agent/registry.py:305-317). No new wire field is needed for the stop prompt; the busy event's existing 'steerable' arg is filled from the same capability.
  - instruction: Test: the prompt's \[t\] label is chosen from the resolved target's capability on the daemon path with a stub daemon that sends no capability information.
  - honesty: On the daemon path the prompt shows the right \[t\] label for pi, codex and openai-compat targets without any new daemon message, and reading the capability starts no harness process.
- The 30 s timeout needs a timed single-key read, which does not exist: `_read_raw_key` blocks forever (nvsh/panel.py:1408-1442) and KeyWatcher.poll is timed but discards every byte except a lone Esc (nvsh/keys.py:133-157). The prompt gets a select()-based timed read that returns t, s, Esc, Ctrl+C, EOF or timeout, with escape sequences drained as today. Boundary c7 is narrowed accordingly: Esc disambiguation, KeyWatcher and typeahead discard stay unchanged, but a timed read may be added beside them.
  - instruction: Test on a pty: no key for 30 s returns timeout; an arrow key neither answers nor leaves bytes behind; Ctrl+C returns the stop path.
  - honesty: The timed read never spins on EOF or a hung-up tty, restores termios on every exit path including SIGHUP/SIGTERM, and tests/`test_keys.py` still passes unmodified.
- Keys typed before the prompt is on screen never answer it: pending input is discarded when the prompt is drawn, so a double-tapped Esc or a key-repeat opens the prompt and leaves it open rather than instantly choosing 'keep going' (or, with Ctrl+C twice, instantly stopping). Today Esc Esc means 'stop, then kill'; after this change it must not silently mean 'nothing happened'.
  - instruction: Test on a pty: write Esc Esc back-to-back within 20 ms; assert the prompt is visible and unanswered.
  - honesty: Two presses arriving within the terminal's key-repeat interval leave the prompt open and visible; a deliberate second press after the prompt is drawn is honoured.
- At the correction line after \[t\], an empty line, Ctrl+C or end of input means 'never mind': nothing is sent, nothing is cancelled and the turn keeps going -- the existing `read_tell` rule (nvsh/panel.py:1108-1133). End of input at the choice prompt itself (a hung-up terminal) also means keep going locally and never stop or steer, matching reliable-agent-stop deviation d1; the daemon's client-gone watch ends the turn.
  - instruction: Test: EOF at the choice prompt and at the correction line both leave cancel, `force_stop` and steer uncalled.
  - honesty: No combination of EOF, empty input or Ctrl+C at the correction line results in a cancel or a steer being sent.
- The turn can finish while the prompt is open (the agent keeps working; only rendering is paused). Choosing \[s\] or \[t\] after that must not misreport: the rest of the answer is still rendered, a stop sent to an already-finished turn does not mark the result interrupted or exit 130, and a correction typed then simply becomes the next request.
  - instruction: Test: a fake that emits DONE while the prompt is open; choose s; assert the full text is rendered and the exit status is 0.
  - honesty: When DONE is already queued at the moment of the choice, no outcome of the prompt loses rendered text or reports 130 for a turn that completed.

## Honesty conditions

- Every one of the four outcomes (steer, stop, keep going, timeout) is reachable from a single first press on a pty, and no code path cancels the harness before the operator has picked.
- nvsh/keys.py has no functional diff, and tests/`test_keys.py` passes unmodified.
- nvsh/daemon.py `cancel_shell`/`kill_shell`/`_force_stop_turn` and every adapter's cancel()/`force_stop`() have no functional diff; tests/`test_agent_conformance.py` stop cases and tests/`test_close_kills_tree.py` pass unmodified.
- The behaviour is only ever seen by an interactive operator: non-interactive invocations never reach the panel's prompt.
- The 70 s figure and the missing choice are reproducible on 0.13.1 and are recorded in this session's thor capture log.
- Each key and the timeout behaves as described on a real pty, not only against a stubbed reader.
- The operator's stated reasons (2026-09-17: 'clearer for the user and avoid accidental clicks') hold in the shipped behaviour: an accidental press costs nothing (keep going or the 30 s timeout leave the turn untouched, c6/c19), and the operator can redirect the agent from the panel instead of killing it (c4/c18).
- Every number in the signal (1 s, 3 s, 30 s +/- 1 s, 100%) is asserted by a test that fails when the number is violated.
- A client paused for the full 30 s against a fast-streaming fake loses no events, is not aborted as client-gone, and other shells' queued requests resume afterwards.

## Success signals

- The prompt appears within 1 s of the first press; after \[s\] the 'stopping…' line appears within 1 s and a further press returns the prompt within 3 s, as in 0.13.0; an unanswered prompt resumes at 30 s (+/- 1 s); keep going and a delivered steer exit 0 with interrupted unset; 100% of registered adapters declare Capabilities.steer and only pi and codex declare it true.
  - instruction: Timing assertions on a pty in tests/`test_panel_stop.py`; a conformance test iterating registry.ADAPTERS for the capability.

## Scope / boundaries

- Lone-Esc versus escape-sequence detection, cbreak handling, typeahead discard and the SIGHUP/SIGTERM terminal restore in nvsh/keys.py are not changed; the prompt reuses `_suspending`, `_read_choice_key` and the `_busy_legend` pattern from nvsh/panel.py rather than reading keys a new way.
- Force-stop, process-tree kill, the daemon's cancel/kill controls, slot removal after a kill, the per-run cancelled-flag reset and every adapter's cancel() are unchanged; the prompt only decides whether and when the existing cancel is called.

## Non-goals

- No automatic kill timer and no auto-apply: the prompt never stops, kills or steers on its own, and reliable-agent-stop c13 (`NVSH_TURN_TIMEOUT` is the only automatic bound) stands.
- The busy prompt (\[t\] steer \[r\] replace \[Esc\] exit, shown when a new request meets a running turn) keeps its keys, wording and rules; only the source of its steerable flag changes.
- No new steering ability is built into harnesses that lack a mid-turn channel (acp, claude, qwen print mode, agy): this work routes to what each harness already supports and says so honestly.
- The bash hook, triggers, rate limiting and the proposal prompt's own \[t\] are out of scope.

## Assumptions

- While the prompt is open the client reads nothing from the daemon socket. This is safe for 30 s: the daemon's `_write` is a blocking write with no send timeout (nvsh/daemon.py:511-517), so a full socket buffer back-pressures the turn thread rather than dropping events; `_peer_is_gone` does not fire for a client that is merely not reading (daemon.py:1079-1093); the 300 s turn cap keeps counting, so a pause can at most consume 30 s of it. kill and cancel travel on their own connections and are unaffected.
  - instruction: Test: pause a client for 5 s against a daemon streaming faster than the socket buffer drains; assert no event is lost or reordered and the turn is not aborted as client-gone.

## Scope exploration

- `s1` — `nvsh/panel.py (_on_press, _StopState, _stream_loop)`: First-press behaviour is decided only in Panel.`_on_press` (705-717): handle()==1 prints `STOPPING_TEXT` (line 91) and calls `cancel_once`(); any later press calls `force_stop_once`() and ends the stream. `_stream_loop`/`_next_item` already tolerate `_on_press` blocking. No timer exists between presses: it is a pure press count.
  - seeds: `c2`
- `s2` — `nvsh/client.py (_StopTarget, _Turn.audited, _stream_request)`: A press reaches the harness through `_StopTarget`.cancel/`force_stop` (client.py:1335-1369): in-process agent.cancel() for one-shot (responder.`bind_agent`), else the daemon cancel/kill controls. Both are passed into panel.stream at client.py:1583-1584. There is no callback for steering from a press today.
  - seeds: `c3`
- `s3` — `nvsh/client.py + nvsh/client_transport.py (steer paths)`: Mid-turn steer machinery already exists: Responder.steer (`client_transport.py`:101-121) reaches agent.steer() in-process or the daemon's steer RPC (`client_transport.py`:650); /steer uses it (client.py:1923-1942); the proposal prompt's \[t\] uses it via `_injector` (client.py:1156-1186); unsent steers become the next request via `_follow_up_prompt` (client.py:1589). Nothing wires a Ctrl+C/Esc press to it.
  - seeds: `c4`
- `s4` — `nvsh/agent/base.py + adapters (steer contract)`: NvshAgent.steer(text)->bool defaults to False (base.py:326-336). Only pi (pi.py:782, rpc prompt with streamingBehavior=steer) and codex (codex.py:729, turn/steer) deliver mid-turn. acp, claude, qwen print mode, fake, demo inherit False; agy (agy.py:490) overrides to return False; openai-compat (`openai_compat.py`:197) returns False but queues the text and resends prior prompt+reply+correction on the next run. Capabilities (base.py:283-307) has no steer field.
  - seeds: `c5`
- `s5` — `nvsh/daemon.py (_overrides_steer, busy prompt)`: The busy prompt's steerable flag comes from `_overrides_steer` (daemon.py:454-456), a test for 'the adapter overrides steer()', so openai-compat and agy are offered \[t\] steer although they cannot steer mid-turn. The busy prompt goes to the owning shell or a dead owner's shell (daemon.py:1484-1487); a live other shell queues.
  - seeds: `c5`
- `s6` — `nvsh/panel.py (_Feeder back-pressure, ticker, interrupted flag)`: `_Feeder` hands over one event at a time and waits for ack (panel.py:1261-1285), so while a prompt is open no events are lost but none are rendered either: output pauses, the harness keeps working. `_on_press` sets result.interrupted=True on any press (panel.py:707), which every caller maps to exit 130 (client.py:1739,1826,1904); keep going and a delivered steer need that flag left clear.
  - seeds: `c6`
- `s7` — `nvsh/keys.py (KeyWatcher) + panel prompt helpers`: KeyWatcher is cbreak with ISIG on (keys.py:116-117); lone Esc is told from CSI/SS3 by `ESC_TIMEOUT`=0.05 (keys.py:31,62-80). Panel.`_suspending` (719-741) already hands stdin to a prompt and takes it back; `_read_choice_key` (1202-1205), `_read_busy_choice` (1189-1200) and `_busy_legend` (1181-1187) are an existing single-key prompt with a near-identical legend.
  - seeds: `c7`
- `s8` — `nvsh/daemon.py (cancel_shell, kill_shell) + adapter cancel()`: `cancel_shell` (daemon.py:926-940) calls slot.agent.cancel() and keeps the slot; `kill_shell` -> `_force_stop_turn` (942-994) removes the slot so the next request builds a fresh agent. Conversation.transcript is nvsh's own log and is never replayed to the backend (daemon.py:304-335). After cancel() the same adapter runs again without start() (0.13.1).
  - seeds: `c8`
- `s9` — `nvsh/keys.py _enabled + nvsh/panel.py non-tty paths`: keys.`_enabled` (keys.py:39-48) makes KeyWatcher a no-op off a tty or under TERM=dumb, so only Ctrl+C (SIGINT) can register a press there; `_on_press` has no separate non-tty branch today. `_read_line_key` (panel.py:1461-1472) reads a cooked line off a tty and treats EOF as ignore.
  - seeds: `c9`
- `s10` — `docs/specs/2026-09-16-reliable-agent-stop.md + .devague/frames/reliable-agent-stop.json`: Two-press behaviour is fixed by claims c1, c4, c5, c17, c20, c21, c22 (1 s / 3 s timings, h5) and c34. Left intact: c6 (Esc disambiguation), c35 (typeahead dropped), c13 (no auto kill timer), c9/c24/c36 (130 vs declined 3), and the busy prompt claims c7/c8/c18/c25/c31/c37.
  - seeds: `c10`
- `s11` — `docs/shell-integration.md, docs/daemon.md, README.md, five harness prompt files`: All describe 'first press asks the harness to stop and shows stopping… press again to kill; second press kills' at the cited lines. scripts/harness-smoke.py fails CI if the prompt files drift. nvsh/explain/catalog.py has no entry describing the press behaviour. The five demo .cast files contain neither 'stopping' nor 'press again' (grep: 0 files), so no re-recording is needed.
  - seeds: `c11`
- `s12` — `tests/test_panel_stop.py, test_client_stop.py, test_agent_stop.py, test_agent_conformance.py`: About 8-10 test functions assert immediate cancel on the first press or the exact `STOPPING_TEXT`. tests/`test_keys.py` and tests/`test_fakes_ignore_cancel.py` test mechanisms and are untouched. No test anywhere references steer() (grep of tests/`test_agent_conformance.py` and tests/`_fake_adapters.py`: no hits); `test_capability_report` checks only five boolean fields.
  - seeds: `c12`
- `s13` — `nvsh/agent/audit.py (STOP_KINDS)`: `STOP_KINDS` is cancel, `force_kill`, steer, replace, `busy_exit`, declined, `doctor_apply` (audit.py:19-29); any other kind is rejected (audit.py:122). 'steer' is currently written only by the busy prompt.
  - seeds: `c13`
- `s14` — `nvsh/client.py (interrupted -> 130 before follow-up)`: `handle_failure` returns 130 as soon as result.interrupted is set, before the 'if inspections or steers' follow-up block (client.py:1739-1758); ask (1826) and the slash path (1904) do the same. A correction typed after a stop is therefore discarded today.
  - seeds: `c22`
- `s15` — `nvsh/agent/agy.py cancel() (warm respawn)`: Warm agy has no protocol cancel: cancel() discards queued events, marks the process unusable and sends SIGINT, so the next run() closes and respawns it (agy.py:493-515). Conversation continuity across a cancel cannot be assumed for agy; openai-compat replays the prior exchange only when steer() was called first (`openai_compat.py`:177-208).
  - seeds: `c23`
- `s16` — `challenge pass / security lens: nvsh/client.py + nvsh/client_transport.py steer path`: Operator-typed text is not redacted anywhere today: `_redact` is used only at client.py:1607 (inspection output). /ask, /steer and the proposal \[t\] share the gap; this frame fixes it only for the new prompt.
  - seeds: `c29`
- `s17` — `challenge pass / hidden-dependencies lens: nvsh/agent/codex.py steer()`: codex.steer() returns False when the app-server is not running (exec fallback) or no thread/turn id exists yet (codex.py:729-755), so a static Capabilities.steer=True is not a guarantee. `_injector` already handles a False return by queueing the text and noting 'could not be steered' (client.py:1156-1186).
  - seeds: `c30`
- `s18` — `challenge pass / adjacent-systems lens: nvsh/agent/registry.py + client capability lookup`: On the daemon path the adapter lives in the daemon; the client sees capabilities only through the busy event's 'steerable' arg (client.py:1474-1502). registry.`_tool_calling` (registry.py:305-317) is the existing precedent for reading a capability client-side by constructing the adapter without starting it.
  - seeds: `c31`
- `s19` — `challenge pass / concurrency lens: nvsh/panel.py _read_raw_key + nvsh/keys.py KeyWatcher.poll`: No timed key read that returns ordinary keys exists. `_read_key` puts the tty in raw mode, where Ctrl+C arrives as byte 0x03 and is raised as KeyboardInterrupt (panel.py:1431-1442); KeyWatcher.poll uses select with a deadline but consumes and discards non-Esc bytes. h7 ('keys.py has no functional diff') is stricter than the design can honour if the timed read is placed in keys.py.
  - seeds: `c32`
- `s20` — `challenge pass / overlooked-actors lens: muscle memory from 0.13.x (Esc Esc, Ctrl+C Ctrl+C)`: 0.13.x taught two presses = stop then kill (docs/shell-integration.md:373-378, README.md:54). With Esc bound to both 'open the prompt' and 'keep going', an unfiltered double-tap becomes a no-op and defeats the feature's stated purpose of avoiding accidental outcomes. keys.py already discards typeahead while streaming (keys.py:16,137), so the precedent exists.
  - seeds: `c33`
- `s21` — `challenge pass / failure-modes lens: nvsh/panel.py read_tell + hung-up terminal`: `read_tell` reads a cooked line with its own SIGINT handler that means 'never mind' (panel.py:1108-1133). `_read_raw_key` returns '' on EOF and d1 already fixes 'end of input at a proposal means ignore, never approve' (docs/shell-integration.md:400-401). c3's 'Ctrl+C at the choice prompt means stop' must not be read as applying to the correction line.
  - seeds: `c34`
- `s22` — `challenge pass / concurrency lens: nvsh/daemon.py _write + _watch_turn`: Examined for parked item v2. The serving loop writes each event with wfile.write/flush and treats only BrokenPipe/ConnectionReset/ValueError as a gone client (daemon.py:505-517); there is no send timeout. The run lock is held for the paused duration, so other shells wait up to 30 s longer. Residual: not measured under a real 6 KB/s reasoning stream.
  - seeds: `c35`
- `s23` — `challenge pass / lifecycle lens: turn ends during the prompt (nvsh/panel.py _Feeder, nvsh/daemon.py cancel_shell)`: `_Feeder` holds at most one un-acked event, so the client cannot see that a turn has finished while it is paused (panel.py:1261-1285). `cancel_shell` returns nothing and only calls slot.agent.cancel() (daemon.py:926-940); a cancel that lands after the turn ended just sets the adapter's flag, which is harmless only because run() clears it per turn since 0.13.1 -- this frame depends on that fix.
  - seeds: `c36`
- `s24` — `challenge pass / migration lens: mixed client and daemon versions`: Clean pass. A newer client retires a stale daemon on connect via the version handshake (`_retire_stale_daemon`, `client_transport.py`:360-380), and nvsh setup stops the daemon on upgrade, so an old daemon computing 'steerable' with `_overrides_steer` cannot outlive the first request. Residual: a shell opened before the upgrade keeps the old hook file, which this frame does not touch.
- `s25` — `challenge pass / operations lens: nvsh overview + audit visibility of a paused turn`: Examined, not changed. nvsh overview shows 'active turn: shell N (target, elapsed)' (nvsh/cli/`_commands`/overview.py:119) and has no notion of a client paused at a prompt; the pause is client-side and invisible to the daemon. The audit line per outcome (c13) is the only record. Residual: an operator on another terminal cannot tell a paused turn from a slow one.
- `s26` — `challenge pass / adjacent-systems lens: demo casts, explain catalog, bash hook, Ctrl+G`: Clean pass. docs/demos/\*.cast contain neither 'stopping' nor 'press again' (grep: 0 files), nvsh/explain/catalog.py has no entry describing the press behaviour, and the bash hook and readline bindings never see keys typed while the Python panel owns the terminal.

## Decisions

- On a harness that cannot steer mid-turn (Capabilities.steer false: openai-compat, acp, claude, qwen print mode, agy), \[t\] is 'stop and correct': nvsh cancels the running turn politely, then sends the operator's typed correction as the next request with the prior exchange as context. On pi and codex \[t\] delivers the text into the running turn and nothing is cancelled.
- A choice prompt left unanswered for 30 s dismisses itself as 'keep going': nothing is sent to the harness, rendering resumes, and the audit log records the timeout.
- Exit status: a turn that was steered from the prompt and then finishes normally exits 0 like an uninterrupted turn; 130 is reserved for \[s\] stop (and the kill that may follow it); keep going never changes the status.
- Prompt wording: 'nvsh: paused -- \[t\] steer  \[s\] stop  \[Esc\] keep going'; where the harness cannot steer mid-turn the first key reads '\[t\] stop & correct'.
- A Ctrl+C typed while another nvsh prompt is already open (a proposal's keys or the busy prompt) goes straight to stop, exactly as it does today; the choice prompt opens only for a press made while the panel is streaming.
- There is no configuration switch to restore the instant stop: one behaviour is documented and tested, and sessions without a terminal already stop instantly.

## Open parks

- [unknown_nonblocking] The 30 s timeout and the paused panel interact with `NVSH_TURN_TIMEOUT` (300 s) and with daemon-side buffering while the client renders nothing: confirm in implementation that a paused client cannot stall or drop the harness stream.
- [follow_up] Whether real pi and real codex visibly act on a mid-turn steer sent from the stop prompt was not probed on the fleet; only the fakes are exercised. Verify on thor/orin after merge.
- [follow_up] Pre-existing gap outside this frame: operator-typed text from /ask, /steer and the proposal prompt's \[t\] is sent to the harness unredacted, against CLAUDE.md's 'everything that leaves the process is redacted first'. Needs its own issue.
