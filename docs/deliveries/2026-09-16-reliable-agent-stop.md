# Delivery Summary — reliable agent stop

plan: `reliable-agent-stop` · run: `complete` · date: `2026-09-16`
baseline: `devague summary skeleton`

## Intent

> Stopping the agent is reliable: Ctrl+C and Esc stop a running agent on every backend, a second press kills a harness that ignores the stop, a new request from the same shell supersedes its own hung turn, a declined run exits with a distinct status, overview shows what the agent is doing and doctor fixes a hung agent

After: a first Ctrl+C or Esc cancels the running turn on every backend and path, a second press kills an unresponsive harness, a new request from the same shell offers steer / replace / exit instead of silent queueing, overview shows the active turn, and doctor --apply clears a hung turn

## Planned Work

Quoted verbatim from the `devague summary` skeleton (t22 and t23 were added mid-run by approved deviations `d7` and `d14`):

- `t1` — fake harnesses gain an ignore-cancel mode that also spawns a sleeping grandchild
- `t2` — process-tree spawn and kill escalation in `_subprocess` and a `force_stop`() on NvshAgent
- `t3` — declined exit code constant and stop audit events
- `t4` — overview shows the live agent turn without autostarting the daemon
- `t5` — codex adapter: `force_stop` kills the app-server and cancel is unchanged
- `t6` — acp adapter (kiro, qwen): `force_stop` kills the ACP process tree
- `t7` — agy adapter: warm cancel really stops the turn and `force_stop` kills
- `t8` — stream-json and http adapters: `force_stop` via `kill_tree` (claude, qwen-p, openai-compat)
- `t9` — daemon force-stop control message releases the run lock
- `t10` — daemon busy event for the owning shell or a dead-owner turn: steer/replace/exit
- `t11` — terminal key watcher module: cbreak hold, lone-Esc disambiguation, safe restore
- `t12` — pi adapter: `force_stop` kills the rpc process tree and the next run respawns
- `t13` — cross-adapter conformance: ignoring-harness stop and respawn for every family
- `t14` — panel stop state: first press stops politely, panel stays up, second press kills; Esc equals Ctrl+C
- `t15` — panel busy prompt: steer / replace / exit, steer only when steerable
- `t16` — client wiring: one-shot and daemon two-press stop
- `t17` — client wiring: busy prompt, declined exit code and stop audit
- `t18` — hook keeps the operator's exit status after declined and stopped runs
- `t19` — doctor detects a hung turn and --apply fixes it with owner authority
- `t20` — end-to-end success signal: stopping line within 1s, tree gone and prompt back within 3s
- `t21` — docs, harness prompts and version bump
- `t22` — agy cold mode: spawn in its own process group and kill the tree
- `t23` — normal close() kills the harness process tree on every persistent adapter, and tests stop leaking fake grandchildren

## Actual Delivery

All 23 tasks merged into `spec/reliable-agent-stop` with a `--no-ff` merge from an isolated worktree, the full suite green before and after each merge. PR [#16](https://github.com/agentculture/nvsh/pull/16) (0.13.0) carries the branch.

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `NVSH_FAKE_IGNORE_CANCEL` / `NVSH_FAKE_GRANDCHILD` modes in `tests/fakes/*`, `tests/test_fakes_ignore_cancel.py` (later extended by `d4`, `d5`, `d13`) |
| `t2` | delivered | `kill_tree()` and `start_new_session` spawn in `nvsh/agent/_subprocess.py`, default `NvshAgent.force_stop()` in `nvsh/agent/base.py` |
| `t3` | delivered | `EXIT_DECLINED = 3` in `nvsh/cli/_errors.py`, `AuditLog.record_stop()` in `nvsh/agent/audit.py` |
| `t4` | delivered | `Agent activity` section and `agent_activity` JSON in `nvsh/cli/_commands/overview.py`, no autostart |
| `t5` | delivered | codex `force_stop()` kills the app-server tree; next run reports `new session` |
| `t6` | delivered | ACP `force_stop()` denies dialogs, sends `session/cancel`, kills the tree; reader-queue reset bug fixed |
| `t7` | delivered | agy warm `cancel()` signals the child; `force_stop` respawns (with `d5`); review fix retires a cancelled warm process |
| `t8` | delivered | claude spawns in its own session; openai-compat shuts the socket down on stop; qwen-p unchanged (with `d4`) |
| `t9` | delivered | daemon `kill` control and `Daemon.kill_shell`; `client_transport.kill()` |
| `t10` | delivered | daemon `busy` event, `busy_choice` control, dead-owner detection, 60 s fallback to the queue |
| `t11` | delivered | `nvsh/keys.py` `KeyWatcher` and `read_choice_key` |
| `t12` | delivered | pi `force_stop()` kills the rpc tree and respawns with `new session` |
| `t13` | delivered | `test_ignoring_harness_stop_and_respawn` over 8 adapter variants (criterion 2 narrowed by `d6`) |
| `t14` | delivered | panel two-press stop state, feeder thread, Esc watcher (with `d1`, `d2`, `d3`) |
| `t15` | delivered | `Panel.show_busy` steer/replace/exit, steer only when steerable |
| `t16` | delivered | client `_StopTarget` routes presses to the in-process agent or the daemon (with `d8`) |
| `t17` | delivered | client busy handler, declined exit 3, stop audit (with `d9`–`d12`) |
| `t18` | delivered | hook `$?`/`PIPESTATUS` tests for client exits 3 and 130; no `hook.bash` change needed |
| `t19` | delivered | `agent_turn_not_hung` check and `nvsh doctor --apply`, `kill_active` control (review added turn identity) |
| `t20` | delivered | `tests/test_agent_stop.py`: 7 families x daemon/one-shot through a real pty and bash (with `d13`) |
| `t21` | delivered | docs (`docs/shell-integration.md`, `docs/daemon.md`, `README.md`), five harness prompt files, CHANGELOG and 0.13.0; `docs/current-spec.md` not hand-edited (it is a `devague today` projection) |
| `t22` | delivered | agy cold spawn in its own process group, `_terminate` uses `kill_tree` |
| `t23` | delivered | `escalate_close` reaps the harness process group after a graceful exit; test teardown kills fake pids; zero leaked `sleep 600` per suite run |

## Mid-work Decisions

- `d1` — panel non-tty fallback `_read_line_key` treated EOF on a binary stream (b'') as Enter = approve; t14 changed EOF to ignore — pre-existing safety bug found on a hung-up pty during t14; a hung-up terminal could approve a proposed command, violating propose-don't-run
- `d2` — a Ctrl+C/Esc press while `on_proposal` runs (e.g. an approved command executing) is acted on after `on_proposal` returns instead of ending the stream immediately — the SIGINT handler now records presses into a stop state handled by the main loop; the running command still receives SIGINT from the terminal
- `d3` — tests/`test_panel.py`::`test_ctrl_c_returns_to_a_prompt_within_one_second` replaced by `test_ctrl_c_says_stopping_within_one_second_and_a_second_press_returns_the_prompt` — the old test asserted one Ctrl+C ends the stream, contradicting amended c22 (panel stays up while stopping)
- `d4` — t8 edited tests/fakes/qwen (a t1 file outside t8's brief): `hang_if_ignoring_cancel` falls back to sleep when stdin is already at EOF (QwenAgent spawns with stdin=DEVNULL) — under the real adapter the fake exited immediately and orphaned its grandchild, so t8's acceptance test could not exercise `force_stop`; 5-line test-fixture fix
- `d5` — t7 extended tests/fakes/agy (a t1 file) with `NVSH_FAKE_SIGNAL_LOG`, `NVSH_FAKE_READY_FILE` and per-turn `sleep_before`, and fixed stale `_cancelled`/`_closed` flags in agy.py run()/`_run_warm`() — asserting the warm child received a stop needed a signal log and a readiness handshake (signal raced interpreter startup); the stale flags made the respawned turn return no events
- `d6` — t13 acceptance criterion 2 holds only for pi, codex and acp; for claude, qwen-p, openai-compat and agy cancel() already escalates, so removing `force_stop`() does not fail their cases — those adapters' cancel() already kills or closes; the stop-and-respawn contract (c3, c15) is still proven for all 8 variants; user accepted without code change
- `d7` — added task t22: agy cold mode spawns in its own process group and its terminate uses `kill_tree` — t13 found agy cold spawn (agy.py:256) lacks `start_new_session` and `_terminate` kills a single pid, so c27 was unmet for agy cold; user chose to fix in wave 4
- `d8` — one-shot teardown force-stops the adapter (kills the process tree) after any stop press, before close(), even if only the first polite press was made — with codex (and acp per its docstring) a polite cancel ends the turn and close() lets the harness exit on stdin EOF, orphaning tool grandchildren; the adapter drops its process handle in close(), so the tree kill must happen inside `one_shot`; only fires after a press
- `d9` — panel.stream gained an `on_busy` hook routing BUSY events through the key-watcher suspension (panel.py edited by t17) — BUSY events arrive on the feeder thread while the main thread reads keys; client.py cannot call `show_busy` safely without a panel hook
- `d10` — there is no 'nvsh ask' CLI verb; the declined exit code is observable as the return of client.ask()/`handle_failure` and slash --json `exit_code`, not as a direct 'nvsh ask' exit status — c9 named 'direct nvsh ask', which does not exist; t17 did not add a CLI verb
- `d11` — client.ask (/ask, Ctrl+G, @target) now puts proposals to the operator on the panel like the hook path, including auto-inspect and the steer follow-up turn; before, ask passed no approvals — without it 'ignore at a proposal' could never happen on the ask path, so c9's declined code was unreachable there
- `d12` — five existing tests (4 in tests/`test_client.py`, 1 in tests/`test_agent_demo.py`) now expect exit 3 instead of 0 after an ignored proposal; /fix and /explain (`_on_last_failure`) also return 3 when declined — c9 makes proposal ignore a declined run; consistency across ask entry points
- `d13` — t20 added `NVSH_FAKE_STALL_TURN`=1 to tests/fakes/pi, codex-app-server and acp (t1 files) so a turn stays open and silent without an approval prompt — without it the fake turns finish or park on an approval, where Ctrl+C cancels the key read and the two-press stop path is never reached
- `d14` — added task t23: normal close() kills the harness process tree on pi/codex/acp/agy, and test teardown stops leaking fake grandchildren — lapse l5: close() let persistent harnesses exit on stdin EOF and orphan their tool processes (tests leaked ~16 sleep 600 per run); user chose to fix adapters' close(), not only the tests
- PR #16 review (Qodo 1-8, SonarCloud) fixed after the waves, in commits `3fdbd08`, `41bc7e9` and `12da446` — no deviation record covers these; they are review fixes inside already-delivered tasks: doctor `--apply` exits healthy after a successful repair (t19); `_force_stop_turn` refuses a turn that is no longer active (t9/t10); `kill_active` carries the confirmed turn identity and refuses `changed` (t19, closes risk `r7`); only JSON `true` confirms (t19); oversized shell ids are treated as live (t10); Ctrl+C at a raw proposal or busy prompt stops the agent instead of declining (t14/t17); a cancelled warm agy process is retired and each stdout reader is bound to its own process and queue (t7); `SubprocessAgent` reaps its process group after a normal exit (t8/t23); SonarCloud complexity, typing and composite-assertion findings refactored without behaviour change.
- Branch naming: this run used `agent/stop/<id>` because `agent/t1`… already existed from an earlier plan; those old branches were left untouched.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t14` (`d1`) | pre-existing safety bug found on a hung-up pty during t14; a hung-up terminal could approve a proposed command, violating propose-don't-run | `acceptable` |
| `t14` (`d2`) | the SIGINT handler now records presses into a stop state handled by the main loop; the running command still receives SIGINT from the terminal | `needs-follow-up` |
| `t14` (`d3`) | the old test asserted one Ctrl+C ends the stream, contradicting amended c22 (panel stays up while stopping) | `acceptable` |
| `t8` (`d4`) | under the real adapter the fake exited immediately and orphaned its grandchild, so t8's acceptance test could not exercise `force_stop`; 5-line test-fixture fix | `acceptable` |
| `t7` (`d5`) | asserting the warm child received a stop needed a signal log and a readiness handshake (signal raced interpreter startup); the stale flags made the respawned turn return no events | `acceptable` |
| `t13` (`d6`) | those adapters' cancel() already kills or closes; the stop-and-respawn contract (c3, c15) is still proven for all 8 variants; user accepted without code change | `acceptable` |
| `t22` (`d7`) | t13 found agy cold spawn (agy.py:256) lacks `start_new_session` and `_terminate` kills a single pid, so c27 was unmet for agy cold; user chose to fix in wave 4 | `acceptable` |
| `t16` (`d8`) | with codex (and acp per its docstring) a polite cancel ends the turn and close() lets the harness exit on stdin EOF, orphaning tool grandchildren; the adapter drops its process handle in close(), so the tree kill must happen inside `one_shot`; only fires after a press | `risky` |
| `t17` (`d9`) | BUSY events arrive on the feeder thread while the main thread reads keys; client.py cannot call `show_busy` safely without a panel hook | `acceptable` |
| `t17` (`d10`) | c9 named 'direct nvsh ask', which does not exist; t17 did not add a CLI verb | `needs-follow-up` |
| `t17` (`d11`) | without it 'ignore at a proposal' could never happen on the ask path, so c9's declined code was unreachable there | `risky` |
| `t17` (`d12`) | c9 makes proposal ignore a declined run; consistency across ask entry points | `acceptable` |
| `t20` (`d13`) | without it the fake turns finish or park on an approval, where Ctrl+C cancels the key read and the two-press stop path is never reached | `acceptable` |
| `t23` (`d14`) | lapse l5: close() let persistent harnesses exit on stdin EOF and orphan their tool processes (tests leaked ~16 sleep 600 per run); user chose to fix adapters' close(), not only the tests | `risky` |
| `t19` | PR review: `kill_active` now requires the confirmed turn identity for a live-owner kill; a confirm-only request is refused | acceptable |
| `t14` / `t17` | PR review: Ctrl+C at a raw proposal or busy prompt is a stop press (exit 130), not a decline (exit 3) | acceptable |
| `t7` | PR review: a cancelled warm agy process is retired and respawned on the next turn, so the warm session does not survive a cancel | acceptable |

## Evidence

Validated by `/validate-delivery` at commit `0fc7d24` (2026-09-17): obligations `o1`–`o21` (claims c2–c12, c14, c15, c22, c26–c32), evidence `e1`–`e43`, deltas `b1`–`b10`, all filed `proposed` (llm origin). nvsh has no behavioral pytest marker or folder; evidence cites the named proving tests from PR #16's claim-to-test table.

- tests: full suite `uv run pytest -n auto` — pass (2057 passed, 5 skipped: live pi/agy/qwen smokes and the known cross-repo skip), run twice at `3d5d207`
- tests: the 41 proving tests for o1–o21 (82 cases) — pass at `0fc7d24`
- tests: `tests/test_agent_stop.py::test_two_presses_stop_every_family_on_both_paths` (14 cases, real pty + bash) — pass; with `_StopTarget.force_stop` made a no-op the pi/codex/acp cases fail on both paths (t20 sensitivity check)
- tests: `tests/test_client_stop.py::test_ctrl_c_at_a_raw_prompt_exits_130_cancels_once_and_never_declines` — pass (`e42`); `tests/test_agent_agy.py::test_cancelled_warm_turn_does_not_contaminate_next_run` — pass (`e43`)
- leak check: `ps -o etimes=,args= -u $(id -u)` filtered to `sleep 600` started during a full run — 0
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r nvsh` — clean; `teken cli doctor . --strict` — 26 passed; `markdownlint-cli2` on changed docs — 0 errors; `scripts/harness-smoke.py --stage config` — 6 passed; `scripts/scan-secrets.py` — clean
- CI on PR #16 (before the review fixes): lint, test (x2), test-publish, harness-smoke, version-check, GitGuardian, SonarCloud quality gate — pass
- commits: `dfb2661..15ebdaf` (main..branch at the time of writing)
- PRs / issues: #16; Qodo review threads 4030452481, 4030452490, 4030452496, 4030452511, 4030452526, 4030452537, 4030452549, 4030452562 — replied and resolved

## Delivery Claims

Lapses `l1`–`l5` are still proposed (not yet evidence); the claims they touch are capped below rather than rounded up. Nothing was run against real harnesses on the fleet, so every claim is about behaviour against the fake harnesses.

| Claim | Confidence | Evidence |
|-------|------------|----------|
| Ctrl+C/Esc stop a running agent on the daemon and one-shot paths, against fake harnesses that ignore cancel (c2, c5) | high | `e1`–`e3`, `e7`, `e29` · test `tests/test_agent_stop.py::test_two_presses_stop_every_family_on_both_paths` |
| The operator's originally reported Ctrl+C failure is fixed on real hardware | unverified | not reproduced on thor/orin/spark (risk `r4`) — not claimed done |
| Every adapter's stop ends the turn and the next run works (c3) | medium | `e4`, `e43` · capped: `l1` (codex polite interrupt untested), `l3` (agy SIGINT assumed, not observed on real agy) |
| First press is polite and keeps the panel up; second press kills once (c4, c22 timings 1 s / 3 s) | high | `e5`, `e6`, `e29`, `e42` · real force-kill exercised for pi, codex, acp only (`d6`) |
| Lone Esc vs arrow/function keys; terminal restored on SIGHUP/SIGTERM (c6, c28) | high | `e8`–`e10`, `e34`, `e35` |
| Busy prompt steer/replace/exit for the owning or dead-owner shell; other live shells untouched (c7, c8, c31) | medium | `e11`–`e16`, `e39`, `e40` · capped: `l4` (client busy flow tested against a stubbed daemon) |
| Declined exit code 3 via `--json`/audit; operator `$?` unchanged (c9, c10) | high | `e17`–`e21` · no `nvsh ask` verb (`d10`) |
| `nvsh overview` shows the active turn without autostarting the daemon (c11, c32) | high | `e22`, `e41` |
| `nvsh doctor --apply` clears a hung turn with owner authority and turn identity (c12) | high | `e23`–`e26` · commit `3fdbd08` |
| Kill releases the daemon run lock; process trees are killed on stop and on close (c26, c27) | high | `e30`–`e33` · leak check 0 |
| No stop path touches harness settings; ignore-cancel fakes per family (c14, c15) | high | `e27`, `e28` · `l2` pending (fakes first verified with piped stdin only, since fixed by `d4`) |
| Every stop kind is audited once (c30) | high | `e37`, `e38` |
| Persistent harnesses resume their conversation after a kill | unverified | not probed (risk `r3`, assumption c33 says it may be lost) |

Lapse ledger evidence:

pending approval (not yet evidence): `l1`, `l2`, `l3`, `l4`, `l5`

## Remaining Work / Follow-up

- Fleet check on thor, orin and spark (risks `r3`, `r4`, lapse `l3`): reproduce the original Ctrl+C failure, confirm first/second press against real pi, codex, agy and qwen/kiro ACP, observe real agy's SIGINT handling, and see whether sessions resume after a kill — operator, before calling c2 delivered.
- Risk `r6`: daemon runs on a non-default target register no active turn, so `kill` and the busy prompt cannot see them — follow-up issue.
- Risk `r8`: a warm harness that died and is respawned by the next `run()` before `close()` never has its old process group reaped — follow-up issue.
- Deviation `d2` (needs-follow-up): a press while an approved command runs is only acted on when the command returns — confirm in practice this is acceptable.
- Deviation `d10` (needs-follow-up): decide whether a real `nvsh ask` CLI verb is wanted.
- Adjudicate lapses `l1`–`l5`, and confirm or reject the proposed obligations `o1`–`o21`, evidence `e1`–`e43` and deltas `b1`–`b10` — human owner.
- Regenerate `docs/current-spec.md` with `devague today` once the evidence and deltas are adjudicated.
- CI and SonarCloud re-run on the review-fix commits; PR #16 awaits human merge (gate 3).

## Adjudication (2026-09-17)

Recorded after PR #16 was squash-merged to `main` as `d94f0ca` (0.13.0). The sections above are left as written on 2026-09-16; this section supersedes their "proposed" / "pending" wording.

- The operator confirmed every record: lapses `l1`–`l5`, obligations `o1`–`o21`, evidence `e1`–`e43` and deltas `b1`–`b10` are now `approved`. The lapses are evidence, so the confidence caps they impose on c3, c7/c8 and c14/c15 stand.
- `docs/current-spec.md` was regenerated with `devague today` from the approved ledger.
- Risk `r6` — accepted as a known limitation of 0.13.0: `kill`, the turn cap and the client-gone watch reach default-target turns only. No follow-up issue was requested.
- Risk `r8` — accepted as a follow-up, tracked as [#17](https://github.com/agentculture/nvsh/issues/17).
- Deviation `d2` — accepted as the intended behaviour: a press while an approved command runs is acted on when the command returns; the command itself still receives the terminal's SIGINT.
- Deviation `d10` — tracked as [#18](https://github.com/agentculture/nvsh/issues/18) (decide on a first-class `nvsh ask` verb).
- Still open: the fleet check on thor, orin and spark (risks `r3`, `r4`, lapse `l3`) — the operator will run and review it. The two `unverified` delivery claims stay unverified until then.
