# Delivery Summary — stop choice prompt

plan: `stop-choice-prompt` · run: `complete` · date: `2026-09-17`
baseline: `devague summary skeleton`

## Intent

> nvsh asks before it stops the agent. While the agent is working, the first Ctrl+C or Esc no longer cancels the turn straight away: the panel pauses and offers a choice -- \[t\] steer, \[s\] stop, \[Esc\] keep going -- so an accidental press costs nothing and the operator can correct the agent instead of killing it. Stopping is still always two keys away, and a terminal that cannot show the prompt stops immediately as before.

After: The first Ctrl+C or Esc pauses the panel and shows 'nvsh: paused -- \[t\] steer  \[s\] stop  \[Esc\] keep going'. Esc (or 30 s of silence) resumes as if nothing happened; s stops exactly as before, with a further press still killing; t takes one line and either steers the running turn (pi, codex) or stops and resends with the correction (every other harness).

## Planned Work

- `t1` — Declare Capabilities.steer on every adapter and expose a client-side lookup
- `t2` — Add a timed single-key reader for prompts in a new module
- `t3` — Add the `keep_going` stop kind and an origin field to the audit log
- `t4` — Daemon busy prompt reads Capabilities.steer instead of `_overrides_steer`
- `t5` — Panel: first press opens the choice prompt (keep going, stop, timeout, non-tty)
- `t6` — Panel: correction line, never-mind rules, press at another prompt, turn finished during the prompt
- `t7` — Client: wire the prompt -- label from the capability, redacted steer delivery, audit lines
- `t8` — Client: stop-and-correct sequencing, self-contained correction request, runtime fallback, exit codes
- `t9` — Rewrite the cross-family stop test and add an end-to-end pty test of all four outcomes
- `t10` — Update every written description of the first press, and record the amendment
- `t11` — Integrate: full suite, lint gates, version bump and changelog

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `Capabilities.steer` on every adapter (true only for pi and codex), `registry.steer_capable()`; merge `e63bfc2`. Sent back once at the merge gate: it claimed tests for two criteria it had only read (lapse `l2`); both tests were then added. |
| `t2` | delivered | new `nvsh/promptkeys.py` `read_choice()` and `tests/test_promptkeys.py`; merge `04e4b86`. Sent back once to delete a test that shelled out to `git diff` against `main`. Later changed by review fix `rv1` (`9910394`). |
| `t3` | delivered | `keep_going` stop kind, optional `origin` / `reason`, `correction` stored as `correction_chars` only; merge `bfca6d2`. |
| `t4` | delivered | `_overrides_steer` removed; the busy prompt reads `capabilities().steer`; merge `1d0be64`. |
| `t5` | partial | choice prompt, keep going, timeout, stop, no-tty path; merge `8b04eaf`. Criterion 4's `--json` case could not be met inside its files (`d2`) and was completed by `t6` and `t7`. |
| `t6` | delivered | correction line, never-mind rules, Ctrl+C at another prompt, `StreamResult.not_running`, plus `stop_prompt` (`d2`); merge `be5782e`. |
| `t7` | delivered | capability label, redacted steer delivery, one audit line per outcome, `--json` threaded through `nvsh/slash.py` and `nvsh/cli/_commands/slash.py` (`d2`); merge `c0812bf`. |
| `t8` | delivered | stop-and-correct sequencing, self-contained follow-up, runtime fallback, exit codes, plus panel seams `STOP_BEGUN` and `Panel.confirm()` (`d3`); merge `dae06d2`. Three of its tests depended on `pi` being installed and failed in CI; fixed in `d3c434b` (lapse `l4`). |
| `t9` | delivered | criterion 1 moved forward as task `t9a` (`d1`, merge `7e4386b`); the rest -- `tests/test_stop_prompt_e2e.py` with seven end-to-end pty tests and the paused-client test -- in merge `47fe7c2`. |
| `t10` | delivered | docs, README and the five harness prompt files; merge `0ebd388`. Written from the spec before the code existed, so `t11` added what the build decided later. |
| `t11` | delivered | every gate run by the main agent, docs brought up to date, version 0.14.0 and CHANGELOG; commit `9e1ffa4`. One gate run discarded a failing test's name (lapse `l3`). |

## Mid-work Decisions

- `d1` — The rewrite of tests/`test_agent_stop.py`::`test_two_presses_stop_every_family_on_both_paths` (press, s, press) is pulled forward from t9 (wave 5) and lands together with t5 in wave 1, as its own small task on its own branch; t9 keeps the end-to-end pty test, the paused-client test and the no-deleted-test check. — t5 changes the first press on a pty, so all 14 parametrizations of that test fail the moment t5 merges (verified by the main agent on the t5 branch: 14 failed, 2106 passed); the plan only rewrote it four waves later, which would have left the branch red through waves 2-5 and hidden any real regression among known failures. Operator approved 2026-09-17.
- `d2` — t5's criterion 4 is only partly met: nvsh/panel.py has no notion of JSON mode, so the prompt is gated on tty and TERM only. t6 adds an optional `stop_prompt`=True argument to Panel.stream (no prompt when False), and t7 has the client pass False under --json, so 'nvsh slash --json' on a real terminal stops at once as spec claim c9 requires. — The t5 agent found that neither nvsh/panel.py nor the Panel constructor calls in nvsh/slash.py and nvsh/client.py carry a JSON-mode flag, and was correctly forbidden from inventing one in files it does not own. Operator approved the panel-switch route on 2026-09-17 over amending c9.
- `d3` — t8 may add two small, additive public seams to nvsh/panel.py (with pty tests in tests/`test_panel_stop.py`), outside its confirmed file list: a way for the caller to tell the panel that a stop has begun, so the next Ctrl+C/Esc kills instead of re-opening the choice prompt, and a one-key yes/no prompt for 'could not steer -- stop and correct instead?'. No existing panel behaviour changes. — Reading the merged t5/t6/t7 code showed the panel/client seam was under-specified in the plan: `on_steer` returns only a bool, so a client-initiated cancel for stop-and-correct leaves `_StopState`.stopping False and 'a further press still kills' (t8 criterion 1) cannot hold; and nvsh/panel.py has no yes/no helper for t8 criterion 3's 'asks once'. Operator approved the additive route on 2026-09-17 over a client-only workaround.
- `d4` — A new task t8b, after t8, fixes plan risk r7: Panel.stream's finally block restores the previous SIGINT handler before the ticker join, the termios restore and the 'nvsh: interrupted' line, so a Ctrl+C landing during teardown raises KeyboardInterrupt and the operator sees a traceback. Files: nvsh/panel.py and tests/`test_panel_stop.py`; one pty test presses Ctrl+C during teardown. — Pre-existing since 0.13.0 and outside the confirmed plan, but found and measured by this run (t9a: 3 failures in 30 under load), in the file and feature area this plan owns, and most exposed on openai-compat, the fleet default, where cancel() already ends the turn and a double Ctrl+C lands in that window. Operator approved including it on 2026-09-17, sequenced after t8 so the two never edit nvsh/panel.py concurrently.

- Review findings were fixed on this branch before the PR was offered for review, as three further tasks not in the plan: `rv1` (merge `9910394`: the prompt now discards typeahead before it is drawn so a fast answer is never lost, and SIGHUP/SIGTERM during the prompt give the terminal back), `rv2` (merge `b7dc94c`: a correction typed after the turn finished becomes the next request; `_Routing` groups three `_stream_request` parameters) and `rv3` (merge `8055ae8`: composite test assertions split). No deviation record covers these -- they are review feedback on delivered work, captured here directly.
- `t1` and `t2` were each sent back once at the merge gate before being accepted (unproven acceptance criteria; a test that would have blocked future pull requests).
- `t9a` measured the flaky cross-family stop test instead of widening its timings and traced it to a real product bug (plan risk `r7`), which became `d4`.
- The feeder-queue snapshot is now taken under the queue's mutex (`db124ea`) rather than relying on the GIL; prompted by a SonarCloud finding the `rv1` agent had deliberately left open.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t9` (`d1`) | t5 changes the first press on a pty, so all 14 parametrizations of that test fail the moment t5 merges (verified by the main agent on the t5 branch: 14 failed, 2106 passed); the plan only rewrote it four waves later, which would have left the branch red through waves 2-5 and hidden any real regression among known failures. Operator approved 2026-09-17. | `acceptable` |
| `t5` (`d2`) | The t5 agent found that neither nvsh/panel.py nor the Panel constructor calls in nvsh/slash.py and nvsh/client.py carry a JSON-mode flag, and was correctly forbidden from inventing one in files it does not own. Operator approved the panel-switch route on 2026-09-17 over amending c9. | `acceptable` |
| `t8` (`d3`) | Reading the merged t5/t6/t7 code showed the panel/client seam was under-specified in the plan: `on_steer` returns only a bool, so a client-initiated cancel for stop-and-correct leaves `_StopState`.stopping False and 'a further press still kills' (t8 criterion 1) cannot hold; and nvsh/panel.py has no yes/no helper for t8 criterion 3's 'asks once'. Operator approved the additive route on 2026-09-17 over a client-only workaround. | `acceptable` |
| `t6` (`d4`) | Pre-existing since 0.13.0 and outside the confirmed plan, but found and measured by this run (t9a: 3 failures in 30 under load), in the file and feature area this plan owns, and most exposed on openai-compat, the fleet default, where cancel() already ends the turn and a double Ctrl+C lands in that window. Operator approved including it on 2026-09-17, sequenced after t8 so the two never edit nvsh/panel.py concurrently. | `acceptable` |

| `t1` | its first report claimed every acceptance criterion had a test when criterion 4 had only been read and criterion 1's `fake` case was unpinned; sent back, both tests added before merge -- no deviation record, lapse `l2` | acceptable |
| `t8` | three of its tests passed `agent="pi"` and so depended on a binary present on the build machine and absent in CI; caught by CI on the PR, fixed in `d3c434b` -- no deviation record, lapse `l4` | acceptable |
| `t10` | the docs were written in wave 0 from the spec alone, so the refused-steer confirm, redaction, never-mind, full exit-status rules and the teardown press were missing until `t11` added them | acceptable |
| `t11` | the plan asked for the full suite on an idle machine with any failing test named; one of 15 runs reported `1 failed` and the name was discarded by the gate command (lapse `l3`); later identified with reasonable but unproven confidence (plan risk `r9`) | needs-follow-up |

## Evidence

- tests: the 70 behavioural selectors filed as evidence `e1`-`e70` (142 cases) -- all pass at `db124ea`; see obligations `o1`-`o23` in `.devague/frames/stop-choice-prompt.json` and `.devague/deliveries/stop-choice-prompt.json`
- tests: `uv run pytest -n auto` -- 2204 passed, 5 skipped at `db124ea` (main: 2072 passed, 5 skipped); the same result with every harness binary (pi, codex, claude, qwen, kiro-cli, agy) hidden from `PATH`
- tests: `tests/test_agent_conformance.py::test_a_rejected_effort_surfaces_the_cli_stderr_tail[pi]` -- failed in 2 of roughly 25 full local runs, passes alone and in its file under load; predates this plan (risk `r9`)
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r nvsh`, `markdownlint-cli2`, `scripts/scan-secrets.py`, `teken cli doctor . --strict`, `scripts/harness-smoke.py --stage config` -- all clean
- no-diff checks against `main`: `nvsh/keys.py`, `tests/test_keys.py`, `.devague/frames/reliable-agent-stop.json`, and the bodies of `cancel_shell` / `kill_shell` / `_force_stop_turn` in `nvsh/daemon.py`
- CI on PR #23 at `db124ea`: lint, test (x2), test-publish, harness-smoke, version-check, GitGuardian, SonarCloud -- pass; SonarCloud 0 open issues (20 on the first analysis); 3 Qodo threads, all answered and resolved. The first CI run on the PR failed three tests (see `t8`)
- commits: `72f0f44..db124ea` (`main..spec/stop-choice-prompt` at the time of writing)
- PRs / issues: #23; upstream agentculture/devague#121

## Delivery Claims

Nothing in this run was executed against a real harness: every claim below is about behaviour against the fake harnesses on ptys. Approved lapse `l1` touched honesty condition `h4`, which was rejected and replaced by `h28` before the spec converged, so it caps nothing here. Lapses `l2`, `l3` and `l4` are still proposed and are not cited as evidence; the claims they bear on are worded conservatively anyway.

| Claim | Confidence | Evidence |
|-------|------------|----------|
| On a tty the first Ctrl+C/Esc opens the choice prompt within 1 s and nothing reaches the harness before a key (c2, c28) | high | `e1`, `e2`, `e63` · `tests/test_stop_prompt_e2e.py` asserts it from the fake harness's own record |
| `[s]` is byte-for-byte the old first press and a further press kills once, for every adapter family on both paths (c3) | high | `e3`-`e5`, `e64` |
| `[Esc]` and the timeout keep going with no event lost and exit 0 (c6, c19) | medium | `e12`-`e15` · capped: the 30 s value itself is only exercised with `STOP_PROMPT_TIMEOUT` patched short |
| `[t]` steers a steer-capable harness mid-turn (c4) | medium | `e6`, `e7` · capped: proven against a generated fake pi only; real pi and codex unprobed (risk `r1`) |
| Stop & correct cancels once, waits, and sends a self-contained follow-up for every adapter family (c18, c22, c23) | high | `e25`-`e31` |
| The typed correction is redacted before it leaves the process and the audit log keeps only its length (c29) | high | `e32`, `e24` |
| A refused steer asks once and never drops the text silently (c30) | high | `e33`-`e36` |
| A correction typed after the turn finished becomes the next request; `[s]` on a finished turn exits 0 (c36) | high | `e51`-`e55` · commit `b7dc94c` |
| Exit status rules (c20) | high | `e56`-`e60` |
| No tty, `TERM=dumb` and `--json` stop at once with no prompt text (c9, `d2`) | high | `e16`-`e20` |
| `Capabilities.steer` is true for exactly pi and codex, shows in `nvsh agent list --json`, and the busy prompt reads it (c5, c31) | high | `e8`-`e11`, `e37`-`e39` |
| Prompt keys: typeahead discarded before the prompt is drawn, a key typed after it never lost, terminal restored on SIGHUP/SIGTERM (c32, c33) | high | `e40`-`e48` · `e44`, `e47` are sensitivity-strength: the main agent saw them fail against the unfixed code |
| A Ctrl+C during panel teardown no longer raises (`d4`) | medium | `e69`, `e70` (sensitivity-strength) · capped: the system-level flake that revealed it (3 in 30) did not reproduce on the second measurement (0 in 30 before and after), so its real-world rate is unverified (`r7`) |
| A client paused at the prompt loses nothing and is not aborted as client-gone (c35) | medium | `e62`: 452 KB in flight against a 213 KB socket buffer · capped: measured against a fake, not a real reasoning stream (`r2`) |
| Key detection, the daemon's cancel/kill and adapter stop paths are unchanged (c7, c8) | high | `e65`-`e68` · no-diff checks above |
| Docs, README and the five prompt files describe the behaviour and record the amendment (c10, c11) | medium | file `docs/shell-integration.md` · `scripts/harness-smoke.py` pass · not backed by a behavioural test |
| The prompt is clearer for the operator and makes an accidental press free on real hardware (c1, c27) | unverified | not run on thor/orin/spark -- not claimed done |
| Real pi and codex act on a steer sent from the prompt | unverified | not probed (risks `r1`, `r8`) -- not claimed done |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `assumption-for-measurement` | Honesty condition h4 states steer text reaches the adapter 'after redaction'. No redaction exists on that path: client.`_redact` is called only for inspection output (nvsh/client.py:1607), and neither Responder.steer (`client_transport.py`:101-121), `_injector` (client.py:1156-1186) nor nvsh/agent/prompt.py redacts operator-typed text. The phrase was written from the project rule in CLAUDE.md, not from the code. |

pending approval (not yet evidence): `l2`, `l3`, `l4`

## Remaining Work / Follow-up

- Fleet check on thor and orin after 0.14.0 publishes (risks `r1`, `r2`, `r7`, `r8`): the three keys and the timeout on a real terminal over SSH, a real steer on pi and codex, stop & correct on openai-compat, a double Ctrl+C on openai-compat -- operator.
- Operator to adjudicate the proposed records: lapses `l2`, `l3`, `l4`; obligations `o1`-`o23`; evidence `e1`-`e70`; deltas `b1`-`b15`. Then regenerate `docs/current-spec.md` with `devague today` (it projects approved deltas only).
- `t11` follow-up (risk `r9`): open an issue for the intermittent `test_a_rejected_effort_surfaces_the_cli_stderr_tail[pi]`; likely the fake pi exits before its stderr is captured.
- Open an issue: text typed at `/ask`, `/steer` and the proposal prompt's `[t]` still leaves the process unredacted (risk `r3`); only the new correction line is redacted.
- Open an issue: `panel._read_key` (the proposal keypress) puts the tty in raw mode with only a `finally`, the same SIGHUP/SIGTERM exposure `rv1` fixed for the new prompt.
- Open an issue or check on the fleet: `PiAgent` sends a late `abort` after a delivered steer's turn has finished (risk `r8`).
- Upstream agentculture/devague#121: the coverage-boundary wording in `docs/current-spec.md`.
- PR #22 (0.13.2) is still open and edits `nvsh/agent/openai_compat.py`; whichever of #22 and #23 merges second needs a version and CHANGELOG rebase.
- PR #23 awaits human review and merge (gate 3).
