# Delivery Summary — first-class multi-harness with aliases

plan: `first-class-multi-harness-with-aliases` · run: `complete` · date: `2026-09-14`
baseline: `devague summary skeleton`

## Intent

> nvsh treats agy, claude, codex and qwen as first-class harnesses at Pi's
> quality bar (thinking, tool calls, approvals, streaming), with a
> configurable default backend/model/effort, an @backend/model/effort ad-hoc
> target, and operator-defined model and invocation aliases such as
> @reviewer and @local

After: An operator sets one 'default' alias (backend/model/effort) in
config.toml and every automatic failure call uses it with a warm session
where the harness supports one; typing '@claude/sonnet/medium why did this
fail?' or '@reviewer ...' or '@local ...' runs that target one-shot;
whichever of pi, agy, claude, codex or qwen answers, the panel shows its
thinking, tool calls and results, and any command it wants to run arrives as
a proposal gated by nvsh approve, or the harness runs read-only and says so.

## Planned Work

- `t1` — Shared subprocess hygiene: scrubbed child environment and redacted stderr tails for every adapter
- `t2` — Fixture hygiene: version-stamped, scrubbed recorded transcripts and a fixture-lint test
- `t4` — Extend the NvshAgent contract: THINKING event kind, request target, richer Capabilities
- `t5` — Config: per-harness effort/`extra_args`/approval keys, flat \[aliases\] table with reserved 'default', migration from \[agent\] provider
- `t6` — Python side of the @target grammar: backend\[/model\[/effort\]\] and alias marks in triggers.py and slash.py
- `t7` — Bash side of the @target grammar in readline.bash, the sync test, and the Tab-completion boundary doc
- `t8` — Registry: agy and ACP entries (kiro, qwen), forced-target choose(), path and local|hosted metadata
- `t9` — PiAgent: pass --thinking from effort and surface thinking deltas as THINKING
- `t10` — ClaudeAgent on the official CLI: model, effort, partial-message streaming, resume, thinking/`tool_use` mapping, host permission prompts as PROPOSAL
- `t11` — CodexAgent on app-server: JSON-RPC client with threads, turns, reasoning, requestApproval as PROPOSAL, on-request policy
- `t12` — AcpAgent: generic ACP client with kiro and qwen entries, plan-mode default, per-harness approval opt-in, bounded initialize
- `t13` — QwenAgent print-mode fallback on stream-json (no-ACP path)
- `t14` — AgyAgent: stream-json print mode with model/effort, warm --input-format session, read-only tool mapping
- `t15` — Cross-adapter conformance and policy invariants: thinking/tool/proposal/effort cases, auto-approve and settings-write bans, version stamps
- `t16` — Daemon and client: default alias resolution, target on the wire, version handshake, one-shot @targets, warm sessions, child containment
- `t17` — Doctor: per-harness reachability with version, auth state and allowlist warnings, no model calls
- `t18` — Panel and audit: THINKING rendering and target visibility
- `t19` — CLI and slash surfaces: agent list with path/hosted/capabilities, agent use writes the default alias, /agent and setup, catalog and invariants
- `t20` — Docs and prompt files: aliases, targets, harness paths, redaction boundary, drift rule, changelog and version bump
- `t21` — Live verification on spark, thor and orin recorded in docs/verification.md

1 task was rejected during planning (`t3`, a probe task the operator withdrew) — see `devague plan show`.

Waves (from `devague plan waves`): 1 = t1, t2, t4, t5 · 2 = t6, t8, t9, t10,
t11, t12, t13, t14, t18 · 3 = t7, t15, t16, t17, t19 · 4 = t20 · 5 = t21.
Workforce: Claude subagents in isolated worktrees (sonnet; opus for t10,
t11, t12, t15, t16), each merged by the main agent after a kiro-cli glm-5
read-only review and a green full suite before and after `git merge --no-ff`.

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `nvsh/agent/_env.py` `child_env()` drops `CLAUDECODE`/`CLAUDE_CODE_*`; `_subprocess.redacted_tail()`; merge 0dcc40d |
| `t2` | delivered | `tests/test_fixture_hygiene.py` (PII + `# recorded-from:` header rule), `tests/fixtures/README.md`; merge e4d6869 + fix e02f712 (lapse l2) |
| `t4` | delivered | `EventKind.THINKING`, frozen `Target`, `AgentRequest.target`, `Capabilities` +thinking/effort/path/approval/unmediated_file_access, request codec in base.py; merge ba996d1 |
| `t5` | delivered | `[agents.<name>]` effort/extra_args/approval keys, flat `[aliases]` with reserved `default`, `Config.resolve_target`, `set_provider` writes the default alias; merge 25d9de2 |
| `t6` | delivered | `triggers._parse_agent_mark` accepts alias and `backend/model/effort` marks; `slash.agent_mark_items()` palette default-first; merge b653616 (table rows live in `tests/test_triggers_prose.py`, d2) |
| `t7` | delivered | `readline.bash __nvsh_mark_line` mirrors the grammar; pty sync test imports the python table; `docs/shell-integration.md`; merge a6bfddd (d6) |
| `t8` | delivered | `ADAPTERS` gains agy, kiro, qwen(acp), `qwen-p`; `path`/`hosted` on every spec; `choose(forced=)`; merge 03664b4 + integration 0384d17, 9029346 |
| `t9` | delivered | `--thinking <effort>` verbatim, `thinking_delta` → THINKING, `docs/pi-rpc.md`; merge f93bc01 |
| `t10` | delivered | official `claude -p` stream-json with `--permission-prompt-tool stdio`; `control_request can_use_tool` → PROPOSAL, `respond_ui`; `--session-id`/`--resume`; recorded 2.1.270 transcript; merge 4400fe4 (risk r3 resolved) |
| `t11` | delivered | `codex app-server` JSON-RPC client (threads, turns, steer, interrupt, resume), `requestApproval` → PROPOSAL, `-c model_reasoning_effort=`, exec fallback; merge cb8c1f9 |
| `t12` | delivered | `nvsh/agent/acp.py` AcpAgent + `build()`; qwen plan-mode default, `approval = "harness"` opt-in, kiro `--model`; bypass modes refused; merge d6c1452 (d1 recorded) |
| `t13` | delivered | print-mode QwenAgent on stream-json, `--approval-mode plan`, `[status]` heuristic removed; registered as `qwen-p`, read-only; merge d0550f0 (d3) |
| `t14` | delivered | AgyAgent cold/warm, auto-deny → STATUS, `unmediated_file_access=True`, approval `none`; merge f72243e |
| `t15` | delivered | conformance parametrised over seven adapters (thinking/tool/proposal/effort/stderr/declarations); invariants: bypass tokens, settings-write AST ban, version stamps; merge 66eb4c6 + integration 08be225 (d8, d9, l3–l5) |
| `t16` | delivered | daemon resolves `default`, `_targeted()` one-shot, version handshake, `Target` on the wire, shared `escalate_close()` used by every adapter, `docs/daemon.md`; merge 1b32466 (d5, d10, l6–l8) |
| `t17` | delivered | per-harness reachability (version floor, kiro auth via `whoami`, hang = failed check), `check_agent_allowlist` (read-only); merge a1118b3 (d7) |
| `t18` | delivered | panel THINKING dim run + `harness/model/effort · path · warm \| one-shot` header; `AuditLog.record(target=)`; merge 9ecf903 |
| `t19` | delivered | `agent list` with installed/path/hosted/capabilities/default; `setup` and `agent use` write `[aliases].default`; `/agent` mirrors; catalog; merge 78c4fac |
| `t20` | delivered | four prompt files + `AGENTS.colleague.md`, architecture, README, config example, shell-integration; CHANGELOG `[0.10.0]`; version 0.9.2 → 0.10.0; merge ca82953 |
| `t21` | delivered | `docs/verification.md`: nvsh 0.10.0 on spark, thor, orin; every installed harness exercised; codex on thor recorded as not verified (not logged in); commit 59d1e71 (d11) |

Suite after the last merge: 1616 passed, 5 skipped (three live-gated, one
cross-repo gap, one agy live gate); black, isort, flake8, bandit, scan-secrets,
markdownlint, harness-smoke and `teken cli doctor --strict` clean.

## Mid-work Decisions

- Wave-2 constructor convention given to every adapter agent up front:
  `model`, `effort`, `extra_args`, `approval` kwargs; registry factories
  import agy/acp lazily so parallel tasks could not break the import graph.
- kiro-cli glm-5 (`kiro-cli chat --no-interactive --agent wf-glm5
  --trust-tools=`) reviewed every diff before merge; all 20 verdicts were
  MERGE except t20, which it BLOCKed because the docs diff alone did not show
  the code it describes — the code was already merged on the branch, so the
  merge proceeded.
- Two wave-2 integration commits closed seams between parallel tasks (qwen
  print-mode fallback read-only and `path="stream-json"`; registry builds
  qwen/kiro through `acp.build()`), and one wave-3 integration commit
  (doctor may *name* the settings files it reads; qwen-p declares
  `unmediated_file_access`).
- Live tests were run with a pty driver against real hooked shells, with
  every proposal answered `n`; operator configs on the three machines were
  backed up before the alias overlay was written.
- Deviations `d1`–`d11` and lapses `l2`–`l8` were recorded `--origin llm` at
  the moment each surfaced; all are **pending the operator's adjudication**
  (`devague deviate --confirm`, `devague lapse --confirm`).

## Drift From Plan

All entries are recorded deviations, currently `proposed`:

- `d1` (t12, acceptable) — qwen 0.23.3 sends `session/request_permission`
  in `default` mode (the brief said it never does); AcpAgent answers it via
  nvsh approve either way.
- `d2` (t6, acceptable) — grammar table rows live in
  `tests/test_triggers_prose.py`, not `tests/test_triggers.py`.
- `d3` (t13, acceptable) — the no-ACP qwen fallback is the explicit backend
  `qwen-p`, not an automatic registry pick.
- `d4` (t11, needs-follow-up) — for codex and print-mode qwen,
  `approval = "harness"` only changes what `capabilities()` reports.
- `d5` (t16, acceptable) — t16 added the shared `escalate_close()` helper and
  touched five adapters' `close()`.
- `d6` (t7, acceptable) — the bash/python sync test imports the table from
  `tests/test_triggers_prose.py`.
- `d7` (t17, needs-follow-up) — doctor detects auth state only for kiro;
  claude, codex, qwen and agy report "not verified"; their login remediation
  is a generic guess.
- `d8` (t15, needs-follow-up) — AcpAgent's stderr drain races an instant
  launch failure; conformance accepts the exit status alone for qwen/kiro.
- `d9` (t15, needs-follow-up) — ACP effort is filtered against advertised
  options rather than passed verbatim (tension with decision c24).
- `d10` (t16, acceptable) — for stream-json harnesses "one warm process"
  means one adapter + resumed session, not one OS process.
- `d11` (t21, acceptable) — `nvsh agent list --json` shows eight adapters,
  not six.

Plan risk `r7` (follow_up): one `nvsh.daemon --foreground` from
`tests/test_cli_setup.py::test_hook_prints_refresh_notice_once_per_session`
outlives a parallel suite run (900 s idle timeout), and
`test_autostart_waits_for_a_cold_backend_instead_of_falling_back` is
timing-flaky under xdist load.

## Evidence

- tests: `uv run --frozen pytest -n auto -q` at 59d1e71 — 1616 passed, 5 skipped;
  obligations `o1`–`o24` each carry an evidence record `e1`–`e26` naming the
  test node (see `devague evidence --list`); behavioral deltas `b1`–`b7`.
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r nvsh` — clean;
  `python3 scripts/scan-secrets.py` — clean (288 files); `markdownlint-cli2 "**/*.md" …` — 0 errors;
  `scripts/harness-smoke.py --stage config --require config` — 6 passed;
  `uv run --frozen teken cli doctor . --strict` — healthy.
- live: `docs/verification.md` (spark, thor, orin; nvsh 0.10.0 from the built wheel).
- commits: 3537832..59d1e71 on `spec/first-class-multi-harness-with-aliases` (20 task merges, 3 integration commits, 5 devague-state commits).
- PRs / issues: PR opened from this branch (see the PR body); issues #1, #2, #6 for the underlying spec.

## Delivery Claims

Evidence ids are `--origin llm` and `proposed`; confidence below is what the
named test or transcript supports, capped where a pending lapse or deviation
touches the claim.

| Claim | Confidence | Evidence |
|-------|------------|----------|
| pi passes `--thinking <effort>` verbatim and streams THINKING (c3) | high | e1; live: spark default panel in docs/verification.md |
| claude uses the official CLI with stdio permission prompts as PROPOSAL (c5, c54) | high | e2; t10 live allow/deny probe (not checked in); live spark/thor/orin @reviewer |
| codex app-server starts threads on-request, requestApproval → PROPOSAL (c6, c40) | high | e3; live spark @codex (tools + proposal), orin @codex |
| qwen ACP is plan-mode read-only unless `approval = "harness"` (c7, c53) | medium | e4; lapse l3 pending (no recorded plan-mode turn); d1 pending |
| agy runs read-only with auto-deny as STATUS (c8, c37) | high | e5; live spark @agy |
| one flat `[aliases]` table with reserved `default` resolves aliases and literals (c10, c26, c27) | high | e6; live: @reviewer/@local/@fast on all three machines |
| both @target parsers stay in lockstep; no Tab completion for @ (c13, c14, c33) | high | e7, e8 |
| the daemon builds its warm agent from the resolved default; @targets run one-shot (c15, c25, c30) | medium | e9, e10, e11, e26; d10 pending (stream-json warm = session, not process); l6 pending for codex |
| panel renders THINKING dimmed and shows the target header (c16, c48) | high | e12, e20; live headers on all three machines |
| doctor probes each harness's version and auth without a model call (c17, c51) | low | e13; d7 pending: only kiro's auth is actually probed |
| no bypass token is ever sent; settings files are never written (c22, c38, c39) | high | e14, e15 |
| every harness passes the shared conformance suite incl. verbatim effort (c32, c24) | medium | e16; d8, d9 pending (ACP stderr race, ACP effort filtered) |
| child env scrubbed of CLAUDECODE markers; stderr tails redacted (c42, c44) | high | e17; tests/test_agent_subprocess.py::test_stderr_tail_is_redacted_before_the_error_event |
| daemon/client version handshake; child containment on close/idle/uninstall (c45, c47) | medium | e18, e19; l7, l8 pending (cost not benchmarked; uninstall tested by mechanism) |
| setup / agent use write the default alias; agent list shows path+hosted (c11, c49) | high | e21, e23, e24 |
| fixtures scrubbed and version-stamped (c46, c50) | medium | e22; l5 pending (pi and kiro fakes carry no recorded-from stamp) |
| docs move together; identity axis untouched; version bumped (c19, c20) | high | e25; CHANGELOG `[0.10.0]` |
| live verified on spark, thor and orin for every installed harness (c29, c31) | high, with one gap | e26; codex on thor **not verified** (not logged in); pi and agy absent on the Jetsons |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `assumption-for-measurement` | The /think leg captured c22 stating that ACP session/`request_permission` gives qwen an approval channel on the strength of pi-acp's method list, without probing qwen --acp; the challenge probe showed qwen never sends it. (approved; superseded in practice by d1: qwen does send it in default mode) |

pending approval (not yet evidence): `l2`, `l3`, `l4`, `l5`, `l6`, `l7`, `l8`

## Remaining Work / Follow-up

- Operator adjudication of `d1`–`d11`, `l2`–`l8`, `o1`–`o24`, `e1`–`e26`, `b1`–`b7` (`devague deviate|lapse|oblige|evidence|delta --confirm`).
- `d7`: a model-free auth probe for claude, codex, qwen and agy (or a documented "login and retry" remediation per CLI) — owner: follow-up issue.
- `d9`: decide whether ACP effort should be sent verbatim (c24) or stay filtered; if verbatim, surface the harness's rejection as ERROR.
- `d8`: give AcpAgent the same "wait a beat for the stderr drain" that agy.py has, then unpin `STDERR_TAIL_RACE`.
- `d4`: either make `approval = "harness"` meaningful for codex (e.g. `approvalPolicy` untrusted) or reject it in config validation for harnesses without an approval UI.
- `r7`: stop the daemon spawned by `test_hook_prints_refresh_notice_once_per_session`; de-flake the cold-backend autostart test under xdist.
- `l5`: record a real `pi --mode rpc` and `kiro-cli acp` transcript with `# recorded-from:` headers.
- codex on thor: log in (`codex login`) and re-run the thor `@codex` line from docs/verification.md.
- The qwen print-mode process lingers ~26 s after its final `result` line before nvsh returns the prompt (observed on spark); investigate the exit path.
