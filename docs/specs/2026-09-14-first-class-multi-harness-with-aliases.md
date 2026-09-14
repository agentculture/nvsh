# first-class multi-harness with aliases

> nvsh treats agy, claude, codex and qwen as first-class harnesses at Pi's quality bar (thinking, tool calls, approvals, streaming), with a configurable default backend/model/effort, an @backend/model/effort ad-hoc target, and operator-defined model and invocation aliases such as @reviewer and @local
> instruction: Manual check on orin or thor after install; record the panel output in docs/verification.md.

## Audience

- Operators at an interactive bash prompt on Jetson, DGX Spark or RTX Spark who already have one or more of pi, agy, claude, codex, qwen or kiro installed and want nvsh to use the harness, model and effort they choose, per call or by default, without losing thinking, tool calls or approvals.
  - instruction: docs/config.example.toml + README section reviewed by the operator.

## Before → After

- Before: Today only pi is a real NvshAgent: claude/codex/qwen adapters return plain text with no thinking, tool calls, proposals, model or effort (nvsh/agent/claude.py, codex.py, qwen.py), agy does not exist, effort has no config key (nvsh/config.py:43), the daemon builds one backend from \[agent\] provider only, and @marks accept a bare adapter name (nvsh/triggers.py:368-376).
  - instruction: grep -rn 'THINKING\|effort\|agy' nvsh/ on main returns nothing.
- After: An operator sets one 'default' alias (backend/model/effort) in config.toml and every automatic failure call uses it with a warm session where the harness supports one; typing '@claude/sonnet/medium why did this fail?' or '@reviewer ...' or '@local ...' runs that target one-shot; whichever of pi, agy, claude, codex or qwen answers, the panel shows its thinking, tool calls and results, and any command it wants to run arrives as a proposal gated by nvsh approve, or the harness runs read-only and says so.
  - instruction: Manual verification on spark (pi + claude + agy present) recorded in docs/verification.md.

## Why it matters

- Operators on these machines already pay for or run several harnesses; nvsh must not make Pi the only good citizen (issue #6: 'Pi.dev is the default harness, not the architecture'), and a per-call target lets a cheap local model handle routine failures while a strong hosted model is one '@reviewer' away.
  - instruction: Manual check on spark: default alias = pi/associate, reviewer alias = claude/opus/medium; cause a failure, then '@reviewer explain the last failure'; record both panels in docs/verification.md.

## Requirements

- The Pi adapter itself is raised to the bar before others are measured against it: pi rpc '`thinking_`\*' assistantMessageEvent sub-types are currently ignored (docs/pi-rpc.md:92-95) and no effort flag is passed, although pi 0.85.1 --help exposes '--thinking <off|minimal|low|medium|high|xhigh|max>' and 'provider/id:`<thinking>`' model patterns; `build_argv` (pi.py:355-374) gains both.
  - instruction: tests/`test_pi_agent.py`: argv contains --thinking; a scripted `thinking_delta` yields EventKind.THINKING.
  - honesty: PiAgent.`build_argv` passes --thinking `<effort>` when \[agents.pi\] effort is set and maps `thinking_`\* deltas to THINKING events.
- The NvshAgent contract grows to carry the new signals: EventKind (base.py:24-33) gains a THINKING kind rendered distinctly from `TEXT_DELTA`; AgentRequest (base.py:60-74) gains a target (backend, model, effort) so a request can name its harness; Capabilities (base.py:173-181) gains 'thinking' and 'effort' booleans so unsupported features degrade explicitly as issue #6 requires.
  - instruction: tests for base.py codec: old dict without target still parses; THINKING survives the round trip.
  - honesty: EventKind.THINKING, AgentRequest.target and Capabilities.thinking/effort round-trip through `event_to_dict`/`event_from_dict` and `request_from_dict` without breaking older wire lines.
- ClaudeAgent (official 'claude -p --output-format stream-json') is the only Claude path: it passes --model and --effort from \[agents.claude\] (claude 2.1.270 --help: '--effort <low|medium|high|xhigh|max>'), uses --include-partial-messages for token streaming, --resume/--session-id for the warm-session and cold-resume paths, --permission-prompts host with a permission prompt tool nvsh hosts so tool calls become PROPOSALs, and maps thinking, `tool_use` and `tool_result` content parts to THINKING/`TOOL_CALL`/`TOOL_RESULT` instead of only type=assistant text parts (claude.py:38-56 today).
  - instruction: tests/fakes/claude replays a stream-json transcript with thinking and `tool_use` blocks; conformance suite asserts the kinds.
  - honesty: ClaudeAgent argv includes --model, --effort and --include-partial-messages from \[agents.claude\], and thinking/`tool_use`/`tool_result` content parts map to THINKING/`TOOL_CALL`/`TOOL_RESULT`.
- CodexAgent drives 'codex app-server' (JSON-RPC over stdio, codex-cli 0.147.0, schema from 'codex app-server generate-json-schema') instead of 'codex exec --json': initialize, thread/start, turn/start, turn/steer, turn/interrupt, thread/resume give a warm, steerable, cancellable session; item/reasoning/textDelta maps to THINKING, item/agentMessage/delta to `TEXT_DELTA`, item/commandExecution/\* to `TOOL_CALL`/`TOOL_RESULT`, and item/commandExecution/requestApproval (execCommandApproval/applyPatchApproval responses) is the approval channel that becomes a PROPOSAL gated by nvsh approve; model and effort are passed as '-c model=' and '-c `model_reasoning_effort`='. 'codex exec' remains only as the degraded fallback.
  - instruction: tests/fakes/codex replays reasoning + tool events; conformance suite asserts the kinds.
  - honesty: CodexAgent argv includes -m and -c `model_reasoning_effort`=`<effort>`, and codex reasoning and tool msg types map to THINKING/`TOOL_CALL`/`TOOL_RESULT`.
- Qwen is driven through its native 'qwen --acp' by the generic AcpAgent (thinking, tool calls, sessions verified live 2026-09-14) in 'plan' mode by default, with \[agents.qwen\] approval = "harness" switching to qwen's own 'default' mode as an explicit, audited opt-out of nvsh-mediated approval; the plain-text '-p' QwenAgent (qwen.py:21-36) is removed or reduced to a no-ACP fallback at plan time; auto/auto-edit/yolo are never set by nvsh.
  - instruction: tests/fakes/qwen switched to stream-json lines; grep for '\[status\] ' in qwen.py returns nothing.
  - honesty: QwenAgent runs with --output-format stream-json and --model, and its thinking/tool events map to THINKING/`TOOL_CALL`/`TOOL_RESULT`; the '\[status\] ' heuristic is deleted.
- A new AgyAgent adapter (Google Antigravity CLI, binary 'agy', 1.0.1 installed here as an aarch64 Go binary; never mentioned in this repo) is added as a SubprocessAgent: 'agy -p `<prompt>` --output-format stream-json' emits NDJSON events init/`step_update`(`text_delta`, usage.`thinking_tokens`)/result (probed 2026-09-14); it accepts --model and --effort, and --continue/--conversation `<id>` for resume; it has NO --append-system-prompt or --system-prompt flag ('flags provided but not defined'), so the brief is embedded in the prompt as codex does.
  - instruction: tests/fakes/agy replays the probed event stream; conformance suite green; a live 'agy' smoke on spark answers 'say OK'.
  - honesty: AgyAgent runs 'agy -p <brief+prompt> --output-format stream-json --model `<m>` --effort `<e>`', maps `step_update` `text_delta` to `TEXT_DELTA` and result to DONE/ERROR, and warms a session with --input-format stream-json using {"event":"user","message":{"role":"user","content":...}} lines.
- Config gains: 'effort' in `_VALID_AGENT_BACKEND_KEYS` (nvsh/config.py:43); a default target on \[agent\] (provider + model + effort, spelled either as three keys or one 'default = "pi/associate/medium"' string); a \[models\] table of model aliases (alias -> per-backend model id, needed because the same model is 'opus' to claude, 'Claude Opus 4.6 (Thinking)' to agy) and a \[targets\] table of invocation aliases (alias -> backend/model/effort, e.g. reviewer = "claude/opus-5/medium", local = "qwen/worker"); unknown keys stay loudly rejected (`_reject_unknown` config.py:183-188) and docs/config.example.toml + `default_toml`() (config.py:72-102) document them.
  - instruction: tests/`test_config.py`: round-trip save/load with \[aliases\]; unknown alias sub-key rejected; docs/config.example.toml shows default and reviewer.
  - honesty: config.py accepts 'effort' in \[agents.`<name>`\], a flat \[aliases\] table whose values are 'backend/model/effort' strings, and rejects unknown keys as before; `default_toml`() documents both.
- registry.ADAPTERS (nvsh/agent/registry.py:87-123) registers 'agy' with binary 'agy' so 'nvsh agent list', doctor's `agent_configured`, the @-mark palette (slash.`agent_mark_items`, slash.py:363-374) and the conformance suite all see it; registry.choose (registry.py:147-173) accepts a forced target instead of only configured-or-openai-compat fallback.
  - instruction: tests/`test_agent_registry.py`: forced target present -> that adapter; absent -> error naming the binary.
  - honesty: registry.ADAPTERS contains agy; registry.choose(config, forced=`<target>`) returns the forced adapter when installed and an explicit error (not the openai-compat fallback) when it is not.
- The @-mark grammar (deviation d23) is extended in BOTH parsers that must stay in lockstep — bash `__nvsh_mark_line` (nvsh/shell/readline.bash:140-159) and Python triggers.`_parse_agent_mark` (nvsh/triggers.py:368-376) — from a bare adapter name to '@`<target>`' where target is 'backend\[/model\[/effort\]\]' or a \[targets\] alias; '/' joins the allowed charset, alias names join the palette returned by 'nvsh complete --json' so @reviewer and @local resolve exactly like @qwen, and the rewrite target '/ask --agent `<name>`' (readline.bash:153, slash.`split_agent_flag` slash.py:109-127) carries the full target.
  - instruction: Table-driven rows in tests/`test_readline_bash.py` and tests/`test_slash.py` for the new grammar.
  - honesty: Both parsers accept '@backend\[/model\[/effort\]\] text' and '@`<alias>` text', rewrite to '/ask --agent `<target>` text', and reject unknown names; the bash/python sync test passes.
- The configured default target (backend/model/effort) is what the daemon builds its warm agent from (Daemon.`_build_agent` nvsh/daemon.py:651-667 via registry.choose) and what client one-shot runs use, so the default's effort and model are honored on every automatic failure call, not only on ad-hoc marks.
  - instruction: tests/`test_daemon.py`: a config with default = "claude/sonnet/medium" builds ClaudeAgent with those argv values.
  - honesty: Daemon.`_build_agent` resolves the 'default' alias to (backend, model, effort) and builds the warm agent from it; one-shot runs resolve the same way.
- The panel (nvsh/panel.py:542-588 `_render_event`) renders THINKING as a dimmed, clearly separated run that ends before the first `TEXT_DELTA`, under the same `NO_COLOR`/TERM=dumb/non-tty guards and hard-coded SGR rules as the rest of the panel, and `TOOL_CALL`/`TOOL_RESULT` rendering (panel.py:376-388, exitCode key sniffing) is checked against each harness's result shape rather than pi's alone.
  - instruction: tests/`test_panel.py`: snapshot for both modes.
  - honesty: Panel renders THINKING dimmed and closes the run before the first `TEXT_DELTA`; with `NO_COLOR` or TERM=dumb it renders plain text with a 'thinking:' prefix and no SGR.
- doctor's `agent_reachable` (nvsh/`doctor_checks.py`:402-443) gains a probe per harness instead of the pi/openai-compat-only dispatcher that warns 'has no reachability probe' for everything else (`doctor_checks.py`:415-422): binary present + version + authenticated state (agy prints 'Print mode: not authenticated'; claude/codex/qwen have equivalents), never a paid model call.
  - instruction: tests/`test_doctor_checks.py` per harness with fake binaries; no network access in the test.
  - honesty: doctor `agent_reachable` returns a real check for agy, claude, codex and qwen (binary, version, auth state) and never calls a model.
- Test and catalog contracts are extended together: scripted fakes 'tests/fakes/agy' and 'tests/fakes/acp' (the latter standing in for kiro, qwen --acp, claude-code-acp and pi-acp) plus ViaFake adapters registered in tests/`test_agent_conformance.py` ADAPTERS (lines 80-86); tests/`test_cli_agent.py` '`reports_all_five`' grows to the new adapter count; new verbs get nvsh/explain/catalog.py ENTRIES (catalog.py:887-936) and a line in tests/`test_invariants.py` `_VERBS` (invariants.py:152-168) so they are proven traceback-free with pi/tmux/fzf/spark absent from PATH.
  - instruction: uv run pytest -n auto green; uv run teken cli doctor . --strict green.
  - honesty: tests/fakes/agy exists, AgyAgentViaFake is in the conformance ADAPTERS list, `test_cli_agent` reports six adapters, and every new verb appears in catalog ENTRIES and `test_invariants`.`_VERBS`.
- Docs move together under the harness-smoke drift rule: CLAUDE.md, AGENTS.override.md, .pi/SYSTEM.md, QWEN.md (CLAUDE.md:246-250; AGENTS.colleague.md repeats the same Pi/Nemotron prose at lines 42-51), plus docs/architecture.md:153-157, README.md:52-54, docs/config.example.toml, docs/shell-integration.md (mark grammar) and CHANGELOG.md; the committed spec docs/specs/2026-09-13-...md records 'PiAgent as the default' at lines 31,35,49, so this frame's export is a new decision record, not an edit of that one.
  - instruction: uv run python scripts/harness-smoke.py --stage config --require config; grep each file for 'aliases'.
  - honesty: CLAUDE.md, AGENTS.override.md, .pi/SYSTEM.md, QWEN.md, AGENTS.colleague.md, docs/architecture.md and README.md describe the alias/target mechanism consistently and scripts/harness-smoke.py passes.
- 'Propose, don't run' holds for every harness through a real approval channel where one exists: codex app-server item/commandExecution/requestApproval (policy on-request), claude --permission-prompts host with nvsh's permission prompt tool, pi rpc via `pi_ext`/approval.ts, and ACP session/`request_permission` for agents that send it (kiro to be verified). Where no channel exists (qwen --acp 0.23.3 never sends `request_permission`; agy headless auto-denies commands) the harness runs read-only with `tool_calling`=False, and an operator may opt a harness into its own agent-side approval with \[agents.`<name>`\] approval = "harness", which is recorded in capabilities and the audit log. The harnesses' auto-approve switches (--dangerously-skip-permissions, yolo, danger-full-access, --full-auto, --trust-all-tools) are never passed.
  - instruction: grep the adapters for dangerously-skip-permissions, yolo, full-auto: zero hits; conformance case per harness.
  - honesty: For each harness the adapter either emits PROPOSAL for a command the model wants to run (gated by nvsh approve) or declares `tool_calling`=False and is launched in its read-only/plan mode; no auto-approve flag is ever in any argv.
- A generic AcpAgent adapter (nvsh/agent/acp.py) speaks ACP JSON-RPC over stdio to a configured command: initialize (with a bounded reply timeout), session/new, session/prompt, session/resume, session/`set_mode`; session/update chunks map `agent_thought_chunk` to THINKING, `agent_message_chunk` to `TEXT_DELTA`, `tool_call`/`tool_call_update` to `TOOL_CALL`/`TOOL_RESULT`; an agent's session/`request_permission` request becomes a PROPOSAL answered from nvsh approve, and an agent that never sends one (qwen 0.23.3, probed) is run in its read-only mode unless the operator opts in per \[agents.`<name>`\] approval = "harness". It registers as 'acp' with \[agents.acp\] command = \["`<argv>`"\] plus model/effort forwarded as ACP config options where exposed, and ships two named entries: kiro (command = \["kiro-cli","acp"\]) and qwen (command = \["qwen","--acp"\]).
  - instruction: tests/fakes/acp replays session/update chunks and one session/`request_permission`; run the conformance suite; then 'nvsh agent use acp' with \[agents.acp\] command = \["qwen","--acp"\] and cause a failure in a hooked shell.
  - honesty: AcpAgent passes the conformance suite against a scripted fake ACP agent (tests/fakes/acp) and against qwen --acp live on spark, with THINKING, `TOOL_CALL` and PROPOSAL events observed.
- AgyAgent: agy 1.0.1 headless auto-denies any tool needing the 'command' permission ('a tool required the "command" permission that headless mode cannot prompt for, so it was auto-denied. Add an allow-rule under permissions.allow in settings.json', probe 2026-09-14) while read tools such as `list_dir` run unprompted; its `step_update` 'tool' events carry `tool_name` and `tool_info`.parameters/output, which map to `TOOL_CALL`/`TOOL_RESULT`. Therefore agy is registered with `tool_calling`=False for commands (read-only), and nvsh never writes allow-rules into agy's settings.json.
  - instruction: tests/fakes/agy replays the probed tool turn; grep the adapter for settings.json: zero writes.
  - honesty: AgyAgent reports `tool_calling`=False for command execution, maps 'tool' `step_updates` to `TOOL_CALL`/`TOOL_RESULT`, and never writes to agy's settings.json.
- Harness-side persistent auto-approval must not silently bypass nvsh approve: on this box agy's settings.json already carries 18 permissions.allow command rules (e.g. command(uv), command(git commit)) that make agy run those commands without any prompt, and claude (settings.json permissions.allow), codex (config.toml `approval_policy`/`sandbox_mode`) and qwen/kiro (--trust-tools, trustedWorkspaces) have equivalents. Each adapter launches with the strictest per-turn policy its protocol offers (codex app-server AskForApproval 'untrusted' or 'on-request' plus sandbox read-only; claude --permission-mode default and --permission-prompts host; qwen ACP `set_mode`; kiro without --trust-all-tools), and doctor reports any harness-side allowlist it can read as a warning naming the file, so the operator knows which commands would run unmediated.
  - instruction: tests per adapter assert the policy flags; tests/`test_doctor_checks.py` with a fixture settings.json containing permissions.allow.
  - honesty: Each adapter's argv/protocol policy is the strictest its harness offers, and doctor emits a warning naming any harness-side allowlist file it finds.
- CodexAgent starts every app-server thread with approvalPolicy 'on-request' (schema AskForApproval enum: untrusted | on-request | never | granular, from 'codex app-server generate-json-schema') and sandbox read-only, so item/commandExecution/requestApproval fires for commands and nvsh approve answers execCommandApproval; 'never' and danger-full-access are never sent.
  - instruction: tests/fakes/codex-app-server replays item/commandExecution/requestApproval; argv/initialize params asserted.
  - honesty: Every codex app-server thread is started with approvalPolicy on-request and sandbox read-only, and a requestApproval round-trips into a PROPOSAL.
- The stderr tail surfaced on a non-zero exit is redacted for every adapter, not only pi: nvsh/agent/`_subprocess.py` keeps a raw tail (lines 37, 97-106) while pi.py:466-470 redacts; SubprocessAgent, AcpAgent and the app-server client all pass their tails and any error payloads through nvsh.redact before an ERROR event or log line carries them.
  - instruction: Parametrised conformance case over all adapters.
  - honesty: A scripted fake that prints an `HF_TOKEN` on stderr and exits 1 yields an ERROR event whose text carries the redacted form for every adapter.
- Every adapter launches its harness with CLAUDECODE and `CLAUDE_CODE_`\* scrubbed from the child environment (and the equivalent nesting markers of other harnesses when known), because a shell opened from inside a Claude Code session inherits them and the claude harness then refuses to start (probe 2026-09-14: session/new failed 'Query closed before response received' until the vars were unset).
  - instruction: Unit test on the env builder shared by SubprocessAgent, AcpAgent and the app-server client.
  - honesty: Child environments passed to every harness lack CLAUDECODE and `CLAUDE_CODE_`\* keys.
- The daemon wire gains a version handshake: the client sends its nvsh version with each request and the daemon answers with its own (grep for version/protocol in nvsh/daemon.py and nvsh/`client_transport.py` finds none today); on mismatch after an upgrade the client stops the stale daemon and starts a new one instead of sending a 'target' the old daemon ignores; `request_from_dict` keeps accepting the old shape.
  - instruction: tests/`test_daemon.py` mismatch case; codec test for the old shape.
  - honesty: A client whose version differs from the running daemon's restarts the daemon and the request succeeds; an old-shape request dict still parses.
- Harness CLI versions are negotiated and recorded, not assumed: ACP initialize (protocolVersion, agentInfo.version) and codex app-server initialize results are logged and checked against the adapter's supported range; agy/claude/qwen/kiro versions are read at doctor time and written into the audit log; each tests/fakes transcript is stamped with the CLI version it was recorded from (agy 1.0.1, codex 0.147.0, qwen 0.23.3, claude 2.1.270, kiro-cli 2.0.0), so a drifted CLI fails loudly in doctor instead of silently mis-parsing.
  - instruction: tests/`test_doctor_checks.py`; a fixture-lint test reads the headers.
  - honesty: doctor prints each harness's version and the adapter's supported range; every fixture file header names the CLI version it was recorded from.
- Process containment: ACP agents and codex app-server run as direct children of the nvsh daemon or one-shot client (never via 'codex app-server daemon', which is a user-level daemon outside nvsh's lifetime); close() applies pi's wait/terminate/kill escalation (pi.py:815-830) to all of them, and 'nvsh uninstall' and daemon idle-exit leave no harness process behind (verified with pgrep in the tests).
  - instruction: tests with fake long-running binaries assert the process table.
  - honesty: After 'nvsh uninstall' and after daemon idle-exit, pgrep finds no qwen/agy/codex/kiro/claude-code-acp child started by nvsh.
- Observability: every request's resolved target (harness, model, effort, path acp|app-server|stream-json|rpc, warm|one-shot) is written to the audit log (nvsh/agent/audit.py) and shown in the panel header line, and 'nvsh context --show-context' prints it, so an operator can tell after the fact which model answered and what it was allowed to do.
  - instruction: tests/`test_panel.py` header snapshot; tests on nvsh/agent/audit.py.
  - honesty: The audit log line and the panel header for a request name harness, model, effort, path and warm|one-shot.
- 'nvsh setup' and 'nvsh agent use `<name>`' (nvsh/cli/`_commands`/setup.py, agent.py:56-81, and the /agent use slash command) write the 'default' alias instead of \[agent\] provider, and an existing config with only \[agent\] provider is read as default = "`<provider>`" with the backend's configured model and no effort, so upgraded installs keep working without editing config.toml.
  - instruction: tests/`test_config.py` migration case; tests/`test_cli_agent.py` use-writes-alias case.
  - honesty: After upgrade, a config.toml containing only \[agent\] provider = 'pi' resolves default = pi with the pi model and no effort; 'nvsh agent use claude' writes \[aliases\] default.
- Recorded fixtures under tests/fakes and tests/fixtures are scrubbed before commit: no absolute home paths (steward portability check fails on ~/ and /home/`<user>`/), no conversation/session ids, no account e-mails or tokens (agy's init event and codex thread ids carry them), enforced by scripts/scan-secrets.py and a fixture-lint test.
  - instruction: Run the test against a deliberately dirty fixture in a temp dir.
  - honesty: The fixture-lint test fails on any /home/`<user>`, ~/, e-mail or 32-hex id inside tests/fakes and tests/fixtures.
- doctor's per-harness reachability check reports auth state without a model call: kiro-cli 2.0.0 on this box answers 'You are not logged in, please log in with kiro-cli login' and 'kiro-cli acp' then prints nothing on stdout (probe 2026-09-14), agy prints 'Print mode: not authenticated', so an unauthenticated harness is a doctor failure with the login command as remediation, never a hung panel.
  - instruction: tests/`test_doctor_checks.py` with a fake that prints the not-logged-in text; AcpAgent timeout test.
  - honesty: An unauthenticated harness makes doctor `agent_reachable` fail with the login command as remediation, and an ACP initialize that gets no reply within the timeout surfaces as ERROR, not a hang.

## Honesty conditions

- A failure on a Jetson with only the default alias configured is answered by the default harness with visible thinking, and '@reviewer' answers from the aliased harness/model/effort, both verified in a hooked shell.
- No new Tab completion is attempted for '@' words; docs/shell-integration.md states the limitation for the extended grammar.
- culture.yaml, .claude/skills and .pi/skills are byte-identical to main after the change.
- With no config.toml present, nvsh agent list --json shows pi as configured and the daemon builds PiAgent.
- Each \[agents.`<name>`\] table accepts model, effort and `extra_args`, and the adapter passes exactly those; unsupported ones are reported in capabilities, never silently dropped.
- Effort strings reach the harness argv unchanged (e.g. 'xhigh' for claude, 'medium' for codex), and a harness-rejected value surfaces as an ERROR event with the CLI's stderr tail.
- An '@target' call never touches the warm daemon session, and the default target's session survives across two consecutive failures for claude, qwen and agy, and resumes by id for codex.
- \[aliases\] is flat: alias = "backend/model/effort" with model and effort optional; 'default' is reserved and required when \[agent\] provider is absent.
- No YAML import or dependency appears in nvsh/ or pyproject.toml.
- The audience can express any of: default alias, per-harness flags, @target, @alias, without reading source.
- main today has no THINKING kind, no effort key and no agy adapter.
- The after-state holds end to end on one hooked shell with two harnesses installed.
- A routine failure answered by the local default and a hard one escalated with '@reviewer' to a hosted model both work from the same hooked shell, so the operator never has to leave the prompt to change harness.
- The conformance suite is parametrised over five adapters and the thinking/tool-call/proposal/effort cases are present for each.
- The parser tables contain the positive and negative target rows named in the claim.
- nvsh agent list --json shows, per harness, which path is active (acp | app-server | stream-json | rpc) and why the fallback was taken when the ACP adapter binary is absent.
- No code path in nvsh/ opens a harness settings or trust file for writing.
- Harnesses with their own file tools are marked 'unmediated file access' in capabilities and the README states the redaction boundary.

## Success signals

- Every registered harness passes the same conformance suite (tests/`test_agent_conformance.py`) extended with thinking, tool-call, proposal and effort cases: 6 of 6 harnesses (pi, agy, claude, codex, qwen, kiro) emit THINKING and `TOOL_CALL` from their scripted fakes (the ACP-driven ones through one scripted fake ACP agent), and 'nvsh agent list --json' reports capabilities with 0 harnesses claiming `tool_calling`=True that cannot route a tool call through nvsh approve.
  - instruction: pytest -k conformance --collect-only lists them.
- A one-line '@`<target>`' or alias mark resolves to the intended backend/model/effort 100% of the time in the table-driven tests for both parsers (tests/`test_readline_bash.py` and tests/`test_slash.py`), including '@claude/sonnet/medium', '@reviewer', '@local' and the negative cases (unknown alias, email-like '@' in argument position).
  - instruction: grep the two test files for '@claude/sonnet/medium' and '@reviewer'.

## Scope / boundaries

- Tab completion of @-targets stays unavailable: bash's hostname completion claims any first word starting with '@' before programmable completion runs (readline.bash:217-220, docs/shell-integration.md:201-204), so @reviewer is Enter-only like @qwen today; '/ask --agent `<target>`' remains the completable spelling.
  - instruction: Doc check plus an existing readline test that '@' words are not offered by `__nvsh_complete_initial`.
- The mesh identity axis is untouched: culture.yaml 'backend: claude' and the four harness prompt files select the resident/interactive harness (docs/harness-selection.md:37-101) and tests/`test_invariants.py`:49-71 freezes culture.yaml, .claude/skills and .pi/skills against main; the default/alias/target mechanism lives only on the NvshAgent runtime axis (nvsh/agent/\*, config.toml).
  - instruction: tests/`test_invariants.py`::`test_mesh_identity_files_match_main` passes.
- Pi stays the shipped default when nothing is configured (issue #6 acceptance: 'Pi.dev is the default agent when no adapter is explicitly configured'; `_DEFAULT_AGENTS` pi/nemotron/associate at config.py:51-53); 'choose a default agent, mode and effort' therefore means a configurable default triple whose shipped value is pi/associate plus a chosen effort, and any cloud harness (agy, claude, codex) is opt-in because Jetsons are often air-gapped (CLAUDE.md 'Headless over SSH').
  - instruction: tests/`test_agent_registry.py` existing default test still passes unchanged.
- nvsh never edits, creates or overrides a harness's own settings or trust files (agy/claude settings.json, codex config.toml, kiro trust settings, qwen settings): it only passes launch flags and protocol-level policy, and reports what it finds.
  - instruction: grep -rn 'settings.json\|config.toml\|trust' nvsh/ shows reads only; a test asserts the files are unchanged after a full run.
- Redaction covers only what nvsh sends (capture.py `redact_report` is the single choke point for captured output, plus the device context); a harness with its own file-reading tools (agy `list_dir`/`view_file` ran unprompted in the probe; ACP agents fall back to their own fs when nvsh advertises fs.readTextFile=false) can read .env, tokens and keys from disk directly, outside nvsh's redaction. The spec states this as an accepted, documented boundary per harness, and capabilities/doctor mark such harnesses 'unmediated file access'.
  - instruction: nvsh agent list --json shows the flag for agy, kiro, claude-acp; README section reviewed.

## Non-goals

- No new install paths: installers.TOOLS stays {pi, node, uv, tmux} (nvsh/installers.py:151-180, pinned by tests/`test_installers.py`:44) and 'nvsh agent install' keeps pi as its only target; agy, claude, codex and qwen remain detect-only (PATH presence via registry.installed, registry.py:126-131), with doctor telling the operator what is missing.

## Assumptions

- The Pi quality bar is concrete: the AgentEvent kinds `TEXT_DELTA`, `TOOL_CALL`, `TOOL_RESULT`, PROPOSAL, STATUS, DONE, ERROR (nvsh/agent/base.py:24-33) plus cancel and persistent session as declared by Capabilities (pi.py:853-860); today PiAgent maps pi rpc events to those kinds (pi.py:679-722) and the approval extension `pi_ext`/approval.ts routes every bash `tool_call` through nvsh approve.
- agy's stream-json shapes are now verified for text, thinking-token usage and tool turns (`step_update` `step_type` 'tool' with `tool_name`, `tool_info`.parameters and `tool_info`.output); its permission behaviour in print mode is auto-deny for command tools with no runtime prompt (see the agy requirement seeded by the challenge pass), so no PROPOSAL can be derived from agy today.
- Automatic failure calls can now hit a hosted, paid harness whenever the 'default' alias names one; the existing trigger rate limit (nvsh/triggers.py) is the only cost bound. Operators on air-gapped Jetsons keep pi/local by default (boundary c21), and 'nvsh agent list' marks each harness local|hosted so a hosted default is a visible choice, not a surprise.

## Scope exploration

- `s1` — `nvsh/agent/pi.py + nvsh/agent/pi_ext/approval.ts + docs/pi-rpc.md`: Pi is the reference: text deltas, `tool_execution_start`/end, `extension_ui_request` proposals, cancel, `new_session`/`switch_session` all mapped (pi.py:679-722, 777-813); the approval extension forwards every bash `tool_call` to 'nvsh approve check' (approval.ts:38-56).
  - seeds: `c2`
- `s2` — `pi 0.85.1 --help (--thinking, --model provider/id:<thinking>) vs pi.py build_argv`: Pi has a native effort knob nvsh never passes, and drops thinking deltas on the floor; 'same quality as pi' must first mean Pi shows thinking and honors effort.
  - seeds: `c3`
- `s3` — `nvsh/agent/base.py (EventKind, AgentRequest, Capabilities) + nvsh/daemon.py request_from_dict:237-253`: No thinking event kind exists, AgentRequest carries only kind/prompt/command/`exit_code`/`failure_id`/ask, and Capabilities has five booleans; every new signal must be added here first because all adapters, the daemon wire codec (`event_to_dict` base.py:106-170) and the panel key off these types.
  - seeds: `c4`
- `s4` — `nvsh/agent/claude.py + claude 2.1.270 --help`: Adapter hard-codes 'claude -p `<prompt>` --append-system-prompt `<brief>` --output-format stream-json' (claude.py:26-36) and drops every non-text content part; the installed CLI already offers model, effort, partial-message streaming, resume and permission-prompt routing.
  - seeds: `c5`
- `s5` — `nvsh/agent/codex.py + codex-cli 0.147.0 exec --help`: Adapter runs 'codex exec --json `<prompt>`' with the brief embedded in the prompt (no system-prompt flag, codex.py:26-31) and ignores every msg.type except four; -m and -c `model_reasoning_effort`=... exist on the installed CLI.
  - seeds: `c6`
- `s6` — `nvsh/agent/qwen.py + qwen 0.23.3 --help`: Qwen is the most degraded adapter (no structured protocol at all) although the installed CLI has stream-json output, --model, --approval-mode {plan,default,auto-edit,auto,yolo} and session resume.
  - seeds: `c7`
- `s7` — `agy 1.0.1 binary ($HOME/.local/bin/agy): --help, 'agy -p --output-format stream-json' probe, --model/--effort error text`: agy's --help hides --output-format/--model/--effort but all three work; model names are display strings with the effort baked in ('Gemini 3.8 Flash (High)', 'Claude Opus 4.6 (Thinking)', 'GPT-OSS 120B (Medium)'); it needs Google sign-in ('Print mode: not authenticated'), so `local_model`=False and it is never an offline default.
  - seeds: `c8`
- `s8` — `agy stream-json init event (tools list, permission_mode) from the 2026-09-14 probe`: Tool-call and approval event shapes are unknown for agy; the text turn proves streaming and thinking-token accounting only.
  - seeds: `c9`
- `s9` — `nvsh/config.py + docs/config.example.toml + tests/test_config.py`: Only \[agent\] provider, \[agents.`<name>`\] provider/model/`base_url`/`api_key_env`/`api_key_file`, \[sessions\] max and \[triggers\] exist; no effort, alias, profile or mode concept anywhere; unknown top-level keys are rejected, so new tables must be added to `_VALID_TOP_KEYS` (config.py:33).
  - seeds: `c10`
- `s10` — `nvsh/agent/registry.py (ADAPTERS, choose, install_offer)`: Registration is the single source of the harness palette; choose() has no forced-target path; `install_offer` returns None for anything but pi.
  - seeds: `c11`
- `s11` — `nvsh/installers.py + tests/test_installers.py:44`: Install offers are pi-only by design and test-pinned; parity of runtime quality does not require parity of installation.
  - seeds: `c12`
- `s12` — `nvsh/shell/readline.bash:126-163 + nvsh/triggers.py:300-376 + nvsh/slash.py:103-141,363-374 + docs/shell-integration.md:105-149`: '@name question' already rewrites to '/ask --agent name' at readline time; the grammar allows \[A-Za-z\]\[A-Za-z0-`9_`-\]\* only and requires a registered adapter name, so '@claude/sonnet/medium' and '@reviewer' currently fall through as ordinary lines; the rules are implemented twice and tested for sync (tests/`test_readline_bash.py`, tests/`test_slash.py`).
  - seeds: `c13`
- `s13` — `nvsh/shell/readline.bash:217-238 + docs/shell-integration.md:201-204`: A documented bash limitation, not an nvsh bug; the new syntax inherits it unless a different sigil is chosen.
  - seeds: `c14`
- `s14` — `nvsh/daemon.py:651-765 (_build_agent, _acquire, _activate) + nvsh/client.py:423-449,595-617 + nvsh/client_transport.py:232-278`: One agent process per daemon lifetime bound to the daemon's Config; a per-request '@name' override deliberately forces one-shot because the warm daemon would answer from the default backend (client.py:603-608); AgentRequest carries no target on the wire.
  - seeds: `c15`
- `s15` — `nvsh/panel.py:195-217,376-388,542-588`: Seven kinds rendered, no thinking path; tool result exit-code sniffing scans exitCode/`exit_code`/exit/returncode (pi spells it exitCode).
  - seeds: `c16`
- `s16` — `nvsh/doctor_checks.py:102-127,255-443`: `agent_configured` is generic; `agent_reachable` is pi- and openai-compat-specific with a warning stub for claude/codex/qwen.
  - seeds: `c17`
- `s17` — `tests/test_agent_conformance.py + tests/fakes/* + tests/test_cli_agent.py + nvsh/explain/catalog.py + tests/test_invariants.py`: Every adapter must pass streaming-order, cancel-mid-stream, error-propagation, capability-report and teardown via a scripted fake binary; `_VERBS` and ENTRIES are hand-maintained lists a new verb must be added to.
  - seeds: `c18`
- `s18` — `docs/harness-selection.md + culture.yaml + tests/test_invariants.py:49-71`: Two different 'backend' notions exist; only the NvshAgent runtime one is in play here.
  - seeds: `c19`
- `s19` — `CLAUDE.md, AGENTS.override.md, .pi/SYSTEM.md, QWEN.md, AGENTS.colleague.md, docs/architecture.md, README.md, docs/specs/2026-09-13-nvsh-bash-hook-agent-on-error.md`: Default-backend prose is duplicated in at least eight committed files, four of them CI-gated for drift.
  - seeds: `c20`
- `s20` — `gh issue 6 (acceptance criteria) + nvsh/config.py:51-53 + CLAUDE.md hook constraints`: Issue #6 is still OPEN and its acceptance list pins Pi as the zero-config default and demands discoverable capabilities with explicit degradation.
  - seeds: `c21`
- `s21` — `claude/qwen/codex/agy headless permission flags (from --help and the agy init event) vs nvsh/agent/loop.py:50-68 + nvsh/agent/audit.py`: Today claude/codex/qwen never emit PROPOSAL, so loop.`run_loop`'s approval gate and the audit trail are dead paths for them; each CLI has a headless permission mechanism but whether it can block a tool call and hand the decision to nvsh (as approval.ts does) is unproven.
  - seeds: `c22`
- `s22` — `pyproject.toml dependencies = [] + nvsh/config.py tomllib`: No YAML parser is available without adding a dependency; the operator said 'yaml' for aliases and this needs an explicit yes/no on TOML.
  - seeds: `c27`
- `s23` — `claude/qwen --input-format stream-json, agy --input-format stream-json probe (2026-09-14), codex exec resume / app-server`: Warm-session support verified per harness from --help and a live agy probe: claude, qwen and agy can hold one process across turns; codex cannot in exec mode.
  - seeds: `c25`
- `s24` — `codex app-server protocol (codex-cli 0.147.0; 'codex app-server generate-json-schema' -> scratchpad/codex-schema, 20k-line v2 schema)`: app-server is a full Pi-equivalent surface: thread/start|resume|fork, turn/start|steer|interrupt, item/reasoning/textDelta + summaryTextDelta, item/agentMessage/delta, item/commandExecution/outputDelta + requestApproval, item/fileChange/\*, execCommandApproval and applyPatchApproval request/response, thread/tokenUsage/updated; marked \[experimental\] in --help, and 'codex app-server daemon' can host it once per user.
  - seeds: `c6`
- `s25` — `ACP availability: qwen 0.23.3 --acp (native), pi-acp 0.0.33 (installed globally, wraps pi), @zed-industries/claude-code-acp 0.16.2 and @zed-industries/codex-acp 0.16.0 on npm, @agentclientprotocol/sdk 1.4.0; pi-acp dist uses session/new|prompt|resume|update|set_mode|request_permission and agent_thought_chunk/agent_message_chunk/tool_call/tool_call_update`: One ACP client covers four of the five harnesses today (qwen natively; pi, claude, codex through Zed's adapters) with thinking, tool calls, permission requests and sessions in one protocol; agy has no ACP adapter found. The sibling culture repo's `culture_core`/clients/acp/ is only 36 lines of config and constants (the client proper lives in its cultureagent dependency), so nvsh writes its own stdlib ACP client rather than citing.
  - seeds: `c34`, `c35`
- `s26` — `kiro-cli 2.0.0 ($HOME/.local/bin/kiro-cli): 'kiro-cli acp --help'; culture docs/reference/harnesses/acp.md:27,67`: Kiro ships a native ACP server ('kiro-cli acp' with --agent, --model, --trust-tools; --trust-all-tools is the auto-approve switch nvsh must never pass); no effort flag is exposed, so effort is unsupported and declared so. The culture repo already registers Kiro as an ACP harness, confirming the command shape.
  - seeds: `c36`, `c35`
- `s27` — `challenge pass / security lens: agy stream-json probe with a run_command turn (scratchpad/agyprobe, 2026-09-14)`: agy has no runtime approval channel in print mode; command tools are auto-denied and only a persistent settings.json allowlist unlocks them; read tools run without any prompt; tool event shape now known.
  - seeds: `c37`
- `s28` — `challenge pass / security lens: harness-side allowlists ($HOME/.gemini/antigravity-cli/settings.json permissions.allow x18; $HOME/.claude/settings.json allow=0; $HOME/.codex/config.toml no approval_policy; kiro-cli acp --trust-tools)`: Persistent per-harness allowlists execute commands with no nvsh gate; the spec had no claim about them.
  - seeds: `c38`, `c39`
- `s29` — `challenge pass / security lens: codex app-server AskForApproval schema (scratchpad/codex-schema)`: The approval channel exists but only under 'untrusted'/'on-request'; the policy value is part of the contract, not a default to trust.
  - seeds: `c40`
- `s30` — `challenge pass / security lens: redaction reach (nvsh/capture.py:30-51,390; nvsh/agent/pi.py:466-470; nvsh/agent/_subprocess.py:37,97-106)`: Redaction is applied to captured output and to pi's stderr only; other adapters' stderr tails and harness-side file reads are outside it.
  - seeds: `c41`, `c42`
- `s31` — `challenge pass / adjacent-systems lens: claude-code-acp 0.16.2 probe (scratchpad/acp) inside and outside a Claude Code environment`: A nested-session guard and a bundled second SDK are hidden dependencies of the claude ACP path.
  - seeds: `c43`, `c44`
- `s32` — `challenge pass / lifecycle lens: daemon upgrade path (nvsh/daemon.py, nvsh/client_transport.py; 'uv tool install --force' is the documented reinstall)`: An upgraded client talking to a pre-upgrade warm daemon would silently lose the target field; no version check exists.
  - seeds: `c45`
- `s33` — `challenge pass / lifecycle lens: CLI version drift (agy 1.0.1 installed vs 1.2.2 current with different flags; codex app-server marked [experimental])`: Every parser is version-coupled; nothing records or checks versions today.
  - seeds: `c46`
- `s34` — `challenge pass / operations lens: child-process lifetime (nvsh/agent/pi.py:815-830 close escalation; codex app-server daemon subcommand)`: Only PiAgent has a teardown escalation; ACP and app-server children need the same, and app-server's own daemon mode must stay out of scope.
  - seeds: `c47`
- `s35` — `challenge pass / observability lens: nvsh/agent/audit.py + nvsh/panel.py header + nvsh context`: The audit log records proposals/decisions/outcomes but not which harness/model/effort produced them.
  - seeds: `c48`
- `s36` — `challenge pass / hidden-dependency lens: config writers (nvsh/cli/_commands/setup.py, nvsh/cli/_commands/agent.py:56-81, nvsh/slash.py /agent use, nvsh/config.py set_provider)`: Three writers of \[agent\] provider exist and must move to the alias table together; migration of existing configs was unstated.
  - seeds: `c49`
- `s37` — `challenge pass / data-flow lens: recorded transcripts (agy init event cwd=/home/spark/..., conversation_id; qwen ACP sessionId; codex thread ids)`: Live transcripts leak home paths and ids into fixtures unless scrubbed.
  - seeds: `c50`
- `s38` — `challenge pass / failure-mode lens: unauthenticated harnesses (kiro-cli whoami; agy 'Print mode: not authenticated'; kiro acp stdout silent)`: An unauthenticated ACP agent blocks silently on stdin; nvsh needs a bounded initialize timeout and a doctor auth check.
  - seeds: `c51`
- `s39` — `challenge pass / overlooked-actors lens: cost and air-gap (nvsh/triggers.py rate window; registry local_model capability)`: A hosted default turns every qualifying failure into a paid call; nothing in the frame said so.
  - seeds: `c52`
- `s40` — `challenge pass / concurrency lens: nvsh/daemon.py _acquire/_activate (sessions.max=1 pool), nvsh/client.py:282-317 stale-state handling, one ACP/app-server stdio pair per slot`: Clean pass: one warm process per slot with switch/new session, @targets always one-shot in-process, so no two writers share a harness stdin; residual risk only if sessions.max>1 pools processes of different harnesses under one target key.
- `s41` — `challenge pass / reversibility lens: config.toml alias table + rc block`: Clean pass: the alias table is additive, \[agent\] provider stays readable, and nothing touches the rc block or the hook; rollback is 'nvsh agent use pi' or deleting the \[aliases\] table.
- `s42` — `challenge pass / cheap-probe lens: what was and was not probed on 2026-09-14`: Probed live: agy print+stream-json (text, tool, command auto-deny, input schema), qwen --acp (auto/default/plan modes, thought chunks), claude-code-acp (session/new inside and outside a Claude Code env; prompt rejected by API 400), codex app-server schema (offline), kiro-cli (auth only). Not probed: pi-acp, codex app-server live turn, claude native --permission-prompts host round-trip, agy --input-format multi-turn with tools.

## Decisions

- Per-harness configuration, not a shared enum: each \[agents.`<name>`\] table lists the flags nvsh passes to that harness (model, effort, and any extra argv), and each adapter provides what its CLI can and declares the rest unsupported through Capabilities; nvsh does not invent a cross-harness 'mode'. (Operator decision 2026-09-14, resolving q1.)
  - instruction: tests/`test_config.py` + per-adapter argv tests.
- Effort and model values are passed through verbatim to the harness (pi --thinking, claude --effort, codex -c `model_reasoning_effort`=, agy --effort); nvsh validates nothing beyond non-empty strings. Aliases exist to spare configuration, not to normalize values. (Operator decision 2026-09-14, resolving q2.)
  - instruction: Adapter argv tests plus one scripted fake exiting non-zero on a bad effort.
- Ad-hoc @targets always run one-shot (the current d23 rule stays). The default target warms a daemon session when its harness supports one: claude 2.1.270 (--input-format stream-json keeps one print-mode process open across turns; --resume/--session-id), qwen 0.23.3 (--input-format stream-json, --session-id/--resume/--continue), agy 1.0.1 (--input-format stream-json verified: each stdin line is {"event":"user","message":{"role":"user","content":"..."}} and yields `step_update`/result events; --continue/--conversation `<id>` for cold resume), codex 0.147.0 (app-server thread/start + turn/start over one stdio process, thread/resume across restarts). Any ACP-served agent warms through session/new + session/prompt + session/resume. Harnesses without a warm path fall back to cold per-turn processes with resume where available. (Operator decision 2026-09-14, resolving q3; amended for codex app-server and ACP.)
  - instruction: tests/`test_daemon.py` + tests/`test_client.py`: override -> `one_shot`=True; two requests -> one process for stream-json harnesses.
- Aliases are one flat table: each alias maps a name to a 'backend/model/effort' string (reviewer = "claude/opus-5/medium", local = "qwen/worker"), and a reserved alias 'default' is the single definition of the default harness, model and effort used when nothing else is specified, so the default is defined once and not duplicated across \[agent\]/\[agents.\*\]. '@default `<text>`' is a valid mark like any other alias (one-shot, per the d23 rule). 'default' is listed first wherever aliases appear, and 'nvsh agent list', '/agent' and the completion palette show it resolved to its backend/model/effort so the operator always sees what a plain failure will be answered by. (Operator decision 2026-09-14, resolving q4; amended for @default and visibility.)
  - instruction: tests/`test_config.py`: parse 'reviewer = "claude/opus-5/medium"' and 'local = "qwen/worker"'; missing default falls back to pi.
- Aliases live in the existing $`XDG_CONFIG_HOME`/nvsh/config.toml (a flat alias table with a reserved 'default' entry): the runtime package has no third-party dependencies (pyproject dependencies = \[\]) and parses config with stdlib tomllib (nvsh/config.py:27,241); no YAML file and no YAML dependency. (Operator decision 2026-09-14.)
  - instruction: grep -r yaml nvsh pyproject.toml returns nothing; dependencies stays \[\].
- ACP (Agent Client Protocol) support is in scope: nvsh gains a generic ACP adapter so any ACP-speaking agent can be a harness, not only the five named CLIs. (Operator decision 2026-09-14.)
- Official protocols first, ACP where the vendor ships it: qwen via 'qwen --acp' and Kiro via 'kiro-cli acp' (kiro-cli 2.0.0, --model, --agent, never --trust-all-tools) are driven by nvsh's generic ACP client; codex via its app-server; claude via its official 'claude -p' stream-json print mode; agy via its stream-json print mode; pi via its rpc mode (the existing reference adapter). Third-party ACP wrappers (claude-code-acp, codex-acp, pi-acp) are not used. (Operator decision 2026-09-14, resolving q5; amended after the challenge pass.)
  - instruction: Run 'nvsh agent list --json' with claude-code-acp absent then present; the path column flips and the reason names the missing binary.
- claude-code-acp is out of scope: it is a third-party (Zed) wrapper bundling its own Claude Code SDK build (2.1.44 in the 2026-09-14 probe, rejected by the API with a 400) separate from the operator's official claude CLI 2.1.270, and it only started with CLAUDECODE/`CLAUDE_CODE_`\* scrubbed. Claude uses its official CLI (decision c54).
- qwen ships read-only by default: AcpAgent sets session/`set_mode` 'plan' for qwen and reports `tool_calling`=False; an operator may opt into qwen's own agent-side approval with one config line (\[agents.qwen\] approval = "harness", which sets mode 'default' and marks the harness 'unmediated' in capabilities and the audit log). Principle: nvsh is not opinionated where the operator can decide by config. (Operator decision 2026-09-14, resolving q2.)
- Claude uses its official CLI only: the native 'claude -p --output-format stream-json' path (with --model, --effort, --include-partial-messages, --permission-prompts host, --resume) is primary and claude-code-acp is not used, because it is not an official Anthropic release and adds a second SDK to the risk surface. General rule: prefer the harness vendor's official protocol; use ACP only where the vendor ships it natively (qwen --acp, kiro-cli acp); third-party ACP wrappers (claude-code-acp, codex-acp, pi-acp) are out of scope. (Operator decision 2026-09-14, resolving q3.)

## Hard questions

- contradiction with Live probe 2026-09-14 (scratchpad/qwenacp): qwen 0.23.3 'qwen --acp' ran `run_shell_command` ('echo nvsh-probe-3') to completion in BOTH its default ACP mode 'auto' and after session/`set_mode` 'default', and never sent session/`request_permission`; only 'plan' mode refuses. So ACP does not give qwen a propose-don't-run channel today.? (resolved: Resolved by operator decision c53: qwen runs read-only (plan) by default; agent-side approval is an explicit per-harness config opt-in.)
- qwen has no client-side approval channel over ACP (0.23.3): does qwen ship read-only in 'plan' mode (`tool_calling`=False), or is its own 'default' mode (agent-side approval that never reaches nvsh) acceptable for this harness, breaking 'propose, don't run'? (resolved: qwen: plan-mode read-only by default; agent-side approval opt-in via \[agents.qwen\] approval = "harness". -> c53)
- ACP-first for claude depends on an npm package that is not installed, not offered by nvsh (non-goal: install paths stay pi-only) and pins its own SDK: keep ACP-first for claude and add claude-code-acp to the install offers, or make the native 'claude -p --output-format stream-json --permission-prompts host' path primary for claude and ACP optional? (resolved: Claude native only; no claude-code-acp; ACP only where vendor-native. -> c54)

## Open parks

- [unknown_nonblocking] kiro-cli acp's initialize handshake, thought chunks and `request_permission` behaviour are unverified: the installed kiro-cli 2.0.0 is not logged in, so the probe could not run; verify during the build after 'kiro-cli login'.

## Resolved vagueness

- [unknown_blocking] Whether each of claude, codex, qwen and agy can, in headless print mode, pause on a tool call and route the approve/deny decision through 'nvsh approve' the way `pi_ext`/approval.ts does (claude: --permission-prompt-tool needs an MCP server nvsh would have to host; codex exec: no approval callback seen in --help; qwen: --approval-mode plan only refuses; agy: `ask_permission` tool event shape unprobed). If a harness cannot, its '`tool_calling`' capability must be declared False and it runs read-only. — resolved: Operator decision 2026-09-14: ACP `request_permission` for qwen/claude/pi/kiro via ACP, codex app-server requestApproval, pi rpc extension; agy read-only until its permission events are probed; bare print-mode fallbacks run read-only
- [unknown_nonblocking] agy 1.0.1 accepts '--input-format stream-json' (one process, many turns) and requires each stdin line to carry an 'event' field with an object payload, but the accepted event name/payload for a user turn was not found by probing (`user_message`/`user_input`/prompt variants all 'ignoring unsupported stream input message event'); the AgyAgent warm path needs the schema from agy's docs or a newer release (1.2.2 is current), with cold '-p' plus '--conversation `<id>`' as the fallback. — resolved: Verified live 2026-09-14 on agy 1.0.1: with --input-format stream-json each stdin line is {"event":"user","message":{"role":"user","content":"`<text>`"}} (Claude Code's shape); other event names are ignored with a warning and a 'user' event without 'message' errors. The AgyAgent warm path is therefore buildable; recorded on c25.
