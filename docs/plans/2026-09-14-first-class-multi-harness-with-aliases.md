# Build Plan — first-class multi-harness with aliases

slug: `first-class-multi-harness-with-aliases` · status: `exported` · from frame: `first-class-multi-harness-with-aliases`

> nvsh treats agy, claude, codex and qwen as first-class harnesses at Pi's quality bar (thinking, tool calls, approvals, streaming), with a configurable default backend/model/effort, an @backend/model/effort ad-hoc target, and operator-defined model and invocation aliases such as @reviewer and @local

## Tasks

### t1 — Shared subprocess hygiene: scrubbed child environment and redacted stderr tails for every adapter

- instruction: Only nvsh/agent/`_env.py` (new), nvsh/agent/`_subprocess.py` and tests/`test_agent_subprocess.py`. pi.py keeps its own redaction for now; later adapters call `child_env`() and the shared tail helper.
- covers: c42, h38, c44, h39
- acceptance:
  - nvsh/agent/`_env.py` exposes `child_env`() that drops CLAUDECODE and every `CLAUDE_CODE_`\* key; SubprocessAgent uses it and passes the stderr tail through nvsh.redact before any ERROR event or log line
  - tests/`test_agent_subprocess.py`: a fake that prints `HF_TOKEN`=abc on stderr and exits 1 yields an ERROR whose text contains the redacted form and not the token; the env test asserts the keys are absent

### t2 — Fixture hygiene: version-stamped, scrubbed recorded transcripts and a fixture-lint test

- instruction: Only tests/`test_fixture_hygiene.py`, scripts/scan-secrets.py and tests/fixtures/README.md describing the header. Existing fixtures under tests/fixtures/platform are files, not transcripts; exempt the header rule for non-transcript fixtures by directory.
- covers: c50, h45
- acceptance:
  - tests/`test_fixture_hygiene.py` fails on any /home/`<user>`, ~/, e-mail address or 32-hex id inside tests/fakes and tests/fixtures, and on a recorded transcript without a '# recorded-from: `<cli>` `<version>`' header
  - scripts/scan-secrets.py covers tests/fixtures; a deliberately dirty fixture in a temp dir makes the test fail

### t4 — Extend the NvshAgent contract: THINKING event kind, request target, richer Capabilities

- instruction: Only touch nvsh/agent/base.py and add tests/`test_agent_base_codec.py`. Keep the wire codec omitting default fields so old daemons and clients stay compatible. Unknown EventKind values still degrade to STATUS.
- covers: c4, h3
- acceptance:
  - nvsh/agent/base.py: EventKind gains THINKING; AgentRequest gains target (backend, model, effort, alias) defaulting to None; Capabilities gains thinking, effort, path, approval ('nvsh'|'harness'|'none') and `unmediated_file_access`
  - `event_to_dict`/`event_from_dict` and `request_from_dict` round-trip the new fields; a pre-change dict without them still parses (tests/`test_agent_base_codec.py`)
  - no other file changes; existing tests pass

### t5 — Config: per-harness effort/`extra_args`/approval keys, flat \[aliases\] table with reserved 'default', migration from \[agent\] provider

- instruction: Only nvsh/config.py, docs/config.example.toml, tests/`test_config.py`. Validate alias values as non-empty segments split on '/', at most three. Effort and model are opaque strings, never validated against an enum (decision c24). Keep `set_provider` working but have it write \[aliases\].default.
- covers: c10, h8
- acceptance:
  - nvsh/config.py accepts effort, `extra_args` (list of str) and approval ('nvsh'|'harness') in \[agents.`<name>`\]; unknown keys are still rejected with the existing message
  - \[aliases\] is a flat table of alias = 'backend\[/model\[/effort\]\]' strings; Config.`resolve_target`(name) returns (backend, model, effort, alias); 'default' resolves from \[aliases\].default, else from legacy \[agent\] provider + that backend's model with no effort
  - save/load round-trips \[aliases\]; `default_toml`() and docs/config.example.toml document default, reviewer and local examples; tests/`test_config.py` covers each case; no yaml import anywhere

### t6 — Python side of the @target grammar: backend\[/model\[/effort\]\] and alias marks in triggers.py and slash.py

- instruction: Only nvsh/triggers.py, nvsh/slash.py, tests/`test_slash.py`, tests/`test_triggers.py`. Allow '/' inside the target token but keep the first character rule \[A-Za-z\]. Do not touch readline.bash here; the bash task mirrors the grammar.
- depends on: t5
- covers: c13
- acceptance:
  - nvsh/triggers.py `parse_mark` accepts '@backend/model/effort text', '@backend text' and '@`<alias>` text' where alias comes from config aliases; unknown names still fall through as ordinary lines; e-mail-like '@' in argument position is untouched
  - nvsh/slash.py: `split_agent_flag` returns the full target string; `agent_mark_items`() lists 'default' first, then other aliases, then adapters; tests/`test_slash.py` and tests/`test_triggers.py` carry table rows for '@claude/sonnet/medium', '@reviewer', '@local', '@default' and the negative cases

### t7 — Bash side of the @target grammar in readline.bash, the sync test, and the Tab-completion boundary doc

- instruction: Only nvsh/shell/readline.bash, tests/`test_readline_bash.py`, docs/shell-integration.md. Reuse the palette from 'nvsh complete --json' so aliases are matched without a bash-side list.
- depends on: t6
- covers: h10, c33, h28, c14, h11
- acceptance:
  - nvsh/shell/readline.bash `__nvsh_mark_line` accepts the same target grammar as triggers.`parse_mark` and rewrites to '/ask --agent `<target>` `<rest>`'; tests/`test_readline_bash.py` table rows match tests/`test_slash.py` rows one-to-one and the bash/python sync test passes
  - no Tab completion is added for '@' words; docs/shell-integration.md documents the extended grammar and repeats the hostname-completion limitation

### t8 — Registry: agy and ACP entries (kiro, qwen), forced-target choose(), path and local|hosted metadata

- instruction: Only nvsh/agent/registry.py and tests/`test_agent_registry.py`. Resolve aliases through Config.`resolve_target`; the adapter factories accept (model, effort, `extra_args`, approval).
- depends on: t5
- covers: c11, h9, c21, h18
- acceptance:
  - registry.ADAPTERS gains agy (binary agy), kiro (binary kiro-cli, path acp) and qwen switches to path acp; each spec carries path ('rpc'|'stream-json'|'app-server'|'acp') and hosted (bool)
  - registry.choose(config, forced=target) returns the forced adapter when its binary is installed and raises a CliError naming the binary otherwise, never falling back to openai-compat; with no config, choose() still returns pi
  - installers.TOOLS and `install_offer` are unchanged (non-goal c12); tests/`test_agent_registry.py` and tests/`test_installers.py` pass

### t9 — PiAgent: pass --thinking from effort and surface thinking deltas as THINKING

- instruction: Only nvsh/agent/pi.py, tests/`test_pi_agent.py`, tests/fakes/`pi_scripted`, docs/pi-rpc.md. Keep the strict-ack command loop untouched.
- depends on: t4
- covers: c3, h2
- acceptance:
  - nvsh/agent/pi.py `build_argv` adds '--thinking `<effort>`' when effort is set; `message_update` `thinking_delta` maps to EventKind.THINKING; capabilities report thinking=True, effort=True, path='rpc', approval='nvsh'
  - tests/`test_pi_agent.py`: argv test and a scripted `thinking_delta` yielding THINKING; docs/pi-rpc.md updated

### t10 — ClaudeAgent on the official CLI: model, effort, partial-message streaming, resume, thinking/`tool_use` mapping, host permission prompts as PROPOSAL

- instruction: Only nvsh/agent/claude.py, tests/fakes/claude, tests/`test_agent_claude.py`. First establish how 'claude -p --permission-prompts host' delivers the prompt in stream-json mode (`control_request` over --input-format stream-json versus --permission-prompt-tool); record the finding in the PR and resolve the matching plan risk.
- depends on: t4, t1
- covers: c5, h4
- acceptance:
  - nvsh/agent/claude.py argv includes --model, --effort, --include-partial-messages, --permission-prompts host and --resume/--session-id when warming; thinking, `tool_use` and `tool_result` content parts map to THINKING/`TOOL_CALL`/`TOOL_RESULT`
  - a permission request from claude becomes a PROPOSAL answered from nvsh approve (allow/deny round-trip proven against tests/fakes/claude); --dangerously-skip-permissions never appears; capabilities report path='stream-json', approval='nvsh'
  - tests/fakes/claude replays a recorded-from-2.1.270 transcript with thinking and `tool_use` blocks; conformance suite green; no claude-code-acp anywhere (decisions c43, c54)

### t11 — CodexAgent on app-server: JSON-RPC client with threads, turns, reasoning, requestApproval as PROPOSAL, on-request policy

- instruction: Only nvsh/agent/codex.py, tests/fakes/codex-app-server, tests/`test_agent_codex.py`. Use the schema from 'codex app-server generate-json-schema' for field names. Never start 'codex app-server daemon'.
- depends on: t4, t1
- covers: c6, h5, c40, h36
- acceptance:
  - nvsh/agent/codex.py drives 'codex app-server' over stdio: initialize, thread/start (approvalPolicy on-request, sandbox read-only), turn/start, turn/steer, turn/interrupt, thread/resume; item/reasoning/textDelta -> THINKING, item/agentMessage/delta -> `TEXT_DELTA`, item/commandExecution/\* -> `TOOL_CALL`/`TOOL_RESULT`, requestApproval -> PROPOSAL answered via execCommandApproval
  - model and effort travel as -c model= and -c `model_reasoning_effort`=; 'never', danger-full-access and --full-auto never appear; 'codex exec --json' remains as the fallback when app-server initialize fails
  - tests/fakes/codex-app-server replays a recorded-from-0.147.0 transcript; conformance suite green

### t12 — AcpAgent: generic ACP client with kiro and qwen entries, plan-mode default, per-harness approval opt-in, bounded initialize

- instruction: Only nvsh/agent/acp.py, tests/fakes/acp, tests/`test_agent_acp.py`. Advertise fs.readTextFile=false, writeTextFile=false and no terminal capability so agents use their own tools; mark such harnesses `unmediated_file_access`=True.
- depends on: t4, t1
- covers: c35, h30, c7
- acceptance:
  - nvsh/agent/acp.py speaks initialize/session/new/session/prompt/session/resume/session/`set_mode` over stdio with a bounded initialize timeout; `agent_thought_chunk` -> THINKING, `agent_message_chunk` -> `TEXT_DELTA`, `tool_call`/`tool_call_update` -> `TOOL_CALL`/`TOOL_RESULT`, session/`request_permission` -> PROPOSAL answered from nvsh approve
  - qwen entry sets mode 'plan' and `tool_calling`=False unless \[agents.qwen\] approval = 'harness' sets mode 'default' and marks approval='harness' (decision c53); kiro entry runs 'kiro-cli acp' with --model, never --trust-all-tools; auto/auto-edit/yolo are never set
  - tests/fakes/acp replays recorded-from qwen 0.23.3 chunks plus one `request_permission`; conformance suite green; a live 'qwen --acp' smoke on spark observes THINKING and `TOOL_CALL`

### t13 — QwenAgent print-mode fallback on stream-json (no-ACP path)

- instruction: Only nvsh/agent/qwen.py, tests/fakes/qwen, tests/`test_agent_qwen.py`. This adapter is the fallback the registry picks when the ACP path is disabled; keep it small.
- depends on: t4, t1
- covers: h6
- acceptance:
  - nvsh/agent/qwen.py runs 'qwen --output-format stream-json --model `<m>` -p `<prompt>`' with --approval-mode plan; thinking/tool events map to THINKING/`TOOL_CALL`/`TOOL_RESULT`; the '\[status\] ' heuristic is gone
  - tests/fakes/qwen replays recorded-from-0.23.3 stream-json lines; conformance suite green

### t14 — AgyAgent: stream-json print mode with model/effort, warm --input-format session, read-only tool mapping

- instruction: Only nvsh/agent/agy.py, tests/fakes/agy, tests/`test_agent_agy.py`. The system brief rides in the prompt (no system-prompt flag). Treat the 'auto-denied' stderr line as a STATUS, not an ERROR.
- depends on: t4, t1
- covers: c8, h7, c37, h33
- acceptance:
  - nvsh/agent/agy.py runs 'agy -p <brief+prompt> --output-format stream-json --model `<m>` --effort `<e>`'; `step_update` `text_delta` -> `TEXT_DELTA`, `step_type` tool -> `TOOL_CALL`/`TOOL_RESULT`, result -> DONE/ERROR; warm sessions use --input-format stream-json with {"event":"user","message":{"role":"user","content":...}} lines and --conversation `<id>` for cold resume
  - capabilities report `tool_calling`=False for commands, `unmediated_file_access`=True, hosted=True; the adapter never reads or writes agy's settings.json; --dangerously-skip-permissions never appears
  - tests/fakes/agy replays the recorded-from-1.0.1 transcripts (text turn, tool turn, auto-deny); conformance suite green; a live 'agy' smoke on spark answers 'say OK'

### t15 — Cross-adapter conformance and policy invariants: thinking/tool/proposal/effort cases, auto-approve and settings-write bans, version stamps

- instruction: Only tests/`test_agent_conformance.py`, tests/`_fake_adapters.py`, tests/`test_invariants.py`. Do not modify adapters here; file a deviation if an adapter fails a case.
- depends on: t9, t10, t11, t12, t13, t14, t2
- covers: c32, h27, c22, h19, c38, h34, c39, h35, c41, h37, c18, h15, c46, h41
- acceptance:
  - tests/`test_agent_conformance.py` is parametrised over pi, agy, claude, codex, qwen, kiro(acp) ViaFake adapters and asserts THINKING and `TOOL_CALL` emission, PROPOSAL or declared `tool_calling`=False, verbatim effort in argv (decision c24), and an ERROR carrying the CLI stderr tail on a rejected effort
  - tests/`test_invariants.py` asserts no adapter argv contains dangerously-skip-permissions, yolo, danger-full-access, --full-auto or --trust-all-tools, and no code path under nvsh/ opens a harness settings/trust file for writing
  - every adapter declares approval and `unmediated_file_access`; every tests/fakes transcript carries the recorded-from header; the CLI-version range per adapter is asserted

### t16 — Daemon and client: default alias resolution, target on the wire, version handshake, one-shot @targets, warm sessions, child containment

- instruction: Only nvsh/daemon.py, nvsh/client.py, nvsh/`client_transport.py`, docs/daemon.md and their tests. Import the shared escalation helper from `_subprocess.py`.
- depends on: t4, t5, t8
- covers: c15, h12, c45, h40, c47, h42
- acceptance:
  - Daemon.`_build_agent` resolves 'default' via Config.`resolve_target` and builds the warm agent with (model, effort, `extra_args`, approval); a request carrying a non-default target always runs one-shot in the client (decision c25); two consecutive failures reuse one warm process for stream-json/ACP/app-server harnesses
  - the client sends its nvsh version and the daemon answers with its own; on mismatch the client stops the stale daemon, starts a new one and the request succeeds; old-shape requests still parse
  - close() escalation (wait/terminate/kill) applies to every adapter; after 'nvsh uninstall' and after idle-exit no child harness process remains (tests with fake long-running binaries)

### t17 — Doctor: per-harness reachability with version, auth state and allowlist warnings, no model calls

- instruction: Only nvsh/`doctor_checks.py`, tests/`test_doctor_checks.py`, tests/fixtures/doctor/. Run CLIs with a short timeout and treat a hang as a failed check with remediation.
- depends on: t8
- covers: c17, h14, c51, h46
- acceptance:
  - `doctor_checks`.`check_agent_reachable` dispatches per harness: binary present, version parsed and compared with the adapter's supported range, auth state detected from the CLI's own 'not logged in'/'not authenticated' output, and login command as remediation; no network or model call in tests
  - a readable harness-side allowlist (agy/claude settings.json permissions.allow, codex config.toml `approval_policy`, kiro trust settings) yields a warning check naming the file; doctor never writes any of them

### t18 — Panel and audit: THINKING rendering and target visibility

- instruction: Only nvsh/panel.py, nvsh/agent/audit.py and their tests. Keep hard-coded SGR sequences; never tput.
- depends on: t4
- covers: c16, h13, c48, h43
- acceptance:
  - panel renders THINKING as a dimmed run closed before the first `TEXT_DELTA`; with `NO_COLOR` or TERM=dumb it prints plain 'thinking:' lines with no SGR; the header line names harness, model, effort, path and warm|one-shot
  - audit.py records the resolved target on every proposal/decision/outcome line; tests/`test_panel.py` snapshots both modes and tests/`test_audit.py` checks the fields

### t19 — CLI and slash surfaces: agent list with path/hosted/capabilities, agent use writes the default alias, /agent and setup, catalog and invariants

- instruction: Only nvsh/cli/`_commands`/agent.py, nvsh/cli/`_commands`/setup.py, the /agent handler in nvsh/slash.py, nvsh/explain/catalog.py, tests/`test_cli_agent.py`, tests/`test_cli_setup.py`.
- depends on: t5, t8, t6
- covers: c49, h44
- acceptance:
  - 'nvsh agent list --json' shows every adapter with installed, path, hosted|local, capabilities and the resolved 'default' alias first; 'nvsh agent use `<name>`' and 'nvsh setup' write \[aliases\].default; '/agent' mirrors it; a config with only \[agent\] provider still resolves
  - explain catalog has an entry for every new verb path; tests/`test_invariants.py` `_VERBS` includes them; tests/`test_cli_agent.py` updated to the new adapter count; hosted harnesses are marked visibly

### t20 — Docs and prompt files: aliases, targets, harness paths, redaction boundary, drift rule, changelog and version bump

- instruction: Docs only. Use the version-bump skill. Quote the qwen opt-in line and the 'never edits harness settings' boundary verbatim from the spec.
- depends on: t15, t16, t17, t18, t19, t7
- covers: c20, h17, c28, h25, c19, h16
- acceptance:
  - CLAUDE.md, AGENTS.override.md, .pi/SYSTEM.md, QWEN.md, AGENTS.colleague.md, docs/architecture.md, README.md, docs/config.example.toml and docs/shell-integration.md describe the default alias, @targets, per-harness paths and the redaction boundary consistently; scripts/harness-smoke.py passes
  - culture.yaml, .claude/skills and .pi/skills are byte-identical to main; CHANGELOG.md has the entry and pyproject.toml is bumped (minor)

### t21 — Live verification on spark, thor and orin recorded in docs/verification.md

- instruction: Manual, operator-run; paste redacted panel output. Record any harness that could not be verified (e.g. kiro not logged in) as such, never as passed.
- depends on: t20
- covers: c1, h1, c30, h26, c31, h32, c29, h29
- acceptance:
  - docs/verification.md records: on main, grep shows no THINKING/effort/agy; after install, a failure in a hooked shell with default = pi/associate shows a thinking run; '@reviewer explain the last failure' answers from the aliased harness; '@claude/sonnet/medium' runs one-shot; 'nvsh agent list --json' shows six adapters with paths
  - at least two harnesses verified live on spark and the default verified on one Jetson

## Risks

- [unknown_nonblocking] kiro-cli acp initialize/thought/`request_permission` behaviour is unverified (installed kiro-cli 2.0.0 not logged in); the kiro entry may need read-only mode if it never sends `request_permission`. (task t12)
- [unknown_nonblocking] codex app-server is marked \[experimental\] in codex-cli 0.147.0; protocol field names may drift between releases, so the adapter must fail closed on an unknown protocol version and fall back to 'codex exec'. (task t11)
- [unknown_nonblocking] How 'claude -p --permission-prompts host' delivers a permission prompt in stream-json mode (`control_request` over --input-format stream-json vs --permission-prompt-tool needing an MCP server) is not yet established; if only an MCP server works, nvsh must host a tiny stdio MCP server in stdlib. (task t10)
- [unknown_nonblocking] pi --thinking may be ignored or rejected by the local nemotron 'associate' model served through Pi; the adapter should treat a rejected thinking level as a STATUS, not a failure. (task t9)
- [follow_up] tests/`_fake_adapters.py` is shared across the adapter wave; the conformance task owns it and adapter tasks add ViaFake classes in their own test modules to keep the wave file-disjoint. (task t15)
