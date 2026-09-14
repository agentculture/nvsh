# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). This project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.1] - 2026-09-14

### Changed

- Recorded the owner's adjudication of the 0.11.0 "setup works with any installed agent" run in devague state: plan task `t5` confirmed, deviations `d1` and `d2` approved, reasoning lapses `l1`–`l9` approved, obligations `o1`–`o15` and evidence `e1`–`e15` approved, and delta `b1` approved.
- Re-filed and approved the three behavioral deltas that cite `d1` and `d2` (bare-machine pi/node bootstrap offers, bare `--agent` adapter names, and `setup --json` staying parseable when the daemon stop prints), which devague refuses to file against an unapproved deviation.
- Committed the re-exported plan and its implementation split artifact, and refreshed the delivery summary so its deviations, lapses and claims read as approved instead of pending.
- Superseded delta `b1` ("probe lists both qwen and qwen-p"), which the PR #12 review fix made false, with `b5` (one probe row per shared binary), backed by obligation `o16` and passing evidence `e16`; all three approved by the owner. Added `docs/current-spec.md`, the `devague today` projection of current behavior, which now shows `b5` and keeps `b1` only as its lineage.

## [0.11.0] - 2026-09-14

### Added

- `nvsh setup --agent <target>`: make an alias, a bare adapter name (`claude`) or a `backend[/model[/effort]]` literal the default in one run; a missing binary fails with exit 2 naming it and writes nothing.
- `nvsh setup` without `--agent` probes `PATH` for every adapter (`registry.probe()`): one installed harness becomes the default silently, several prompt once on a terminal with tool-calling adapters first and `read-only / plan mode` labels, `--yes` never answers the pick, `--json`/non-tty takes the first row; `openai-compat` is the pick only when no harness binary exists.
- Setup reports `agent.hosted` and prints `<name> is hosted: on a failure the redacted command, output and device context leave this machine` for claude/codex/agy/kiro picks, runs doctor's reachability probe for the pick (`agent.reachable`, 2-second budget, never fatal), and warns `nvsh is not tested on macOS/zsh yet (see issue #11)` on Darwin or a zsh login shell.
- `nvsh agent use` accepts all eight adapters; `nvsh agent install <name>` knows `npm install -g` for claude, codex, qwen and qwen-p, pi's own command, and says `no known installer` for agy, kiro and openai-compat; every install goes through `run_install` and the audit log.
- `installers.missing_tools(chosen=...)` scopes the pi/node offers to a pi pick; `installers.harness_install_step()`.
- Doctor's `agent_configured` check reports what `[aliases].default` resolves to and says whether it came from the alias or the legacy `[agent] provider`.
- README rewritten in the announcement-first shape (Install, Set up, Work with it, Safety first, What nvsh never does, What lands where) with absolute links for the PyPI page; a test pins the heading order and forbids relative links.

### Fixed

- A fallback pick no longer sticks: an `[aliases].default = "openai-compat"` written when nothing was installed is re-probed on the next `nvsh setup` once a harness appears (it used to be kept forever because a binary-less adapter always counted as installed).
- Setup stops a running daemon after changing the default, so a warm session never keeps the previous harness; the child's `daemon: not running` line no longer lands on setup's `--json` stdout.
- `registry.choose()` consults the probe before falling back to `openai-compat`, so a box with claude or codex installed is never steered to an OpenAI-compatible endpoint and an API key it does not need.

## [0.10.1] - 2026-09-14

### Fixed

- `Panel.stream` now does all its setup (termios save, SIGINT handler install, ticker start) inside its try block, so a Ctrl+C that lands before the handler is installed still ends as an interrupted stream with `cancel` called once instead of an uncaught KeyboardInterrupt.
- `Panel.stream` (and the line readers) capture the previous SIGINT handler before installing their own, and teardown restores it first, so a second Ctrl+C during teardown never leaves the panel handler installed.
- The flaky CI test `test_ctrl_c_returns_to_a_prompt_within_one_second` now announces READY from inside the event generator, closing the race that made the Publish workflow's test job fail on main (run 34837468240, `assert -2 == 130`). Two regression tests added.

## [0.10.0] - 2026-09-14

### Added

- First-class multi-harness support: eight registered `NvshAgent` adapters in `nvsh/agent/registry.py` (`pi`, `qwen` over ACP, `qwen-p` read-only stream-json fallback, `claude`, `codex` app-server with `exec` fallback, `agy` read-only stream-json, `kiro` over ACP, `openai-compat`), each reporting its protocol path (rpc/stream-json/app-server/acp/http) and hosted/installed state via `nvsh agent list --json`
- A flat `[aliases]` config table (`$XDG_CONFIG_HOME/nvsh/config.toml`) mapping short names to `backend[/model[/effort]]` targets, with `default` reserved for a bare `nvsh --agent default`; `nvsh agent use <name>` and `nvsh setup` write `[aliases].default`
- The `@target` grammar at the prompt (`@name` for a registered alias/adapter, `@backend/model/effort` for a literal) on both the Python side (`triggers.py`, `slash.py`) and the bash readline layer, rewritten to `/ask --agent <target>`; an ad-hoc target runs one-shot, the default target rides the daemon's warm session
- Per-harness config knobs (`model`, `effort`, `extra_args`, `approval`) passed verbatim to each backend's own flags (`pi --thinking`, `claude --effort`, `codex -c model_reasoning_effort=`, `agy --effort`, ACP `set_config_option`), plus a generic `AcpAgent` (`nvsh/agent/acp.py`) speaking ACP JSON-RPC for `qwen` and `kiro`
- A `THINKING` event kind through the `NvshAgent` contract, rendered dim in the panel, plus a panel header (`harness/model/effort · path · warm|one-shot`) and an audit log that records the resolved target for every turn
- Doctor checks for per-harness version/auth reachability and harness-side allowlist warnings (never writes to a harness's own settings)
- Scrubbed child environments for every subprocess-backed adapter (`CLAUDECODE`/`CLAUDE_CODE_*` stripped, `nvsh/agent/_env.py`) and redacted stderr tails; a fixture-hygiene scan for recorded transcripts
- Daemon/client wiring for the resolved target on the wire, a version handshake, and shared close escalation across concurrent shells
- Docs updated across all four harness prompt files (`CLAUDE.md`, `AGENTS.override.md` + `.pi/SYSTEM.md`, `QWEN.md`, `AGENTS.colleague.md`), `docs/architecture.md`, `README.md`, `docs/config.example.toml` and `docs/shell-integration.md` to describe the default alias, `@target` grammar, per-harness adapter paths and the redaction/settings-write boundary consistently

## [0.9.2] - 2026-09-13

### Added

- docs/architecture.md recording the hook-vs-wrap decision (issue #1 milestone 1) and a docs/platforms.md skeleton for the platform-detection value/source table
- The bash-hook shell itself, built across this PR's task waves: trigger rules (`nvsh/triggers.py`), the redactor (`nvsh/redact.py`), platform detection, the capture layer (session log + OSC 133 slicing + tmux pipe-pane), the bash hook core and readline layer (Enter macro, `/`+Tab palette, Ctrl+G), the `NvshAgent` contract with a `PiAgent` adapter for `pi --mode rpc`, the harness chooser/registry, the pi approval extension, the per-user session daemon, the failure client and inline panel, `nvsh setup`/`uninstall`/`on`/`off`/`hook`, the slash-command registry (`nvsh slash`/`complete`, `/agent` `/help` `/undo` `/approve` `/doctor` …), and doctor's platform/backend-reachability/hook-health/terminfo checks. Shipped and unit/integration-tested in this PR; real-hardware verification on spark, thor and orin is recorded in `docs/verification.md`.

- Three-machine verification record `docs/verification.md` (DGX Spark in Ghostty, Jetson AGX Thor and AGX Orin over ssh, with and without tmux), `tests/test_timing.py` (success-path overhead and time to first agent text with the fake agent), two asciicast demos under `docs/demos/` with a stdlib recorder `scripts/record-cast.py`
- `nvsh setup` detects missing helper tools (pi, node, uv, tmux), prints their install commands, and installs them on confirmation or `--yes` (`nvsh/installers.py`)

### Fixed (found by the hardware verification, recorded as deviations d2-d8)

- Under bash-preexec (Ghostty on bash < 5.3, kiro-cli / fig / amazon-q) the hook reads `BP_PIPESTATUS`, so pipelines keep their per-stage statuses; doctor accepts that layout
- The failed command's output slice is the open OSC 133 `C..` region, not the previous command's closed one, so the agent diagnoses the right output under Ghostty
- `nvsh doctor`: `bindings_present` parses `bind -s`/`bind -X` (which the shell now exports), `agent_reachable` never prints the endpoint URL, and `[FAIL]` on screen always means `healthy=false`
- nvsh's own hidden slash dispatch (`/doctor` reporting unhealthy) never triggers an agent turn; `nvsh slash` exits 0 for a handled command
- Ctrl+G streams the agent's answer on the tty instead of discarding it, in emacs and vi keymaps
- A cold daemon start waits for the daemon to answer instead of timing out after 5 s and falling back to a one-shot run; a second autostart never spawns a rival daemon (lock file); fallback reasons are stated
- openai-compat reads its bearer from `api_key_file` (default `$XDG_CONFIG_HOME/nvsh/api_key`, must be 0600) when the env var named by `api_key_env` is unset, so a headless Jetson over ssh needs no rc export; the auto-call rate limiter prints one line when it holds back instead of staying silent (deviation d10, reported live from orin)
- The daemon no longer wedges behind one turn: a turn whose client disconnected is aborted (pending approval dialogs denied, backend cancelled), turns are capped at a wall clock (`NVSH_TURN_TIMEOUT`, default 300 s), queued requests are told they are waiting, and `nvsh daemon status` shows the active turn and queue (deviation d12, observed live on spark)
- In the one-shot fallback, Enter/Esc on a proposal answers the in-process agent's dialog (it used to go to the absent daemon, so nothing ran); pi's rpc lifecycle events no longer print as `... agent_start` lines (deviation d11, observed live on spark)
- The panel repaints a dim `... waiting for the agent (Ns)` line while the backend is silent (plain single line off a tty), acknowledges every proposal key immediately (`nvsh: running ...` / explaining / details / ignored), and renders a fallback as a visible `nvsh: falling back - <reason>` line (deviation d13, operator report)
- The proposal keys gain `[s] +session` and `[u] +user`: run and approve the command class for this session or persist it (widened to `<first word> *`) for the user; `sudo`/destructive commands refuse both and fall back to run-once; session approvals now actually persist for the login session under `$XDG_RUNTIME_DIR/nvsh/session-approvals.toml` (they used to die with the process, so `nvsh approve add --session` was a no-op) (deviation d15, operator request)
- Daemon-routed pi turns answer again (first text in about 2 s instead of a 120 s timeout and one-shot fallback): the pi adapter never pipelines rpc commands (each waits for its ack; pi 0.85 silently drops pipelined `new_session`+`prompt`), `start()` handshakes with `get_state`, session files are the ones pi chose, and a pi that exits or never handshakes surfaces as an ERROR with its exit code and redacted stderr tail (deviation d14)
- Every adapter now carries a shipped system brief (who nvsh is, propose-one-command rules, never investigate the harness) and a per-platform playbook (`nvsh/agent/playbooks.py`: DGX Spark unified memory, Jetson tegrastats/jtop/nvpmodel, RTX, generic Linux) ahead of the detected facts; pi/claude/qwen get it through `--append-system-prompt`, openai-compat as a system message, codex at the head of the prompt (deviation d19)
- A sentence typed at the prompt (`what are the memory levels?`, exit 127) is routed as a request to the agent instead of a failed command (`triggers.prose_request`), and the brief tells the agent to run the inspections itself and answer with numbers rather than list commands (deviation d20)
- pi's approval extension gets `NVSH_BIN` and the XDG dirs in pi's environment and treats a failed `nvsh approve` spawn as a named block instead of an endless dialog, so `[u]`/`[s]` approvals are honoured for the rest of the turn even when `nvsh` is not on pi's PATH (deviation d21)
- Proposal keys gain `[t] tell` (steer the agent mid-turn: pi `prompt` with `streamingBehavior: steer`, next-message on adapters without a mid-turn channel) and a `/steer <text>` command; `[e]` asks the agent to explain when the model gave no rationale, and pi's streamed text before a tool call becomes the rationale; `[d]` shows the approval state, the exact patterns `[s]`/`[u]` would store, backend, conversation and slice size; the panel prints `... running: <command>` / `... finished (exit N)`, a scope line saying what `[s]`/`[u]` allow, and a header that names the hand-off (`forwarding to pi/associate`, `asking pi/associate: <sentence>`) (deviations d16, d17, d18, d19 panel, d22)
- Security hardening from the PR review: `--rc` paths are validated (`rcfile.RcPath`: no `..`, no symlink escaping `$HOME`, owned regular file only), the runtime-dir fallback is a per-uid 0700 directory verified for ownership before use (`nvsh/runtimedir.py`, shared by capture, setup, daemon, approvals and the hook), and the only path to `bash -c` is `_run_approved`, which accepts a rendered `Proposal` with an explicit approval decision and rejects control bytes
- Review fixes (Qodo, PR #8): the capture wrapper's cleanup trap survives `exec` and abandons the log when capture cannot start; the tmux pipe-pane target is shell-quoted; steer and ui-response controls match the shell id exactly; cancelling a queued request touches only its own slot; `/retry` runs in the recorded cwd; uninstall stops the daemon before unlinking its socket; `NVSH_DISABLE=0` keeps the hook on; the redactor consumes whole quoted env values; `sudo*`/`rm*`/leading-`*` patterns are refused and re-checked at match time; the rate limiter is serialised under a per-user lock; malformed slash quoting is a user error; adapter stderr is drained on a thread (no deadlock); malformed platform JSON no longer erases the facts block; `nvsh approve remove` reports a missing pattern
- SonarCloud cleanup on the PR: cognitive-complexity hot spots split into named helpers across the pi adapter, daemon, client, panel, slash, triggers, doctor, config, setup and platform detection (behaviour pinned by the suite), redundant exception clauses and unused parameters removed, composite test assertions split, `case` statements gain default arms, and rc reads/writes go through a directory descriptor opened on the validated home (`os.open(..., dir_fd=...)`), which clears the last path-traversal findings and pins the write to the directory that was validated
- With pi, `proposal.command` is the bare tool-call command, never the rendered panel text; a command-less proposal is never executed
- An explicit AI mark at the prompt: a line whose first token is `?` asks the default agent and `@pi`/`@qwen`/`@claude`/`@codex`/`@openai-compat` asks that harness for that one request (`/ask --agent <name>` from the slash path). The readline Enter macro rewrites both to the hidden `nvsh slash "/ask …"` dispatch, so bash never prints `?: command not found` and the typed line still reaches history; `triggers.prose_request` recognises the same forms for hook-only shells. Marks are explicit like Ctrl+G — never held back by, and never consuming, the auto-call rate limit — and an unavailable harness is one line (`nvsh: @qwen is not available: …`) rather than a silent fallback (deviation d23, operator request)
- Proposal approvals gain a *specific* form and per-stage patterns: `[S]`/`[U]` store `<command> <first argument> *` (`ssh orin *`, `docker ps *`) where `[s]`/`[u]` keep the d15 behaviour (exact line / `<first word> *`), and a command line is split into stages on `|`, `&&`, `||`, `&`, `;` and newlines (quotes honoured, so `ssh orin "ps | head"` is one stage) with one pattern stored per stage and **every** stage required to match before anything is auto-approved — a broad `ls *` no longer covers `ls | sudo tee /etc/x`, a subshell or `$(...)` is opaque and never auto-approved, and `nvsh approve check` names the unapproved stage. One pure helper (`approvals.pattern_for`/`patterns_for`) backs the panel scope line, the client store write, the details view and the pi extension, which now forwards six choices (`once`, `session`, `session-specific`, `user`, `user-specific`, `deny`) to `nvsh approve add --scope`, the single writer (deviation d24, operator request from the Spark)
- A multi-stage proposal now shows its stages and lets the operator approve only some of them: the panel numbers them above the scope lines (`stages: 1 'ls /srv/models'  2 'grep -i orin'`), each scope family names the pattern it would store per stage (`[u] 'ls *' | 'grep *'`, `[s]` reads `exact`, a `sudo`/`rm` stage reads `(not approvable)`), and after `s`/`S`/`u`/`U` one cooked line `stages [all,1,2]:` reads the pick — `all`, a number, or a comma/space list, with anything unreadable re-asked once and then treated as `all`. Only the chosen stages are stored, the ack names exactly those patterns (`'grep -i *' approved for this session (stage 2 of 2)`), `[d]` gains a per-stage table, and the command still runs once whatever was picked. `nvsh approve add --scope <choice> --stages 1,2` is the same choice from the CLI; because pi's `ctx.ui.select` carries only one value string, a partial pick rides to the extension as `session-specific:1,2` and is split back apart there (deviation d26, operator request from the Spark)

### Changed

- README.md, CLAUDE.md, AGENTS.override.md, AGENTS.colleague.md, QWEN.md and .pi/SYSTEM.md now describe the bash-hook architecture instead of the retired PTY-wrapper/login-shell design; CLAUDE.md's Login-shell constraints section is replaced by Hook constraints; README's opening names Jetson AGX Orin/Thor, DGX Spark, bash, Ghostty and ssh, describes the implemented hook, daemon and verbs, and adds a What leaves the machine section
- `doctor`'s `capture_active` check now reports `info` severity (not `warning`) when `NVSH_HOOK_VERSION` is absent — running `nvsh doctor` outside a hooked shell is expected, not unhealthy; it still warns when hooked but not capturing

## [0.9.1] - 2026-09-13

### Changed

- CLAUDE.md re-initialized from the seed into a full runtime prompt for nvsh (shell -> agent for Jetson / DGX Spark / RTX Spark), incorporating issues #1 and #2 and the goal of running nvsh as the default login shell: login-shell safety, PTY wrapper around a real bash, trigger rules, propose-before-run, pluggable offline-first backends, and redaction.
- README.md rewritten around nvsh's purpose and goals instead of the agent template.
- QWEN.md, AGENTS.override.md, AGENTS.colleague.md and .pi/SYSTEM.md now describe nvsh and its current scaffold state; the template re-initialization steps are removed.
- `nvsh learn` (text and --json), the `explain` root entry, and `nvsh --help` now describe nvsh instead of a clonable agent template.

## [0.9.0] - 2026-09-06

### Added

- **Four agent harnesses, each reading exactly one root file.** Claude
  Code→`CLAUDE.md`; Pi/`associate`→`AGENTS.override.md` (context) plus
  `.pi/SYSTEM.md` (system prompt, which *replaces* Pi's default);
  colleague→`AGENTS.colleague.md`; Qwen Code→`QWEN.md`. Deliberately **no**
  `AGENTS.md` — `AGENTS.override.md` is what stops Pi inheriting `CLAUDE.md`.
- One skill tree, four loaders: `.qwen/skills`, `.colleague/skills` and
  `.pi/skills` are relative symlinks onto `.claude/skills`. No forked scripts,
  no duplicated docs.
- `docs/harness-selection.md` (the two selections), plus
  `docs/automation-contract.md` and `docs/harness-invocations.yaml` (the four
  forced invocations as a machine-readable contract), and
  `docs/harness-verification.md` (the instrumented run).
- `scripts/harness-smoke.py` — a per-harness CI check that fails when **any one**
  of the four configs is broken, so three-quarters of a clone cannot rot
  unnoticed. A skipped check is reported as not-verified, never as a pass.
- `scripts/scan-secrets.py` — CI gate against committed credentials and
  non-localhost endpoints, with planted-secret tests proving it catches.

### Changed

- `culture.yaml` declares `backend: claude`. Because no code path rewrites that
  key, the template's declaration is what every clone inherits — the previous
  `colleague` value is why ~30 siblings carry a backend disagreeing with their
  seeded prompt file.
- `backend-fingerprints.yaml` re-synced from steward: list-valued prompts, so
  `acp` accepts `QWEN.md` and `colleague` accepts `AGENTS.override.md` and
  `.pi/SYSTEM.md`.

### Fixed

- `CLAUDE.md`, `README.md`, `QWEN.md`, `AGENTS.override.md`,
  `AGENTS.colleague.md` and `docs/skill-sources.md` had claimed this repo was a
  colleague resident, contradicting `culture.yaml`. Every harness prompt now
  names `backend: claude` / `CLAUDE.md` as the mesh resident, while stating
  that its own harness stays interactively available regardless.
- `.pi/settings.json`'s `skills` key is **inert** — that key belongs to a
  `package.json` manifest, not `settings.json`. Syscall instrumentation on a
  fresh clone: settings alone opened **0** `SKILL.md`; the `.pi/skills` symlink
  opened **19**. Replaced with the symlink, settings file removed.
- `doctor` no longer accepts an interactive harness's prompt file as the mesh
  resident's. Four harnesses ride three backend names, so `AGENTS.override.md`,
  `.pi/SYSTEM.md` and `QWEN.md` are *recognized* under `colleague`/`acp` — but
  the Culture daemon reads exactly one file per backend, and a clone carrying
  only Pi's files under `backend: colleague` used to report healthy with no
  resident prompt at all. `prompt_file_present` now requires the resident
  prompt; the others are reported by a new `harness_prompts` info check.
- `scan-secrets.py` closes three evasions: values containing `@`/`:`/`%`/`=`/`?`
  are now matched in full instead of stopping at a base64-ish alphabet (a
  quoted JSON key is matched too, which it never was); the placeholder
  exemption is a whole-value judgement, so a high-entropy literal merely
  *containing* `fake`/`example` is still reported; and endpoint hosts are
  parsed with `urlsplit`, so a bracketed IPv6 authority such as
  `http://[2001:db8::1]:8080` can no longer slip past the localhost allowlist.
- `harness-smoke.py`: a live probe is satisfied by **stdout only** (stderr
  noise mentioning `yes` or a prompt filename no longer counts as an answer,
  and a yes/no probe must answer exactly `yes`); a nonzero exit from `steward
  doctor` or `guild create` can no longer reach a passing branch on
  success-shaped JSON; failure/skip/waiver diagnostics go to stderr, leaving
  stdout to results; and an unknown `--stage` exits 1 (user error) rather than
  argparse's 2 (reserved for environment failures).

Known gaps carried into this release (not a changelog category — recorded
here so the release is not read as claiming more than it delivers): colleague
loads **0 of 19** skills from the nested tree, blocked on two upstream defects
filed with a reproduction as
[`agentculture/colleague#494`](https://github.com/agentculture/colleague/issues/494)
(the template ships the correct shape regardless); and retrofitting
already-provisioned siblings is an explicit **non-goal**, so
`nvsh#25` stays open for the existing fleet.

## [0.8.0] - 2026-09-05

### Added

- **`validate-delivery` skill** (origin `devague`, re-broadcast by
  `guildmaster`) — the validation leg between `/assign-to-workforce` and
  `/summarize-delivery`: run the confirmed plan's behavioral tests agent-side,
  then file obligations, evidence, and behavioral deltas. Record-only; the
  `devague` CLI never runs a test.
- **`scripts/` wrappers for the five prompt-only workflow skills** — `scope`,
  `challenge`, `deviate`, `validate-delivery`, `summarize-delivery`. Each
  forwards its arguments to the `devague` CLI verbatim, so upstream owns the
  surface. Every clone now ships a complete, convention-clean skill directory
  instead of inheriting the script-less shape.

### Changed

- **Re-synced all eight `devague`-origin workflow skills** — `scope`, `think`,
  `challenge`, `spec-to-plan`, `assign-to-workforce`, `deviate`,
  `validate-delivery`, `summarize-delivery` — from devague `0.24.1`
  (`ec15362`), matching guildmaster's canonical copies. Notable upstream
  content: `/scope` fans out to read-only exploration subagents at 5+ candidate
  surfaces, and `assign-to-workforce split-plan --write` persists a durable
  gate-2 record.
- **All eight `SKILL.md` files are now byte-verbatim with upstream.** devague
  ships `type: command` on all eight itself, so nothing is added to the
  frontmatter here. The only divergence is the five wrapper scripts, which are
  additions, not edits.
- **`docs/skill-sources.md` updated for the re-sync** — count corrected from
  seven skills to eight (`validate-delivery` had no row), all eight rows
  repointed to `../guildmaster/.claude/skills/`, and pins refreshed to devague
  `0.24.1`.
- **The 2026-07-15 "vendor directly from devague" divergence is superseded.**
  That decision existed to keep guildmaster's `scripts/*.sh` wrappers out of
  this repo. The wrappers are now wanted: without them a clone ships a
  `SKILL.md` with no sibling `scripts/` and fails a
  `test_skills_convention`-style gate (guildmaster#95). The old section is
  retained, marked superseded, with its stale re-sync recipe replaced — that
  recipe would have deleted the wrappers this release adds and skipped
  `validate-delivery` entirely.

### Fixed

- **`scope.sh` usage advertised an invalid claim kind** — `--kind non-goal`
  (hyphen) is rejected by `devague capture`, which accepts `non_goal`. Anyone
  copying the wrapper's usage line got `invalid choice: 'non-goal'`. Corrected
  in both this repo and guildmaster.

## [0.7.0] - 2026-08-24

### Added

- **`resume <task-id|last> [--detach]` verb** in `ask-colleague` — pick a cut / timed-out / SIGTERM'd run back up from its persisted artifact, continuing on the original `colleague/<id>` work branch.
- **Per-seat thinking effort** in `ask-colleague` — `--effort` (acting seat), `--seat-effort S=R` (any seat), `--role NAME` (colleague#416). Rule of thumb: `--effort off` for small well-specified briefs, default for ordinary work, `xhigh` for open-ended judgement.
- **Review diff front-loading** — `ask-colleague review` embeds a filtered, bounded diff directly in the prompt instead of relying on the colleague run to fetch it.

### Changed

- **`ask-colleague` re-vendored byte-verbatim from `agentculture/colleague` @ 1.63.0** (cite-don't-import) — all five files (`SKILL.md`, `scripts/ask-colleague.sh`, `prompts/{explore,review,write}.md`). Every repo scaffolded from this template (`guild create` instantiates it) shipped the Qwen3.6-era wrapper until now.
- **Default colleague model is `unsloth/Qwen3.8-27B-NVFP4`** (was the Qwen3.6 pin). The lobes gateway on `:8001` no longer serves 3.6, so the previous default only worked via colleague's auto-refresh warning path.
- **`docs/skill-sources.md` ledger row** for `ask-colleague` updated to the 1.63.0 sync (was `2026-06-12 (colleague 1.7.0, direct)`) and its verb list extended with `plan` / `resume` / the pilot verbs.

## [0.6.1] - 2026-07-20

### Added

- **Worktree location convention** in `CLAUDE.md` — every worktree you create
  by hand (workforce fan-out lanes, scratch checkouts) lives in
  `../.worktrees.nvsh/<name>/`, one
  repo-named directory beside the checkout, replacing a shared `../worktrees/`
  folder. This workspace holds many sibling projects, so a generic shared
  folder accumulates orphaned trees from several repos at once with nothing
  indicating ownership — a stale-tree sweep can't tell a live lane from junk.
  Matches the convention already documented in sibling repo `reachy-mini-cli`.
  Adds branch-prefix guidance (scope the prefix to the work; plain `agent/*`
  collides with leftovers from earlier fan-outs and fails `git worktree add
  -b`), and notes that the vendored `assign-to-workforce` skill uses both the
  shared path *and* `agent/<task-id>` branches in its fan-out example — it is
  cited verbatim and must not be edited, so both are overridden when following
  it. Teardown guidance names `git worktree remove <path>` as the verb that
  actually deletes a worktree; `git worktree prune` only clears metadata for
  directories that are already gone. Tool-managed throwaways are explicitly
  out of scope: `ask-colleague`'s read-only verbs create a detached worktree
  under `${TMPDIR:-/tmp}` and reap it on an EXIT trap, so they never persist
  to need an owner.

## [0.6.0] - 2026-07-18

### Added

- **Four devague-origin skills re-vendored into `.claude/skills/`**
  (cite-don't-import), synced to the fixed devague source
  (devague#74/#75/#76):
  - `challenge` — a risk-scaled blind-spot discovery pass that runs between
    `/think` and `/spec-to-plan`, routing findings back through the existing
    deterministic moves as human-adjudicated proposals.
  - `scope` — the idea→scope leg that surveys the surfaces an idea touches
    before framing, seeding the Announcement Frame with provenance-backed
    boundary/non-goal/assumption claims.
  - `deviate` — stops an in-flight `assign-to-workforce` run when execution
    must diverge from the confirmed plan and records the divergence as a
    first-class, append-only deviation record.
  - `summarize-delivery` — closes the loop after an `assign-to-workforce`
    run with a planned-vs-actual accountability artifact.

  These four originate in `devague` and are re-broadcast via guildmaster; see
  `docs/skill-sources.md` for provenance.

## [0.5.0] - 2026-06-24

### Added

- **Memory-discipline "Conventions and workflow" section in `CLAUDE.md`** — a
  per-task *recall-before / remember-after* convention (scope localized to this
  repo's nick) so the vendored `remember` / `recall` skills are actually used,
  not just present: `/recall` before non-trivial work to build on prior
  decisions instead of re-deriving them, and `/remember` when a non-obvious
  decision, constraint, fix-and-why, or hard-won gotcha surfaces. The section
  documents this repo's memory as **in-repo and public** — records resolve to
  `<repo-root>/.eidetic/memory` (committed, team- and mesh-shared). Inserted
  idempotently (skipped if already present), slotted under an existing
  "Conventions and workflow" heading when one exists, else appended.

### Changed

- **Refreshed the `remember` + `recall` wrappers from eidetic-cli 0.10.0**
  (cite-don't-import) — picks up eidetic's **project-local store default**: the
  files backend now resolves per record by visibility — PUBLIC records inside a
  git repo go to `<repo-root>/.eidetic/memory` (committed, team-shared), PRIVATE
  records (or any record outside a repo) go to `$HOME/.eidetic/memory` (never
  committed), an explicit `EIDETIC_DATA_DIR` still wins, and recall reads both
  stores and merges. Also carries the 0.9.3 hardening (interactive-stdin guard,
  `help` as a search term, SIGPIPE-safe suffix parsing). **Recipe policy
  override (the wrappers here are NOT byte-verbatim):** the injected default
  visibility is flipped from eidetic's `private` to **`public`**, so a plain
  `/remember` lands the note in `./.eidetic/memory` in this repo, kept as part
  of the repo — pass `--visibility private` to route a record to `$HOME`
  instead. `remember` drives `eidetic remember` (idempotent upsert of one JSON
  record or an NDJSON batch on stdin); `recall` drives `eidetic recall` with
  four search modes (exact / approximate / keyword / hybrid). Each `SKILL.md` is
  localized only in the illustrative `--scope <nick>` examples (Provenance keeps
  "First-party to eidetic-cli"). Runtime dep: the `eidetic` CLI on PATH (else a
  local eidetic-cli checkout with `uv`) — **`eidetic >= 0.10.0`** for the
  in-repo routing; on an older CLI the public records still work but are stored
  in `$HOME/.eidetic/memory` instead of in-repo. Propagated by rollout-cli's
  `eidetic-memory` recipe.

## [0.4.0] - 2026-06-23

### Added

- **Vendored the `remember` + `recall` memory skills from eidetic-cli**
  (cite-don't-import) — the write/read halves of eidetic's shared
  `$HOME/.eidetic/memory` surface, so this agent (Claude and its colleague
  backend) can persist facts across sessions and recall them later, sharing
  one store.
  `remember` drives `eidetic remember` (idempotent upsert of one JSON record or
  an NDJSON batch on stdin, dedup by id + content hash); `recall` drives
  `eidetic recall` with four search modes — exact / approximate / keyword /
  hybrid — each hit carrying text, full provenance metadata, a relevance score,
  and a freshness signal. The `.sh` wrappers are byte-verbatim from eidetic-cli
  (their first-party origin); each `SKILL.md` is localized only in the
  illustrative `--scope <nick>` examples (Provenance keeps "First-party to
  eidetic-cli"). Both default to this agent's PRIVATE scope, reading the suffix
  from `culture.yaml`. Runtime dep: the `eidetic` CLI on PATH (else a local
  eidetic-cli checkout with `uv`). Propagated by rollout-cli's `eidetic-memory`
  recipe.

## [0.3.4] - 2026-06-20

### Fixed

- Identity docs and self-description strings still claimed `backend: claude`
  (prompt file `CLAUDE.md`), but this template was promoted to a colleague
  resident in #14/#15: `culture.yaml` declares `backend: colleague` (Qwen) with
  `AGENTS.colleague.md` as the resident prompt. Corrected the stale claim in
  `CLAUDE.md` (Identity section), `README.md`, `docs/skill-sources.md`, and the
  two CLI description strings (`overview` artifacts and `explain doctor`). The
  `doctor` backend→prompt-file mapping and the tests were already on
  `colleague`; this aligns the prose and self-description with them.

## [0.3.3] - 2026-06-20

### Fixed

- pyproject.toml: correct the `license` field and PyPI classifier from MIT to
  Apache-2.0 to match the `LICENSE` file. The README License section was already
  corrected in 0.3.2, but the package metadata was missed; the built wheel now
  reports `License-Expression: Apache-2.0`.

## [0.3.2] - 2026-06-18

### Added

- ask-colleague skill: `monitor`/`guide`/`stop` pilot verbs plus a `--watch`
  flag to dispatch, watch the live feed of, send mid-flight guidance to, and
  cooperatively stop a running colleague flight (re-vendored from colleague).

### Changed

- README: correct the License section from MIT to Apache 2.0 to match the
  `LICENSE` file.

## [0.3.1] - 2026-06-13

### Changed

- CLAUDE.md: add a convention to reach for the `ask-colleague` skill reflexively
  for explore/review/write/grade — read-only `review`/`explore` are always safe;
  side-effecting `write` needs the user's go-ahead.

## [0.3.0] - 2026-06-13

### Added

- AGENTS.colleague.md resident prompt file (backend colleague <-> AGENTS.colleague.md)

### Changed

- Promote agent identity to a colleague resident: culture.yaml backend
  claude -> colleague with a pinned model. The `doctor` backend-consistency
  map gains `colleague` -> AGENTS.colleague.md.

## [0.2.1] - 2026-06-12

### Changed

- **Re-vendored the `ask-colleague` skill from colleague (now 1.7.0, up from the
  0.39.2 sync)** — the wrapper had drifted multiple releases behind origin. Picks
  up the `clean` verb (reap stale/corrupt `colleague/*` branches + orphaned
  `.colleague/` artifacts a crashed run left behind), the `--json` flag on every
  verb (result JSON on stdout, diagnostics/digest on stderr), the
  `_colleague_via_uv` local-dev resolution that honors `--repo`, and the
  tri-state (0/1/2) exit-code contract. `scripts/ask-colleague.sh` + `prompts/`
  are byte-identical to the origin; `SKILL.md` diverges only in the one
  consumer-identifying Provenance clause (`nvsh vendors from
  guildmaster`). `docs/skill-sources.md` sync row updated to
  `2026-06-12 (colleague 1.7.0, direct)`. Refs: colleague#183, #186.

## [0.2.0] - 2026-06-06

### Added

- **`ask-colleague` skill** (`.claude/skills/ask-colleague/`) — the first-party front door to the `colleague` CLI (the renamed `convertible`). On top of `explore` / `review` / `write` it adds a `feedback` verb (grade a finished work item — the ROI loop), and `write` now **previews by default** in a throwaway worktree (no side effects) unless `--apply` / `--pr` is given. Reach for it reflexively — `review` for a diverse second opinion on a committed diff before opening a PR, `explore` for a fresh read of an unfamiliar area.

### Changed

- **Replaced the `outsource` skill with `ask-colleague`.** `outsource` was renamed to `ask-colleague` upstream ([colleague#148](https://github.com/agentculture/colleague/pull/148)). Because guildmaster has not re-broadcast the rename yet (its kit still ships the old `outsource`), `ask-colleague` is vendored **directly from the sibling `colleague` checkout** rather than from guildmaster — a tracked local divergence recorded in `docs/skill-sources.md`, parallel to the `agex` → `devex` one. Vendored verbatim except one consumer-identifying clause in the Provenance paragraph.
- **Ledger + CLAUDE.md + `.gitignore`:** point `docs/skill-sources.md` and the CLAUDE.md Skills section at `colleague` / `ask-colleague`, swap the *optional* runtime prerequisite `convertible` → `colleague` (env prefix `CONVERTIBLE_*` → `COLLEAGUE_*`, with the legacy names kept as a deprecated fallback), and gitignore the `.colleague/` run-artifact dir the skill writes (plus the stale `.agex/`).

## [0.1.4] - 2026-05-31

### Added

- **Vendor the `outsource` skill** (`.claude/skills/outsource/`) from
  guildmaster's canonical copy (origin
  [`agentculture/convertible`](https://github.com/agentculture/convertible),
  re-broadcast via guildmaster — guildmaster
  [#51](https://github.com/agentculture/guildmaster/pull/51)). Every agent
  cloned from this template now inherits the ability to hand a scoped task to a
  *different* engine/mind: `explore` (read-only investigation), `review` (a
  diverse second opinion on the committed diff), and `write` (delegate a small
  implementation). `explore`/`review` run isolated in a throwaway `git worktree`;
  `write` refuses a dirty tree. Fulfils
  [#8](https://github.com/agentculture/nvsh/issues/8).
- **Ledger + CLAUDE.md:** record `outsource` in `docs/skill-sources.md`
  (origin = convertible, re-broadcast via guildmaster; vendored verbatim — it
  already carries `type: command`) and document its *optional* runtime
  dependency on the `convertible` CLI (the skill exits with an install hint if
  absent, so a clone that never uses it is unaffected).

### Changed

### Fixed

## [0.1.3] - 2026-05-31

### Changed

- Expanded the clone-and-rename instructions in `CLAUDE.md`: added `README.md` to
  the rename targets and a portable `git grep` discovery command so a cloner can
  find every occurrence of the template name (hard-coded in ~100 places across the
  package, including the CLI command files and `_ISSUES_URL` in
  `nvsh/cli/__init__.py`) rather than renaming by hand.
- Synced `README.md`'s "Make it your own" checklist with `CLAUDE.md`: it now lists
  `README.md` itself as a rename target and points to `CLAUDE.md`'s discovery
  command as the authoritative procedure, so the two onboarding checklists no
  longer drift.

## [0.1.2] - 2026-05-30

### Changed

- Renamed the PR-lifecycle CLI references `agex` / `agex-cli` to `devex` (same
  tool, new name) across `CLAUDE.md`, `docs/skill-sources.md`, `.gitignore`, and
  the vendored `cicd`, `assign-to-workforce`, and `communicate` skills — the
  `cicd` scripts now invoke `devex pr`.
- Logged the vendored-skill in-place patch as a local divergence in
  `docs/skill-sources.md`; the matching canonical rename is tracked upstream for
  guildmaster in
  [agentculture/guildmaster#48](https://github.com/agentculture/guildmaster/issues/48)
  so a future re-sync reconciles cleanly.
- Aligned the documented `devex` version floor to `>=0.21` across the vendored
  `cicd` `SKILL.md` and `workflow.sh` install hint (were `>=0.1`), matching
  `docs/skill-sources.md` and the `await`-era feature set; flagged upstream on
  guildmaster#48.

### Fixed

- SonarCloud now reports code coverage — added `relative_files = true` to
  `[tool.coverage.run]` so `coverage.xml` emits repo-relative paths that map to
  `sonar.sources=nvsh` (absolute / `.venv` paths were dropped
  as unmappable). Mirrors the sibling `convertible` setup.

## [0.1.1] - 2026-05-26

### Changed

- **CI gates on the SonarCloud quality gate**
  ([issue #3](https://github.com/agentculture/nvsh/issues/3)) —
  added `sonar.qualitygate.wait=true` to `sonar-project.properties` so a failing
  gate fails the `test` job when `SONAR_TOKEN` is set. Token-less repos and fork
  PRs remain green (the scan step is guarded by `if: env.SONAR_TOKEN != ''`).

## [0.1.0] - 2026-05-26

### Added

- **Onboarded into the AgentCulture mesh** ([issue #1](https://github.com/agentculture/nvsh/issues/1)).
- **Agent-first CLI** cited from teken's (`afi-cli`) `python-cli` reference
  (`teken cli cite`) — verbs `whoami`, `learn`, `explain`, `overview`, `doctor`,
  and the `cli` noun group. Runtime is self-contained (`dependencies = []`);
  `teken>=0.8` is a dev dependency only. Passes the seven-bundle agent-first
  rubric (`teken cli doctor . --strict`). `doctor` checks the agent-identity
  invariants (prompt-file-present, backend-consistency, skills-present).
- **Mesh identity**: `culture.yaml` (`suffix: nvsh`,
  `backend: claude`) and the matching `CLAUDE.md` prompt file.
- **Canonical guildmaster skill kit** (11 skills) vendored under
  `.claude/skills/` (cite-don't-import): `agent-config`, `assign-to-workforce`,
  `cicd`, `communicate`, `doc-test-alignment`, `pypi-maintainer`, `run-tests`,
  `sonarclaude`, `spec-to-plan`, `think`, `version-bump`. Every `SKILL.md`
  carries `type: command` (load-bearing for the culture/claude backend);
  `cicd` / `communicate` consumer-identifying prose adapted, all script bodies
  verbatim. Provenance in `docs/skill-sources.md`. Three skills (`think`,
  `spec-to-plan`, `assign-to-workforce`) originate in `devague`, re-broadcast
  via guildmaster.
- **Build + deploy baseline**: `pyproject.toml` (hatchling), `tests/` (pytest,
  xdist, coverage), `.github/workflows/{tests,publish}.yml` (CI rubric/lint gate,
  PyPI Trusted Publishing), `.flake8`, `.markdownlint-cli2.yaml`,
  `sonar-project.properties`, and `.claude/skills.local.yaml.example`.

### Changed

### Fixed
