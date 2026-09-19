# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). This project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.17.0] - 2026-09-19

### Added

- Tier 2 (opt-in): a small resident LFM2.5 runs a bounded inspect, then propose / explain / escalate loop (`nvsh/tiers/lfm.py`); a command failure starts there. Configure `[tiers.lfm] model` to switch it on. See `docs/tier2.md`.
- Tier 2 container launcher (`nvsh/tiers/runtime_docker.py`): the only `docker run` in nvsh, built from config and platform detection only; localhost-only port, image by digest, per-user name and port, `--gpus all` on DGX Spark and `--runtime nvidia` on Jetson, engines `llama-server`, `vllm`, `sglang`, or attach to a server you already run.
- `@lfm`, an eleventh adapter for Tier 2 alone; excluded from the setup probe and refused as a default, like `needle`.
- `nvsh uninstall` removes tier records and prefetched files, stops and removes the Tier 2 container, and says which images it left.
- `nvsh tiers bench` reports accuracy per expected operation and per phrasing class; the development corpus is now a 318-entry grid.
- LFM2.5 fine-tune recipe and dataset builder (`docs/lfm-finetune.md`, written and not yet run), a reading of the LFM Open License (`docs/lfm-license-notes.md`), and DGX Spark baselines (`docs/benchmarks/`).

### Changed

- Text is bounded before it is redacted in the tier router and Tier 2, because the redactor is slow on very long unbroken input.
- An unconfigured `@needle` or `@lfm` now says what is missing instead of naming a binary that is not on PATH.

### Fixed

- Measured and recorded: stock LFM2.5 models rarely use propose or escalate, and a LoRA-tuned Needle3 cannot ship until an upstream export fault is fixed (cactus-compute/needle#134). Both tiers stay opt-in.

## [0.16.1] - 2026-09-19

### Added

- Delivery record for the tiered local response work so far (`docs/deliveries/2026-09-19-tiered-local-response-with-needle3-and-lfm2-5.md`): a partial run — PR A (#32) and PR B (#34) merged, Tier 2 (PR C) not started. All 25 plan tasks accounted for; two of the five success targets are recorded as **not met** (correct-operation rate and should-escalate rate, measured on the DGX Spark), one as unverified.
- devague ledger: 17 behavioral obligations, 36 evidence records (34 pass, 2 fail) and 7 behavioral deltas from `/validate-delivery`; operator-approved lapses l5 to l7; `docs/current-spec.md` regenerated.

## [0.16.0] - 2026-09-19

### Added

- Tier 1 end to end (opt-in: `[tiers] enabled = true`, off by default). A request typed at the prompt (`/ask`, Ctrl+G) is first offered to Needle3, a 121M local model that picks one typed operation; nvsh grounds the arguments against the machine, renders the command from the operation table, and shows it as an ordinary proposal that states the operation and arguments it understood. Nothing runs without approval; `sudo` and destructive-command handling are unchanged. A failed command never goes to Tier 1, and `@target` / `--agent` requests bypass the tiers.
- `nvsh[needle]`, `nvsh[lfm]` and `nvsh[tiers]` install flavors. The base install keeps zero runtime dependencies.
- Needle3 runs in its own child process: started on first use, killed and restarted if it dies or hangs (10 s), unloaded when idle. Telemetry off, Hugging Face offline, `complete()` only. The pinned engine is extracted from the verified wheel and the verified stock weights are staged into an nvsh-owned directory, so a first request never downloads anything and the operator's own `cactus-needle` cache is never touched.
- Tier router (`nvsh/tiers/router.py`): floor, table, grounding, rendering, then an optional yes/no log-probability check against a local LFM2.5 (deviation d1) whose per-operation baselines are measured from the table. Records name the tier, the decline reason and the check's numbers.
- Daemon residency: a `tier` request is answered without taking the agent turn lock, so a Tier 1 answer returns while another shell's agent turn is running; models load on first use and unload after `idle_unload_seconds`. `nvsh daemon status` reports the tiers.
- The panel header names who answered: `needle`, or `needle -> claude/...` when it was escalated. Declining a tier's proposal offers to send the same request to the full agent. The audit log names the tier that proposed a command.
- `@needle` / `--agent needle`: an explicit Tier-1-only adapter. Excluded from `nvsh setup`'s probe and refused as a persisted default, like `demo`.
- `nvsh tiers stats | export | prefetch | bench`. `bench` runs the committed corpus through the real router against a fixture machine declared in the corpus (`--live` for this host) and reports per-request rows, accuracy per request kind, escalation precision/recall, latency, memory, calibration of the check, and a pass/miss/not-measured line per target.
- `nvsh doctor`: `tiers_configured`, `tier_files_present`, `tier_hashes_match`; a damaged pins file fails the check instead of crashing doctor.
- Needle3 fine-tune recipe (`docs/needle-finetune.md`, `scripts/needle-finetune/build_dataset.py`, `try_table.py`) with the adoption rule (held-out only) and the `jetson-ai-lab` naming for published weights and data (deviation d2). `docs/tiers-improving-accuracy.md`: the measured baseline, what did not work, and the data plan.

### Changed

- Measured, not assumed: through the shipped path stock Needle3 picks the right operation on 4 to 5 of 9 explicit asks and declines 2 to 3 of 5 should-escalate asks (0 wrong mutating picks, warm p95 78 ms). Rewording descriptions and reordering tools do not move it. Routing therefore stays opt-in; accuracy work is data and fine-tuning, tracked in the docs above. Loading a tuned model (tuned pin, operation-table hash check, config override) is not built yet.

## [0.15.0] - 2026-09-19

### Added

- Foundations for local response tiers (issues #30, #31): a spec, a challenged and converged plan, and the first ten tasks. Nothing here changes how nvsh behaves at the prompt yet — no tier is wired into a request, and `[tiers] enabled` defaults to `false`.
- `nvsh/ops/`: a typed operation table (16 operations, each read-only or mutating, with typed arguments and a `validate()` that never raises), per-platform rendering of an operation to an argv list (through the `spark`/`thor`/`orin` device CLIs when they are on PATH, a plain system command otherwise, and nothing at all when no single exiting command exists), and argument grounding: a service or container name from a model is only ever compared with what `systemctl`/`docker` list, never passed to a command, and an option-like or unprintable value renders nothing.
- `nvsh/tiers/`: the tier contract (`decide()` turns a model's raw calls into one validated decision or a decline with a reason; zero or several calls decline; a NaN or missing confidence counts as absent), tier measurement records (redacted, size-capped with rotation, request text stored only on opt-in), a memory floor read from `/proc/meminfo`, pinned fetch and prefetch (sha256- and size-checked downloads of the Needle3 engine and weights, bounded at the pinned size, resolved offline afterwards), and a stdlib tool-call chat client for an OpenAI-compatible server on localhost, with next-token log-probability scoring for a yes/no verifier.
- `[tiers]` and `[tiers.lfm]` config tables: routing switch, confidence floor, memory floor, idle unload, records cap, and the Tier 2 engine (`llama-server`, `vllm` or `sglang`), mode and a base URL whose host must be this machine (parsed, not prefix-matched).
- Platform detection reports the `thor` and `orin` device CLIs on PATH the way it already reported `spark`; sources are recorded in `docs/platforms.md`.
- `docs/specs/2026-09-19-tiered-local-response-with-needle3-and-lfm2-5.md`, the matching plan and implementation split under `docs/plans/`, with the measurements behind them: Needle3 selects in 28-170 ms on a DGX Spark but mis-selects on failure text and can score a wrong pick 1.0, so confidence is recorded and never the safety mechanism.

## [0.14.3] - 2026-09-18

### Changed

- The conformance suite no longer tolerates a missing stderr tail for `qwen` and `kiro` (`STDERR_TAIL_RACE` is empty), and a new test makes the race certain instead of rare by delaying each adapter's stderr reader; it fails for `pi`, `qwen` and `kiro` without the fix.

### Fixed

- A harness that dies at launch no longer loses the reason it gave. `pi`, the ACP adapters (`qwen`, `kiro`) and the codex app-server read the child's stderr on a separate thread and built the launch-failure message as soon as they noticed the process was gone, so under load the message could read `exited with code 2; no stderr` although the CLI had said exactly what was wrong (about one whole-suite run in a dozen for `pi`, roughly one in ten for ACP; it failed CI on #22). They now wait, briefly and only once the process has exited, for the reader to reach end-of-file (`_subprocess.settle_stderr`).

## [0.14.2] - 2026-09-18

### Fixed

- `openai-compat` shows a reasoning model's thoughts instead of looking hung. vLLM streams them as `delta.reasoning` (other servers: `delta.reasoning_content`) ahead of any `content`, and the adapter dropped every such chunk, so the panel sat on `... waiting for the agent` for the whole think — over a minute for Nemotron on thor on 2026-09-17, while 380 KB of stream arrived. They are now THINKING events, rendered as the same dimmed run the `claude` adapter already gets; they are never added to the reply kept for steer context. The adapter now declares `thinking=True` in its capabilities (shown by `nvsh agent list --json`) to match.

## [0.14.1] - 2026-09-18

### Changed

- Records only, no code change: the operator approved the stop-choice-prompt records (lapses l2-l4, obligations o1-o23, evidence e1-e70, deltas b1-b15), `docs/current-spec.md` is regenerated from the approved ledger with `devague today` and now describes the stop choice prompt, and the delivery summary gains an adjudication section.

## [0.14.0] - 2026-09-17

### Added

- Stopping the agent now asks first. While the agent works on a terminal, the first Ctrl+C or lone Esc pauses the panel with `nvsh: paused -- [t] steer  [s] stop  [Esc] keep going` instead of cancelling at once, so an accidental press costs nothing. `[Esc]`, or 30 s without an answer, keeps going and leaves the exit status alone; `[s]` stops exactly as before (`stopping… press again to kill`, a further press kills the process tree); `[t]` takes one line. A Ctrl+C typed at the prompt, or while a proposal or busy prompt is open, goes straight to stop. Sessions without a terminal, `TERM=dumb` and `--json` still stop at once.
- `[t]` steers the running turn on `pi` and `codex`. On every other harness it reads `[t] stop & correct`: nvsh cancels the turn, waits for it to end, and sends the correction as a self-contained follow-up request (original request, a note that the previous attempt was stopped, and the correction). If a steer-capable harness refuses the steer at runtime, nvsh says so and asks once whether to stop and correct; the text is never dropped silently.
- `Capabilities.steer`: every adapter now declares whether it can take a correction mid-turn (true only for `pi` and `codex`); it shows in `nvsh agent list --json`, and `registry.steer_capable()` reads it without starting a harness.
- `nvsh/promptkeys.py`: a timed single-key reader for prompts (select-based, discards typeahead, restores the terminal on every exit path).
- Audit log: new stop kind `keep_going`, optional `origin` (`stop_prompt` / `busy_prompt`) and `reason` (`key` / `timeout`) fields, and `correction_chars` — the length of a typed correction, never its text.

### Changed

- The correction typed at the stop prompt is redacted before it leaves the process.
- Exit status: 0 after keep going or a delivered steer; the follow-up turn's own status after stop & correct; 130 only for `[s]` on a running turn and for a kill.
- `nvsh slash --json` and `nvsh hook --json` now tell the panel they are in JSON mode, so the prompt is never shown there.
- Docs, README and all five harness prompt files describe the new behaviour; `docs/shell-integration.md` records the amendment to the reliable-agent-stop spec (claims c1, c4, c5, c17, c20, c21, c22, c34).

### Fixed

- The busy prompt no longer offers `[t] steer` on harnesses that cannot steer mid-turn: the daemon decided steerability from whether an adapter overrode `steer()`, which `openai-compat` and `agy` do only to return False. It now reads `Capabilities.steer`.
- A Ctrl+C landing while the panel is shutting down no longer raises a Python traceback. `Panel.stream` restored the previous SIGINT handler before joining its ticker thread and restoring the terminal, so a late press — typical after a double Ctrl+C on a harness whose cancel ends the turn at once, such as `openai-compat` — hit the default handler. The panel's handler now covers the whole teardown and records the press as an interrupt. Present since 0.13.0; found while tracing a flaky test.

## [0.13.1] - 2026-09-17

### Changed

- Records: the operator confirmed the reliable-agent-stop records (lapses l1-l5, obligations o1-o21, evidence e1-e43, deltas b1-b10 are now approved), `docs/current-spec.md` is regenerated from the approved ledger with `devague today`, plan risks r6 (accepted as a known limitation) and r8 (follow-up, #17) are resolved, and the delivery summary gains an adjudication section; the `nvsh ask` verb question (deviation d10) is tracked as #18.
- Regression tests drive a second `run()` after `cancel()` without a second `start()` (the conformance cases restarted the adapter and only asserted a trailing DONE, which the bug also produced), plus a daemon-level test that cancels a warm `openai-compat` turn and asks again.

### Fixed

- A polite stop (first Ctrl+C/Esc) no longer silences the warm session. The `openai-compat`, `claude`, `qwen-p`, `codex exec`, cold `agy` and `fake` adapters kept their cancelled flag set after a stop; the daemon calls `start()` once per warm session, not once per turn, so every later request printed its header and then nothing (exit 0) until the daemon restarted. `run()` now clears the flag itself. Found on thor on 2026-09-17.
- `@default <question>` is labelled `warm` in the panel header instead of `one-shot`: it names a target but still rides the daemon's warm session.
- Retyping the same failed line now reaches nvsh again. The hook told a new command from a redrawn prompt by comparing `HISTCMD`, but under `HISTCONTROL=ignoredups`/`ignoreboth` (the Ubuntu default) a line identical to the previous one is never recorded, so `HISTCMD` did not move and the retry was dropped without a word — no agent call, no held-back note. The hook now counts executed commands with a `PS0` token (expanded once per command that runs, never for an empty Enter; pure bash, no fork, no DEBUG trap), composes with an existing `PS0` (Ghostty, OSC 133), falls back to `HISTCMD` if `PS0` is replaced, and `nvsh off` takes the token back out. Found on thor on 2026-09-17.
- Review fixes on the two items above: `run()` clears the cancelled flag eagerly and hands the stream to a `_turn` generator, so a stop that lands between `run()` and the first step still ends that turn (a generator body would have erased it); the hook matches the whole `PS0` counter token rather than the variable name, does not trust a counter that was off `PS0` while the command ran (that prompt is decided by `HISTCMD`), and `nvsh off` no longer fails under `set -u` when `PS0` is unset.

## [0.13.0] - 2026-09-16

### Added

- Esc stops the agent mid-work exactly like Ctrl+C. The first press asks the harness to stop through its own channel and the panel stays up with `stopping… press again to kill`; a second press kills the harness's whole process tree (`NvshAgent.force_stop()` on every adapter, daemon `kill` control). Lone Esc is told apart from arrow/function keys (`nvsh/keys.py`); typeahead during streaming is dropped; the terminal is restored on SIGHUP/SIGTERM.
- Busy prompt: a new request from a shell whose own turn is still running, or whose owner shell is gone, offers `[t] steer` (pi, codex), `[r] replace` or `[Esc] exit` instead of queueing silently (daemon `busy` event and `busy_choice` control). Other shells' turns are never touched.
- Declined exit code `EXIT_DECLINED = 3` for exit at the busy prompt or Esc/ignore at a proposal, visible as `exit_code` in `nvsh slash --json` and in the audit log; `nvsh slash` still exits 0 and the operator's `$?` is unchanged.
- `nvsh overview` shows the daemon's active turn (shell, target, elapsed) and queue without autostarting the daemon.
- `nvsh doctor` check `agent_turn_not_hung`; `nvsh doctor --apply` force-stops a hung turn (a dead-owner turn directly, a live other shell's turn only after a confirm naming it; daemon `kill_active` control).
- Audit `event: "stop"` lines for cancel, force_kill, steer, replace, busy_exit, declined and doctor_apply.

### Changed

- Harness children (claude, qwen-p, pi, codex app-server, ACP, agy warm and cold) start in their own session, and `close()` reaps the harness's process group after the graceful stdin-close wait, so tool subprocesses are never orphaned.
- `/ask`, Ctrl+G and `@target` requests put proposals to the operator on the panel like an automatic failure does (deviation d11).
- The panel's stream loop runs the event source on a worker thread so Esc and a second press are seen even while no event arrives.

### Fixed

- Ctrl+C did not stop a one-shot agent: the client only sent a socket cancel to the daemon and never called the in-process adapter's `cancel()`.
- Harnesses that ignore their protocol cancel (pi, codex, ACP, warm agy) kept running and held the daemon's run lock for every shell until the 300 s turn cap; warm agy's cancel sent nothing to the child.
- On a hung-up terminal, end of input at a proposal read as Enter (approve); it now ignores (deviation d1).
- ACP adapter: a stale end-of-output marker in the reader queue made the next `initialize` fail after a stop.
- openai-compat: stopping a stalled stream waited out the 5 s socket timeout; it now shuts the socket down at once.
- PR review: Ctrl+C at a proposal or busy prompt stopped nothing and counted as a decline; it now stops the agent (exit 130).
- PR review: `nvsh doctor --apply` exited unhealthy after a successful repair; a delayed kill could stop a later request; `kill_active` now carries the confirmed turn identity, accepts only JSON `true` as confirmation and survives oversized shell ids.
- PR review: late output from a cancelled warm agy turn could leak into the next turn; claude/qwen-p left tool subprocesses running after a normal exit.

## [0.12.1] - 2026-09-14

### Fixed

- `tests/test_cli_setup.py::test_hook_prints_refresh_notice_once_per_session` no longer leaks one `nvsh.daemon` process per full test run: it now sets `NVSH_NO_DAEMON=1`, since the refresh notice it checks is printed by the hook itself (plan risk `r4` of the README demo recording run).

## [0.12.0] - 2026-09-14

### Added

- README opens with a recording of the failure panel in action: `docs/demos/demo-spark.svg` embedded after the tagline, with the same session on Jetson AGX Thor and AGX Orin linked beside it. The three `.cast` sources and renders are committed under `docs/demos/`.
- A `demo` harness adapter (`nvsh/agent/demo.py`, ninth in `ADAPTERS`) that replays a committed fixture (`nvsh/agent/demo_fixture.json`) through the real daemon, panel, approval loop and audit log. The reply names the detected platform kind and the script from the failing command line, and says it is scripted. `demo` is excluded from `nvsh setup`'s probe and refused as a persisted default by `nvsh agent use demo` and `nvsh setup --agent demo`; `nvsh doctor` fails `default_target_not_demo` when a hand-edited config points the default at it.
- `scripts/demo-record.py`: a sandboxed, scrubbed, re-runnable recorder (throwaway HOME and XDG dirs, pinned PS1, planted `./run-model.sh` that fails with exit 126, `chmod +x` approved with Enter, retry succeeds) that runs unchanged on every device; `scripts/demo-render.sh`: `.cast` to animated SVG through a pinned `svg-term-cli`, with the `agg` GIF fallback documented. `docs/demos/README.md` documents the three-command re-record loop; re-recording is a manual maintainer step with no CI regeneration.
- `tests/test_demo_casts.py` fails when the panel legend or the fixture wording no longer matches the committed recordings, so a panel change forces a re-record.

### Changed

- `scripts/record-cast.py` sets the pty window size with `TIOCSWINSZ`, pins the header timestamp with `--timestamp`, and starts the child from an env allowlist with `--clean-env`, so two scripted runs are diffable. `scripts/` is now linted by black, isort and flake8 in CI.
- Every "eight adapters" mention (README, the four harness prompt files, tests) now says nine.

### Fixed

- Nothing user-facing. (The recorder's scrub no longer rewrites a user or host name that sits inside the detected platform kind, e.g. `spark` inside `dgx-spark`.)

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
