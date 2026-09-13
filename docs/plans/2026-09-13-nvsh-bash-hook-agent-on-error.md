# Build Plan — nvsh bash hook + agent on error

slug: `nvsh-bash-hook-agent-on-error` · status: `exported` · from frame: `nvsh-bash-hook-agent-on-error`

> nvsh elevates your existing bash into a self-healing NVIDIA shell: install one sourced hook, keep bash exactly as it is, and when a command fails an NVIDIA-opinionated agent (Pi.dev driving the local associate model by default) appears inline with a diagnosis and a proposed fix you approve, retry and verify; /ask, /fix, /doctor and Ctrl+G reach the same agent on demand, and successful commands never pay a millisecond for it.

## Tasks

### t1 — Docs and prompts: architecture decision, platforms doc skeleton, README, four harness prompts

- instruction: Write docs/architecture.md from the spec's c2, c18, c28, c27, c43, c51 and the challenge scope entries s4-s6, s15-s19. Keep every ~/ path out of committed docs (steward portability rule). Do not touch culture.yaml, .claude/skills or .pi/skills. Bump the version with the version-bump skill in this task (patch is fine) so every later task inherits it.
- covers: c14, h12, c31, h24, c30, h23
- acceptance:
  - docs/architecture.md exists, opens with the before-state citing README.md:10-14 and the retired PTY wording of 0.9.1, records hook-over-wrap with reasons and the parked login-shell mode
  - CLAUDE.md, AGENTS.override.md, AGENTS.colleague.md, QWEN.md and .pi/SYSTEM.md all describe the hook architecture; the 'Login-shell constraints' section is replaced by 'Hook constraints'; 'uv run python scripts/harness-smoke.py --stage config --require config' passes
  - README's first paragraph names Jetson AGX Orin/Thor, DGX Spark, bash, Ghostty and ssh, marks shell verbs as in progress, and has a 'What leaves the machine' section; markdownlint-cli2 passes

### t2 — Trigger rules module nvsh/triggers.py with table-driven tests

- instruction: Pure functions, no I/O. Pipeline status: honour the shell's pipefail setting as given in the event; do not re-derive. Program class comes from the first word of the command (vim, htop, jtop, less, ssh, docker, kubectl, tmux, screen, su, sudo -i) — keep the list in one frozenset so t8 can reuse it for the hook.
- covers: c4, h3
- acceptance:
  - tests/`test_triggers.py` is a table over (command, exit, pipestatus, pipefail, `program_class`, `rate_state`) with one row per rule in CLAUDE.md's trigger list plus ssh/docker exec/kubectl exec/tmux/screen/su/sudo -i pass-through; every non-error row yields decision 'skip' with a reason
  - 130, 141, grep/diff exit 1, false, test/\[ \], and a command run while the shell is inside a script produce 'skip'; 'ls /nope' exit 2 produces 'ask'; the rate limiter refuses a second auto call inside the configured window and records it
  - nvsh/triggers.py imports only stdlib and exposes decide(event) -> Decision with fields action, reason, `rule_id`

### t3 — Redactor nvsh/redact.py with corpus tests

- instruction: Stdlib re only. Keep patterns in a list of (name, regex) so --show-context can report which rules fired. Redaction of rc files is not needed because the collector never reads them; add a test asserting the collector API has no rc reader (t9).
- covers: c11, h10
- acceptance:
  - tests/`test_redact.py` has a corpus file with `HF_TOKEN`=, --api-key, Authorization: Bearer, `OPENAI_API_KEY`=, .env-style KEY=value, ssh private-key blocks, and JSON 'apiKey' fields; every secret is replaced by a typed marker and no secret substring survives
  - redact(bytes) -> bytes handles invalid UTF-8 and escape sequences without raising; a property test asserts idempotence

### t4 — Platform detection nvsh/platform/ (file-first) plus docs/platforms.md

- instruction: Prefer file reads (/etc/dgx-release, /etc/`nv_tegra_release`, /proc/device-tree/model and compatible, /sys/class/dmi/id/`product_name`, /usr/local/cuda/version.json, /proc/driver/nvidia/version, /proc/meminfo, /usr/include/aarch64-linux-gnu/`cudnn_version`.h, /etc/docker/daemon.json). Subprocess only for nvidia-smi, nvpmodel -q, dpkg-query, docker info, with timeouts. Call 'spark status --json' when on PATH and merge, never require it.
- covers: c10, h9, c15, h13
- acceptance:
  - detect() returns a Platform with kind in {dgx-spark, jetson, rtx, generic}, every value carrying (value, `source_path_or_command`); fixture trees for spark, thor and orin (copied from the real files) make the tests reproduce DGX Spark/DGX OS 7.x/CUDA 13.0.2/driver 580.126.09, AGX Thor/R38.2/cuDNN 9.12/TensorRT 10.13, AGX Orin/R39.2 with CUDA toolkit and tensorrt reported absent
  - nvidia-smi memory \[N/A\] is mapped to unified-memory mode and MemAvailable is reported as the pressure signal; nvpmodel, cuDNN, TensorRT, tmux, pi, spark (dgx-spark-cli) are reported present/absent, never omitted
  - docs/platforms.md lists every value, its source, whether it is a file read or subprocess, and the verifying command for spark, thor and orin

### t5 — Config and approval store: nvsh/config.py, nvsh/approvals.py

- instruction: Two small modules, both stdlib. Expose Approvals.decide(cmd) -> 'user' | 'session' | 'ask' so t11's extension and t13's client share it via `nvsh approve check <cmd> --json`.
- covers: c9, h8, c50, h41, c21, h17
- acceptance:
  - config loads $`XDG_CONFIG_HOME`/nvsh/config.toml with tomllib, defaults to \[agent\] provider='pi' and \[agents.pi\] provider='nemotron' model='associate', sessions.max=1, and never reads or stores API keys; missing file yields defaults
  - approvals: approved.toml (0600) holds glob patterns; matches(cmd) uses fnmatch on the full command line; add() refuses 'sudo \*', 'rm \*', bare '\*' and patterns starting with 'sudo'/'rm'; the default list contains 'nvidia-smi \*', 'docker ps\*', 'journalctl \*', 'systemctl status \*', 'df \*', 'free \*', 'nvsh \*'; a session list lives in memory only
  - every example config under docs/ or nvsh/ uses <http://localhost> placeholders; scripts/scan-secrets.py passes

### t6 — Agent contract nvsh/agent/ with FakeAgent and conformance tests

- instruction: Keep the loop (nvsh/agent/loop.py) UI-free: it takes an approve callback and an executor. The audit log path is $`XDG_STATE_HOME`/nvsh/audit.jsonl. This is the module every later task plugs into; keep the public surface tiny.
- covers: c7, c8, h7, c48, h39
- acceptance:
  - nvsh/agent/base.py defines NvshAgent (start, run -> iterator of events, cancel, close, capabilities), AgentRequest(kind failure|slash|explicit, prompt, command, `exit_code`, `failure_id`), AgentContext, events `text_delta`|`tool_call`|`tool_result`|proposal|status|done|error; nothing imports pi
  - FakeAgent replays a scripted event list; tests/`test_agent_conformance.py` runs the same suite (streaming order, cancel mid-stream, error propagation, capability report, teardown) against every registered adapter via a fixture
  - tests/`test_approval_loop.py` drives FakeAgent through inspect -> proposal 'sudo rm -rf /x' -> the proposal is rendered and never executed until approve() returns True -> retry -> verify; the audit log (jsonl) records proposal, decision, outcome; a proposal is never written into `READLINE_LINE`

### t7 — Readline layer nvsh/shell/readline.bash: Enter macro, /+Tab palette, argument completion, Ctrl+G, all keymaps

- instruction: Bind with 'bind -m emacs', 'bind -m vi-insert', 'bind -m vi-command'. Enter: bind -x on an unused sequence plus a C-m macro ending in C-j. Completion data comes only from 'nvsh complete' (t16); never duplicate the command list in bash. HISTCONTROL: append ignorespace if absent, and document it.
- depends on: t2
- covers: c5, h28, c6, h5, c39, h32
- acceptance:
  - under script(1) with CR keystrokes, in emacs mode and after 'set -o vi': '/doctor' and '/ask why' route to ' nvsh slash `<line>`' and 'history' shows the original line only; '/notacmd', 'ls -d /tmp' and a quoted two-line echo pass through unchanged; Ctrl+G prints the panel hook and restores the line
  - '/'+Tab lists slash commands merged with real paths, '/do'+Tab completes /doctor, '/tm'+Tab completes /tmp/, '/doctor '+Tab lists its arguments from 'nvsh complete --json', 'ec'+Tab still lists echo; bash-completion's -D loader is untouched and complete -I is registered by nvsh
  - a deliberately erroring hook function still lets Enter accept the line (degrade, never lock)

### t8 — Capture layer nvsh/capture.py: session log, OSC 133 slicing, tmux pipe-pane, bounding

- instruction: The wrapper command rendered for the hook is: exec script -qfc "$BASH" "$log" with `NVSH_WRAPPED`=1 exported. Bound first, then strip, then redact (t3). Provide 'nvsh capture --show' for --show-context.
- depends on: t3
- covers: c29, h21, c47, h38
- acceptance:
  - `open_session_log`() creates $`XDG_RUNTIME_DIR`/nvsh/`<shell-pid>`.log with mode 0600, refuses to start when already under script/tmux or when `NVSH_WRAPPED` is set, and removes the log at shell exit; killing script(1) mid-session is simulated and the reader returns 'no capture' without raising
  - `last_slice`(log) returns exactly the bytes between the last OSC 133 C and D markers (fixture from the scratchpad experiment: 'hello-out' + 'ls: cannot access'), capped at 64 KB with head+tail and a truncation marker, escape sequences stripped, invalid UTF-8 replaced; the log path never appears in the returned context
  - inside tmux ($TMUX set) the source is 'tmux pipe-pane -o' to the same log; a test proves only the slice, never the whole log, reaches the redactor

### t9 — PiAgent nvsh/agent/pi.py: rpc subprocess, launch hygiene, abort

- instruction: subprocess.Popen with pipes, a reader thread and a queue; no asyncio. Ship the fake pi as tests/fakes/pi so CI never needs node. Capabilities: streaming, toolCalling, cancellation, persistentSession, localModel true.
- depends on: t6
- covers: c7, h6, c41, h33, c46, h37
- acceptance:
  - PiAgent builds argv 'pi --mode rpc --no-context-files --no-extensions --no-skills --no-prompt-templates --no-approve --session-dir $`XDG_STATE_HOME`/nvsh/pi-sessions -e <approval ext>' (asserted in a test); the reader splits on LF only and tolerates U+2028; `message_update` text deltas, `tool_execution_`\* and `extension_ui_request` map to NvshAgent events
  - cancel() sends {type: abort} and returns within 1 s; a killed pi process yields an error event and never hangs the caller; after a run no file appears under ~/.pi/agent/sessions (test uses HOME in tmp)
  - the conformance suite from t6 passes against PiAgent using a fake 'pi' script on PATH that speaks the rpc protocol from docs/rpc.md

### t10 — Harness chooser and adapters: nvsh/agent/registry.py, `openai_compat.py`, claude.py, codex.py, qwen.py

- instruction: Registry is a dict of factories like `culture_core`/cli/agents.py:788. Do not add dependencies. The install offer for pi runs 'npm install -g @earendil-works/pi-coding-agent' only after a keypress and only when npm exists.
- depends on: t6
- covers: c24, h18, c16, h14
- acceptance:
  - 'nvsh agent list --json' reports pi, qwen, claude, codex, openai-compat with installed status from PATH; 'nvsh agent use `<name>`' writes config; with none installed the failure panel text offers 'install pi' or 'choose another harness'
  - OpenAICompatAgent uses stdlib urllib against an OpenAI-compatible /v1/chat/completions endpoint with streaming, 5 s connect timeout, bearer from an env var name in config (never a literal), and passes the conformance suite against a local fake HTTP server
  - claude ('claude -p --output-format stream-json'), codex ('codex exec --json') and qwen ('qwen -p') adapters are thin subprocess mappers with capability declarations and pass the conformance suite via fake scripts on PATH; 'nvsh setup' on a host without node (orin) selects openai-compat and prints why

### t11 — pi approval extension nvsh/agent/`pi_ext`/approval.ts with user/session allowlists

- instruction: Keep the TypeScript under 120 lines and dependency-free (pi's ExtensionAPI only, per docs/extensions.md:1338-1387 and 751-793). All policy lives in Python (t5); the extension only forwards.
- depends on: t5, t9
- covers: c50, h41
- acceptance:
  - the extension registers a `tool_call` handler: for the bash tool it runs 'nvsh approve check `<cmd>` --json'; 'user' or 'session' allows, 'ask' raises `extension_ui_request` with choices once/session/user; a blocked call returns {block: true, reason}; non-bash tools are blocked in v1
  - a Node-free test parses the .ts for the required handlers; a live test (skipped when pi is absent) drives the fake-agent script from t6 through nvidia-smi (runs), 'apt install foo' (asks; approve-for-session then runs; new daemon asks again) and 'sudo rm -rf /x' (asks even after approve-for-user is attempted) with all three in the audit log
  - the .ts file ships inside the wheel (hatch include) and is referenced by absolute path from PiAgent

### t12 — Session daemon nvsh/daemon.py: per-user socket, per-shell conversations, sleep/switch, fallback

- instruction: socketserver.ThreadingUnixStreamServer, JSON lines, stdlib only. Protocol: {shell, kind, request} -> stream of events. Keep the daemon runnable in the foreground for tests. Reap stale sockets at start.
- depends on: t9, t10
- covers: c43, h34, c51, h42
- acceptance:
  - starting a shell creates no nvsh or pi process; the first qualifying failure starts one daemon on $`XDG_RUNTIME_DIR`/nvsh/daemon.sock (0600) and one agent process; closing the last shell (EXIT trap) or an idle timeout stops both (pgrep-based test)
  - two simulated shells A and B: failure in A, failure in B, then '/ask what did you just see' from A answers from A's conversation only; one pi process throughout; conversations sleep via `switch_session`/`new_session` and resume on the shell's next request; sessions.max=2 allows a second process
  - with the daemon missing or crashed the client falls back to a one-shot adapter run; with pi absent the daemon reports 'pi unavailable' and uses the configured fallback adapter

### t13 — Failure client and panel nvsh/client.py + nvsh/panel.py: context assembly, streaming, approve keys, retry, verify, Ctrl+C

- instruction: This is the user-visible piece: keep first text on screen as soon as the first `text_delta` arrives. Use the loop from t6, the approvals from t5 for allowlisted read-only inspectors the client runs itself, and the daemon protocol from t13 with the one-shot fallback.
- depends on: t8, t4, t12
- covers: c1, h1, c33, h26, c38, h31, c46, h37
- acceptance:
  - the client assembles AgentContext from the hook args, the redacted output slice, and the platform block (every value with its source); 'nvsh context --show' prints exactly the bytes that would be sent; a test with sockets disabled proves a localhost endpoint is still reached (air-gapped mode)
  - the panel renders with hard-coded SGR only, never tput; under TERM=dumb, `NO_COLOR`=1, a non-tty, and TERM=xterm-ghostty without terminfo it prints readable text without errors; keys: Enter approve, e explain, d details, Esc ignore; a proposal is shown as the exact command and never pre-typed
  - Ctrl+C while streaming aborts the agent run, restores the terminal and returns to a working prompt within 1 s; '/fix' afterwards still finds the last failure (state file under $`XDG_STATE_HOME`/nvsh/last-failure.json); retry re-runs only after Enter and verify reports the new exit status

### t14 — Slash command registry and verbs: nvsh/slash.py, 'nvsh slash', 'nvsh complete', /ask /fix /explain /retry /context /agent /help /undo /approve

- instruction: Register commands in one module; the CLI verbs are thin. /help prints the registry. Keep /remember-good etc. out (deferred, c22).
- depends on: t13
- covers: c5, c6
- acceptance:
  - SlashCommand registry (name, aliases, description, arg schema, completion provider, handler, safety policy) drives 'nvsh complete --json' ({items: \[{value, description}\]}) with platform-aware filtering (Jetson-only /power /clocks hidden on the Spark) and '/doctor `<Tab>`' arguments; the bash layer holds no command list
  - 'nvsh slash "/ask why"' routes to the client with kind slash; /undo drops the last agent turn and its proposal from the conversation and never runs anything on the machine (test asserts no executor call); /approve manages the session and user lists; each verb has a catalog entry and 'uv run teken cli doctor . --strict' passes

### t20 — Bash hook core nvsh/shell/hook.bash: `PROMPT_COMMAND` first, capture exec, Ghostty composition, kill switch

- instruction: Pure bash 5.1+, every function ends with '|| return 0', never set -e. Prepend with `PROMPT_COMMAND`=(`__nvsh_hook` plus the existing elements), handling the string form. Source ghostty.bash from `GHOSTTY_RESOURCES_DIR` when `TERM_PROGRAM`=ghostty and `__ghostty_hook` is undefined; emit OSC 133 C/D yourself otherwise. The capture exec wrapper from t8 is invoked from here; the log reader stays in Python.
- depends on: t2
- covers: c3, h2, c26, h19, c35, h29
- acceptance:
  - tests/`test_hook_bash.py` sources the rendered hook in bash --norc -i under script(1) with CR keystrokes: 1000 successful commands fork no nvsh process (execve counter shim on PATH) and prompt latency median under 5 ms over baseline; 'false | true' and 'true | false | true' record PIPESTATUS '1 0' and '0 1 0'; with set -o pipefail the recorded status is 1
  - with ghostty.bash sourced first, declare -p `PROMPT_COMMAND` shows the nvsh hook first and `__ghostty_hook` once; no DEBUG trap is installed by nvsh; `NVSH_DISABLE`=1 makes the file a no-op (no hook, no capture)
  - the hook calls the Python entrypoint only when the bash-side pre-filter passes (exit not in 0/130/141, non-empty line, not inside a sourced script), passing exit, PIPESTATUS, line, cwd and log path

### t21 — Setup, uninstall, on/off verbs: nvsh/cli/`_commands`/setup.py and the rc editor nvsh/rcfile.py; wheel includes

- instruction: Render hook files from package resources (importlib.resources) into the user's data dir. Every verb: register(sub), CliError, `emit_result`, --json, catalog entry. Also add the thin 'nvsh hook' verb the bash hook calls on failure (parse args, hand to t13's client).
- depends on: t20, t7, t8
- covers: c36, h30, c44, h35, c45, h36, c13, h11
- acceptance:
  - nvsh setup inserts one marked block right after the rc's interactive guard (detected by the 'If not running interactively' comment or a test on $-; fallback: top of file) containing the exec wrapper and the source of the rendered hook files, writes a timestamped backup, and is idempotent (second run: byte-identical rc); bash -ic true timing stays within 10% of the pre-setup baseline in a test with a heavy fake rc
  - the rendered hook file carries the nvsh version; a client run with an older stamp prints the refresh notice once per session; nvsh uninstall removes the block (restoring the backup if the block was edited), hook files, sockets, logs and daemons and leaves the rc byte-identical to the backup; nvsh off and nvsh on unbind and rebind in the current shell
  - the wheel built from a clean checkout contains nvsh/shell/\*.bash and nvsh/agent/`pi_ext`/\*.ts (hatch include) and nvsh setup works from a clean venv install; dependencies stays empty

### t17 — Doctor extensions: platform, backend reachability, in-shell hook health, terminfo

- instruction: Extend the existing checks list in doctor.py in the same shape; in-shell checks read state the hook exports (`NVSH_HOOK_VERSION`, `NVSH_LOG`, TMUX) plus bind -p output passed by /doctor from bash. Keep `_PROMPT_FILE` untouched (tests pin it).
- depends on: t4, t10, t21
- covers: c49, h40, c9, c38
- acceptance:
  - 'nvsh doctor --json' adds checks `platform_detected`, `agent_configured`, `agent_reachable` (distinguishing pi-missing, endpoint-unreachable, endpoint-401 with distinct remediations), `hook_sourced`, `hook_first_in_prompt_command`, `bindings_present` (per active keymap, via bind -p), `capture_active` (script or tmux, with log path and mode), `daemon_status`, `terminfo_present`; each has id, passed, severity, message, remediation
  - in a hooked test shell with the hook deliberately moved to last position, `hook_first_in_prompt_command` fails with the fix command; on a host without xterm-ghostty terminfo the remediation is the infocmp-plus-tic line; the wheel-install case (no culture.yaml) still runs every new check

### t18 — CI green: lint set, bandit, scan-secrets, teken strict, harness-smoke, coverage, invariants untouched

- instruction: This is the integration gate task: fix what the earlier tasks left red, add nothing new. Confirm the version bump from t1 is still ahead of main.
- depends on: t14, t17, t21, t11
- covers: c17, h15, c20, h16, c21, h17, c13, h11
- acceptance:
  - black/isort/flake8/bandit/markdownlint/scan-secrets/teken cli doctor --strict/harness-smoke all pass as CI runs them; pytest -n auto --cov=nvsh at or above 60%; python -X importtime of nvsh.cli lists only stdlib modules
  - git diff on culture.yaml, .claude/skills, .pi/skills and doctor.`_PROMPT_FILE` is empty; tests/`test_pi_settings.py` and tests/`test_harness_registries.py` pass; removing pi, tmux, fzf and spark from PATH in a test leaves every verb working with diagnostics only

### t19 — Three-machine verification, timing tests and demos (spark in Ghostty, orin with and without tmux, thor)

- instruction: This task is run by the operator with the agent assisting (screenshots and GIFs welcome). Anything that fails here goes back as a /deviate, not a silent fix.
- depends on: t18
- covers: c32, h25, c34, h27, c30, h23, c1, h1, c16, h14, c26, h19, c5, h28, c36, h30
- acceptance:
  - docs/verification.md holds a dated checklist executed on spark (Ghostty), thor and orin (ssh, with and without tmux): setup, no visible change on success, ls /nope panel with the real error slice, /doctor, Ctrl+G, slash-Tab palette, vi mode, uninstall restoring the rc; each line marked pass/fail with the observed output
  - tests/`test_timing.py` measures under 5 ms added on success and under 2 s to first agent text with the fake agent; the measured numbers for spark are recorded in docs/verification.md next to the targets; two recordings (Spark CUDA-OOM-style failure in Ghostty, orin missing-package over ssh) are stored under docs/demos/
  - declare -p `PROMPT_COMMAND` on the Spark shows both hooks once each with nvsh first, Ghostty jump-to-prompt works across a panel, and no flicker or duplicated prompt is observed; on orin the openai-compat fallback produces the panel

## Risks

- [unknown_nonblocking] pi rpc cold start and node RSS on Thor unmeasured; if a warm process is too heavy on Jetsons, the daemon's sessions.max default may need to be 0 there (one-shot only). Attaches to t12. (task t12)
- [unknown_nonblocking] Nemotron 'associate' behind the lobes-gateway has not been exercised for OpenAI tool calling; if it fails, the approval extension (t12) still works for text-only proposals and t15 must degrade to 'diagnosis only'. (task t11)
- [unknown_nonblocking] The Enter macro redraws the prompt line after every command; real-Ghostty flicker or duplicated OSC 133 markers would force the alternative 'run dispatch inside bind -x' variant in t8 (terminal is raw there; needs stty handling). (task t7)
- [follow_up] orin has no node: pi cannot be installed there without also installing node; t11's openai-compat fallback is the v1 answer, install-node is a follow-up. (task t10)
