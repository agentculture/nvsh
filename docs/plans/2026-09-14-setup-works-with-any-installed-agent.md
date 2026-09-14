# Build Plan — setup works with any installed agent

slug: `setup-works-with-any-installed-agent` · status: `exported` · from frame: `setup-works-with-any-installed-agent`

> nvsh setup works with the agent you already have: it picks an installed harness (claude, codex, qwen, kiro, agy or pi) or takes `--agent <target>`, offers to install only what that pick needs, mentions an API key only when openai-compat is actually the pick, and the README says how in the devague announcement-first shape

## Tasks

### t1 — registry: probe() and an installed-first choose()

- instruction: Files: nvsh/agent/registry.py, tests/`test_agent_registry.py` only. Add probe() next to installed(); derive `tool_calling` from the adapter's capabilities the same way `build_adapter_rows` does (nvsh/cli/`_commands`/agent.py) without importing the CLI module. The reason string must list the probe result verbatim, e.g. 'pi not on PATH; installed: claude, codex; picked claude'. Do not touch setup.py.
- covers: c3, h2
- acceptance:
  - registry.probe(which) returns the installed adapters (openai-compat excluded) ordered tool-calling adapters first, then ADAPTERS order, each row carrying name, hosted and `tool_calling`
  - registry.choose() with default config and which={claude,codex} returns ('claude', reason naming the probe); which={} still returns openai-compat; a configured-and-installed provider still wins
  - tests/`test_agent_registry.py` has a table over which sets {}, {claude}, {claude,codex}, {qwen,claude}, {pi,claude}; existing tests keep passing

### t2 — README rewrite in the announcement-first shape

- instruction: Files: README.md, tests/`test_docs_architecture.py` only. Model it on ../devague/README.md: commands and tables, not prose. Set up shows 'nvsh setup', 'nvsh setup --agent claude' and 'nvsh setup --agent codex'. Do not edit docs/architecture.md.
- covers: c6, h5, c13, h9, c18, h12, c28, h20
- acceptance:
  - README.md H2s in order: Install, Set up, Work with it, Safety first, What nvsh never does, What lands where, License; first paragraph is the announcement naming an operator who already has one agent CLI
  - wc -l README.md < 120; no paragraph over 4 lines outside Safety first; no '\](docs/' or '\](CLAUDE.md' relative links; no ~/ paths; markdownlint-cli2 README.md exits 0
  - Safety first says nothing leaves the machine except to the agent you trust, that with claude or codex that agent is a third party, and that redaction runs first
  - tests/`test_docs_architecture.py` gains a test asserting the H2 order and the absence of relative docs links

### t3 — installers: offers scoped to the pick, harness install specs

- instruction: Files: nvsh/installers.py, tests/`test_installers.py` only. Keep TOOLS as the uv/tmux/node/pi table but make node and pi conditional on chosen == 'pi' (node also when the chosen harness `needs_node` and npm is missing). Put the npm package names in one dict in installers.py, not registry.py, so this stays file-disjoint from the registry task. A curl-pipe-sh is still never executable.
- covers: c2, h1
- acceptance:
  - installers.`missing_tools`(which, chosen='claude') never returns pi or node when claude is on PATH; chosen='pi' keeps today's node+pi behaviour; uv and tmux are offered regardless
  - installers.`harness_install_step`(name, which) returns an executable npm step for claude (@anthropic-ai/claude-code), codex (@openai/codex), qwen (@qwen-code/qwen-code), the existing pi command, and a non-executable 'no known installer' step for agy, kiro and openai-compat
  - tests/`test_installers.py` covers every branch above; no third-party import is added

### t4 — setup: --agent, probe-and-ask, sticky-fallback fix, daemon stop

- instruction: Files: nvsh/cli/`_commands`/setup.py, tests/`test_cli_setup.py` only. Inject the prompt like `run_install` injects confirm (a callable defaulting to input()) so tests never read stdin. Resolve --agent through registry.choose(cfg, forced=...) which already fails loudly. Keep `_write_rc_block` untouched. Reuse `_stop_daemon` from the uninstall path.
- depends on: t1, t3
- covers: c4, h3, c24, h18, c8, c20, h14, c22, h16
- acceptance:
  - nvsh setup --agent codex/gpt-5/high writes \[aliases\].default = 'codex/gpt-5/high'; with codex off PATH it raises CliError exit 2 naming the binary and writes nothing
  - no --agent: one probed harness becomes default silently; several on a tty prompt once with the ordered list (non-tool-calling rows marked 'read-only / plan mode'); --yes does not answer the pick; non-tty/--json takes the first row
  - two consecutive setups, first which={} then which={claude}, end with \[aliases\].default = 'claude'; a kept default is re-probed only when it is openai-compat with no `base_url`
  - install offers come from installers.`missing_tools`(chosen=`<pick>`); setup with only claude on PATH has no pi row and `key_hint` None; after writing the alias setup calls `_stop_daemon` and reports `daemon_stopped`
  - tests/`test_cli_setup.py` covers one-harness, several-harnesses (tty prompt and non-tty), --agent hit and miss, and the sticky-fallback pair; no test still asserts pi-always-offered

### t5 — agent verbs: use and install cover all eight adapters

- instruction: Files: nvsh/cli/`_commands`/agent.py and its tests (tests/`test_cli.py` or a new tests/`test_cli_agent.py`) only. Route through installers.`run_install` so the audit log gets a row, replacing the inline subprocess.run.
- depends on: t3
- covers: c5, h17
- acceptance:
  - nvsh agent use `<name>` accepts every key of registry.ADAPTERS and its help string lists all eight
  - nvsh agent install `<name>` prints the step from installers.`harness_install_step`; runs it only after confirm or --yes when executable; agy and kiro print 'no known installer'
  - tests parametrize both verbs over registry.ADAPTERS

### t6 — doctor reports the resolved default target; prompt files drop the pi default

- instruction: Files: nvsh/`doctor_checks.py`, tests/`test_doctor`\*.py, the five prompt files only. Wording change in the prompts is one sentence each; keep CLAUDE.md's 'Nemotron must not be hard-wired' line.
- depends on: t4
- acceptance:
  - doctor's `agent_configured` check reports the backend \[aliases\].default resolves to (falling back to \[agent\] provider) and its message names which one it used
  - CLAUDE.md, QWEN.md, AGENTS.override.md, AGENTS.colleague.md and .pi/SYSTEM.md say setup picks the installed harness; uv run python scripts/harness-smoke.py --stage config --require config passes

### t7 — setup: hosted line, reachability report, macOS/zsh warning

- instruction: Files: nvsh/cli/`_commands`/setup.py (after the setup core task merges), tests/`test_cli_setup.py`. Monkeypatch platform.system and the SHELL env in tests. The reachability call must be injectable and default to a 2-second budget so setup never hangs on a dead endpoint.
- depends on: t4
- covers: c26, h19, c17, h10
- acceptance:
  - setup --json carries agent.hosted; for a hosted pick the text output contains '`<name>` is hosted: on a failure the redacted command, output and device context leave this machine' and pi has no such line
  - setup runs `doctor_checks`.`check_agent_reachable` for the pick and reports agent.reachable {passed, message} without changing the exit code
  - when platform.system() == 'Darwin' or SHELL ends in zsh, setup output and --json 'warnings' carry 'nvsh is not tested on macOS/zsh yet (see issue #11)' and the rc path is still the bash rc

### t8 — explain catalog follows the new setup and agent surface

- instruction: Files: nvsh/explain/catalog.py and the test that greps it. Keep the entries factual and short; the catalog is what the rubric and the README point at.
- depends on: t7, t5
- covers: c7, h6
- acceptance:
  - nvsh explain setup mentions --agent, the probe-and-ask rule, the hosted line and the macOS/zsh warning; nvsh explain agent install lists the install targets
  - uv run teken cli doctor . --strict passes; catalog wording is asserted in the CLI introspection tests

### t9 — verification, before-state evidence, version bump

- instruction: Files: docs/verification.md, pyproject.toml, CHANGELOG.md. Use the version-bump skill. The fleet run is agent-side and its output is pasted verbatim into docs/verification.md.
- depends on: t2, t8, t6
- covers: c1, h11, c19, h13, c21, h15, c11, h8, h7
- acceptance:
  - docs/verification.md records: on main, setup --no-install --json with only claude on PATH reported openai-compat + `key_hint` + a pi offer while agent list showed claude installed (the challenge-pass probe); on the branch, setup then a failing command reaches claude with no pi or key prompt, run on one fleet machine
  - grep -rn 'settings.json|config.toml|trust' nvsh/ shows reads or docstrings only
  - uv run pytest -n auto, black, isort, flake8, bandit, markdownlint and teken doctor all pass; pyproject version bumped minor with a CHANGELOG entry

## Risks

- [unknown_nonblocking] npm install -g may need sudo or an nvm shim on a given machine; the executable flag cannot know in advance, so a failed harness install must surface its returncode and `output_tail` rather than be retried (task t2)
- [unknown_nonblocking] qwen over ACP and kiro may need a first interactive login a headless reachability probe cannot detect (frame park v3) (task t5)
- [follow_up] the rc block's nvsh() shell function calls 'command nvsh', not "$`NVSH_BIN`"; when nvsh is installed outside PATH (a venv, uv tool dir not yet on PATH) the hook still fires but 'nvsh on/off' at the prompt fails with command not found (seen in the t9 fleet run on spark)
