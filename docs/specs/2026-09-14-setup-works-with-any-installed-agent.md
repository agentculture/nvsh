# setup works with any installed agent

> nvsh setup works with the agent you already have: it picks an installed harness (claude, codex, qwen, kiro, agy or pi) or takes `--agent <target>`, offers to install only what that pick needs, mentions an API key only when openai-compat is actually the pick, and the README says how in the devague announcement-first shape
> instruction: docs/verification.md gains this scenario run on one fleet machine

## Audience

- an operator on a Jetson or DGX Spark who already has one agent CLI installed (claude, codex, qwen, kiro, agy or pi) and wants nvsh to use it, plus the reader landing on README.md from PyPI or GitHub
  - instruction: README opens by naming this operator; nvsh setup on a box with only claude installed picks claude with no pi or API-key mention

## Before → After

- Before: nvsh setup always offers to install pi, falls back to openai-compat and prints an API-key hint whenever pi is off PATH, has no --agent flag, and README.md opens with a 20-line status paragraph and a Goal essay
  - instruction: reproduce on a box without pi: nvsh setup --no-install --json shows agent.name openai-compat with a `key_hint` and installs listing pi
- After: nvsh setup --agent claude (or a probe-and-ask when several harnesses are installed) writes \[aliases\].default, offers only the installs that pick needs, mentions a key only when openai-compat is the pick, warns on macOS/zsh, and README.md reads like devague's: announcement, Install, Set up, Work with it, Safety first, What nvsh never does, What lands where
  - instruction: nvsh setup --agent claude --no-install --json: agent.name == claude, `key_hint` null, installs has no pi row; markdownlint-cli2 README.md passes

## Why it matters

- the multi-harness work (0.10.0) made eight adapters first-class, but the on-ramp still assumes pi, so an operator with claude or codex is steered to install a second agent and to find an API key they do not need
  - instruction: compare nvsh agent list --json (eight installed) against nvsh setup output on the same box today

## Requirements

- nvsh/installers.py TOOLS hard-codes node+pi as always-missing helper tools, so setup offers to install pi even when claude/codex are on PATH; setup must offer installs only for the backend it picked (none when that backend is already installed)
  - instruction: tests/`test_installers.py`: which={claude} chosen=claude -> names == {uv,tmux}
  - honesty: installers.`missing_tools`(chosen=...) never returns pi or node when the chosen backend is not pi and its binary is on PATH
- nvsh/agent/registry.py choose() falls straight to openai-compat when the configured provider (default pi) is off PATH, even with claude/codex installed; the unforced pick must prefer any installed adapter and reach openai-compat only when no harness binary is on PATH
  - instruction: tests/`test_agent_registry.py` table over which sets: {}, {claude}, {claude,codex}, {pi,claude}
  - honesty: registry.choose() with default config and which={claude,codex} returns claude or codex, never openai-compat
- nvsh/cli/`_commands`/setup.py `cmd_setup` has no --agent flag; setup must accept `--agent <target>` (alias or `backend[/model[/effort]]`), fail loudly if that binary is missing, and write it to \[aliases\].default in the same run
  - instruction: tests/`test_cli_setup.py`: read config.toml after setup; assert CliError code on a which that lacks codex
  - honesty: nvsh setup --agent codex/gpt-5/high writes \[aliases\].default = 'codex/gpt-5/high' and fails with exit 2 naming the binary when codex is off PATH
- nvsh/cli/`_commands`/agent.py: 'agent use' help lists five names and 'agent install' refuses everything but pi; both must cover all eight ADAPTERS, with install commands for the npm-installable harnesses (claude, codex, qwen)
  - instruction: parametrize over ADAPTERS in tests/`test_cli.py`; assert the help string lists all eight
  - honesty: nvsh agent use `<name>` accepts every key of registry.ADAPTERS and nvsh agent install `<name>` prints an install command for pi, claude, codex and qwen and 'no known installer' for agy and kiro
- README.md is rewritten in the devague README shape: one-paragraph announcement, Install, Set up (nvsh setup --agent claude|codex|...), Work with it, Safety first (nothing leaves the machine except to the agent you trust; with claude or codex that agent is a third party; redaction runs before anything is sent), What nvsh never does, What lands where; no status/history prose
  - instruction: a test in tests/`test_docs_architecture.py` reads README.md headings; markdownlint-cli2 README.md exits 0
  - honesty: README.md has exactly these H2s in order: Install, Set up, Work with it, Safety first, What nvsh never does, What lands where (plus License), and the first paragraph is the announcement
- nvsh/explain/catalog.py `_SETUP`, `_AGENT_USE` and `_AGENT_INSTALL` text follows the new flags and install targets, or teken cli doctor --strict and the introspection tests fail
  - instruction: run the rubric gate in CI; grep the catalog text in tests/`test_cli.py`
  - honesty: uv run teken cli doctor . --strict passes and nvsh explain setup / agent install mention --agent and the new install targets
- tests/`test_cli_setup.py` and tests/`test_installers.py` encode the current behavior (missing set == {pi,node,uv,tmux}; fallback to openai-compat with a key hint) and are rewritten with the new pick rule
  - instruction: grep tests/ for 'openai-compat', 'fallback' and '{"pi", "node"' after the rewrite
  - honesty: uv run pytest -n auto passes with no test still asserting the pi-always-offered or openai-compat-fallback behavior
- nvsh setup warns (text and --json 'warnings') when the login shell is not bash or the OS is macOS: 'nvsh is not tested on macOS/zsh yet (see issue #11)'; the hook block is still written only to the bash rc
  - instruction: tests/`test_cli_setup.py` monkeypatches platform.system and SHELL; asserts warnings and the rc path
  - honesty: nvsh setup --json on a run where platform.system()=='Darwin' or $SHELL ends in zsh carries warnings: \['nvsh is not tested on macOS/zsh yet (see issue #11)'\] and still edits only the bash rc
- a fallback pick must not stick: today setup writes \[aliases\].default = openai-compat and keeps it on every later run because registry.installed() is always True for a binary-less adapter; the new probe re-runs whenever the kept default is openai-compat with no \[agents.openai-compat\] `base_url`, so installing claude after a bad first setup is picked up by the next setup
  - instruction: tests/`test_cli_setup.py`: first run which={} -> openai-compat; second run which={claude} -> claude; scratch probe in the challenge pass reproduced the sticky case
  - honesty: two consecutive nvsh setup runs, the first with no harness on PATH and the second with claude, end with \[aliases\].default = claude
- when the pick is a hosted harness (AdapterSpec.hosted: claude, codex, agy, kiro) setup prints one line: '`<name>` is hosted: on a failure the redacted command, output and device context leave this machine' -- the Safety first promise stated at the moment of choice, not only in the README
  - instruction: tests/`test_cli_setup.py` asserts the line for claude and its absence for pi; --json carries agent.hosted
  - honesty: nvsh setup --agent claude --json carries agent.hosted true and the text output contains 'is hosted'; the same for pi has hosted false and no such line
- README.md is the PyPI long description (pyproject readme = README.md); links to docs/ and issues use absolute GitHub URLs so the PyPI page has no dead links
  - instruction: grep README.md for '\](docs/' after the rewrite: zero hits
  - honesty: README.md contains no '\](docs/' or '\](CLAUDE.md' relative link after the rewrite

## Honesty conditions

- the announcement holds end to end: from a clean config on a box with claude installed, nvsh setup then a failing command reaches claude with no pi or key prompt in between
- no code path under nvsh/ opens a harness settings or trust file for writing (claude/agy settings.json, codex config.toml, kiro trust, qwen settings)
- tests/`test_docs_architecture.py`'s no-tilde-path check passes on the new README.md and docs/architecture.md is byte-identical to main
- README.md's first paragraph names an operator who already has one agent CLI installed, and nvsh setup on a box with only claude installed picks claude with no pi or API-key mention
- on main today, nvsh setup --no-install --json on a box without pi reports agent.name openai-compat with a `key_hint` and an installs row for pi (recorded as the before-state in the spec)
- nvsh setup --agent claude --no-install --json reports agent.name claude, `key_hint` null and no pi install row; markdownlint-cli2 README.md exits 0
- nvsh agent list --json reports every adapter installed on the same box where nvsh setup on main still offers pi and hints at a key
- tests/`test_cli_setup.py` covers the one-harness and several-harnesses cases with a fake which, and wc -l README.md is below 120

## Success signals

- on a box with 1 harness installed and no config, nvsh setup picks it with 0 install offers for other agents and 0 API-key lines; with >1 installed it asks once; README.md is under 120 lines with no paragraph over 4 lines outside Safety first
  - instruction: tests/`test_cli_setup.py` covers the 1-harness and >1-harness cases with a fake which; wc -l README.md < 120

## Scope / boundaries

- nvsh still never edits, creates or overrides a harness's own settings or trust files (docs/specs/2026-09-14 spec, docs/config.example.toml header); a setup --agent claude pick passes launch flags only
  - instruction: grep -rn 'settings.json\|config.toml\|trust' nvsh/ shows reads or docstrings only; tests/`test_agent_conformance.py` boundary test stays green
- README.md stays free of ~/ paths (tests/`test_docs_architecture.py` `OWNED_DOCS`) and docs/architecture.md's historical 'README.md:10-14' citation is left as-is; the README rewrite must pass markdownlint-cli2 with the repo's .markdownlint-cli2.yaml
  - instruction: git diff main -- docs/architecture.md is empty in the PR

## Non-goals

- the adapters themselves (nvsh/agent/claude.py, codex.py, qwen.py, agy.py, acp.py, pi.py, `openai_compat.py`) and the rc-block/hook mechanics (nvsh/rcfile.py, nvsh/shell/hook.bash, readline.bash) are not changed; this is a setup/selection and README change only
- the runtime package keeps zero third-party dependencies (pyproject dependencies = \[\]); harness install commands are plain argv strings, no package-manager library

## Assumptions

- nvsh/`doctor_checks.py` `check_agent_configured` reads \[agent\] provider only; it should report the resolved default target (\[aliases\].default) so doctor agrees with what setup wrote
- CLAUDE.md, QWEN.md, AGENTS.override.md, AGENTS.colleague.md and .pi/SYSTEM.md say Nemotron/pi is the default backend; their wording changes to 'the installed harness setup picked' so scripts/harness-smoke.py and the four prompts do not drift
- the daemon resolves 'default' from the Config it was constructed with (daemon.py `default_target`, no reload); after setup changes \[aliases\].default a warm daemon keeps the old harness, so setup must stop a running daemon (as uninstall does via nvsh daemon stop) or say so
  - instruction: grep nvsh/daemon.py for a config reload; if none, setup calls `_stop_daemon` after writing the alias and reports `daemon_stopped`
- installed means the binary is on PATH, not that it is logged in; an unauthenticated claude or codex pick fails on the first real failure. setup runs doctor's `agent_reachable` check for the pick and reports the result without blocking
  - instruction: reuse `doctor_checks`.`check_agent_reachable`; assert setup --json carries agent.reachable for the pick

## Scope exploration

- `s1` — `nvsh/installers.py (TOOLS tuple, lines ~140-175)`: node and pi are unconditional ToolSpec entries; `missing_tools`() lists them whenever pi is absent, regardless of which adapter setup chose or which harness is already installed
  - seeds: `c2`
- `s2` — `nvsh/agent/registry.py choose() (lines 276-332)`: configured-provider-or-openai-compat is the whole decision; installed() is checked only for the configured provider, never for the other seven adapters
  - seeds: `c3`
- `s3` — `nvsh/cli/_commands/setup.py (cmd_setup, register)`: register() adds only --rc/--json/--yes/--no-install; the agent pick is registry.choose(cfg) with no forced target, and `_agent_key_hint` already scopes the API-key line to openai-compat only
  - seeds: `c4`
- `s4` — `nvsh/cli/_commands/agent.py (cmd_agent_install, register lines 210-216)`: help text 'One of: pi, qwen, claude, codex, openai-compat' and 'Install target (only pi today)'; `cmd_agent_install` hard-codes registry.`PI_INSTALL_CMD`
  - seeds: `c5`
- `s5` — `README.md (169 lines) vs ../devague/README.md (119 lines)`: current README opens with a 20-line Status paragraph and a Goal essay; devague's opens with a two-sentence announcement then Install / Set up / Work with / Why it works / What lands where, mostly commands and tables
  - seeds: `c6`
- `s6` — `nvsh/explain/catalog.py (_SETUP lines 556-590, _AGENT_* entries 921-924)`: `_SETUP` names pi/node/uv/tmux as the fixed helper-tool set and describes the pick as registry.choose() with no --agent; CLAUDE.md says every verb needs a catalog entry the rubric checks
  - seeds: `c7`
- `s7` — `tests/test_cli_setup.py (lines 145-169, 200, 522-551), tests/test_installers.py`: assertions pin names == {pi,node,uv,tmux} and the openai-compat fallback key hint; these are the behaviors the idea changes, so the tests change with them
  - seeds: `c8`
- `s8` — `nvsh/doctor_checks.py check_agent_configured (lines 125-150)`: uses config.`agent_provider`, while setup writes its pick to \[aliases\].default and deliberately leaves \[agent\] provider alone (setup.py comment), so doctor can report pi while default resolves to claude
  - seeds: `c9`
- `s9` — `CLAUDE.md + four harness prompt files (grep 'nvsh setup', 'openai')`: all five describe setup and the pi default consistently today; CLAUDE.md's repo conventions require updating the other three whenever project facts in CLAUDE.md change
  - seeds: `c10`
- `s10` — `docs/config.example.toml header + docs/specs/2026-09-14-first-class-multi-harness-with-aliases.md`: the settings-files boundary is a confirmed claim of the shipped multi-harness spec and restated in the example config; setup choosing a harness must not cross it
  - seeds: `c11`
- `s11` — `nvsh/agent/* adapters and nvsh/rcfile.py + nvsh/shell/*.bash`: nvsh agent list --json on this box shows all eight adapters installed and working; setup's rc-block writing is independent of the agent pick (`_write_rc_block` runs before registry.choose)
  - seeds: `c12`
- `s12` — `tests/test_docs_architecture.py + .markdownlint-cli2.yaml`: README.md is in `OWNED_DOCS` for the no-tilde-path check; the README.md:10-14 string is asserted on docs/architecture.md, not README, so a README rewrite does not break it as long as architecture.md is untouched
  - seeds: `c13`
- `s13` — `pyproject.toml / CLAUDE.md 'no third-party dependencies'`: installers.py already uses only shutil and subprocess; adding claude/codex/qwen npm install specs stays within that
  - seeds: `c14`
- `s14` — `nvsh/agent/registry.py PI_INSTALL_CMD (only install command on record)`: only pi has a recorded install command; claude (@anthropic-ai/claude-code), codex (@openai/codex) and qwen (@qwen-code/qwen-code) are npm packages, agy and kiro-cli have no recorded installer
- `s15` — `nvsh/shell/hook.bash header + nvsh/rcfile.py + setup.py _default_rc`: hook is 'pure bash 5.1+' on `PROMPT_COMMAND`/PIPESTATUS/PS0 with readline bindings; `_default_rc` is hard-wired to .bashrc; nothing zsh exists on disk -- deferred to a GitHub issue, this milestone only warns on macOS/zsh
  - seeds: `c15` (rejected)
- `s16` — `docs/specs/2026-09-13-nvsh-bash-hook-agent-on-error.md line 113 + gh issue 1 lines 37-39,128`: zsh integration was explicitly deferred out of the 0.9.x hook scope, while issue 1 asks for bash/zsh (precmd) from the start; the user is now running nvsh from zsh on a mac, so the deferral ends here
  - seeds: `c15` (rejected)
- `s17` — `nvsh/installers.py _apt_step (lines 88-101) and _uv_install_commands`: the only package manager probed is apt-get (plus snap/curl for uv); on macOS which('apt-get') is None so every node/tmux step is executable=False and `run_install` never calls confirm -- this is why setup asked nothing on the mac
  - seeds: `c16` (rejected)
- `s18` — `challenge pass / lifecycle lens: setup.py keep_existing (cmd_setup) + registry.installed() + scratch probe with PATH={claude,node}`: probe: first setup -> openai-compat with key hint and a pi offer; second setup -> '\[aliases\].default = openai-compat kept'; a wrong first pick is permanent until nvsh agent use
  - seeds: `c24`
- `s19` — `challenge pass / adjacent-systems lens: nvsh/daemon.py default_target (lines 674-696), setup.py _stop_daemon`: daemon reads self.config, set once; setup already has `_stop_daemon` but only uninstall calls it
  - seeds: `c25`
- `s20` — `challenge pass / data-flow lens: registry.ADAPTERS hosted flag + README Safety first decision (q3)`: hosted is already recorded per adapter and shown by agent list, but setup's output never says the pick sends context off-box
  - seeds: `c26`
- `s21` — `challenge pass / failure-mode lens: registry.installed() + doctor_checks.check_agent_reachable (line 753+)`: a reachability probe per harness already exists in doctor and honours \[aliases\].default; setup does not call it
  - seeds: `c27`
- `s22` — `challenge pass / overlooked-actors lens: pyproject.toml readme field + README.md relative links`: the PyPI reader is an audience the frame named but the relative docs/ links only work on GitHub
  - seeds: `c28`
- `s23` — `challenge pass / security lens: installers.run_install + audit log, npm install -g for claude/codex/qwen`: clean: new harness installs ride the same confirm-then-run-then-audit path as pi; a curl-pipe-sh is never run; residual: npm -g may need sudo or an nvm shim per machine, same as pi today
- `s24` — `challenge pass / reversibility lens: uninstall leaves config.toml; nvsh agent use rewrites the default`: clean once c24 lands: a wrong pick is undone by nvsh agent use or a re-run; uninstall deliberately keeps config
- `s25` — `challenge pass / observability lens: setup agent.reason string + audit log`: clean: every pick already carries a verbatim reason and every install an audit row; the probe list itself should appear in agent.reason (covered by c23's instruction)

## Decisions

- default resolution on first setup: probe PATH for every adapter; one hit is the default; several hits on a terminal always prompt the operator with the list, tool-calling adapters first then ADAPTERS order, each non-tool-calling entry marked with a read-only/plan-mode warning; --yes answers install prompts only and never the pick; non-tty/--json takes the first entry of that ordered list; `--agent <target>` skips the probe; openai-compat is the pick only when no harness binary exists
  - instruction: encode as registry.probe() + a setup prompt; test all four branches

## Hard questions

- Do uv and tmux stay in the setup install offers? tmux is described as the inline panel and daemon target, uv only matters for a checkout install (resolved: user: yes, uv and tmux stay in the offers; user reports setup never asked about them on their mac (zsh) -- installers.py only knows apt-get, so on macOS every step is non-executable and no prompt fires)
- When several harnesses are installed and no --agent is given, what order wins: keep pi first (offline-first principle in CLAUDE.md), the ADAPTERS registration order, or prompt the operator interactively? (resolved: user: on first setup 'default' is set by probing installed harnesses and by asking; read as: probe PATH for all adapters, one hit becomes default, several hits prompt the operator with the probed list (both), --agent skips the prompt, non-tty/--json takes the first probed hit)
- Does the rewritten README keep a 'What leaves the machine' section (redaction, no network on success)? It is the one prose block a security-minded operator reads first (resolved: user: keep the section but rephrase: a 'Safety first' header, saying nothing leaves the machine except to the agent you trust, and that with claude or codex that agent is a third party)
- several hits, non-tty: ADAPTERS order is pi, qwen, qwen-p, claude, codex, agy, kiro, so a box with qwen and claude auto-picks qwen (plan mode, `tool_calling` False). Should the probe rank `tool_calling`=True adapters first and skip qwen-p (a fallback) entirely? (resolved: user: on a terminal the operator always picks from the probed list; the list is ordered with tool-calling adapters first, then ADAPTERS order; non-tool-calling entries carry a warning (read-only / plan mode); qwen-p is not skipped, just listed with the warning)
- nvsh setup --yes answers install prompts today; when several harnesses are installed does --yes also take the first probed harness, or does the pick prompt still fire? (resolved: user: --yes answers install prompts only; the harness pick prompt still fires on a terminal)

## Open parks

- [unknown_nonblocking] install commands for agy and kiro-cli are not known from this repo (registry.py has only `PI_INSTALL_CMD`); until verified, agent install for those two prints 'no known installer' rather than guessing
- [unknown_nonblocking] whether qwen's ACP session or kiro need a first interactive login that a headless setup probe cannot detect; only claude and codex auth state were reasoned about

## Resolved vagueness

- [unknown_blocking] which zsh hook behaviors carry over unchanged (output capture via script(1), OSC 133 markers, slash-command readline interception) and which need zsh-specific work is unknown until a zsh prototype runs on the mac; docs/architecture.md measured only bash — resolved: user: zsh/macOS deferred to a GitHub issue; the hook behaviors question travels with it
