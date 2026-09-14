# Delivery Summary — setup works with any installed agent

plan: `setup-works-with-any-installed-agent` · run: `complete` · date: `2026-09-14`
baseline: `devague summary skeleton`

## Intent

Make `nvsh setup` work with the agent the operator already has: probe `PATH`
for every adapter, take `--agent <target>`, offer installs only for the pick,
mention an API key only when `openai-compat` is actually the pick, and
rewrite the README in the devague announcement-first shape. The plan
(`docs/plans/2026-09-14-setup-works-with-any-installed-agent.md`, nine tasks
in five waves) was fanned out by /assign-to-workforce, one worktree per
task, TDD-gated merges, on branch `spec/setup-works-with-any-installed-agent`
(PR #12, version 0.11.0).

## Planned Work

Quoted verbatim from the `devague summary` skeleton. `t5` was briefly
omitted there after a backtick-only amend flipped it back to `proposed`; the
owner re-confirmed it at adjudication (0.11.1), so all nine tasks are
confirmed.

- `t1` — registry: probe() and an installed-first choose()
- `t2` — README rewrite in the announcement-first shape
- `t3` — installers: offers scoped to the pick, harness install specs
- `t4` — setup: --agent, probe-and-ask, sticky-fallback fix, daemon stop
- `t5` — agent verbs: use and install cover all eight adapters
- `t6` — doctor reports the resolved default target; prompt files drop the pi default
- `t7` — setup: hosted line, reachability report, macOS/zsh warning
- `t8` — explain catalog follows the new setup and agent surface
- `t9` — verification, before-state evidence, version bump

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `registry.probe()` (installed adapters, tool-calling first) and `choose()` consulting it before the `openai-compat` fallback; merge `c1b4911` |
| `t2` | delivered | README.md rewritten (103 lines, required H2 order, absolute links) plus heading-order and no-relative-link tests; merge `8ed0942`; one flag corrected afterwards (`nvsh context --show`) |
| `t3` | delivered | `installers.missing_tools(chosen=...)`, `harness_install_step()`, `NPM_PACKAGES`; merge `ab876f1` |
| `t4` | delivered | `nvsh setup --agent`, probe-and-ask with an injectable prompt, sticky-fallback fix, daemon stop; merge `0e2e5f7` (see `d1`) |
| `t5` | delivered | `agent use` over all eight adapters, `agent install` through `harness_install_step` + `run_install` + audit; merge `48a16fd` |
| `t6` | delivered | doctor `agent_configured` resolves `[aliases].default` and names its source; five prompt files reworded; merge `49a9d22` |
| `t7` | delivered | hosted disclosure line, `agent.reachable` (2 s budget, never fatal), macOS/zsh warning; merge `96e546c` |
| `t8` | delivered | `_SETUP` and `_AGENT_INSTALL` catalog entries rewritten, two introspection tests; merge `bca7d35` |
| `t9` | delivered | `docs/verification.md` 0.11.0 section (before/after probes and an end-to-end failure reaching claude on spark), 0.11.0 bump, CHANGELOG; commits `ad77ad5`, `463f887` |

## Mid-work Decisions

Both deviation records were approved by the owner at adjudication (0.11.1)
and are quoted as the recorded ground truth.

- `d1` (approved) — `t4` passes `chosen=None` to `installers.missing_tools`
  when the probe is empty (the `openai-compat` fallback), so a bare machine
  still gets the pi and node offers — scoping offers to `openai-compat` would
  leave a machine with no harness and no way to install one from setup.
- `d2` (approved) — post-wave-2 fixes by the main agent outside any task's
  file list: `registry._forced_backend` accepts a bare adapter name (the
  README and spec promise `nvsh setup --agent claude`; only `/`-targets and
  aliases resolved), and `_stop_daemon` discards the child's stdout so
  `setup --json` stays parseable. Found by the after-state probe and by
  `t7`'s reported lapse.
- The README's `nvsh context --show-context` came from the main agent's
  brief; the real flag is `--show`. Corrected on the branch after the `t2`
  merge (`commit` in the `main..HEAD` range, "docs(readme): nvsh context
  --show is the real flag"); lapse `l3`.
- This run's worktree branches are `agent/setup-tN`, because `agent/t1..t3`
  still exist from the 0.9.2 fan-out (squash-merged, so not deletable by
  ancestry); the stale branches were left alone.
- zsh/macOS was scoped out at the spec gate and filed as issue #11; this
  milestone only warns.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t4` (`d1`) | empty-probe path keeps the pi/node bootstrap offers instead of scoping to the `openai-compat` pick | acceptable |
| `t1`, `t4` (`d2`) | two spec promises (`h3` bare `--agent claude`, `h14` parseable `--json`) needed fixes in files the split had assigned to already-merged tasks | acceptable |
| `t2` | README shipped one wrong flag from the brief; fixed post-merge | acceptable |
| `t9` | fleet run was on spark only; thor and orin were not re-run (unchanged hook, same wheel) | needs-follow-up |

## Evidence

- tests: 41 acceptance-criterion tests selected across
  `tests/test_agent_registry.py`, `tests/test_installers.py`,
  `tests/test_cli_setup.py`, `tests/test_cli_agent.py`, `tests/test_cli.py`,
  `tests/test_docs_architecture.py`, `tests/test_doctor_checks.py` — pass at
  `463f887` (filed as `e1`–`e13`, `e15` via /validate-delivery)
- tests: `uv run pytest -n auto -q` — 1737 passed, 5 skipped (three runs;
  one earlier run hit a known timing-sensitive daemon test that did not
  reproduce)
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c
  pyproject.toml -r nvsh`, `markdownlint-cli2 "**/*.md"`,
  `scripts/scan-secrets.py`, `scripts/harness-smoke.py --stage config`,
  `teken cli doctor . --strict` — all clean
- manual: `docs/verification.md` "Setup works with any installed agent
  (0.11.0)" — before-state on main, after-state on the branch, end-to-end
  failure reaching claude on spark (filed as `e14`)
- commits: `a12921d..463f887` on `spec/setup-works-with-any-installed-agent`
- PRs / issues: PR #12; issue #11 (zsh/macOS, deferred)
- deltas: `b1` (approved, then superseded by `b5`) — `probe()` listed
  both `qwen` and `qwen-p` when the `qwen` binary was present; `b5`
  (proposed, with obligation `o16` and evidence `e16`) — the PR #12 review
  fix keeps one probe row per shared binary, so a qwen-only PATH picks
  `qwen` silently; `b2` (`d1`) — an empty
  probe still offers node and pi as the bootstrap path; `b3` (`d2`) — a
  bare adapter name is accepted by `--agent`; `b4` (`d2`) — the daemon
  stop's output never precedes setup's `--json` payload

## Delivery Claims

Lapses `l1`–`l9` were approved at adjudication. None touches a claim's
verification: each records a process slip (a brief fact, an assumption or a
skipped red control) that was caught and corrected before merge — `l3`'s
wrong flag was fixed in the README, and `l8`'s missing red control was
supplied by the main agent — so no confidence below is capped by them.
Obligations `o1`–`o15` and evidence `e1`–`e15` are approved.

| Claim | Confidence | Evidence |
|-------|------------|----------|
| with only claude on PATH, `nvsh setup` picks claude, offers no pi, prints no key hint, writes `[aliases].default = claude` | high | `tests/test_cli_setup.py::test_setup_one_probed_harness_becomes_the_default_silently` · `docs/verification.md` after-state · `e4`, `e14` |
| a fallback `openai-compat` default no longer sticks once a harness appears | high | `tests/test_cli_setup.py::test_setup_re_probes_a_sticky_openai_compat_default` · `e5` |
| `nvsh setup --agent codex/gpt-5/high` writes the literal target; a missing binary is exit 2 and writes nothing; bare `--agent claude` works | high | `tests/test_cli_setup.py::test_setup_agent_flag_writes_the_literal_target`, `::test_setup_agent_flag_with_the_binary_missing_is_an_env_error`, `tests/test_agent_registry.py::test_choose_forced_accepts_a_bare_adapter_name` · `e3` |
| several installed harnesses prompt once on a tty, tool-calling first, read-only rows labelled; `--yes` never answers the pick | high | `tests/test_cli_setup.py::test_setup_several_harnesses_prompt_once_on_a_tty`, `::test_setup_yes_does_not_answer_the_harness_pick` · `e4` |
| hosted picks are disclosed at setup; reachability is reported; macOS/zsh warns | high | `e6`, `e7`, `e8` |
| `agent use` / `agent install` cover all eight adapters | high | `tests/test_cli_agent.py::test_agent_use_accepts_every_registered_adapter` · `e10` |
| README is in the announcement-first shape, under 120 lines, PyPI-safe links | high | `tests/test_docs_architecture.py::test_readme_h2_order` · `e12`, `e13` |
| a real failure reaches claude with no pi or key prompt from a clean config | medium | manual run on spark, `docs/verification.md`; one machine, one harness · `e14` |
| the same holds on thor and orin | unverified | not re-run this milestone |
| the interactive pick prompt behaves on a real terminal | medium | covered by tests with an injected prompt only; not exercised on a tty in the fleet run |

## Remaining Work / Follow-up

- Owner adjudication — done in 0.11.1: `t5`, `d1`, `d2`, `l1`–`l9`,
  `o1`–`o15`, `e1`–`e15` and `b1` approved; the three deltas citing
  `d1`/`d2` re-filed as `b2`–`b4` and approved; plan re-exported.
- PR #12 review (merged as `4162f2f`): ten Qodo findings fixed, SonarCloud
  gate OK with 0 issues.
- Owner adjudication of `o16`, `e16` and `b5` (filed in 0.11.1 when the
  current-spec projection showed `b1` was stale); then run `devague today`
  and commit `docs/current-spec.md`.
- Issue #11 — zsh and macOS: `.zshrc`, a zsh hook (`precmd`, `$pipestatus`,
  zle binding), brew installers, a mac in the verification fleet.
- Plan risk `r3` (follow-up) — the rc block's `nvsh()` function calls
  `command nvsh`, not `"$NVSH_BIN"`; `nvsh on/off` fails at the prompt when
  nvsh is installed outside `PATH`.
- Plan risks `r1`, `r2` (non-blocking) — npm global installs may need sudo
  or an nvm shim; qwen/kiro may need a first interactive login a headless
  probe cannot see.
- Re-run the end-to-end scenario on thor and orin, and the several-harness
  prompt on a real tty, before the next verification pass.
