# Delivery Summary — README demo recording

plan: `readme-demo-recording` · run: `complete` · date: `2026-09-14`
baseline: `devague summary skeleton`

## Intent

Put a repeatable recording of nvsh's failure panel at the top of the README,
re-recordable from one committed script whenever the panel changes. The plan
`readme-demo-recording` (ten tasks in five waves) was fanned out by
`/assign-to-workforce` from the spec at
`docs/specs/2026-09-14-readme-demo-recording.md`; every wave merged into
`feat/readme-demo-recording` behind the TDD gate. This artifact is the review
map for the final PR.

> nvsh's README opens with a recording of the failure panel in action,
> re-recorded from one committed script whenever nvsh changes

After: The README opens with a moving picture of a failure turning into a
proposed fix; a maintainer who changes the panel runs one script per device
and one render command and commits the result.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Register a demo adapter that replays a committed fixture through the real daemon and client path
- `t2` — Refuse demo as a persisted default in agent use, setup --agent and doctor
- `t3` — Replace every hard-coded 'eight adapters' with nine across README, the four harness prompt files, harness-selection doc and tests
- `t4` — Make scripts/record-cast.py deterministic enough to diff two runs
- `t5` — Sandboxed re-record driver that produces one scrubbed .cast per device
- `t6` — Render a committed .cast into the README image
- `t7` — Document the three-command re-record loop in docs/demos/README.md
- `t8` — Record the scenario on Spark, Thor and Orin, render, and commit casts plus images
- `t9` — Embed the Spark recording in README with a scripted-demo caption and links to Thor and Orin
- `t10` — Version bump, CHANGELOG entry, and PR

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `nvsh/agent/demo.py` (`DemoAgent` on `FakeAgent`), `nvsh/agent/demo_fixture.json`, `ADAPTERS['demo']` with `binary=None`, `PROBE_EXCLUDED`, fixture packaged in the wheel; `tests/test_agent_demo.py` (25 tests incl. an in-process daemon end-to-end). Later extended with a `{platform}` placeholder (see `d5`). |
| `t2` | delivered | `nvsh agent use demo` and `nvsh setup --agent demo` raise `CliError(1)` with the fixture hint; `doctor` gains `default_target_not_demo` and a demo reachability branch; `tests/test_demo_default_refused.py` (19 tests). One pre-existing parametrized test in `tests/test_cli_agent.py` reconciled at merge to exclude `demo`. |
| `t3` | delivered | "eight" → "nine" in `README.md`, `CLAUDE.md`, `QWEN.md`, `AGENTS.override.md`, `AGENTS.colleague.md`, `.pi/SYSTEM.md`; tests renamed and asserting nine names, with a help-wrap normaliser. `docs/harness-selection.md` needed no change (it never listed the count). |
| `t4` | delivered | `scripts/record-cast.py` sets `TIOCSWINSZ`, adds `--timestamp` and `--clean-env`; `tests/test_record_cast.py`; `scripts/` added to black/isort/flake8 in `.github/workflows/tests.yml` (bandit left off `scripts/`: it does not pass cleanly there). |
| `t5` | delivered | `scripts/demo-record.py` (sandbox HOME + all XDG dirs incl. `XDG_STATE_HOME`, pinned PS1, planted `./run-model.sh`, three-line feed, scrub, daemon stop); `tests/test_demo_record.py` incl. a real two-run end-to-end. Scrub later taught to keep tokens inside the platform kind (t8 fix). |
| `t6` | delivered | `scripts/demo-render.sh` (pinned `svg-term-cli@2.1.1`, `--no-window`, 100x34, agg fallback in the header); `tests/test_demo_render.py`. |
| `t7` | delivered | `docs/demos/README.md` "Re-recording" section: three commands per device, manual step, no CI regeneration, scripted fixture named. |
| `t8` | delivered | `docs/demos/demo-{spark,thor,orin}.cast` recorded on the real devices on 2026-09-14 (Spark locally, Thor and Orin over ssh from a `git archive` of the branch) and their `.svg` renders; `tests/test_demo_casts.py` (18 drift tests). The two verification casts are byte-identical to `main`. |
| `t9` | delivered | `README.md` embeds `demo-spark.svg` after the tagline with a scripted-demo caption and absolute links to the Thor and Orin renders, plus a "demos" bullet in the More list. The visual check on github.com and TestPyPI is pending the PR (evidence `e15`, filed `fail`/unchecked). |
| `t10` | delivered | `pyproject.toml` 0.11.1 → 0.12.0, the CHANGELOG entry, and PR #14 opened by the `cicd` skill; merge is the human's gate 3. |

## Mid-work Decisions

All five deviation records are `proposed` (LLM-filed) and await the owner's
`devague deviate --confirm`; they are listed as decisions taken, not as
approved ground truth.

- `d1` — the retry after the approved fix is a third fed line (`./run-model.sh` typed again), not something Enter triggers — `handle_failure` runs the proposal and ends the turn; `/retry` is a separate slash verb.
- `d2` — `scripts/demo-render.sh` passes `--no-window` instead of the literal `--window off` — svg-term-cli's parser rejects `--window off`.
- `d3` — `demo-record.py` uses the `nvsh` on `PATH` only when its `--version` matches the imported package, else a sandbox wrapper running this interpreter `-m nvsh` — the dev box carries an older uv-tool nvsh without the demo adapter.
- `d4` — every accepted line appears twice in a stripped transcript — recorded as needs-follow-up when found, then settled by inspection: the second copy is preceded by `\r` + erase-to-EOL, an in-place redraw after the `bind -x` Enter callback; the rendered SVG shows one line. No visible defect; no code change.
- `d5` — the demo reply gains a `{platform}` placeholder filled from the context's platform kind, so each device's recording names its real platform (`dgx-spark`, `jetson`). Without it the three recordings were visually identical and `h8` unverifiable.
- The first Spark recording read "dgx-operator": the user-name scrub rewrote `spark` inside `dgx-spark`. `scrub_rules` now keeps a token that sits inside the detected platform kind; the Spark cast was re-recorded. No deviation record; captured here and as delta `b4`.
- `t2`'s literal contract broke `tests/test_cli_agent.py::test_agent_use_accepts_every_registered_adapter[demo]` (a file t2 was told not to edit); the main agent excluded `demo` from that parametrization at merge, since refusing demo is the confirmed claim `c23`.
- PR #14 review (Qodo, ten inline findings, all FIX): fixed in `b627bb1` and recorded as delta `b6` with evidence `e18`–`e20`. The demo's script token is command-position only and shell-safe (a `./model.sh;id` token can no longer ride into the approved `chmod`), fixtures are shape-checked and `[agents.demo] fixture` is a valid config key, setup refuses demo through an alias or `@`/model form and never keeps a stored demo default, the recorder scrubs across pty read boundaries and `--clean-env` drops `NVSH_*`/`XDG_*`, and the driver pins `XDG_CACHE_HOME`/`NVSH_NO_DAEMON` and refuses to write an incomplete cast. Six SonarCloud code smells fixed in `11b4c92`/`9ef269f` (the first of those was pushed ungated: lapse `l6`).
- `tests/test_cli_setup.py::test_hook_prints_refresh_notice_once_per_session` leaks one `nvsh.daemon` per full test run (pre-existing); five leaked daemons were stopped and the leak filed as plan risk `r4` (follow_up).

## Drift From Plan

No deviation record is approved yet, so every entry below is worked out from
the task contract; the `dN` in parentheses is the proposed record that covers it.

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t5` (`d1`) | the criterion "Enter, wait for the retry" reads as if approval re-runs the command; nvsh does not, so the driver feeds the retry as a third line and the recordings show that | acceptable |
| `t6` (`d2`) | `--no-window` instead of `--window off`; same negation, the literal flag is rejected by svg-term-cli | acceptable |
| `t5` (`d3`) | `NVSH_BIN` resolution is version-gated instead of "prefer PATH", to avoid recording an older installed nvsh | acceptable |
| `t8` (`d5`) | `nvsh/agent/demo.py` and the fixture were changed after t1 merged to add the platform placeholder; `h8` was not verifiable without it | acceptable |
| `t4` | `h9` says two runs differ only in stamps; they differ in pty chunk boundaries too, so equality is asserted on the concatenated stream (delta `b2`) | acceptable |
| `t2` | one test outside t2's file set edited at merge to exclude `demo` from "agent use accepts every adapter" | acceptable |
| `t9` | rendering on github.com and TestPyPI unchecked until the PR exists (evidence `e15`) | needs-follow-up |

## Evidence

- tests: `uv run pytest -n auto -q` at `13b2edf` — 1836 passed, 5 skipped; at `b627bb1` (after review fixes) — 1858 passed, 5 skipped (pre-existing live-harness skips)
- tests: `tests/test_agent_demo.py` (25), `tests/test_demo_default_refused.py` (19), `tests/test_demo_casts.py` (18), `tests/test_demo_record.py` (11, incl. the two-run end-to-end), `tests/test_record_cast.py` (3), `tests/test_demo_render.py` (5), `tests/test_docs_architecture.py` (8) — all pass at `9fbebba`
- lint: `black --check`, `isort --check-only`, `flake8` on `nvsh tests scripts`; `bandit -c pyproject.toml -r nvsh`; `markdownlint-cli2 "**/*.md"` (repo excludes); `scripts/scan-secrets.py` (320 files clean); `teken cli doctor . --strict` (pass); `scripts/harness-smoke.py --stage config --require config` (6 passed) — all green at `123a9ce`
- devices: `python3 -m nvsh doctor --json` on thor and orin reported `detected platform: jetson`; local `nvsh doctor --json` reported `dgx-spark`; no daemon or sandbox left on either device after recording
- render: `docs/demos/demo-spark.svg` viewed in Chrome from a local server — one line per command, animation plays
- commits: `ff52eda..b627bb1` (30 commits on `feat/readme-demo-recording`)
- devague: obligations `o1`–`o15`, evidence `e1`–`e16`, deltas `b1`–`b5`, deviations `d1`–`d5`, lapses `l1`–`l5`, risks `r1`–`r4`
- PRs / issues: [#14](https://github.com/agentculture/nvsh/pull/14) — lint, harness-smoke, version-check and GitGuardian green at open; `raw.githubusercontent.com` serves the branch SVG as `image/svg+xml` and it animates in Chrome (evidence `e17`); the README embed targets `main`, so on the PR branch it shows alt text until merge; TestPyPI `0.12.0.dev49` renders the README, caption and Thor/Orin links (viewed in Chrome), with the image likewise resolving only after merge

## Delivery Claims

Confidence is capped by the lapse ledger: `l1`/`l2` are approved; `l3`–`l5`
are proposed and therefore not yet evidence, but the claims they touch are
capped anyway rather than defaulting to high.

| Claim | Confidence | Evidence |
|-------|------------|----------|
| README opens with an animated recording of a failing command becoming a proposed fix, recorded on a DGX Spark | high | file `README.md` · file `docs/demos/demo-spark.svg` · test `tests/test_demo_casts.py::test_cast_and_svg_are_committed[spark]` |
| the same session is committed for Jetson AGX Thor and AGX Orin, each naming its real platform | high | files `docs/demos/demo-thor.cast`, `demo-orin.cast` · test `tests/test_demo_casts.py::test_cast_names_the_device_platform` · evidence `e5`, `e6` |
| the recording is regenerated by one committed driver in a sandbox that never touches the operator's rc file, config or daemon | high | file `scripts/demo-record.py` · test `tests/test_demo_record.py::test_demo_record_end_to_end` · evidence `e7` |
| a `demo` adapter replays the fixture through the real daemon, panel, approval loop and audit log | medium | test `tests/test_agent_demo.py::test_a_failing_command_with_demo_as_default_streams_through_the_daemon` · evidence `e1` — capped: lapse `l3` (daemon path verified by code reading, not by breaking it) |
| `demo` is never auto-picked by setup and is refused as a persisted default | high | tests `tests/test_agent_demo.py::test_probe_never_returns_demo_even_with_everything_on_path`, `tests/test_demo_default_refused.py` · evidence `e2`, `e3` |
| a panel-legend or fixture change fails tests until the casts are re-recorded | high | test `tests/test_demo_casts.py::test_cast_carries_the_current_panel_legend` · evidence `e6` |
| two recorder runs are diffable (identical output stream, pinned size and timestamp) | medium | test `tests/test_record_cast.py::test_two_feed_runs_are_identical_except_stamps` · evidence `e10`, delta `b2` — capped: lapse `l4`, chunk boundaries vary |
| the committed casts contain no hostname, user name or LAN address | high | test `tests/test_demo_casts.py::test_cast_is_scrubbed_of_hosts_and_lan_addresses` · evidence `e8` |
| one command renders a cast to the committed SVG | medium | file `scripts/demo-render.sh` · evidence `e13` (manual, needs npx and network) |
| the image renders on github.com and on the PyPI project page | low | evidence `e17`: the raw asset renders and animates; the README embed resolves only after merge (`e15` stays unchecked until then); TestPyPI page not yet viewed |
| a maintainer with no prior context reproduces a device recording from `docs/demos/README.md` | unverified | evidence `e16` filed unmet — the loop was executed only by the agent that wrote it |
| animated SVG plays in the GitHub mobile app | unverified | park `v2` / risk `r2` — no observation |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `provenance-missing` | Stated in the q2 decision that github.com animates CSS-animated SVG images in READMEs from general knowledge, without loading such a README in this session |
| `l2` | `assumption-for-measurement` | Assumed a binary-less demo adapter would count as installed in registry.installed() without checking what shutil.which(None) returns |

pending approval (not yet evidence): `l3`, `l4`, `l5`

## Remaining Work / Follow-up

- `t10` — PR #14 is open; after merge, view `README.md` on github.com and the PyPI project page by eye and re-file `e15` as pass or fail (owner: the human at gate 3, or the next session).
- Owner adjudication: `devague deviate --confirm d1 d2 d3 d5` (and `d4`, settled as no visible defect), `devague lapse --confirm l3 l4 l5`, `devague evidence --confirm`/`--reject` for `e1`–`e16`, `devague delta --confirm` for `b1`–`b5`, `devague plan confirm` is already complete.
- `r4` (follow_up) — `tests/test_cli_setup.py::test_hook_prints_refresh_notice_once_per_session` should stop the daemon it starts; pre-existing, out of this plan's scope.
- `r1` — no size budget for the rendered images (currently 12–17 KB each; not a problem today).
- `r2` — verify SVG animation in the GitHub mobile app once the PR is viewable.
- `e16` — have a maintainer who did not build this follow the re-record loop once and file the result.
