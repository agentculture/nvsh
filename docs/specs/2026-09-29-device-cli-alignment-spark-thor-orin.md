# device-CLI alignment spark thor orin

> nvsh drives Spark, Thor and Orin through one device-CLI UI: spark, thor and orin answer the same read-only verbs with the same --json envelope, so nvsh's operation table renders uniformly on all three; the only per-platform difference is which extra you install (nvsh\[spark\], nvsh\[thor\], nvsh\[orin\])

## Audience

- Operators running nvsh on DGX Spark, AGX Thor and AGX Orin (the fleet: spark, ssh thor, ssh orin), and nvsh's agents/Tier 1-2 which call the device CLI through render.py
  - instruction: check README and docs/platforms.md name the three platforms and the extras

## Before → After

- Before: Only spark and thor answer machine verbs; orin 0.5.0 has only template verbs, so `_ORIN_VERBS` = {} and every read-only op on an Orin falls back to system commands (`thermal_stats` and `machine_status` render to nothing); `power_get` works only on thor; installing a device CLI is manual and undocumented in nvsh
  - instruction: reproduce: on orin, uv run --frozen nvsh ... render `machine_status` returns None; orin --help lists 6 verbs
- After: pip install 'nvsh\[orin\]' (or \[thor\], \[spark\]) on each box installs nvsh plus its device CLI; every read-only op in nvsh/ops/table.py that has a device-CLI form renders to `<cli> <verb> --json` on all three boxes, with the same envelope keys
  - instruction: on each box: uv pip install `nvsh[<platform>]` then run the per-op render table test and each `<cli> <verb> --json`

## Why it matters

- Tier 1/2 and the agents pick operations from one table (nvsh/ops/table.py); an op that renders on one box and not another makes the same request succeed on Spark and fail on Orin, and the no-op-name-switching rule means the fix must be in the CLIs, not in nvsh
  - instruction: cite nvsh CLAUDE.md tiers paragraph: never switch on an operation name

## Requirements

- jetson-orin-cli (ssh orin, ~/git/jetson-orin-cli, main d55eb39, 0.5.0) gains the dgx-spark-cli machine surface: status, memory, gpu, disk, thermal, containers, network, processes (probe envelope {subject,available,source,warnings,sections,data}), swap {overview,status,history,sample,grow}, monitor group, plus power — ported from jetson-thor-cli 0.5.0, which already ported it from dgx-spark-cli (thor CLAUDE.md:124)
  - honesty: orin's ported collectors run on the Orin itself (not only on x86 test fixtures) and degrade to available:false rather than raising when a sysfs path differs from Thor's
- The shared UI contract is dgx-spark-cli 0.7.1's: every verb takes --json after the verb; results on stdout, CliError {code,message,remediation} on stderr with a hint: line; exit 0/1/2; argparse errors exit 1; probe verbs never raise and degrade to available:false and exit 0 (spark/cli/`_errors.py`, `_output.py`, probe/`_report.py`)
  - honesty: A diff of the error/output modules (cli/`_errors.py`, cli/`_output.py`, probe/`_report.py`) across the three CLIs shows only name differences
- nvsh fills `_ORIN_VERBS` in nvsh/ops/render.py:57 (currently {} because orin 0.5.0 has no machine verbs, docs/platforms.md:161) and bumps the measured-version comment at render.py:23-33
  - honesty: `_ORIN_VERBS` lists only verbs orin actually exposes at the pinned version, measured on the box and recorded in docs/platforms.md
- nvsh gains optional extras spark, thor and orin in pyproject.toml \[project.optional-dependencies\], each pulling one PyPI device CLI (dgx-spark-cli 0.7.1, jetson-thor-cli 0.5.0, jetson-orin-cli 0.5.0 are all published); base dependencies stay \[\]
  - honesty: pip install nvsh with no extra still installs zero third-party packages, and each extra resolves to exactly one device CLI
- nvsh tells Thor from Orin by the detected board model/L4T, not only by which binary is first on PATH: `_classify` (nvsh/platform/`_detect.py`:244-254) returns jetson for both, and `_CLI_CANDIDATES` (render.py:65-69) tries thor before orin
  - honesty: The Thor/Orin split reads a source recorded in docs/platforms.md (e.g. /proc/device-tree/model or `nv_tegra_release`) and falls back to PATH order when the model is unreadable
- dgx-spark-cli (/home/spark/git/dgx-spark-cli, 0.7.1) gains a power verb using the probe envelope; nvsh adds `power_get` -> power to `_SPARK_VERBS` in render.py
  - honesty: spark power reports only values the GB10 actually exposes (nvidia-smi power/clock fields, measured on spark) and says available:false for fields it cannot read, never guessed

## Honesty conditions

- The three device CLIs are released to PyPI at versions that include the aligned verbs before nvsh's extras pin them (orin >= 0.6.0, spark >= 0.8.0)
- git status on thor before and after the work shows the usb frame, vendored skills and skill-sources diff unchanged
- Each PR is opened from the remote checkout's branch off current origin/main, not from the /home/spark/git copies
- The three platforms are exactly the fleet boxes reachable today: spark (local), ssh thor, ssh orin
- The before-state is reproduced on current main: render.py `_ORIN_VERBS` == {} and orin 0.5.0 --help lists only whoami/learn/explain/overview/doctor/cli
- No change in this work adds an operation-name switch in nvsh code; all per-platform differences live in render.py's verb tables
- The after-state is observed on the boxes themselves, installed from PyPI/TestPyPI, not only in tests
- The key check runs over ssh on thor and orin and locally on spark, with its output kept in the PR descriptions
- The table test fails if a verb is added to or removed from any `_`\*`_VERBS` table without updating its expected rows
- The detection test covers a fixture with both binaries on PATH for each board model

## Success signals

- On each of spark, thor and orin, all 10 read-only device verbs (status, memory, gpu, disk, thermal, containers, network, processes, 'swap status', power) exit 0 with --json and emit an object with keys subject, available, source, warnings, sections, data (swap status keeps its own documented shape)
  - instruction: script: `for v in ...; do <cli> $v --json | python3 -c 'check keys'; done` over ssh on each box
- render.py maps 10 of 10 read-only device ops for each of spark, thor and orin (0 None renders when the CLI is present), covered by a table test in tests/`test_ops_render.py` per platform kind x CLI present/absent
  - instruction: uv run pytest tests/`test_ops_render.py` -v
- Detection picks thor on a Thor and orin on an Orin even when both binaries are on PATH: 2 of 2 fleet boxes report the right device CLI in nvsh doctor --json / --show-context
  - instruction: ssh thor and ssh orin: uv run --frozen nvsh doctor --json, check device-cli value

## Scope / boundaries

- thor's in-flight uncommitted work on ssh thor (.devague frame 'thor usb: port power visibility + keep-powered-through-sleep', vendored skills, docs/skill-sources.md diff) is left untouched; alignment branches off main f2248e7
- Work happens in the remote checkouts (ssh thor ~/git/jetson-thor-cli, ssh orin ~/git/jetson-orin-cli); the local /home/spark/git copies are stale (thor local 9b53573 predates 0.5.0; orin local is on an old rename branch f965cd7) and are not the source of truth

## Non-goals

- No mutating device-CLI verbs are added for nvsh's mutating ops (`power_set`, `service_restart`, `container_restart` keep their system fallbacks in render.py); the only mutating CLI verb stays swap grow, dry-run unless --apply
- nvsh's runtime package keeps zero third-party dependencies; the device CLIs are optional extras and nvsh still works when none is installed (render.py system fallbacks)
- rtx-spark-cli is not aligned in this slice and gets no nvsh extra; it stays under issue 48
- `gui_status`/`gui_set`, `jetson_stats` (jtop), `disk_hogs` operations, docs/ops.md generation and Tier 2 retraining stay in issue 48, outside this slice

## Assumptions

- All three CLIs answer the same verb set including power: thor's power (nvpmodel, `jetson_clocks`, hwmon rails) is ported to orin, and dgx-spark-cli gains power for GB10 power/clock state; shared verbs' data keys stay identical, platform-specific fields live inside data

## Scope exploration

- `s1` — `dgx-spark-cli 0.7.1 (/home/spark/git/dgx-spark-cli)`: Reference UI: 8 top-level probe verbs + monitor + swap, shared probe envelope (probe/`_report.py`:28-52), CliError contract (cli/`_errors.py`:19-42); argparse prog says dgx-spark-cli while binary is spark; no power verb; no nvsh mention
  - seeds: `c3`, `q2` (question, resolved)
- `s2` — `jetson-thor-cli 0.5.0 (ssh thor, main f2248e7)`: Already aligned: same 8 probe verbs + power + monitor + swap, same envelope and error contract, ported from dgx-spark-cli; root --help still template text; uncommitted usb frame in flight
  - seeds: `c7`, `c8`
- `s3` — `jetson-orin-cli 0.5.0 (ssh orin, main d55eb39)`: Template baseline only: whoami/learn/explain/overview/doctor/cli; no probe, swap, monitor or power; zero deps; backend colleague; d55eb39 is a skills sync, not a verb
  - seeds: `c2`
- `s4` — `local ~/git copies of thor/orin CLIs`: Stale: thor local 9b53573 predates 0.5.0; orin local on branch rename-command-to-orin f965cd7, main not fetched
  - seeds: `c9`
- `s5` — `nvsh/ops/render.py`: `_SPARK_VERBS` (40-50), `_THOR_VERBS` = spark + power (52-55), `_ORIN_VERBS` = {} (57); `_CLI_CANDIDATES` jetson=(thor, orin) (65-69); renders \[cli, \*verb, --json\], else system fallback; mutating ops never use a device CLI
  - seeds: `c4`, `c10`
- `s6` — `nvsh/platform/_detect.py`: PATH-only spark/thor/orin detection (191-203); only parses spark status --json 'available' (`_subprocess.py`:99-109); `_classify` has no thor/orin split (244-254)
  - seeds: `c6`
- `s7` — `nvsh pyproject.toml extras`: Only needle/lfm/tiers; dependencies = \[\]; no spark/thor/orin extra anywhere in docs; all three CLIs are on PyPI (dgx-spark-cli 0.7.1, jetson-thor-cli 0.5.0, jetson-orin-cli 0.5.0)
  - seeds: `c5`, `c11`, `q1` (question, resolved)
- `s8` — `rtx-spark-cli (local 0.1.5, PyPI 0.3.0)`: Local checkout has only template verbs; PyPI 0.3.0 not inspected; no render.py candidate for rtx
  - seeds: `q3` (question, resolved)
- `s9` — `GitHub issue 48 + issue 43`: 48 already plans: align orin to thor's --json verbs, fill `_ORIN_VERBS`, say Spark has no power form, rtx row, docs/ops.md generated from render.py; 43 is sibling-repo verb issues (t25)
  - seeds: `q4` (question, resolved)

## Hard questions

- Is rtx-spark-cli (0.3.0 on PyPI; local checkout 0.1.5 has only template verbs, no render.py row) in this alignment, or left for issue 48? (resolved: rtx-spark-cli is out of scope; stays under issue 48 (operator decision 2026-09-29 (AskUserQuestion)))
- Is this a slice of open issue 48 (align Spark/Thor/Orin, docs/ops.md, gui/jtop/disk-hogs ops, retrain Tier 2) to be tracked there, or a new issue scoped to UI alignment only? (resolved: Build as a slice of issue 48: UI alignment, orin verbs, spark power, extras, thor/orin detection; gui/jtop/disk-hogs/Tier-2 retrain stay in 48 (operator decision 2026-09-29 (AskUserQuestion)))
- What does nvsh\[spark\]/nvsh\[thor\]/nvsh\[orin\] mean: (a) pip extras that install the matching device CLI from PyPI, (b) the per-platform verbs allowed to differ (e.g. power on Jetson only), or both? (resolved: Both: pip extras nvsh\[spark|thor|orin\] install the matching PyPI device CLI, and platform-only verbs are the only UI difference (operator decision 2026-09-29 (AskUserQuestion)))
- Should Spark gain a power verb (GB10 power/clock state) so `power_get` is uniform, or stay Jetson-only as issue 48 suggests? (resolved: Spark gains a power verb (GB10 power/clock state) so `power_get` renders uniformly on all three (operator decision 2026-09-29 (AskUserQuestion)))

## Open parks

- [unknown_nonblocking] Orin power rails and nvpmodel modes differ from Thor (and between AGX/NX/Nano); orin power data keys can only be measured on the orin box
- [unknown_nonblocking] thor/orin CLIs live in ~/.local/bin, not on PATH for non-interactive ssh (issue 48); PATH-only detection (`_detect.py`:191-203) may miss them outside an interactive shell
