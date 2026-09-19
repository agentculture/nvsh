# Delivery Summary — Tiered local response with Needle3 and LFM2.5

plan: `tiered-local-response-with-needle3-and-lfm2-5` · run: `partial` · date: `2026-09-19`
baseline: `devague summary skeleton`

## Intent

> nvsh answers routine operator requests and routine failures locally in well under a second: a tiny Needle3 reflex picks a typed deterministic operation, a small resident LFM2.5 agent handles the middle ground, and the configured full agent is called only when neither can — so even a Jetson Orin Nano gets a useful nvsh with no large model at all.

After: An explicit operator request (`/ask`, Ctrl+G, a `?`-marked line) goes through tiers in order: Tier 0 shell and slash commands, Tier 1 Needle3 selecting one typed operation, Tier 2 a small resident LFM2.5 agent running a bounded inspect-interpret-propose loop, Tier 3 the configured full NvshAgent. A qualifying command FAILURE skips Tier 1 and starts at Tier 2, because Tier 1 selects an operation from a short intent and measurably mis-selects (including mutating operations) when shown command/exit/stderr text. Each tier either answers with a normal nvsh Proposal or declines, and a decline moves the request up one tier. A machine with only Tier 1 (or Tiers 1-2) installed is a supported configuration, not a degraded one.

## Planned Work

Quoted verbatim from the `devague summary` skeleton. The human approved a three-PR split at gate 2: PR A (foundations), PR B (Tier 1 end to end), PR C (Tier 2, benchmarks, docs). PR A and PR B are merged; PR C has not started, which is why this run is `partial`.

- `t1` — Operation table model and validation: `nvsh/ops/__init__.py`, `nvsh/ops/_model.py`, `nvsh/ops/table.py`
- `t2` — Per-platform rendering of operations to argv, plus `thor`/`orin` CLI detection: `nvsh/ops/render.py`, `nvsh/platform/_detect.py`, docs/platforms.md
- `t3` — Argument grounding: `nvsh/ops/ground.py`
- `t4` — Tier measurement records with rotation: `nvsh/tiers/__init__.py`, `nvsh/tiers/records.py`
- `t5` — Tier contract, decision validation and fixture tier: `nvsh/tiers/base.py`, `nvsh/tiers/fake.py`
- `t6` — Config `[tiers]` table: `nvsh/config.py`, docs/config.example.toml
- `t7` — Memory floor check: `nvsh/tiers/memfloor.py`
- `t8` — Pinned fetch and prefetch: `nvsh/tiers/fetch.py`, `nvsh/tiers/pins.json`
- `t9` — Needle3 child-process tier: `nvsh/tiers/needle.py`, `nvsh/tiers/needle_worker.py`
- `t10` — Install flavors: pyproject.toml extras `needle`, `lfm`, `tiers`
- `t11` — Tier router: `nvsh/tiers/router.py`
- `t12` — Daemon residency: tiers in `nvsh/daemon.py`
- `t13` — Client routing and panel tier header: `nvsh/client.py`, `nvsh/client_transport.py`, `nvsh/panel.py`
- `t14` — Explicit `needle` adapter: `nvsh/agent/needle.py`, `nvsh/agent/registry.py`, tests/`_fake_adapters.py`
- `t15` — Doctor checks for tiers: `nvsh/doctor_checks.py`
- `t16` — Tool-call chat client over stdlib HTTP: `nvsh/tiers/toolchat.py`
- `t17` — Docker runtime launcher for Tier 2: `nvsh/tiers/runtime_docker.py`, docs/platforms.md
- `t18` — Tier 2 bounded loop: `nvsh/tiers/lfm.py`
- `t19` — Explicit `lfm` adapter and Tier 2 wiring: `nvsh/agent/lfm.py`, `nvsh/agent/registry.py`, `nvsh/tiers/router.py`
- `t20` — `nvsh tiers` CLI verbs — stats, export, prefetch: `nvsh/cli/_commands/tiers.py`, `nvsh/explain/catalog.py`
- `t21` — Uninstall removes tier state: `nvsh/cli/_commands/setup.py`
- `t22` — Benchmark corpus and runner: `nvsh/tiers/bench.py`, `nvsh/tiers/corpus/*.json`, `nvsh tiers bench`
- `t23` — Needle3 LoRA fine-tune recipe: `scripts/needle-finetune/`, docs/needle-finetune.md
- `t24` — Run the benchmark on DGX Spark, AGX Thor and AGX Orin and commit results: docs/benchmarks/
- `t25` — Docs, prompt files, sibling-repo issues, version bump

## Actual Delivery

All 25 plan tasks, keyed by task id. "PR A" is #32 (nvsh 0.15.0, `a026db7`); "PR B" is #34 (nvsh 0.16.0, `75c7461`).

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | PR A. 16 typed operations, `validate()` never raises. |
| `t2` | delivered | PR A. Table-driven argv rendering, `thor`/`orin` CLI detection; the lead added a guard so an option-like, empty or unprintable value renders nothing. |
| `t3` | delivered | PR A. Rewritten in review: every candidate examined, case variants decline as ambiguous, the untrusted value is only compared. |
| `t4` | delivered | PR A. JSONL, mode 0600, redacted, rotated, locked; an infinite loop in the cap enforcement was found and fixed before merge. PR B added a `verifier` field. |
| `t5` | delivered | PR A. `decide()`, `Tier`, `FakeTier`; NaN or out-of-range confidence counts as absent. PR B added `Explanation`, `Decline.inspections`, `NOT_RENDERABLE`. |
| `t6` | delivered | PR A. `[tiers]` and `[tiers.lfm]`; the localhost check parses the hostname (lapse `l3`). |
| `t7` | delivered | PR A. |
| `t8` | delivered | PR A. Pins by revision + sha256 + size, bounded download, https-only redirects; verified against the real Hugging Face URLs. PR B ran `nvsh tiers prefetch --yes` live. |
| `t9` | delivered | PR B. Child process, length-prefixed JSON, lock, reply-id check, 10 s timeout, kill and restart. Plus offline staging (see Drift). |
| `t10` | delivered | PR B. Extras `needle`, `lfm`, `tiers`; `dependencies = []`. The "broken needle import reaches the full agent" criterion is met by a router test rather than in the packaging test file. |
| `t11` | delivered | PR B. Router with the `d1` verifier step. |
| `t12` | delivered | PR B. `TierManager`, `tier` / `tier_decision` wire kinds, no `_run_lock`, idle unload; leases added in review. |
| `t13` | delivered | PR B. Tier step in front of `_send`, panel header, decline-then-escalate offer, audit names the tier. Exercised live against a real daemon and the real engine. |
| `t14` | delivered | PR B. `@needle`, tenth adapter, `inproc`, not persistable as default (agent use, setup, doctor). It has its own test file instead of joining the generic conformance table, like `demo`, because a router never emits the scripted events that table replays. |
| `t15` | delivered | PR B. Three checks; a damaged pins file fails the check instead of crashing doctor. |
| `t16` | delivered | PR A. Rewritten by the lead on `http.client` after review (lapse `l4`); includes `score_next_token` for `d1`. |
| `t17` | blocked | Not started — PR C. |
| `t18` | blocked | Not started — PR C. The router and manager seams it plugs into exist. |
| `t19` | blocked | Not started — PR C. The registry tables it extends exist. |
| `t20` | delivered | PR B. `stats`, `export`, `prefetch` (and `bench` from `t22`). |
| `t21` | blocked | Not started — PR C. `nvsh uninstall` does not yet remove tier records, the prefetched files or the staged engine. |
| `t22` | delivered | PR B. Corpus (21 dev entries, held-out empty for the operator to author), runner, fixture world, per-item rows, calibration, targets. |
| `t23` | partial | PR B. Dataset builder, `try_table.py`, `docs/needle-finetune.md` with the adoption rule. No fine-tune has been run, and loading a tuned model is not built. |
| `t24` | partial | One device only: stock Needle3 measured on the DGX Spark. No committed `docs/benchmarks/`, no Thor or AGX Orin run, no Tier 2, no memory figures, no stock-versus-tuned comparison. |
| `t25` | partial | PR B updated the four prompt files, README, CHANGELOG and added two docs. Sibling-repo issues are not filed (they need the operator's go-ahead); Tier 2 docs wait for PR C. |

## Mid-work Decisions

The two `dN` entries are the approved deviation records, quoted as recorded (angle-bracket placeholders are backticked here so the file lints).

- `d1` — Add a logprob verifier between Tier 1 and the proposal: after Needle3 selects an operation, a local LFM2.5 model is asked a fixed yes/no question (`For <request> the solution is <operation>. Correct?`) and nvsh reads the next-token log-probabilities of yes vs no (one prefill, one token, nothing generated) as the confidence signal; a multiple-choice variant reads the next-token distribution over lettered operations plus 'none'. This replaces Needle's learned confidence head as the propose/ask/escalate signal, works for a fine-tuned Needle that reports no confidence, and is calibrated (temperature/Platt) on the corpus dev split. It is NOT the safety mechanism: approval, the single-call rule, table validation and grounding are unchanged. nvsh/tiers/toolchat.py gains a score() call returning next-token logprobs; the benchmark adds calibration metrics and compares Needle's own confidence vs yes/no vs multiple-choice, on base and post-trained LFM2.5; vLLM is tried before llama.cpp because prefill speed is what matters here. — Measured: Needle3's confidence is a learned probe head, scored a wrong pick 1.0, is None for a fine-tuned model, and the shipped engine exposes no logits; LFM2.5 runtimes expose real token logprobs over the OpenAI-compatible API. Operator proposed the yes/no logit structure and approved this departure on 2026-09-19.
- `d2` — Fine-tuning and sharing, for all tier models: (1) tuned weights and the training data are published under the public Hugging Face org jetson-ai-lab, hard-coded in nvsh's pins with a config override (repo + revision + sha256 together); names follow `<base>-nvsh-<task>`: models jetson-ai-lab/needle3-nvsh-ops and jetson-ai-lab/lfm2.5-350m-nvsh-triage (+ a -GGUF repo for the llama.cpp build), dataset jetson-ai-lab/nvsh-ops; version tags `ops<N>-r<M>`, nvsh pins by commit + sha256, and a tuned pin records the operation-table hash it was trained on (mismatch => doctor reports it and the tier falls back to stock). (2) LFM2.5 fine-tuning moves from non-goal (c18) to documented recipe: instructions for unsloth, unsloth-cli and a remote Hugging Face run; Needle keeps cactus-needle's own JAX LoRA path (unsloth cannot train it). nvsh ships recipes, the dataset exporter and the pins; it still runs no training itself and uploads nothing on its own -- publishing stays an operator action (c17 unchanged). — Operator request 2026-09-19: created the jetson-ai-lab HF org, asked for unsloth/unsloth-cli/HF-remote fine-tune instructions, and to share data and weights with one naming format for all models. Unsloth targets HF-transformers models, so this necessarily brings LFM2.5 tuning into scope.
- Three-PR delivery (plan risk `r5`, approved at gate 2) — one PR would have been too large to review.
- `t10`'s second criterion was tested in `tests/test_tier_router.py` — it needs a router, which `t10`'s own files do not have.
- `t22` was started one wave early — it depends only on the router, which was already merged.
- The router's Tier 2 contract (`Explanation`, inspections on a decline) was defined in PR B — so PR C's loop plugs in without changing the router.
- Operation descriptions were left unchanged — rewording and reordering were tried against the real engine and moved nothing reliably.
- Issue #33 (a larger edge agent on LFM2.5-1.2B/2.6B over a maintained knowledge store) was filed as separate future work, not folded into this plan.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t11` (`d1`) | Measured: Needle3's confidence is a learned probe head, scored a wrong pick 1.0, is None for a fine-tuned model, and the shipped engine exposes no logits; LFM2.5 runtimes expose real token logprobs over the OpenAI-compatible API. Operator proposed the yes/no logit structure and approved this departure on 2026-09-19. | `acceptable` |
| `t23` (`d2`) | Operator request 2026-09-19: created the jetson-ai-lab HF org, asked for unsloth/unsloth-cli/HF-remote fine-tune instructions, and to share data and weights with one naming format for all models. Unsloth targets HF-transformers models, so this necessarily brings LFM2.5 tuning into scope. | `acceptable` |
| `t9` | The plan had the worker pass the weights path to cactus-needle; that marks the model as tuned (confidence `None`, nested subprocess), and the engine library had no offline path at all. The worker now stages the verified engine and weights into an nvsh-owned HOME for the child only. | acceptable |
| `t12` | Review found the idle sweep could close the tiers under an in-flight request; tier sessions now hold a lease, and a departed client cancels its session. Cancellation is cooperative: an in-flight selection (bounded at 10 s) is not interrupted. | acceptable |
| `t22` | The benchmark grounds against a fixture machine declared in the corpus instead of the host, and reports per-request rows and accuracy per request kind; without this a correct grounding refusal counted as a miss. | acceptable |
| `t23` | Recipe delivered, but no tuned model exists and nvsh cannot load one yet (tuned pin, operation-table hash check, config override). | needs-follow-up |
| `t24` | Measured on one device, not three, and only Tier 1. The measurement that exists misses two of the five `c20` targets. | needs-follow-up |
| `t17` `t18` `t19` `t21` | Not started: the approved split puts them in PR C. | needs-follow-up |

## Evidence

Read-only checks run for this summary, at commit `61f8c6d` (main `75c7461` plus ledger-only commits), 2026-09-19:

- tests: the 20 test files covering the delivered tasks — 826 passed, 0 failed. Named node ids are in the evidence ledger (`devague evidence --list`, `e1`-`e32`, and `e37`-`e53` added when review showed six behavioral deltas cited evidence that did not assert their behaviour; those deltas, `b1`-`b4`, `b6`, `b7`, are superseded by `b8`-`b13`, which cite the specific tests).
- tests: full suite on the PR B branch before merge — 2958 passed, 8 skipped (live-harness and one known cross-repo skip).
- lint: `black --check`, `isort --check-only`, `flake8 nvsh tests`, `bandit -r nvsh`, `scripts/scan-secrets.py`, `teken cli doctor . --strict`, `markdownlint-cli2`, `harness-smoke` — all clean on the PR B branch; cognitive complexity counted on every changed file, none over 15.
- CI: PR #34 — every check green, SonarCloud gate OK with 0 open issues, 14 of 14 review threads answered and resolved.
- observation: `nvsh tiers bench --tier needle` on the DGX Spark with cactus-needle 3.0.2 — evidence `e33`-`e36` (two pass, two **fail**).
- observation: live round trip in an isolated sandbox — client, daemon, real Needle3, panel; a `sudo systemctl restart docker.service` proposal kept the privilege-escalation restrictions.
- observation: offline run with an empty HOME — no download, the operator's own `cactus-needle` cache byte-identical before and after.
- commits: `cb5422b..75c7461`
- PRs / issues: #32, #34; issues #30, #31 (source), #33 (future work)

## Delivery Claims

Confidence follows the evidence ledger. Every record filed by the agent is `proposed` until the operator adjudicates it.

| Claim | Confidence | Evidence |
|-------|------------|----------|
| A tier never executes anything; proposals are built only from the table's rendered argv; nvsh never calls `Needle.run` (c9) | high | `e31`, `e32` · `tests/test_tier_needle.py::test_nvsh_never_calls_needle_run` |
| A mutating pick becomes a FIX proposal naming the interpreted operation and arguments, and nothing runs without approval, at confidence 1.0 and `None` alike (c10, c17) | high | `e3`, `e4` |
| Exactly one call or decline; arguments are untrusted, grounded and rendered as argv (c29, c30) | high | `e5`-`e8` |
| A failed command never reaches Tier 1; `@target` bypasses the tiers (c7, Tier 1 share) | high | `e1`, `e2` |
| Needle runs in a child process that is killed and restarted on death or hang (c32) | high | `e9`, `e10` |
| Offline once prefetched; third-party telemetry off (c12) | medium | `e11`, `e12` · observed offline run. One machine, one architecture; no x86_64 engine pin exists. |
| Daemon residency: lazy load, idle unload, answers without the agent turn lock (c13) | high | `e13`-`e16` |
| Optional flavors; base install dependency-free; no tier import at startup (c5, c11) | high | `e17`, `e18` |
| `@needle` is explicit-only and cannot become the default on any path: agent use, setup (flag, `@name`, alias, model-qualified), doctor (c14, Tier 1 share) | high | `e19`, `e20`, `e37`-`e41` |
| The panel names the tier that answered; records and the audit log name the tier (c36, c6) | high | `e21`-`e24` |
| Pinned fetch with hash verification; local redacted export that opens no socket (c31, c23) | high | `e25`-`e27` |
| Memory floor declines instead of loading (c33) | medium | `e28` — fixture floor only; never exercised under real memory pressure. |
| Tier 1 is fast enough: warm p95 78 ms, cold 312 ms (c20, latency) | medium | `e33` — one device; lapse `l6` (small sample). |
| No wrong mutating operation reaches the operator without its interpretation shown (c20) | medium | `e34` — 14 requests, one device. |
| Tier 1 picks the right operation at least 90% of the time (c20) | **not met** | `e35` **fail**: 4 to 5 of 9. Lapses `l1`, `l6` apply. |
| At least 80% of should-escalate prompts are declined (c20) | **not met** | `e36` **fail**: 2 to 3 of 5; Tier 2 and the verifier are not in the path yet. |
| Less than 1 GB added memory (c20, c26) | unverified | not measured in the bench run; an earlier spike saw about 180 MB for Needle alone. |
| The `d1` check improves the propose/decline decision | unverified | the code and its tests exist; it has never run against a model inside the router, and its thresholds are guesses. |
| The fine-tune recipe produces a better model (c39, c40) | unverified | the builder is tested (`e29`, `e30`); no fine-tune has been run and the held-out split is empty. |
| The `d2` publishing convention is written down | low | `e42` — a read of `docs/needle-finetune.md`; documentation only, nothing in nvsh implements it. |
| Tier 2, the Docker runtime, `@lfm`, uninstall of tier state (c15, c24, c25, c28, c34, c35 second half, c37) | unverified | not built — PR C. |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `assumption-for-measurement` | During /think I wrote c7 ('a request or qualifying failure goes through tiers in order') and c20's >= 90% accuracy target without ever feeding Needle3 a failure case or the remodelled 16-operation table; the spike covered 14 explicit asks on the 10-tool table only. The challenge probe shows both were wrong for the stock model. |
| `l2` | `assumption-for-measurement` | c25/c28 state the Tier 2 container starts with `--runtime nvidia` 'the same way' on every device; I wrote that from the Orin alone without checking Docker on the Spark or Thor. |
| `l3` | `assumption-for-measurement` | My precision brief for t6 specified the localhost check as a string prefix match (startswith '<http://localhost>'), which accepts <http://localhost.example.com>; I wrote a security boundary from intuition without testing the counter-example. Caught in my own review before merge and fixed with a parsed-hostname check and tests. |
| `l4` | `grader-unverified` | I merged qwen worker's toolchat.py (complete/score paths) after checking it only against its own fixtures plus one real-server call: the shape-L fixtures were wrong (one token per list item), two functions had cognitive complexity 81 and 58, and the urllib transport followed redirects and could not be stopped before headers. External review (qodo, SonarCloud) found what my review and the senses review both passed. |
| `l5` | `control-absent` | During the t13 live round trip I told the operator 'the live test found a real gap: the client never consulted the tiers' before checking my own instrument. The sandbox wrapper ran 'python -c' from the main checkout, so Python imported nvsh from the current directory (the integration branch without t13) instead of the t13 worktree on PYTHONPATH; the 'gap' was the wrong code under test. Run from the t13 checkout, the live path worked end to end (client -> daemon -> real Needle -> panel, tier ~47 ms warm). I corrected the statement in the next message. |
| `l6` | `n-below-claim` | The /think spike reported stock Needle3 at 7 of 9 correct on explicit asks and that figure shaped c20's >= 90% target and the plan's expectation that Tier 1 would be close to usable; it was measured with hand-written tool functions on 9 prompts, not through the table-generated schemas that ship. Through the shipped path the same model scores 4-5 of 9, and prompt changes move single answers at random. |
| `l7` | `control-absent` | Opened PR #34 after running a cognitive-complexity count only. SonarCloud then failed the gate on reliability with 25 issues I could have found locally: 15 composite assertions in tests (the same S9073 rule that failed PR #32, and which I had written into every brief as a rule but never checked for in the merged code), three return-type mismatches, a 21-parameter function, a backtracking regex, a float equality. Qodo found 14 more, two high (setup can persist needle as the default; the idle sweep can close tiers under an in-flight request). My pre-PR check repeated only the ONE failure class I remembered from #32's lesson instead of the rule list in my own handoff notes. |

## Remaining Work / Follow-up

- **Accuracy (the open problem)** — stock Needle3 is at about half the target. Start from `docs/tiers-improving-accuracy.md`: grow the dev corpus as a grid, the operator authors the held-out split, re-baseline, fine-tune per `docs/needle-finetune.md`, then set the verifier thresholds from the dev split. Owner: next session with the operator.
- `t23` — build tuned-model loading: the tuned pin, the operation-table hash check (doctor reports, tier falls back to stock), the config override that requires repository, revision and sha256 together.
- `t17`, `t18`, `t19`, `t21` — PR C: Docker launcher, Tier 2 bounded loop, `@lfm` adapter, uninstall of tier state.
- `d2` — PR C: LFM2.5 fine-tune recipes (unsloth, unsloth-cli, remote Hugging Face run), the dataset exporter for `jetson-ai-lab/nvsh-ops`, and the LFM Open License read before any tuned LFM is published.
- `t24` — benchmarks on AGX Thor and AGX Orin, memory and image-size figures, llama.cpp against the 8 GB budget, two engines, stock versus tuned; commit the results under `docs/benchmarks/`.
- `t25` — sibling-repo issues (device-CLI verbs for `orin`, a `power` verb for `spark`, service and container-restart verbs) once the operator approves filing them; Tier 2 documentation.
- Known limits to keep visible: `thermal_stats` and `machine_status` render to nothing without a device CLI (plan risk `r8`); a tier selection already in flight is not interrupted when its client leaves; there is no x86_64 engine pin.
- The operator adjudicates the proposed ledger records from `/validate-delivery`: obligations `o1`-`o18`, evidence `e1`-`e53`, deltas `b5` and `b8`-`b13` (`b1`-`b4`, `b6`, `b7` are superseded — reject them), and lapse `l8`.
