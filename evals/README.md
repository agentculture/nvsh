# Tool-Jev DeepEval release gate (issue #64)

This is the operator guide for running the gate: install, the manifest, provider
keys, a smoke run, a full run, stopping and continuing, adding a checkpoint,
where the private outputs live, and the estimated cost per provider. For the
design (what the gate answers, the layout of `evals/tool_jev/`, the request and
answer contracts, the decisions and deviations behind it) read
[`../docs/deepeval-gate.md`](../docs/deepeval-gate.md) instead of duplicating it
here.

Everything below runs from the repository root. Nothing here needs a home
directory or an absolute path baked in — every path is either repo-relative or
comes from an environment variable the operator sets to wherever their own
private data and run directory live.

## Install

```bash
uv sync --group evals
```

This adds the `evals` dependency group (`deepeval==4.2.6` and friends) on top
of the base project. The runtime package (`nvsh`) never gains this dependency
group, and the wheel never ships `evals/`.

Run the suite's own tests (synthetic fixtures only, no network, no keys) with:

```bash
uv run pytest -c evals/pytest.ini --rootdir=. -q
```

## The manifest

One TOML file is the whole run: which candidate/baseline checkpoints to
score, which hosted reference models to compare them against, which of those
sit on the judge panel, which case sets to run, and the per-provider budget
caps. [`tool_jev/manifest.example.toml`](tool_jev/manifest.example.toml) is
the committed, fully-commented example — read every field there before
writing a real one.

The **operator's real manifest is a private file that lives outside this
repository** (in the operator's own private data root — for example, their
`lfm-train` work tree), never inside the checkout and never at a hard-coded
path. Point the runner at it with an environment variable:

```bash
export NVSH_EVALS_MANIFEST=/path/outside/the/repo/manifest.toml
export NVSH_EVALS_PRIVATE_ROOT=/path/outside/the/repo/private-data
```

`NVSH_EVALS_PRIVATE_ROOT` is the base that every case-set path, the ground
snapshot and every relative `predictions_path` resolve against. A manifest
path may also use the literal placeholder string `${NVSH_EVALS_PRIVATE}`,
expanded from `NVSH_EVALS_PRIVATE` (falling back to
`NVSH_EVALS_PRIVATE_ROOT` when unset) — see the comments at the top of
`manifest.example.toml`.

The run directory is separate from the private data root and must also sit
outside every git worktree — the runner refuses a run dir inside one:

```bash
export NVSH_EVALS_RUN_DIR=/path/outside/the/repo/run-dir
```

## Provider keys

Nothing under `evals/` reads a key from a file or a hard-coded path. Every
adapter reads its key from the environment variable the manifest names
(`api_key_env` on each `[[reference]]`), at the moment it makes a call — so
keys never touch disk and never get committed.

For an interactive or manual run, export the keys the manifest's references
need before calling the runner:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPEN_ROUTER_API_KEY=...
export NGC_API_KEY=...            # build.nvidia.com references
export LOBES_GATEWAY_API_KEY=...  # the local lobes gateway
```

For the autonomous docker compose driver (see "Stopping and continuing"
below), keys are passed through by name only, with `grant run --inject`
picking each value up from the calling shell — no key file, no key value on
disk, and nothing added to the image:

```bash
grant run --inject OPENAI_API_KEY=OPENAI_API_KEY \
          --inject ANTHROPIC_API_KEY=ANTHROPIC_API_KEY \
          --inject OPEN_ROUTER_API_KEY=OPEN_ROUTER_API_KEY \
          --inject LOBES_GATEWAY_API_KEY=LOBES_GATEWAY_API_KEY \
          --inject NGC_API_KEY=NGC_API_KEY \
          --inject NVSH_EVALS_ALERT_WEBHOOK=NVSH_EVALS_ALERT_WEBHOOK \
          -- docker compose -f evals/docker/compose.yaml up -d --build
```

Every name is listed under `environment:` in
[`docker/compose.yaml`](docker/compose.yaml) with no value, so a name that
is unset stays unset in the container rather than failing the build.

An OpenAI key must be allowed to use Files and Batch: a restricted key
without the `api.files.write` scope is refused at the batch upload, and
the runner reports that as a `missing_scope` stop, not a bad key.

### Alerts

When `NVSH_EVALS_ALERT_WEBHOOK` holds a Discord (or Discord-compatible)
webhook URL, the driver posts a short message after each step with
anything new: every 10% of each provider's calls answered, every whole
dollar of total spend (with each provider's spend and cap), each provider
or model stop, and the run's end (complete, or stopped to ask). Messages
carry counts, dollars, names and stop reasons only, never case text.
Milestones already sent are kept in `alerts.json` in the run directory, so
a restart never repeats them, and a failed post is retried at the next
step without affecting the run. The URL is a secret: keep it in `grant`
(`grant set NVSH_EVALS_ALERT_WEBHOOK --hidden`), never in a file.

## Smoke run

Before spending real money on the full roster, run a small number of cases
through every reference model and both interfaces, with one judge pass, and
get a measured cost projection instead of an estimate:

```bash
uv run --group evals python -m evals.tool_jev smoke \
    --manifest "$NVSH_EVALS_MANIFEST" --run-dir "$NVSH_EVALS_RUN_DIR" --cases 10
```

This writes `smoke.json` in the run directory: per-model token counts, cost,
whether any reply was cut at its output budget (`OK` or `CAPPED`), and a
projected full-run cost scaled from the measured tokens. Read the projection
before raising any `[budget.<provider>]` cap in the real manifest.

A smoke run dir stays a smoke run under `continue` and `drive`. `run
--expand` turns a smoke run dir into the full run in place, reusing the
smoke answers instead of resending them.

## Full run

```bash
uv run --group evals python -m evals.tool_jev run \
    --manifest "$NVSH_EVALS_MANIFEST" --run-dir "$NVSH_EVALS_RUN_DIR"
```

`--manifest` defaults to `$NVSH_EVALS_MANIFEST` and `--run-dir` to
`$NVSH_EVALS_RUN_DIR`, so once both are exported the flags above are
optional. Check progress at any time, from any other shell, without taking
the run lock:

```bash
uv run --group evals python -m evals.tool_jev status --run-dir "$NVSH_EVALS_RUN_DIR"
```

## Stopping and continuing

A run stops for one of several reasons, each affecting only what it must:

- **Pause** — a rate limit (429), a timeout or a network error pauses that
  provider for about a minute, then it resumes in the same pass; each
  provider sends from its own thread pool, and a round of direct calls stops
  taking new calls after five minutes, so a slow provider (a large local
  model) never holds the others back. A slow provider may need a longer
  per-call timeout: `[budget.<provider>] timeout_seconds` (default 60; the
  example manifest gives `local` 300).

- **Money stop** — a provider returns insufficient credit/quota (402) or its
  `[budget.<provider>] usd_cap` is reached. Only that provider stops; every
  other provider keeps going. Top up or raise the cap, then
  `continue`. Under `drive`, a money-stopped provider is re-probed with
  exactly one call every `--recheck-minutes` (default 30) and stays blocked
  until that probe's answer is recorded — nothing is resent in bulk.
- **Truncation stop** (plan risk r10) — once a model has enough answers and
  too many of them were cut at `max_output_tokens`, that model alone stops
  and asks for an operator decision: raise `max_output_tokens`, lower
  `reasoning` toward `low`/`none`, or replace the model. Replacing a model in
  the manifest is a recorded plan deviation, not a routine config edit.
  `continue` after any of the first two picks it back up automatically.
- **Rejected request** (400/401/403/404/422, or an unsupported parameter) —
  that model stops while the same rejected parameters would still be sent.
  Fix the manifest or the parameters, then resume it explicitly:

  ```bash
  uv run --group evals python -m evals.tool_jev continue \
      --manifest "$NVSH_EVALS_MANIFEST" --run-dir "$NVSH_EVALS_RUN_DIR" \
      --retry-rejected
  ```

- **`uncertain_attempts` / `batch_failures`** — a sync call that may have
  been sent and billed but never answered (a crash or a timeout), or three
  failed batches in a row for one model. These are billed as an uncertain
  charge and reported, then left for the operator to decide: check the
  provider's own usage or batch console, then `continue --retry-rejected`
  once the cause is understood.
- **Stop and ask** — a submitted batch that can no longer be found or ruled
  out (an unresolvable lookup). This is never resubmitted automatically: the
  runner exits 3, and the operator must check that provider's batch console
  by hand before deciding what to do. Resubmitting blind risks a duplicate,
  billed batch.
- **Interrupted (Ctrl+C, exit 130)** — the ledger is left consistent;
  `continue` resumes exactly where it stopped, with no call sent twice.

`continue` re-attaches any still-outstanding batch submissions by their
submit reference (never by resubmitting) and sends whatever is still
pending:

```bash
uv run --group evals python -m evals.tool_jev continue \
    --manifest "$NVSH_EVALS_MANIFEST" --run-dir "$NVSH_EVALS_RUN_DIR"
```

For a run that must survive the operator's own session and machine reboots,
`drive` repeats `continue` on a loop (deviation d2) and is what the
autonomous docker compose service in [`docker/`](docker/) runs:

```bash
uv run --group evals python -m evals.tool_jev drive \
    --manifest "$NVSH_EVALS_MANIFEST" --run-dir "$NVSH_EVALS_RUN_DIR" \
    --start --idle-when-done
```

`--start` begins a full run when the run directory holds none yet (the
container's first boot); `--idle-when-done` waits for a signal instead of
exiting once the run completes or stops to ask, so a `restart:
unless-stopped` service never re-runs the same finished pass (or the same
stop-and-ask lookup) in a loop. Watch it with:

```bash
docker compose -f evals/docker/compose.yaml logs -f
tail -f "$NVSH_EVALS_RUN_DIR/drive.log"
```

## Adding a checkpoint

A new checkpoint is a data change, never a code change: add one
`[[candidate]]` (or `[[baseline]]`) table to the private manifest, naming its
saved predictions file per case set (and, if it has one, its own
`train_split`, so the runner can refuse to score it on a case id it was
trained on — see `evals/tool_jev/cases.py`'s training-overlap check).

Those saved predictions are produced outside this evals suite entirely: the
gate only ever *replays* them (no GPU, no model load). Producing them is
`scripts/lfm-finetune/pipeline.sh measure-final` — see that script and
[`../docs/deepeval-gate.md`](../docs/deepeval-gate.md) for how a checkpoint's
predictions file is generated; this guide does not re-explain that step.

## Where private outputs live

- The **run directory** (`$NVSH_EVALS_RUN_DIR`) holds the call ledger and
  response cache, the run record, the billing log, per-subject traces, the
  per-(subject, policy) metrics, and — once every call is answered —
  `result.json` and `report.md`. It sits outside every git worktree and the
  runner never writes into the repository.
- The **private data root** (`$NVSH_EVALS_PRIVATE_ROOT`) holds the case
  sets, the ground snapshot, saved predictions and the real manifest. It
  also lives outside this repository.
- **Case text never enters git.** Every text that leaves the process is
  redacted through one choke point (`evals/tool_jev/providers/base.py`), and
  the sealed held-out sets never leave the private data root at all — the
  provider layer refuses a held-out case before any network call.
- Only the **aggregate page** (`report.md`, or the parts of it the operator
  chooses) may later be copied by hand into `docs/` — nothing in this suite
  does that automatically, and it must never carry per-case text.

## Cost

These are estimates from list prices, not a live bill — read
[`../docs/deepeval-gate.md`](../docs/deepeval-gate.md)'s cost section and
deviation d3 for how the roster and the numbers below were chosen. The
**smoke run measures real, current token counts** and should be trusted over
any of these figures before setting a budget cap. A cached rerun (an answer
already in the ledger) costs nothing.

| Provider | Notes | Approx. cost |
| --- | --- | --- |
| OpenAI | Batch API, 50% off list | included in the full-run estimate below |
| Anthropic | Batch API, 50% off list | included in the full-run estimate below |
| OpenRouter | sync, cheaper open models (deviation d3) | ~$15 per fresh full run |
| OpenRouter | smoke run only | ~$0.60 |
| build.nvidia.com | hosted, rate-limited | free |
| local (lobes gateway) | on-machine | free |

One fresh full run at medium reasoning, batched, was estimated at roughly
$27–50 depending on the multi-round Track A loop (deviation d1); every
cached rerun after that costs nothing.

## Fixture run

The commands in this section are the acceptance test for this README: they
run a real, complete gate pass end to end using only synthetic data
committed at
[`tool_jev/fixtures/readme/`](tool_jev/fixtures/readme/) — a private-root-
shaped folder with three synthetic, non-held-out cases and synthetic saved
predictions for one candidate and one baseline on each track. There are no
reference models, no judges and no budgets in this fixture manifest, so
**no network call and no provider key are needed** — this is exactly why the
fixture run has no references at all, rather than trying to point one at a
stub server.

```bash
uv sync --group evals

export NVSH_EVALS_PRIVATE_ROOT="$PWD/evals/tool_jev/fixtures/readme"
export NVSH_EVALS_MANIFEST="$NVSH_EVALS_PRIVATE_ROOT/manifest.toml"
export NVSH_EVALS_RUN_DIR="$(mktemp -d)"

uv run --group evals python -m evals.tool_jev run \
    --manifest "$NVSH_EVALS_MANIFEST" --run-dir "$NVSH_EVALS_RUN_DIR"

uv run --group evals python -m evals.tool_jev status --run-dir "$NVSH_EVALS_RUN_DIR"

ls "$NVSH_EVALS_RUN_DIR/result.json" "$NVSH_EVALS_RUN_DIR/report.md"
```

The `run` command should print `complete: .../result.json and .../report.md`
and exit `0`. `evals/tool_jev/tests/test_readme.py` extracts exactly the
fenced commands above and runs them for real in CI, in a run directory
outside the repository, to keep this section from silently drifting out of
date.
