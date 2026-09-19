# Fine-tuning Needle3 for nvsh

Tier 1 of nvsh's local response path is [Needle3](https://huggingface.co/Cactus-Compute/needle3),
a 121M-parameter model that picks **one operation** from nvsh's operation
table (`nvsh/ops/table.py`). This page is the recipe for tuning it on nvsh's
own requests, measuring the result, and publishing it so other machines can
pin it.

Training happens on a development machine. **nvsh itself never trains, never
uploads, and imports nothing from this recipe** — it only loads a `.cact`
file whose hash is pinned. Publishing is something an operator does by hand.

Needle3 is trained with cactus-needle's own JAX LoRA trainer. Unsloth and the
Hugging Face trainers cannot train it (it is not a `transformers` model, and
its weights are a 2-bit `.cact` archive). Those tools are the route for the
Tier 2 model, LFM2.5, which has its own recipe.

## Why tune at all

Measured on 2026-09-19 through the shipped path (real engine, tool schemas
generated from the operation table, dev corpus, fixture world), stock Needle3:

| Metric | Stock Needle3 | Target (spec c20) |
|---|---|---|
| Correct operation, explicit asks | 4 of 9 | at least 90% |
| Should-escalate asks declined | 3 of 5 | at least 80% |
| Wrong mutating picks | 0 | 0 |
| Warm latency, p95 | 78 ms | under 150 ms |

Rewording does not close that gap. Tried on the same day with
`scripts/needle-finetune/try_table.py` (it runs the dev corpus's explicit
requests against the real engine with replacement descriptions, without
touching the table): sharper descriptions moved correct picks from 5 of 9 to
6 of 9 while turning a correct decline into a wrong *mutating* pick; reversing
or shuffling the tool order changed which requests failed but not how many
(7 to 8 of 14 overall). Stock Needle3 is near a coin flip on this table
whatever the wording, so the lever is training data, not prose. Re-run that
script before and after any table change — it takes seconds.

## 1. Set up the training machine

```bash
python3 -m venv needle-train
./needle-train/bin/pip install 'cactus-needle[train]>=3.0.1,<3.1'
# NVIDIA GPU:  ./needle-train/bin/pip install 'cactus-needle[train,gpu]>=3.0.1,<3.1'
```

Use the same `cactus-needle` range nvsh pins in `pyproject.toml`, so the
exported archive matches the engine nvsh loads.

## 2. Build the dataset

```bash
uv run python scripts/needle-finetune/build_dataset.py --out needle-train.jsonl
```

The builder turns the **dev split** of the benchmark corpus
(`nvsh/tiers/corpus/dev.json`) into `needle finetune` JSONL: one line per
explicit request, holding the request (`query`), the tool schemas exactly as
nvsh hands them to Needle at run time (`tools`), and the expected call
(`answers`). A request that should be escalated gets an empty `answers` list,
which is what teaches the model to make no call.

- Failure-shaped entries are left out: a failed command never reaches Tier 1.
- The **held-out split is refused**. Training on it would make the adoption
  rule below meaningless.
- `--bundle FILE` adds records from an `nvsh tiers export` bundle, but only
  records the operator **approved**, that carry request text, and whose
  operation still validates against the table. Request text is only stored
  when `[tiers] store_request_text = true`, which is off by default, so most
  bundles add nothing.

A bundle is redacted, but it is still your machine's history. Read it before
you train on it, and do not publish a model or dataset built from a bundle
you have not read.

`needle finetune --generate N` and `needle generate-data` ask a hosted model
(through OpenRouter) to invent more examples. That sends the tool schemas and
your examples off the box. It is not part of this recipe; if you use it,
review what it produced like any other data, and say so in the model card.

## 3. Train and export

```bash
./needle-train/bin/needle finetune needle-train.jsonl \
    --epochs 3 --lora-rank 16 --lora-alpha 32 --seed 0 \
    --out needle3-nvsh-ops.lora.safetensors

./needle-train/bin/needle build \
    --lora needle3-nvsh-ops.lora.safetensors \
    --out needle3-nvsh-ops.cact
```

`finetune` downloads the Needle3 base checkpoint when `--checkpoint` is not
given, holds out 10% of the examples for validation, and writes the LoRA
adapter. `build --lora` merges the adapter into the base and exports the
2-bit `.cact` archive nvsh loads. Keep `--seed` in the model card so the run
can be repeated.

## 4. Measure it

```bash
nvsh tiers bench --tier needle --split dev      --out stock-dev.json
nvsh tiers bench --tier needle --split held-out --out stock-held-out.json
```

Run both again with the tuned archive in place (next section) and keep all
four result files. Each one records the model file hashes, the nvsh version,
the device and the per-request rows, so a reviewer can see *which* requests
changed, not only the totals.

A tuned Needle reports **no confidence**: cactus-needle does not update its
confidence head during fine-tuning and returns `None` instead. nvsh is built
for that — confidence was never the safety mechanism. The proposal always
shows the operation and arguments it understood, and nothing runs without
approval.

## 5. Adoption rule

A tuned model ships **only if both hold, on the held-out split**:

1. its correct-operation rate is higher than stock Needle3's, and
2. it makes no more wrong mutating picks than stock Needle3 does.

The dev split does not count: the model was trained on it. If the held-out
split is empty (it ships empty; the operator authors it, separately from the
operation descriptions), the rule cannot be evaluated and the tuned model does
not ship. "Not measured" is never a pass.

## 6. Pin it

nvsh loads only files whose size and sha256 match `nvsh/tiers/pins.json`.

```bash
sha256sum needle3-nvsh-ops.cact
stat -c %s needle3-nvsh-ops.cact
```

**Not built yet.** `pins.json` today pins only stock Needle3, and nothing
loads a tuned archive automatically: `NeedleTier(tuned=True, weights_path=...)`
exists, but the tier manager does not pass it. What has to be added before a
tuned model can ship, all in one change:

- a tuned entry in `pins.json`: repository, **commit revision** (never a
  branch or a tag), file name, size, sha256, `tuned: true`, and the hash of the
  operation table it was trained against;
- the table-hash check: a tuned model can only pick operations that existed
  when it was trained, so when the installed table's hash differs from the
  pinned one, `nvsh doctor` reports it and Tier 1 falls back to stock Needle3
  rather than silently ignoring new operations;
- a `config.toml` override that must give repository, revision and sha256
  together — an override can change *which* file is trusted, never *whether*
  it is checked.

## 7. Publish (optional, by hand)

Tuned weights and the data they were trained on are shared under the public
Hugging Face organisation [`jetson-ai-lab`](https://huggingface.co/jetson-ai-lab).
One naming format covers every nvsh model: `<base>-nvsh-<task>`.

| Repository | What it is |
|---|---|
| `jetson-ai-lab/needle3-nvsh-ops` | Needle3 tuned to pick one operation from nvsh's operation table |
| `jetson-ai-lab/lfm2.5-350m-nvsh-triage` | LFM2.5-350M tuned for the Tier 2 inspect / propose / explain / escalate loop |
| `jetson-ai-lab/lfm2.5-350m-nvsh-triage-GGUF` | the llama.cpp build of the same model |
| `jetson-ai-lab/nvsh-ops` (dataset) | the corpus: requests, expected operations, escalation cases |

Versions are git tags `ops<N>-r<M>`: `N` goes up when the operation table
changes in a way that invalidates training (operations added, renamed or
re-typed); `M` is the training run. Tags are for people — nvsh pins the
commit and the hash.

```bash
NEEDLE_HF_REPO=jetson-ai-lab/needle3-nvsh-ops \
    ./needle-train/bin/needle build --lora needle3-nvsh-ops.lora.safetensors \
    --out needle3-nvsh-ops.cact --upload
```

Alongside the archive, upload the LoRA adapter, the exact tool schemas it was
trained against, the four benchmark result files, and a model card with
`base_model: Cactus-Compute/needle3`, `license: apache-2.0`, the training
command and seed, and a plain statement of where the training data came from.
