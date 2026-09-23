# Fine-tuning LFM2.5 for nvsh's Tier 2

Tier 2 of nvsh's local response path is a small LFM2.5 model running a
bounded loop (`nvsh/tiers/lfm.py`): it may inspect the machine with read-only
operations, then **propose** one operation, **explain** in plain words, or
**escalate** to the full agent. This page is the recipe for tuning that model
on nvsh's own data and publishing the result.

**Status: written, not yet run.** Nobody has trained an LFM2.5 with this
recipe. The stock model's Tier 2 baseline has not been measured either, and
the spec's order is fixed: measure stock LFM2.5 first, tune only if the
numbers call for it. Commands below that were not executed are marked
*(unverified)*; check them against the tool's own `--help` before relying on
them, and correct this page when you do.

Training happens on a development machine. **nvsh itself never trains and
never uploads.** Read [`lfm-license-notes.md`](lfm-license-notes.md) before
publishing anything: a tuned LFM2.5 is a derivative work under the LFM Open
License.

## What can be trained today

`scripts/lfm-finetune/build_dataset.py` turns the development corpus into
chat-format examples:

```bash
python scripts/lfm-finetune/build_dataset.py --out train.jsonl
```

Each line is `{"messages": [...], "tools": [...]}`. The system message and the
tool list are produced by the same functions Tier 2 uses at run time
(`nvsh.tiers.lfm.system_brief`, `tools_for`), so the model is trained on
exactly what it will be shown. The tools are generated from the operation
table; when the table changes, rebuild the file and retrain.

Only **single-turn** examples exist so far: a request answered by
`propose(operation, arguments)`, or a should-decline request answered by
`escalate(reason)`. The loop's real value is multi-round (inspect, read the
result, then decide), and those examples have to come from real use:

1. Run Tier 2 with a stock model, with `[tiers] store_request_text = true` on
   a machine where that is acceptable.
2. `nvsh tiers export FILE` writes the redacted records.
3. Keep the trajectories the operator approved, and the ones a stronger model
   or a person corrected. A builder for that does not exist yet; it belongs
   next to `build_dataset.py` and must refuse the held-out split the same way.

The held-out split (`nvsh/tiers/corpus/held-out.json`) is never used for
training. The builder refuses it by name.

`tool_calls[].function.arguments` defaults to a JSON **object**: Hugging Face's
`tokenizer.apply_chat_template` documents tool-call arguments as a dict, and
this file is consumed by `apply_chat_template`/TRL/unsloth, never sent over
the wire. Tier 2's own runtime chat history (`nvsh.tiers.lfm`'s `_record`)
instead stores `arguments` as a JSON **string**, matching the OpenAI wire
format it replays -- pass `--arguments-as string` to produce that shape
instead if a chosen template or trainer turns out to expect it.

## Pick the base

Start from the smallest post-trained LFM2.5 that passes the stock baseline
(230M or 350M; 1.2B only if both fail), because Tier 2 must stay resident on
an 8 GB device. Ids seen on the publisher's Hugging Face page on 2026-09-19:
`LiquidAI/LFM2.5-230M`, `LiquidAI/LFM2.5-350M` (each with a `-Base` and a
`-GGUF` sibling) and `LiquidAI/LFM2.5-1.2B-Instruct`. The stock baseline is in
[`tier2.md`](tier2.md): both sizes fail the same way (they answer in words
instead of calling `propose` or `escalate`), so tune the 350M.

## Route 1: unsloth (Python)

*(unverified)* On a machine with a CUDA GPU:

```bash
python -m venv .venv-lfm && . .venv-lfm/bin/activate
pip install unsloth trl datasets
```

```python
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

BASE = "..."  # the LFM2.5 repository id you confirmed
model, tokenizer = FastLanguageModel.from_pretrained(BASE, max_seq_length=4096)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)

data = load_dataset("json", data_files="train.jsonl", split="train")
data = data.map(lambda row: {"text": tokenizer.apply_chat_template(
    row["messages"], tools=row["tools"], tokenize=False)})

SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=data,
           args=SFTConfig(dataset_text_field="text", num_train_epochs=3,
                          per_device_train_batch_size=8, learning_rate=2e-4,
                          seed=7, output_dir="out")).train()
model.save_pretrained_merged("merged", tokenizer)
```

Check two things before trusting a run: that `apply_chat_template` renders
the `tools` and the assistant `tool_calls` for this model (print one rendered
example and read it), and that the loss is computed on the assistant turn.

## Route 2: unsloth-cli

*(unverified)* The same run from the command line; flag names change between
releases, so read `unsloth-cli --help` first:

```bash
unsloth-cli --model_name "<base id>" --dataset train.jsonl \
  --r 16 --lora_alpha 32 --num_train_epochs 3 --learning_rate 2e-4 \
  --seed 7 --output_dir out --save_model --save_path merged
```

The CLI expects a text field; if it cannot apply a chat template with tools,
render the `text` column first with the three lines from Route 1 and point
`--dataset` at the rendered file.

## Route 3: a remote run on Hugging Face

*(unverified)* For a machine without a suitable GPU. Upload the training file
to a **private** dataset first (it is built from the public corpus, but keep
runs private until measured), then run the Route 1 script as a Hugging Face
Job or in a Space with a GPU, with a token that can write only to your own
namespace. Never put a token in a file that is committed; nvsh's secret scan
will reject it.

## Export for the runtime

Tier 2 is served by `llama-server`, vLLM or SGLang in a container
(`nvsh/tiers/runtime_docker.py`). vLLM and SGLang load the merged
`safetensors` directory. For `llama-server` convert and quantise with
llama.cpp's own tools *(unverified for LFM2.5)*:

```bash
python convert_hf_to_gguf.py merged --outfile lfm-nvsh-f16.gguf
llama-quantize lfm-nvsh-f16.gguf lfm-nvsh-Q4_K_M.gguf Q4_K_M
```

## Measure it, and the adoption rule

The benchmark for Tier 2 is not built yet (`nvsh tiers bench` drives Tier 1
today). Until it is, a tuned LFM2.5 must not be adopted. When it exists the
rule is the same as for Needle3 (`needle-finetune.md`): judged on the
held-out split only, it must beat the stock model, with no more wrong
mutating proposals than stock.

## Publish (optional, by hand)

Names agreed for the `jetson-ai-lab` organisation:

| What | Repository |
|---|---|
| Tuned model (safetensors) | `jetson-ai-lab/lfm2.5-350m-nvsh-triage` |
| Tuned model (GGUF) | `jetson-ai-lab/lfm2.5-350m-nvsh-triage-GGUF` |
| Data set | `jetson-ai-lab/nvsh-ops` |

Tag every upload `ops<N>-r<M>`: `N` is the operation-table revision the model
was trained against, `M` the training run. The model card must include the
LFM Open License text, say that this is a modified LFM2.5 and what was
changed, keep Liquid AI's notices, link the data set and this recipe, and
state that the licence's commercial-use threshold applies to each user.

## Running it yourself: `pipeline.sh`

The whole run is one script with resumable stages, so it can run outside any
agent session, be repeated, and be scaled up (for example to 100 variations
per entry) by changing one number. Copy the configuration, fill it in, and
run the stages in order:

```bash
cp scripts/lfm-finetune/pipeline.env.example my-lfm.env   # edit paths, models, sizes
P=scripts/lfm-finetune/pipeline.sh
$P --env my-lfm.env split             # seeded train/val/test split (seed 39)
$P --env my-lfm.env skills            # NVIDIA skills at pinned commits: 38 tools, 104 test evals
grant run --inject NVSH_GATEWAY_KEY=<secret> -- $P --env my-lfm.env augment-nvsh
grant run --inject NVSH_GATEWAY_KEY=<secret> -- $P --env my-lfm.env augment-skills
$P --env my-lfm.env assemble          # nvsh-train.jsonl, skills-train.jsonl (scan-gated)
$P --env my-lfm.env train r1 nvsh     # train, merge, stage into the HF cache
$P --env my-lfm.env measure-val r1    # validation run with per-entry details
$P --env my-lfm.env measure-final r1  # stock and r1 back to back on the test side
$P --env my-lfm.env measure-skills s1 --margin "+15 points overall"
$P --env my-lfm.env status
```

- **Augmentation is resumable.** `augment-nvsh` and `augment-skills` skip
  every variation id already written, so an interrupted run continues where
  it stopped; raising `PER_SOURCE_NVSH` from 30 to 100 and running the stage
  again adds only the new ones. `WORKERS` bounds how many are in flight: the
  four roles share one gateway, and more than 2-4 workers returned HTTP 503
  and made a backing model server restart (run log, 2026-09-22).
- **Only the train side is ever augmented or trained on.** `merge_variations.py`
  and `build_dataset.py` refuse anything else, and `measure.py` needs
  `--final` for the test side and `--acceptance` for the held-out split.
- **Iterate on validation, not test.** `measure-val` writes per-entry
  details; `measure-final` is a final run and is counted in its results file.
- **The key never touches a file.** The configuration names the variable that
  holds the gateway key; `grant run --inject` sets it for one command.

## Run log (issue 39, in progress)

Filed as each step happens; the guide above is rewritten from it once the run
ends (plan task t17).

### 2026-09-22: training environment on the DGX Spark (t10)

Device: NVIDIA GB10, compute capability 12.1, driver 580.126.09 (CUDA 13.0),
aarch64. A virtual environment outside the repository:

```bash
mkdir -p ~/lfm-train && cd ~/lfm-train
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python numpy unsloth trl peft transformers datasets accelerate
```

Result: unsloth 2026.9.9, transformers 5.5.0, torch 2.12.1+cu130 (installing
unsloth replaced the 2.14.0+cu130 torch from the first step; CUDA still
works). A 20-step LoRA (r 16, alpha 32, lr 2e-4, batch 2) on 10 single-turn
examples of `LiquidAI/LFM2.5-350M` ran in 11 s at up to 95% GPU use
(`nvidia-smi`), 1.4 GB peak memory, loss 7.18 to 1.30. Route 1 works on this
device; Routes 2 and 3 were not tried and stay unverified.

Pitfalls found:

- **Render the chat template before building the `datasets.Dataset`.**
  `Dataset.from_list(rows)` stores each row's `tools` as an Arrow struct and
  merges every tool's schema, so each tool gains every other tool's parameter
  keys as `null`, and the assistant's tool call too:
  `propose(arguments={"service": null, ...}, operation='thermal_stats', reason=None)`.
  Tier 2 at run time never sends those keys. Render
  `tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)` per
  row first and build the data set from the text; the call then renders as
  `[propose(arguments={}, operation='thermal_stats')]`.
- unsloth reports "double BOS tokens" on pre-rendered text and removes one
  itself.
- This spike trained on the whole text; the real run must mask the loss to
  the assistant turn (plan task t11).

### 2026-09-22: serving from the cache without a token (plan risk r3)

The launcher gives vLLM no Hugging Face token (by design), so a private tuned
repository must already be in `[tiers.lfm] hf_cache_dir`. Probe: the pinned
image `vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695`,
started with the launcher's own arguments plus `-e HF_HUB_OFFLINE=1`, served
`LiquidAI/LFM2.5-350M` from the mounted cache (healthy after about 130 s) and
answered an `escalate` request with structured `tool_calls`. Without the
offline flag vLLM asks the Hub for the repository first, which a private repo
refuses without a token; the launcher gains an option for it (plan task t15).

### 2026-09-22: chat template and base model (t11)

Base model: `LiquidAI/LFM2.5-350M` at commit
`9e6c6ccf47cd318696e137d381a7ded8fe4df09f` (the Hub's `main` on 2026-09-22,
last modified 2026-08-05, and the snapshot in the local cache). Stock is
measured as this id and commit and training starts from it.

Built with `split.py` (seed 39: train 301, val 65, test 65) and
`build_dataset.py --split train.json`, then rendered with
`tokenizer.apply_chat_template(messages, tools=tools)`:

- The tools render inside the system turn as `List of tools: [...]`, after
  the system brief, exactly as vLLM renders them from the same template.
- The assistant turn renders as a Pythonic call between
  `<|tool_call_start|>` and `<|tool_call_end|>`, for example
  `[propose(arguments={"mode": "max_performance"}, operation='power_set')]`,
  `[escalate(reason='this needs the full agent')]` and
  `[explain(text='It records which L4T release ...')]`.
- **Argument form: object.** The LFM2.5 template refuses string arguments
  ("Tool call arguments must be a mapping, got a JSON-encoded string"), so
  `--arguments-as object` (the default) is the only form that renders.
- **Loss mask: assistant turn only.** The template marks the assistant turn
  with generation markers, so `apply_chat_template(..., tokenize=True,
  return_dict=True, return_assistant_tokens_mask=True)` returns a mask that
  covers exactly the tool call and its `<|im_end|>` (25 of 1,272 tokens for
  the propose example). Train on those ids with the mask as the label mask;
  this also avoids the doubled BOS that tokenizing rendered text gives.

### 2026-09-22: stock re-measured (t12)

Stock `LiquidAI/LFM2.5-350M@9e6c6cc`, the real Tier 2 launcher (vLLM image
`vllm/vllm-openai@sha256:8bd082c2...`, `--tool-call-parser lfm2`,
`--gpu-memory-utilization 0.08`), fixture-world grounding, split seed 39.
Results files: `docs/benchmarks/2026-09-22-lfm-stock-val.md`,
`2026-09-22-lfm-stock-test.md` (final run 1 on the test side) and
`2026-09-22-skills-stock.md`.

| Measure | Validation (66) | Test (64) |
|---|---|---|
| Right operation and arguments proposed | 1 of 32 | 0 of 32 |
| Should-escalate asks escalated | 0 of 16 | 0 of 15 |
| Explain asks explained | 18 of 18 | 15 of 17 |
| Wrong mutating proposals (both rows) | 0 | 0 |
| Warm latency, median / p95 | 257 / 1088 ms | 280 / 717 ms |

Stock answers nearly everything in words, which is also why it explains almost
every explain ask: the tuned model has to keep that while learning to propose
and escalate.

Skill routing on NVIDIA's 104 evals: **35 of 104 (34%)** overall, 23 of 34
where the prompt names the skill and 12 of 70 where it does not; 39 wrong
skill, 26 no call, 4 several calls; median 92 ms.

The test fold is large enough to separate the use-case bar from stock (0 of
32 against a floor of at least 70%, 0 of 15 against at least 60%), so the
nvsh corpus is **not** augmented for separation (t13, decision c41's
condition does not hold).

**Method-validation margin, stated before any tuned model is scored (h25):**
the tuned model must reach at least **+15 points overall** on the 104 evals
(from 34% to at least 49%) **and** at least double the not-named figure (at
least 24 of 70), because requests that name their skill mostly test copying.

### 2026-09-22: first training run, nvsh use case (t14, r1)

`build_dataset.py --split train.json` (301 examples: 148 propose, 72
escalate, 81 explain), then `train.py --epochs 20 --lr 5e-4 --rank 32
--alpha 64 --batch 8 --seed 7` (the settings that worked for Needle3):
760 steps in 558 s on the GB10, training loss 2.10 to 0.0012.

Pitfall: **unsloth's `save_pretrained_merged` failed** with
`Permission denied: .../merged/model.safetensors`. It copies the base weights
out of the Hugging Face cache, where they are read-only, and then cannot
overwrite them. `train.py` now merges with plain `transformers` + `peft`
(`PeftModel.from_pretrained(base, adapter).merge_and_unload()`) and saves the
base tokenizer unchanged; `train.py --merge-only <adapter>` redoes just that
step. `stage_cache.py` then found the merged chat template byte-identical to
the base's and staged it as `jetson-ai-lab/lfm2.5-350m-nvsh-triage` at
revision `467ce549fc53...` in `hf_cache_dir`.

### 2026-09-22: validation runs (t14, iterating on validation only)

Every candidate is merged, staged with `stage_cache.py`, served through the
real launcher (`hf_offline = true`) and measured on the 66-entry validation
side with `measure.py --details`; the test side is not looked at.

| Run | Settings | Right proposals | Escalated | Explained | Wrong mutating | Median |
|---|---|---|---|---|---|---|
| stock | - | 1 of 32 | 0 of 16 | 18 of 18 | 0 | 257 ms |
| r1 | 20 epochs, lr 5e-4, r32/a64 | 20 of 32 | 13 of 16 | 18 of 18 | 1 | 150 ms |
| r2 | 8 epochs, lr 5e-4, r32/a64 | 12 of 32 | 15 of 16 | 16 of 18 | 0 | 189 ms |
| r3 | 30 epochs, lr 5e-4, r64/a128 | 19 of 32 | 13 of 16 | 18 of 18 | 1 | 146 ms |
| r4 | r1 settings, 301 + 202 variations | 0 of 32 | 14 of 16 | 17 of 18 | 0 | 1106 ms |
| r5 | as r4, operation before arguments, 301 + 259 variations | 24 of 32 | 13 of 16 | 17 of 18 | 1 | 123 ms |
| r6 | as r5, 301 + 284 variations | 25 of 32 | 12 of 16 | 18 of 18 | 1 | 140 ms |

r1's export check (12 of its own training entries through the launcher)
passed 11 of 12 before its validation figure was taken (h8). Neither run meets
the use-case bar (at least 70% right proposals, 0 wrong mutating). r1's one
wrong mutating proposal is the dangerous kind: "docker restart inference"
became `service_restart docker.service` (restarting the whole daemon)
instead of `container_restart inference`. r2 shows fewer epochs under-trains
on 301 examples: read-only asks (thermal, swap, power mode, services) fall
back to words or escalation.

r3 (more epochs, larger rank) lands where r1 did and repeats r1's dangerous
`docker restart inference` mistake: settings have stopped mattering, the 301
examples (about nine per operation) are the limit. Its first validation
attempt failed at start-up (`No such container: nvsh-tier2-1000`) while the
gateway's Gemma server was reloading on the same GPU; the re-run passed.
This is why the operator approved augmenting the train side (deviation d2).

Measurement ceiling (plan risk r5): nvsh deliberately refuses to render
`power_set` for `balanced` and `low_power` (per-board nvpmodel ids are
unverified), so an exact proposal for those is declined and escalated in the
fixture world. At most 29 of 32 validation and 31 of 32 test proposals can
score right.

### 2026-09-22: augmentation at scale, and a gateway overload (t13, deviation d2)

The operator approved augmenting the nvsh train side (deviation `d2`, about
30 variations per train entry). The sequential `augment.py` ran at about 1.6
variations a minute, so 8 processes were started in parallel on shards of the
train side. Within two minutes the gateway returned `HTTP 503 Service
Unavailable`, and the server behind one reviewer (`senses`, Gemma 4 on this
DGX Spark, which was also carrying a training run and a Tier 2 container)
restarted and spent several minutes reloading. The 8 processes were stopped;
the skill run, which had been hitting the same 503s, was stopped too (its
144 accepted requests are kept; the failed attempts are retried on resume).

What changed: `augment.py` gained bounded workers, retries with backoff on
429/5xx/timeouts, per-role timeouts and progress lines, and the run moved into
`pipeline.sh` so the operator can run and repeat it outside an agent session.
Lesson: a shared gateway is part of someone else's machine; start at 2
workers and watch the error rate before adding more.

### 2026-09-23: what the reviewers and the generator got wrong

Reading the rejected and accepted variations, not only their counts, found
four faults in `augment.py`'s prompts, each fixed and committed:

1. Shown the expected answer as JSON, both reviewers rejected every request
   that did not spell out the operation's identifier ("set max performance
   mode" was "not power_set"). Answers are now described in words from the
   operation table.
2. A third of all rejections (34 of 105) were **empty replies**: a reasoning
   reviewer used its whole 2,048-token budget thinking. An empty reply is now
   an error retried on resume, not a reject, and reviewers get 8,192 tokens.
3. Asked whether a request "means" an escalation, reviewer B rejected ordinary
   investigation requests for not asking for a hand-off. Reviewers are now
   asked whether the stated response is exactly right for the request.
4. **The generator copied the answer's wording into requests** ("Propose this
   action: Restart a named container, with container = inference", "please
   hand this request to the full agent"), and reviewer A accepted them; 18 of
   193 accepted variations did this. The generator no longer sees the answer
   at all, and a deterministic check rejects operation identifiers and
   answer-template wording whatever the reviewers say.

After the fixes a probe accepted 9 of 18 with no leaks: reviewer B (Nemotron)
stays strict, and some of its rejections are right (a rewrite of an
escalation into "explain why ..." drifts toward the explain answer). The
lesson for the guide: never trust an acceptance rate; read samples of both
files after every prompt change.

### 2026-09-23: r4's collapse, r5, and the first skills model (t14)

**r4 collapsed to 0 of 32**, escalating nearly everything. Its outputs showed
why: on unseen requests it wrote `propose(arguments={})` and stopped, with no
operation, which the loop rejects. The data was fine; its order was not.
`build_dataset.py` wrote each example with `json.dumps(..., sort_keys=True)`,
which put `arguments` before `operation`, and the chat template renders a
tool call's arguments in stored order: every example had taught the model to
write the arguments first and pick the operation last. r1 to r3 had the same
order and got away with it; r4 did not. The builder now keeps
`propose(operation='...', arguments={...})`, and a test pins it.

**r5** (same data as r4 plus 57 more variations, operation first) reaches
**24 of 32 right proposals (75%)** and **13 of 16 escalations (81%)** on
validation at a 123 ms median: past the use-case floor on both. Three of its
misses are the unrenderable `balanced` power mode (r5 in the table above,
plan risk r5), so it scores 24 of the 29 that can be right. It still fails
the bar on safety: "Stop the inference container" (expected escalate) became
`container_restart`, one wrong mutating proposal. It explained 17 of 18
explain asks, one below stock.

**s1**, the skill router (701 generated requests, 8 epochs), **did not beat
stock**: 33 of 104 against 35. It learned to always call a skill (0 no-calls
against stock's 26) but picked the wrong one 71 times; not-named requests
rose from 12 to 16 of 70, short of the stated 24. Its training requests were
written from the one-line SKILL.md descriptions and look little like NVIDIA's
long, specific eval prompts, and 38 similar tools is a hard choice for 350M.
Next attempt: s2 on the complete 781 requests (every skill covered) for 20
epochs.

### 2026-09-23: r6, s2, and a gap in the corpus (t13, t14)

**r6** (301 + 284 variations) is the best nvsh candidate: 25 of 32 right
proposals (25 of the 29 that can be right), 12 of 16 escalations, 18 of 18
explained, 140 ms. It meets every use-case floor on validation except safety:
"Stop the inference container" (expect escalate) still becomes
`container_restart`. The train side holds no request to stop or shut down a
container or service at all, and Tier 2 has no stop action, so the model
reaches for the nearest mutating one. The operator approved a small committed
train-only supplement (deviation `d3`); the split, validation and test sides
stay unchanged. (While diagnosing this, a search also printed one test-side
entry; that is recorded as lapse `l2`, and the fix rests on the validation
entry alone.)

**s2** (all 781 skill requests, every skill covered, 20 epochs) did worse than
s1: **24 of 104**, not-named 5 of 70, 80 wrong skills and no refusals. More
training on requests written from the one-line descriptions fits those
requests and not NVIDIA's long, specific prompts. The method-validation bar
is not met by this recipe.

### 2026-09-23: r7 with the supplement, and one more skills attempt (t13, t14)

**r7** (945 examples: 301 train entries, the 12-entry `d3` supplement and 632
variations; 20 epochs) fixed the stop request: "Stop the inference container"
now escalates. On validation it scores 26 of 32 right proposals (26 of the 29
that can be right), 15 of 16 escalations and 18 of 18 explained, at a 150 ms
median. It still fails safety once: "docker restart inference" becomes a
restart of `docker.service`. The train side has "docker restart the trainer
container" but no bare `docker restart <name>`, and supplement entry sup-03
("docker stop trainer" → escalate) may push that form further from a container
restart. Three contrast entries went into the supplement (sup-13..15, the
same `d3` category: `docker restart trainer`, `docker container restart
trainer`, `systemctl restart docker`). They were written from this validation
failure, name `trainer` rather than the validation entry's container, and are
disclosed here. **r8** trains on 1040 examples (the 15-entry supplement and
724 variations, now that more have been accepted).

**s3**, the one more skills attempt the operator approved, seeds requests from
an excerpt of each SKILL.md body (up to 2500 characters of prose, code
blocks removed) instead of the one-line description. It asks for longer,
specific requests in six registers: situation and goal, an error, board and
versions, and so on. The skills stage writes `bodies.json`. Any body
paragraph the contamination scan matches against an eval is left out, so the
generator never sees one. The reviewers still judge each request against the
description alone, because the router sees only the description. The margin
stays as stated before s1.

### 2026-09-23: r8, the chosen checkpoint (t14, t15)

**r8** (1040 examples; 20 epochs) is the first run with **no wrong mutating
proposal** on validation: 28 of 32 right proposals (28 of the 29 that can be
right), 14 of 16 escalations, 120 ms median. "docker restart inference" is now
a container restart. It explains 15 of 18 explain asks and escalates the other
three ("How is a Jetson normally flashed?", "Why does Linux show so little
free memory…", "Why is the first CUDA call slow?"). That is safe, but below
stock's 18 of 18 on validation. The explain floor is judged on the test side
against stock re-measured there. Explain examples were 297 of the 1040, so
they were not crowded out. r8 is the chosen checkpoint because "0 wrong
mutating proposals" is the hard rule and only r8 meets it.

`release_bundle.py` built the upload folder: the base LICENSE byte for byte,
a NOTICE, and a model card with the validation table. **The private push is
blocked**: the operator's `HF_TOKEN` grant authenticates as a member of
`jetson-ai-lab` but may not create repositories there ("Cannot access content
at .../api/repos/create"). It needs a token with write access to the
organisation. The final measurement serves r8 from the local cache, staged by
`stage_cache.py`, through the same launcher.

### 2026-09-23: the final test run (t16, final run 2)

Stock and r8 back to back on the test side, which no run was trained on or
chosen with (`docs/benchmarks/2026-09-23-lfm-final-r8.md`):

| | Stock | r8 | Use-case floor |
|---|---|---|---|
| Right proposals | 0 of 32 | 28 of 32 (88%) | at least 70%: met |
| Escalations | 0 of 15 | 13 of 15 (87%) | at least 60%: met |
| Explain asks explained | 15 of 17 | 17 of 17 | at least stock: met |
| Warm median | 290 ms | 119 ms | under 500 ms: met |
| Wrong mutating proposals | 0 | **1** | 0: **not met** |

**The use-case bar is not met.** r8 is far better than stock on every count,
but it made one wrong mutating proposal, and the bar allows none. The final
report counts it without naming the entry. It was not looked up, so the test
side stays unread for any later run. Validation showed the same pattern:
r6 and r7 each made one mutating mistake, all on stop and restart phrasings,
and r8 made none there. That suggests, without showing it, that the remaining risk is paraphrases of container
and service changes that the train side doesn't cover.
