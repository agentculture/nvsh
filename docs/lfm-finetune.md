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

## Pick the base

Start from the smallest post-trained LFM2.5 that passes the stock baseline
(230M or 350M; 1.2B only if both fail), because Tier 2 must stay resident on
an 8 GB device. Confirm the exact repository id on the publisher's Hugging
Face page; do not guess it.

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
