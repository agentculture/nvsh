# LFM Open License: what it means for nvsh

Read 2026-09-19 from the publisher's page (`liquid.ai/lfm-license`), LFM Open
License v1.0. This is an engineer's reading to decide what nvsh may do, not
legal advice. The quotes were extracted from the page by a fetch tool; check
them against the licence file shipped in the model repository before
publishing anything.

## What the licence says

- **Redistribution is allowed, including fine-tunes** (Section 4): "You may
  reproduce and distribute copies of the Work or Derivative Works thereof in
  any medium, with or without modifications", on four conditions:
  - (a) give recipients a copy of the licence;
  - (b) modified files carry prominent notices that you changed them;
  - (c) keep all copyright, patent, trademark and attribution notices;
  - (d) if the work ships a `NOTICE` file, the derivative includes its
    attribution notices.
- **Your own terms on your changes** are allowed; there is no copyleft.
- **No trademark grant** (Section 7), except to describe the origin of the
  work.
- **Commercial use has a revenue threshold** (Section 5): rights for
  commercial use are "conditioned upon You or Your Legal Entity not exceeding
  the Threshold", which is "annual revenue of 10 million United States
  dollars ($10,000,000) or more". "Legal Entity" includes every entity under
  common control. "Commercial Use" is "any use of the Work for direct or
  indirect commercial advantage or monetary compensation". A qualified
  non-profit's research use is exempt.
- **Termination is automatic** on any breach (Section 11).

## What nvsh does about it

- nvsh **downloads LFM2.5 from the publisher and never redistributes the
  stock weights**. `nvsh tiers prefetch` names the licence and its URL before
  it downloads an LFM2.5 file.
- A **tuned LFM2.5 published under `jetson-ai-lab`** is a Derivative Work. Its
  model card must: include the licence text, state that it is a modified
  LFM2.5 and what was changed (the nvsh operations fine-tune, with the data
  set and recipe linked), keep Liquid AI's notices, and describe the origin
  as "fine-tuned from LiquidAI/LFM2.5-…" without using the name as a brand.
  The name pattern `lfm2.5-350m-nvsh-triage` describes origin; keep it that
  way.
- **The threshold follows the user, not the publisher of the fine-tune.** An
  operator whose company is over the threshold and who uses Tier 2
  commercially needs their own agreement with Liquid AI. `docs` and the model
  card say so plainly. Tier 1 (Needle3, Apache-2.0) has no such limit, and
  nvsh works with Tier 2 absent.

## Who publishes (answered by the operator, 2026-09-19)

The operator states that the `jetson-ai-lab` organisation is a small
non-profit community. On that basis it is far below the threshold, and a free
fine-tune published there is within the licence as read above. A
tuned LFM2.5 may therefore be published under `jetson-ai-lab`, with the model
card duties listed in the previous section. Two things stay true:

- the threshold still applies to each **user** of the tuned model, and the
  model card says so;
- if the organisation's standing changes (it comes under a company's
  control), this reading must be redone before the next upload.

Nothing in nvsh uploads by itself; publishing is an operator action.

## The upload folder for a tuned LFM2.5 (issue 39)

`scripts/lfm-finetune/release_bundle.py` builds the folder that is uploaded,
and refuses to build one that misses a duty above:

- `LICENSE` is the base model's own licence file, copied byte for byte, and
  must begin with "LFM Open License v1.0";
- `NOTICE` states that the model is a modified LFM2.5, names the base
  revision, says what was changed, and keeps Liquid AI's notices;
- `README.md` (the model card) names the licence and the base model in its
  front matter, says the model was modified, states the USD 10,000,000
  threshold as a duty of every user, and lists the teacher models that
  generated and reviewed the synthetic training requests;
- the chat template must be byte-identical to the base model's.

The script never uploads. The operator pushes the folder to a **private**
repository with `grant run --inject HF_TOKEN=HF_TOKEN -- hf upload ...`, so
the token lives only in that one process. A model trained on the Jetson
skills data would also need NVIDIA's CC-BY-4.0 attribution (repositories and
commits from the skills `manifest.json`); the nvsh-triage model is not
trained on that data.
