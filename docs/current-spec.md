# Current spec — what the app does today

## Coverage boundary

This projection is complete only over the behavior ledger: 2 of 3 plans have a ledgered delivery (`first-class-multi-harness-with-aliases`, `setup-works-with-any-installed-agent`), spanning `2026-09-14T06:16:12Z` (plan `first-class-multi-harness-with-aliases`) through `2026-09-14T14:25:30Z` (plan `setup-works-with-any-installed-agent`).
1 of 3 frame have no ledgered delivery at all (`nvsh-bash-hook-agent-on-error`) — nothing in this document reflects them.
Anything predating this boundary, or belonging to an unledgered frame, is not reflected here by construction.

## Current behavior

- on an empty probe (openai-compat fallback) setup still offers node and pi as the bootstrap path instead of scoping offers to the pick (`setup-works-with-any-installed-agent:b2`, amended)
  - provenance: caused by `d1` — plan `setup-works-with-any-installed-agent`, frame `setup-works-with-any-installed-agent`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-14 @ 463f887)
- a bare adapter name (claude, @codex) is accepted by --agent / forced targets without an alias (`setup-works-with-any-installed-agent:b3`, added)
  - provenance: caused by `d2` — plan `setup-works-with-any-installed-agent`, frame `setup-works-with-any-installed-agent`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-14 @ 463f887)
- `_stop_daemon` discards the child's stdout/stderr so 'daemon: not running' never precedes setup's --json payload (`setup-works-with-any-installed-agent:b4`, amended)
  - provenance: caused by `d2` — plan `setup-works-with-any-installed-agent`, frame `setup-works-with-any-installed-agent`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-14 @ 463f887)
- probe() keeps one row per shared binary, so qwen-p is no longer listed beside qwen and a qwen-only PATH picks qwen without a prompt (`setup-works-with-any-installed-agent:b5`, amended)
  - provenance: caused by `c23` — plan `setup-works-with-any-installed-agent`, frame `setup-works-with-any-installed-agent`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-14 @ 4162f2f)
  - lineage: `setup-works-with-any-installed-agent:b1`, `setup-works-with-any-installed-agent:b5`

## Ledger status

- proposed deltas awaiting adjudication: 10
- rejected deltas (excluded from this projection): 0
- retired lineages (superseded with no live replacement): 0
