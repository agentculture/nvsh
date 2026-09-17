# Current spec — what the app does today

## Coverage boundary

This projection is complete only over the behavior ledger: 4 of 5 plans have a ledgered delivery (`first-class-multi-harness-with-aliases`, `readme-demo-recording`, `reliable-agent-stop`, `setup-works-with-any-installed-agent`), spanning `2026-09-14T06:16:12Z` (plan `first-class-multi-harness-with-aliases`) through `2026-09-16T18:23:52Z` (plan `reliable-agent-stop`).
1 of 5 frame have no ledgered delivery at all (`nvsh-bash-hook-agent-on-error`) — nothing in this document reflects them.
Anything predating this boundary, or belonging to an unledgered frame, is not reflected here by construction.

## Current behavior

- on a hung-up terminal, end of input at a proposal means ignore, never approve (`reliable-agent-stop:b1`, amended)
  - provenance: caused by `d1` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - ⚠ unproven: no passing evidence on record
    - evidence: none on record
- a Ctrl+C/Esc press while an approved command runs is acted on after the command returns (`reliable-agent-stop:b2`, amended)
  - provenance: caused by `d2` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - ⚠ unproven: no passing evidence on record
    - evidence: none on record
- after any stop press in a one-shot run, the harness's leftover process tree is killed once the turn ends (`reliable-agent-stop:b3`, added)
  - provenance: caused by `d8` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-17 @ 0fc7d24)
- /ask, Ctrl+G and @target requests put proposals to the operator on the panel with the full approval legend (`reliable-agent-stop:b4`, added)
  - provenance: caused by `d11` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-17 @ 0fc7d24)
- the declined exit code is observable via --json `exit_code`, the audit log and client.ask()'s return, not a 'nvsh ask' CLI exit status (no such verb) (`reliable-agent-stop:b5`, amended)
  - provenance: caused by `d10` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-17 @ 0fc7d24)
- normal close() on pi, codex, acp and agy reaps the harness's process group, and claude/qwen-p reap it after a normal exit (`reliable-agent-stop:b6`, added)
  - provenance: caused by `d14` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-17 @ 0fc7d24)
    - evidence: automated — execution: pass (run 2026-09-17 @ 0fc7d24)
- Ctrl+C at a proposal or busy prompt stops the agent (exit 130, cancel once, no declined record) instead of declining (PR #16 review, Qodo 4) (`reliable-agent-stop:b7`, amended)
  - provenance: caused by `c4`, `c9` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - ⚠ unproven: no passing evidence on record
    - evidence: none on record
- `kill_active` carries the confirmed turn identity and the daemon refuses with 'changed' if the active turn differs; a confirmed live-owner kill without identity is refused (PR #16 review, Qodo 3) (`reliable-agent-stop:b8`, amended)
  - provenance: caused by `c12` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-17 @ 0fc7d24)
- nvsh doctor --apply exits healthy after it clears the only failing check (PR #16 review, Qodo 1) (`reliable-agent-stop:b9`, amended)
  - provenance: caused by `c12` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - proof: best strength `execution`
    - evidence: automated — execution: pass (run 2026-09-17 @ 0fc7d24)
- a cancelled warm agy process is retired and the next turn respawns with a fresh queue (PR #16 review, Qodo 5) (`reliable-agent-stop:b10`, amended)
  - provenance: caused by `c3`, `d5` — plan `reliable-agent-stop`, frame `reliable-agent-stop`
  - ⚠ unproven: no passing evidence on record
    - evidence: none on record
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

- proposed deltas awaiting adjudication: 17
- rejected deltas (excluded from this projection): 0
- retired lineages (superseded with no live replacement): 0
