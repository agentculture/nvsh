# Harness selection: two decisions, not one

This page is the spec-level companion to
[`docs/automation-contract.md`](automation-contract.md) (the invocation-level
contract for force-selecting a harness) and
[`docs/harness-invocations.yaml`](harness-invocations.yaml) (the
machine-readable source both that page and CI read). This page does not
duplicate either — it answers a different question: *who reads the harness
config, what does it actually decide, and what does it deliberately not
decide.*

## Who reads the harness config

Three distinct readers care about the harness setup in a clone of this
template, and they care about it for different reasons:

1. **The provisioning operator** running `guild create` (or hand-editing a
   fresh clone) — decides the *initial* `backend` a new sibling declares in
   `culture.yaml`, and which of the four prompt files gets written with
   real content versus left as a template stub.
2. **The clone's own resident agent** — the process the Culture daemon
   starts for this clone, which reads exactly one prompt file (the one
   matching its own backend) as its operating instructions.
3. **The doctor / inventory tooling** — `steward doctor` and this
   template's own `culture-agent-template doctor`, which parse
   `culture.yaml` and check that the declared `backend` has a matching
   prompt file on disk (**prompt-file-present**) and that the pairing is
   correct (**backend-consistency**, e.g. `claude` ↔ `CLAUDE.md`, never
   `claude` ↔ only `AGENTS.colleague.md`). guildmaster's `guild overview`
   / `guild show` read the same file for the same reason: reporting, not
   judging.

None of these three readers cares about — or changes — which harness
binaries a human or script can run interactively. That is a separate
concern, covered next.

## The two selections

It is easy to say "switch harness" and mean either of two different things.
This document exists so that never happens silently. They are stated here
**separately**, and neither is called just "switch harness" on its own:

### Selection 1 — the interactive harness

Chosen by **which binary you run**. `cd` into a clone of this template and
run `claude`, `pi`, `colleague`, or `qwen`. Each reads its own prompt file:

| Binary | Prompt file(s) it reads |
|---|---|
| `claude` | `CLAUDE.md` |
| `pi` | `AGENTS.override.md` (context) + `.pi/SYSTEM.md` (system prompt) |
| `colleague` | `AGENTS.colleague.md` |
| `qwen` | `QWEN.md` |

**All four are live simultaneously** against the same working tree. There is
no config file that "activates" one of them and no file that must be edited
before you can run a given binary — running the binary *is* the selection.
Any of the four can also be **force-chosen** by a human or by automation
(for example, a CI smoke check) regardless of what `culture.yaml` declares.
Forcing a harness this way is strictly **invocation-level** — flags passed
to that one process — and must never mutate `culture.yaml` or any other
tracked file, because a shared checkout may have other callers running
concurrently; a config-mutating force would race them and leave the working
tree dirty for whoever runs next. The exact forced invocations and their
empirical verification live in
[`docs/automation-contract.md`](automation-contract.md).

### Selection 2 — the mesh resident

Chosen by the single `backend` value `culture.yaml` declares:

```yaml
agents:
- suffix: culture-agent-template
  backend: claude
```

This is what the **Culture daemon starts** for this clone, and it is the
only harness selection that `steward doctor` and this template's own
`doctor` verb check. `guild harness use <name>` (guildmaster's write verb)
changes **only this** — it edits `culture.yaml`'s `backend` field and
nothing else. It does not disable, remove, or otherwise affect the other
three prompt files, and it has no effect on which harness a human can run
interactively per Selection 1 above.

### Telling them apart

A reader who only sees "harness switch" mentioned once cannot tell which of
the two is meant. This document — and every place in this repo that
discusses harness selection — states the two separately so that is never
ambiguous:

- If the change is "I ran a different binary in my terminal," that is
  Selection 1. It touches no file.
- If the change is "the mesh daemon now starts a different resident," or
  "`guild harness use` was run," that is Selection 2. It touches exactly
  `culture.yaml`.

`guild harness use <name>` only ever affects Selection 2. It never
force-selects an interactive harness, and running a harness binary directly
never changes Selection 2.

## Prior state (what this arc fixed)

Before this arc, the template was accidentally **two-harness and silently
inconsistent**: `culture.yaml` declared `backend: colleague` while a
217-line `CLAUDE.md` shipped beside an 11-line `AGENTS.colleague.md` — the
fuller, better-maintained prompt file belonged to a harness the mesh
resident wasn't even running. Three separate backend→prompt registries (in
the CLI, in doctor, and in prose) disagreed with each other, `guild create
--backend colleague` raised a `ValueError`, and roughly thirty already
provisioned siblings inherited an unreconciled backend with no check ever
firing to catch it. That contradiction is why `CLAUDE.md` in this repo used
to describe itself as belonging to a `colleague`-resident template — it no
longer does; `culture.yaml` here declares `backend: claude` and `CLAUDE.md`
is this clone's resident prompt file, consistent with each other.

## Non-goal: migrating existing siblings

**This arc does not retrofit already-provisioned siblings.** No task in the
plan that produced this document edits an already-provisioned sibling
repo's `culture.yaml` or prompt files. The scope here is strictly: (a) make
this template internally consistent, and (b) stop *new* siblings created
from this template from inheriting the same two-harness confusion.

The direct consequence is that
[`culture-agent-template#25`](https://github.com/agentculture/culture-agent-template/issues/25)
("Downstream scaffolds still carry the pre-0.3.4 backend: claude seed text —
no re-sync path exists") **stays open**. It tracks the existing fleet of
already-provisioned siblings, which this work does not touch and for which
no re-sync path is introduced here. Nobody should close #25 on the strength
of this document or this arc — it addresses only what a *newly instantiated*
clone of this template looks like going forward, not what any sibling
created before this arc already has on disk.
