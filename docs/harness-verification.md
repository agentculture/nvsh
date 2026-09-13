# Four-harness verification run

Evidence for the plan task `t14`, recorded 2026-09-06 against a **genuinely
fresh `git clone`** of this repo's `feat/four-harness-support` branch — not the
working tree the changes were authored in.

The operator required this run rather than accepting a file-layout argument:
every coexistence claim in the spec had been reasoned from loader source, and
nothing had observed four harnesses actually running over one clone.

## Method

Each harness was run in the clone with **instrumentation**, never by asking the
model what it had loaded. Model self-reports are unreliable evidence here: an
earlier check of this same wiring accepted a model's answer and reached the
wrong conclusion (see *What this run overturned*, below).

| Harness | Instrument |
|---------|-----------|
| Pi | `strace -f -e trace=openat` — which files the process actually opened |
| Qwen Code | `qwen --debug` session log — its own `[SKILL_MANAGER]` lines |
| colleague | `colleague agents list` / `skills list --json` — the same `layers.resolve_skills` path the real prompt composition uses |

## Results

| Harness | Prompt file resolved | Skills loaded | `CLAUDE.md` leaked in? |
|---------|---------------------|---------------|------------------------|
| Pi / associate | `.pi/SYSTEM.md` (system prompt) **and** `AGENTS.override.md` (context) | **19** via `.pi/skills` | **No** — 0 opens |
| Qwen Code | `QWEN.md` (3 refs in log) | **19** project-level | **No** — 0 refs |
| colleague | `AGENTS.colleague.md` | **0** — see below | n/a |
| Claude Code | `CLAUDE.md` | `.claude/skills` natively | n/a |

Each harness resolved **its own** file and no other's. `AGENTS.override.md`
does the job it was chosen for: Pi never opened `CLAUDE.md`.

### colleague skills: a known upstream failure, not a template defect

colleague loads **0 of 19**. Reproduced against colleague 1.76.0 and filed as
[`agentculture/colleague#494`](https://github.com/agentculture/colleague/issues/494).
Two independent blockers, either fatal alone:

* `configdir.collect_files` takes only direct **file** children, so a nested
  skill directory is skipped;
* `layers._within` rejects any symlink target leaving a `.colleague/` root — a
  deliberate control against smuggling local files into a prompt bound for a
  remote engine.

Both fail with **exit 0**, so a wired repo is indistinguishable from an unwired
one. The template ships the nested shape anyway; it becomes correct the moment
upstream lands the fix.

### Qwen over ACP

`qwen --acp` answers a real ACP `initialize` JSON-RPC call
(`protocolVersion: 1`, `agentInfo: qwen-code 0.23.0`), confirming the transport
a `backend: acp` + `acp_command: [qwen, --acp]` mesh resident depends on. A
full round-trip through a running Culture mesh was **not** performed — the ACP
handshake is proven, mesh residency is not.

### steward doctor

`steward doctor --scope self` on the fresh clone returns **one** finding: the
`portability` check, reporting five pre-existing eidetic memory-store references
in the vendored `recall` / `remember` skills that predate this work. There is
**no `backend-consistency` finding and no `prompt-file-present` finding** — the
four-prompt-file clone declaring `backend: claude` passes both harness checks.

## What this run overturned

Wiring Pi's skills through a `skills` array in `.pi/settings.json` **does not
work**. That key belongs to a *package manifest* (`package.json`'s `pi` block),
not to `settings.json`. Syscall counts, from this clone:

```text
.pi/settings.json alone       ->   0 SKILL.md opened
pi --skill .claude/skills     ->  19 SKILL.md opened
.pi/skills symlink, no flag   ->  19 SKILL.md opened
```

The fix is the relative-symlink pattern already proven for `.qwen/skills` and
`.colleague/skills`. The inert `settings.json` was removed rather than shipped
as a file whose only key does nothing.

This is exactly the failure the run existed to catch, and it was caught only
because the check was instrumented rather than asked.
