# Skill upstream sources

culture-agent-template vendors its `.claude/skills/` from **guildmaster** — the
AgentCulture **skills supplier** after the steward → guildmaster cutover
(guildmaster 0.5.0, 2026-05-24). `steward` retains the **alignment** role
(`steward doctor`, the sibling-pattern baseline); only the skills-supplier role
moved. This file tracks provenance so re-syncs stay deterministic.

Eight skills — `think`, `spec-to-plan`, `assign-to-workforce`, `scope`,
`challenge`, `deviate`, `validate-delivery`, and `summarize-delivery` —
originate in
[`agentculture/devague`](https://github.com/agentculture/devague) and are
**re-broadcast** through guildmaster: cite guildmaster's copy, track devague as
the true origin.

All eight now come from guildmaster, wrapper scripts included. The
2026-07-15 decision to vendor `scope` / `challenge` / `deviate` /
`summarize-delivery` **directly from devague**, specifically to keep
guildmaster's added `scripts/*.sh` wrappers out, is **superseded** — see
[below](#superseded-2026-07-15-vendor-directly-from-devague-to-avoid-the-wrappers).
The wrappers are now wanted here: they make every clone ship a complete,
convention-clean skill directory.

One skill, `ask-colleague` (formerly `outsource`), originates in
[`agentculture/colleague`](https://github.com/agentculture/colleague) — the
renamed `convertible`. guildmaster's re-broadcast still carries the old
`outsource` name, so `ask-colleague` is vendored **directly from colleague** as a
tracked local divergence (see [below](#local-divergence--outsource--ask-colleague-2026-06-06)).

Every vendored `SKILL.md` carries `type: command`. culture-agent-template
declares a culture agent (`culture.yaml`, `backend: claude`), and
`core.skill_loader` silently skips any `SKILL.md` lacking `type:` — so the field
is load-bearing, even where guildmaster's upstream copy omits it.

| Skill | Upstream | Origin | Notes | Last synced |
|-------|----------|--------|-------|-------------|
| `cicd` | `../guildmaster/.claude/skills/cicd/` | guildmaster | CI/CD lane layered on `devex pr`: the 5 thin scripts (`workflow.sh`, `pr-status.sh`, `pr-reply.sh`, `_resolve-nick.sh`, `portability-lint.sh`) delegate lint/open/read/reply/delta to `devex` and add the `status` / `await` SonarCloud-gating extensions. Consumer-identifying prose (`guildmaster` → `culture-agent-template`) adapted in the description + heading; upstream history (`Renamed from pr-review in steward 0.7.0; rebased on devex in 0.12.0`) and env-var literals (`STEWARD_*`) kept verbatim. The PR signature resolves at runtime from `culture.yaml` via `_resolve-nick.sh` (→ `culture-agent-template`). Requires `devex` on PATH. | 2026-05-26 (guildmaster 0.6.0) |
| `communicate` | `../guildmaster/.claude/skills/communicate/` | guildmaster | Cross-repo + mesh communication. Consumer-identifying prose adapted in the description (incl. the `- culture-agent-template (Claude)` signature line). **No hard-coded signature literal in the scripts** — `post-issue.sh` is `agtag`-backed and resolves the signing nick from `culture.yaml`; requires `agtag` (>=0.1) on PATH. The supplier `scripts/templates/` (`skill-update-brief.md`, `skill-new-brief.md`) are kept verbatim — inert for a consumer (they cite guildmaster as upstream). Renamed from `coordinate` in steward 0.8.0; absorbed `gh-issues` in 0.9.1. | 2026-05-26 (guildmaster 0.6.0) |
| `version-bump` | `../guildmaster/.claude/skills/version-bump/` | guildmaster | Pure-Python, CWD-aware (`scripts/bump.py`). Verbatim except added `type: command`. | 2026-05-26 (guildmaster 0.6.0) |
| `agent-config` | `../guildmaster/.claude/skills/agent-config/` | guildmaster (origin steward) | Shows a Culture agent's full config; run `scripts/show.sh` directly (no `guild` binary required). `scripts/show.sh` + `data/backend-fingerprints.yaml` verbatim. Verbatim except added `type: command`. | 2026-05-26 (guildmaster 0.6.0) |
| `doc-test-alignment` | `../guildmaster/.claude/skills/doc-test-alignment/` | guildmaster | **STUB** — `scripts/check.sh` exits not-yet-implemented; the contract lives in SKILL.md. Verbatim except added `type: command`. | 2026-05-26 (guildmaster 0.6.0) |
| `pypi-maintainer` | `../guildmaster/.claude/skills/pypi-maintainer/` | guildmaster | Switch a package install between PyPI / TestPyPI / local editable (`scripts/switch-source.sh`). Verbatim except added `type: command`. | 2026-05-26 (guildmaster 0.6.0) |
| `run-tests` | `../guildmaster/.claude/skills/run-tests/` | guildmaster | pytest + xdist + coverage (`scripts/test.sh`). Verbatim except added `type: command`. | 2026-05-26 (guildmaster 0.6.0) |
| `sonarclaude` | `../guildmaster/.claude/skills/sonarclaude/` | guildmaster | SonarCloud API queries (`scripts/sonar.sh`). Verbatim except added `type: command`. | 2026-05-26 (guildmaster 0.6.0) |
| `think` | `../guildmaster/.claude/skills/think/` | **devague** (re-broadcast via guildmaster) | idea→spec leg of the devague workflow chain. Verbatim (already carried `type: command` at guildmaster). Origin/broadcast prose left verbatim. | 2026-09-05 (devague 0.24.1 via guildmaster) |
| `spec-to-plan` | `../guildmaster/.claude/skills/spec-to-plan/` | **devague** (re-broadcast via guildmaster) | spec→plan leg of the devague workflow chain. Verbatim (already carried `type: command`). | 2026-09-05 (devague 0.24.1 via guildmaster) |
| `assign-to-workforce` | `../guildmaster/.claude/skills/assign-to-workforce/` | **devague** (re-broadcast via guildmaster) | plan→parallel-implementation leg of the devague workflow chain. Verbatim (already carried `type: command`). | 2026-09-05 (devague 0.24.1 via guildmaster) |
| `scope` | `../guildmaster/.claude/skills/scope/` | **devague** (re-broadcast via guildmaster) | Explores the scope of a vague idea BEFORE framing it into a spec — the idea→scope leg, the optional opening move ahead of `/think`; surveys the surfaces the idea touches (code, docs, skills, CI, sibling repos) and seeds the coming Announcement Frame with boundary/non-goal/assumption claims that cite what was actually explored. `SKILL.md` verbatim with devague (already carries `type: command`); guildmaster adds `scripts/scope.sh`, vendored here deliberately since 0.24.1. | 2026-09-05 (devague 0.24.1 via guildmaster) |
| `challenge` | `../guildmaster/.claude/skills/challenge/` | **devague** (re-broadcast via guildmaster) | Runs a risk-scaled blind-spot discovery pass over a converged, exported frame BETWEEN `/think` and `/spec-to-plan` (the seventh origin skill, third leg in flow order): pressure-tests the spec through structured lenses, routes every finding back through the existing deterministic moves as proposed-only content the human adjudicates, and on a clean pass records the examined lenses/surfaces and residual uncertainty — never a claim that there are no unknown unknowns. `SKILL.md` verbatim with devague (already carries `type: command`); guildmaster adds `scripts/challenge.sh`, vendored here deliberately since 0.24.1. | 2026-09-05 (devague 0.24.1 via guildmaster) |
| `deviate` | `../guildmaster/.claude/skills/deviate/` | **devague** (re-broadcast via guildmaster) | Stops an in-flight assign-to-workforce run the moment execution must diverge from the confirmed plan, gets explicit human approval for the divergence, and records it as a first-class, append-only deviation record via `devague deviate` before resuming — never folds a deviation silently into drift after the fact. `SKILL.md` verbatim with devague (already carries `type: command`); guildmaster adds `scripts/deviate.sh`, vendored here deliberately since 0.24.1. | 2026-09-05 (devague 0.24.1 via guildmaster) |
| `validate-delivery` | `../guildmaster/.claude/skills/validate-delivery/` | **devague** (re-broadcast via guildmaster) | Runs the confirmed plan's behavioral tests agent-side after `/assign-to-workforce` merges its waves and before `/summarize-delivery` closes the loop, then files what was found — evidence for what passed, behavioral deltas for what the run added, amended, or removed — as first-class, record-only entries via the devague CLI. The CLI never runs a test ([devague#20](https://github.com/agentculture/devague/issues/20)); a failing or partial outcome is never suppressed. `SKILL.md` verbatim with devague (already carries `type: command`); guildmaster adds `scripts/validate-delivery.sh`. | 2026-09-05 (devague 0.24.1 via guildmaster) |
| `summarize-delivery` | `../guildmaster/.claude/skills/summarize-delivery/` | **devague** (re-broadcast via guildmaster) | Closes the loop after an assign-to-workforce run by turning what actually happened into an accountability artifact — planned versus actual delivery, mid-work decisions, plan drift, evidence-backed delivery claims, and remaining work; runs on complete, partial, AND failed runs, reporting failure faithfully rather than smoothing it over. `SKILL.md` verbatim with devague (already carries `type: command`); guildmaster adds `scripts/summarize-delivery.sh`, vendored here deliberately since 0.24.1. | 2026-09-05 (devague 0.24.1 via guildmaster) |
| `ask-colleague` | `../colleague/.claude/skills/ask-colleague/` | **colleague** (renamed from convertible; vendored directly — guildmaster re-broadcast pending) | The first-party front door to the `colleague` CLI: hand a scoped task to a *different* engine/mind via `explore` / `review` / `write`, run the spec→plan→workforce arc via `plan`, pick a cut or timed-out run back up via `resume` (`--detach` to background it), pilot a live work item with `monitor` / `guide` / `stop`, grade a finished work item via `feedback` (the ROI loop), and reap stale/corrupt `colleague/*` branches a crashed run left behind via `clean`. Thinking effort is per-seat (`--effort`, `--seat-effort S=R`, `--role`). Every verb takes `--json` (result JSON on stdout, diagnostics on stderr). `explore`/`review` run isolated in a throwaway `git worktree`; `write` **previews by default** (throwaway worktree, no side effects) and refuses a dirty tree only when applying (`--apply` / `--pr`). Vendored **byte-verbatim** as of the 1.63.0 sync — the Provenance paragraph is consumer-neutral upstream, so the localization noted for earlier syncs no longer applies; verify with `diff -r ../colleague/.claude/skills/ask-colleague .claude/skills/ask-colleague`. Already carries `type: command`. Optional runtime dep: **`colleague`** on PATH. | 2026-08-24 (colleague 1.63.0, direct) |

## Re-sync procedure

```bash
# Diff against upstream before pulling (example: cicd / communicate):
for s in cicd communicate; do
  diff -ru ../guildmaster/.claude/skills/$s .claude/skills/$s
done

# Pull a skill fresh (remove first so dropped scripts don't linger):
rm -rf .claude/skills/<skill>
cp -R ../guildmaster/.claude/skills/<skill> .claude/skills/

# Re-apply the identifier-only adaptations in SKILL.md:
#   - consumer-identifying prose: `guildmaster` → `culture-agent-template` (NOT
#     where it cites guildmaster/steward/devague as the upstream/origin).
#   - add `type: command` to the frontmatter if guildmaster's copy omits it
#     (load-bearing for the culture/claude backend's core.skill_loader).
# No script bodies are edited (cite-don't-import). The communicate signature
# resolves from culture.yaml via agtag — no literal to patch.
```

If a re-sync would lose a culture-agent-template adaptation, lift the change
upstream into guildmaster first (per guildmaster's `docs/skill-sources.md`) and
re-vendor.

### Local divergence — `agex` → `devex` rename (2026-05-30)

The PR-lifecycle CLI was renamed `agex` → `devex` (same tool, new name). The
vendored `cicd` (`SKILL.md`, `workflow.sh`, `pr-status.sh`),
`assign-to-workforce`, and `communicate` (`skill-new-brief.md` template) copies
were **patched in place** for this rename rather than re-vendored — a deliberate
exception to cite-don't-import, made so the `cicd` scripts invoke the real
`devex pr` binary now. The matching canonical rename is tracked upstream for
guildmaster in [agentculture/guildmaster#48](https://github.com/agentculture/guildmaster/issues/48),
so the next clean re-sync from guildmaster reconciles without losing this
change. (Re-sync once guildmaster's renamed copies are broadcast.)

The same in-place patch also bumped the documented `devex` version floor from
`>=0.1` to `>=0.21` in the vendored `cicd` `SKILL.md` + `workflow.sh` (to match
this doc's tooling-prerequisites and the `await`-era feature set) — likewise
flagged for guildmaster on #48.

### Local divergence — outsource → ask-colleague (2026-06-06)

`convertible` was renamed **`colleague`**, and its skill `outsource` →
**`ask-colleague`** (colleague#148; the `wheels` verb also became `backends`, and
`drive` → `work`). `ask-colleague` adds a fourth verb, `feedback` (the ROI loop),
and `write` now **previews by default** (a throwaway worktree, no side effects)
instead of committing to a branch unless you pass `--apply` / `--pr`.

guildmaster has **not** re-broadcast the rename yet — its kit still ships the old
`outsource`. So this template's `outsource/` was removed and `ask-colleague/`
vendored **directly from the sibling `colleague` checkout**
(`../colleague/.claude/skills/ask-colleague/`), not from guildmaster. This is a
tracked exception to "cite guildmaster's copy", parallel to the `agex` → `devex`
divergence above. Re-sync path until guildmaster catches up:

```bash
# Pull ask-colleague fresh from colleague (the origin):
rm -rf .claude/skills/ask-colleague
cp -R ../colleague/.claude/skills/ask-colleague .claude/skills/
# Byte-verbatim as of 1.63.0 — nothing to re-apply. Upstream rewrote the
# SKILL.md Provenance paragraph to be consumer-neutral, retiring the one
# consumer-identifying clause earlier syncs had to patch back in
# (`which colleague vendors from guildmaster` →
#  `which culture-agent-template vendors from guildmaster`).
# Confirm the copy is clean:
diff -r ../colleague/.claude/skills/ask-colleague .claude/skills/ask-colleague
# (already carries `type: command`; no script bodies edited.)
```

**Vendored means vendored.** Findings a reviewer raises against
`scripts/ask-colleague.sh` — bot or human — are fixed **upstream in
`agentculture/colleague` and pulled back in on the next sync**, never patched
here. A local patch is exactly the drift this ledger exists to prevent: the
next re-sync silently reverts it, and in the meantime `diff -r` against the
origin stops being a meaningful check.

Once guildmaster re-broadcasts `ask-colleague`, switch the upstream column back
to `../guildmaster/.claude/skills/ask-colleague/` and re-sync from there.

### Superseded (2026-07-15): vendor directly from devague to avoid the wrappers

**This divergence no longer applies. Do not follow the recipe it described.**

From 2026-07-15 until the devague `0.24.1` re-sync, `scope`, `challenge`,
`deviate` and `summarize-delivery` were vendored **directly from
`../devague/.claude/skills/<skill>/`** rather than from guildmaster. The reason
was guildmaster's added `scripts/*.sh` wrapper per skill (guildmaster
`292feac`) — at the time judged "content this repo never asked for", so citing
guildmaster's copy would have pulled it in silently.

**That judgment is reversed.** The wrappers are now vendored here deliberately,
for all five prompt-only skills (`validate-delivery` joins the other four):

- A clone that ships `SKILL.md` with no sibling `scripts/` fails a
  `test_skills_convention`-style gate — the failure guildmaster itself hit in
  [guildmaster#95](https://github.com/agentculture/guildmaster/issues/95). Every
  repo scaffolded from this template inherited that shape.
- The wrappers are thin and portable: each forwards its arguments to the
  `devague` CLI verbatim, so upstream still owns the surface and no wrapper
  needs editing when a move is added. They contain no absolute paths and no
  cross-repo references.

So the upstream column for all eight is now `../guildmaster/.claude/skills/`,
and the normal procedure applies — there is no longer an exception here.
Re-sync path:

```bash
# All eight come from guildmaster, wrappers included:
for s in scope think challenge spec-to-plan assign-to-workforce \
         deviate validate-delivery summarize-delivery; do
  rm -rf .claude/skills/$s
  cp -R ../guildmaster/.claude/skills/$s .claude/skills/
done
```

Each `SKILL.md` is byte-verbatim with devague (they already carry
`type: command` upstream — do **not** add a second one; the key sits after the
multi-line `description:`, so a first-few-lines check will wrongly report it
missing). The only content guildmaster adds is the five wrapper scripts, which
are additions, not edits — verify with:

```bash
for s in scope challenge deviate validate-delivery summarize-delivery; do
  diff ../devague/.claude/skills/$s/SKILL.md .claude/skills/$s/SKILL.md
done
```

## Tooling prerequisites

- **`devex`** (>=0.21) on PATH — `cicd` delegates the PR lifecycle to `devex pr`.
- **`agtag`** (>=0.1) on PATH — `communicate` issue I/O wraps `agtag issue`.

Both ship on PATH in the standard AgentCulture dev setup (installed per the
devex / agtag READMEs).

- **`colleague`** on PATH — *optional*; only the `ask-colleague` skill needs it,
  and only when invoked (`uv tool install colleague`). The wrapper exits
  with a clear install hint if it is absent, so the skill degrades gracefully
  rather than blocking a clone that never uses it. `ask-colleague` also needs a
  reachable backend — a local vLLM by default, overridable via `--engine` /
  `--model` / `--base-url` or `COLLEAGUE_*` env (the legacy `CONVERTIBLE_*` names
  still work as a deprecated fallback).

## Pending upstream: associate backend

The template is gaining support for a fourth harness, `associate`
(Pi-shaped), alongside Claude, colleague, and Qwen Code (the latter needs no
new backend — it rides the existing `acp` backend). `associate` is not yet a
recognized Culture backend; promoting it is requested upstream in
[`agentculture/cultureagent#51`](https://github.com/agentculture/cultureagent/issues/51)
(daemons live in `cultureagent`; `culture_core/clients/<backend>/` in
`culture` are pure shims over it — neither guildmaster nor this template
implements backend daemons). Until that lands, a clone that wants associate as
its *mesh resident* declares it the way the existing `associate` agent runs
today: `backend: colleague` and `model: associate` in `culture.yaml`, with its
prompt carried via `.pi/SYSTEM.md` and `AGENTS.override.md`. This template
itself declares `backend: claude` — the Pi/associate harness ships here as an
interactively available harness, which is independent of which single backend
the Culture daemon starts.

## Per-harness skills discovery

The canonical skill tree stays exactly one copy, at `.claude/skills/<name>/`.
Each of the four harnesses this template ships for resolves that same tree
its own way — no skill script is ever forked or duplicated per harness:

- **Claude Code** reads `.claude/skills/` natively — no wiring needed.
- **Qwen Code** reads project skills from `.qwen/skills/`, and follows
  symlinks when loading them. `.qwen/skills` is therefore a *relative* git
  symlink to `../.claude/skills`, following the precedent already in place
  for [`sensibo-cli`](https://github.com/agentculture/sensibo-cli) (confirm
  with `git ls-files -s .qwen/` in that repo: mode `120000`). Being relative
  and stored as a real git object (not a copy), it survives a fresh clone on
  another machine and `guild create`'s identifier-rename transform, which
  walks real files and leaves symlinks untouched
  (`guild/scaffold/instantiate.py`). Verified directly: starting
  `qwen -p ... --debug` against a checkout of this template logs
  `[SKILL_MANAGER] Loaded 19 project level skills` sourced from
  `<repo>/.qwen/skills/...`, one line per vendored `SKILL.md` — Qwen Code
  resolves the full kit through the symlink with zero forked scripts.
- **Pi / associate** discovers skills from its user-level config directory,
  a user-level agents directory, or the repo-local `.pi/skills/` or
  `.agents/skills/` — none of which is
  `.claude/skills/` — plus whatever is passed via repeatable `--skill <path>`
  flags. Pointing that flag at the canonical tree does resolve it: running
  `pi --skill .claude/skills -p "list your loaded skills"` in this repo
  returns all 19 vendored skill names (verified directly). Wiring that
  flag into a checked-in `.pi/settings.json` so it happens automatically,
  without a hand-typed flag, is a separate task; until that lands, a plain
  `pi` invocation in this repo sees only Pi's own user-level skills, not the
  template's.
- **colleague** reads skills from `.colleague/skills/`, built via
  `colleague learn-from`. Populating that tree from `.claude/skills/` is a
  separate task; not addressed here.

## colleague skills wiring (nested tree, blocked upstream)

The "Per-harness skills discovery" section above left colleague as a separate
task. This is that task; the note there is superseded by what follows.

`.colleague/skills` is now a **relative git symlink** to `../.claude/skills`
(mode `120000`, byte-identical blob to `.qwen/skills` — confirm with
`git ls-files -s .colleague/skills .qwen/skills`). That exposes all 19 vendored
skills in the nested, directory-per-skill shape
`.colleague/skills/<skill-folder>/SKILL.md`, with no second copy and no forked
scripts, exactly as the Qwen harness already does.

`.colleague/` is otherwise gitignored, because colleague writes run artifacts
there. Git cannot re-include a path beneath an excluded directory, so
`.gitignore` ignores `.colleague/*` and negates `!.colleague/skills`: the config
half of the directory ships with the template, the artifact half stays local.

**colleague 1.76.0 does not load this shape yet — it resolves 0 of the 19.**
Two independent loader rules each block it, and both fail silently
(`colleague skills list` prints `(no skills found)` and exits 0):

1. `configdir.collect_files` takes only direct *file* children of the skills
   dir, so a skill *directory* is skipped and never descended into.
2. `layers._within` fully resolves symlinks and drops any target that leaves a
   `.colleague/` root — a deliberate control against smuggling arbitrary local
   files into a prompt sent to a remote engine.

Fixing either alone still yields 0 skills; both are needed. Reported upstream
with the full isolated reproduction in
[`agentculture/colleague#494`](https://github.com/agentculture/colleague/issues/494),
which also asks for a non-silent signal when a skill doc is seen and then
discarded. Per the standing decision the template is **not** reshaped to fit the
current loader — the nested symlink stays as committed and the fix lands
upstream.

Two consequences worth knowing while that is open:

- A plain `colleague` run in a clone of this template sees none of the
  template's skills — only whatever colleague's user-level config directory
  (its `skills/` subdirectory, resolved from `COLLEAGUE_HOME` when set) holds.
- **Do not run `colleague learn-from claude` in a repo with this symlink.** Its
  destination is `.colleague/skills/<name>.md`, which now resolves *through* the
  symlink, so it would write 19 generated flat docs into the canonical
  `.claude/skills/` tree. `learn-from` (flat generated copies carrying a
  `<!-- learned-from: -->` marker) is the currently-designed colleague answer and
  was deliberately not taken here: for a template every sibling is cloned from,
  it means committing a derived second copy of every skill doc that drifts from
  the canonical tree the moment a skill is re-vendored.
