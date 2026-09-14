# Pi RPC protocol (in-repo reference for `PiAgent`)

This is a short, in-repo summary of `pi --mode rpc`'s wire protocol, kept
next to `nvsh/agent/pi.py` (`PiAgent`) and `tests/fakes/pi` /
`tests/fakes/pi_scripted` so both have a reference that does not depend on a
local Pi.dev install. It is distilled in our own words from the shipped
`docs/rpc.md` in `@earendil-works/pi-coding-agent` (pi 0.84.2, installed via
nvm's global `node_modules` under the operator's home directory on this
box) — read that file for the authoritative, complete protocol.
Field names below are quoted exactly as they appear on the wire.

## Framing

JSONL over stdin/stdout: one JSON object per line, `\n`-terminated. Split
records on `\n` only. An optional trailing `\r` may be stripped. Critically,
**do not** use a line reader that also splits on Unicode line separators
(`U+2028`, `U+2029`) — those are legal, unescaped characters inside a JSON
string, and Node's own `readline` module is explicitly *not*
protocol-compliant here for that reason. `PiAgent`'s reader thread splits on
raw bytes (`BufferedReader.readline()`, which only ever stops at `0x0A`)
before decoding, so this is a non-issue by construction.

## Commands (stdin)

Every command is `{"type": "<name>", ...}`, with an optional `"id"` used to
correlate the eventual `{"type": "response", ...}` acknowledgement.

**Never pipeline: write one command, wait for its `response`, then write
the next.** Measured against pi 0.85.1 on the Spark (nemotron/associate), a
second command line written before the first has been acknowledged loses
*both*: no `response`, no events, ever again — the process stays alive,
keeps reading stdin, and answers nothing. That is deviation d14, where the
session daemon wrote `new_session` and then, a millisecond later, `prompt`:
the operator's panel sat empty until the client's 120 s stream timeout gave
up and re-ran the request one-shot. With the ack waited for, 4/4 runs
answered in ~1.5 s; pipelined, 4/4 produced nothing at all. The ack costs
one round trip (~0.2 s cold). `PiAgent._command()` is the only way commands
are written and it always waits, bounded by `$NVSH_PI_ACK_TIMEOUT`
(default 20 s) — a bound that expires is an error naming pi's exit code and
the redacted tail of its stderr, never silence.

**Two writes are deliberately not ack-waited, and neither breaks the rule
above.** `abort` (`cancel()` must return inside a second) and the mid-turn
steer (`steer()`, deviation d16) are only ever written *during* a turn,
which is precisely when no command is outstanding — the `prompt` that
started the turn was acknowledged before the first event arrived. Nothing
is written ahead of an unacknowledged command, so neither can reproduce
d14. They cannot wait for an ack in any case: during a turn the event
consumer owns pi's stdout queue, and a second reader would race it for the
`response` line. A rejected steer is still surfaced, not swallowed —
`_map_event` turns a `response` for `prompt` with `success: false` into an
`error` event the operator reads.

`PiAgent` sends:

- `{"type": "get_state"}` — the handshake, sent once at `start()`. Its
  `response` proves pi booted *and* that its rpc loop is reading stdin; the
  `data.sessionFile` it carries is how nvsh learns which session file pi is
  writing.
- `{"type": "prompt", "message": "<text>"}` — start a turn. Acked with
  `{"type": "response", "command": "prompt", "success": true|false}`;
  `success: false` means the prompt was rejected outright (rare) — a
  rejected/failed turn after acceptance instead shows up in the event
  stream, not a second response.
- `{"type": "prompt", "message": "<text>", "streamingBehavior": "steer"}` —
  **steer the turn that is already running** (`PiAgent.steer()`, deviation
  d16). pi's own wording: during streaming a `prompt` *must* carry
  `streamingBehavior`, and `"steer"` queues the message to be delivered
  "after the current assistant turn finishes executing its tool calls,
  before the next LLM call" (`"followUp"` waits for the agent to stop
  instead). A bare `prompt` written mid-stream is rejected outright. pi
  also has a dedicated `{"type": "steer", "message": ...}` command with the
  same delivery rule; nvsh uses the `prompt` form so one command shape
  covers both the idle and the mid-turn case.
- `{"type": "abort"}` — cancel the current turn.
- `{"type": "new_session"}` / `{"type": "switch_session", "sessionPath": "<path>"}`
  — start fresh / resume a stored session (`PiAgent.new_session()` /
  `.switch_session()`, driven by the session daemon). **pi names its own
  session file** inside `--session-dir`; no rpc command chooses the name, so
  `new_session()` follows its ack with a `get_state` and returns the
  `sessionFile` for the daemon to remember and hand back to
  `switch_session()` later.
- `{"type": "extension_ui_response", "id": "<id>", ...}` — answer a pending
  dialog request (see below); sent by `PiAgent.respond_ui()`.

## Events (stdout)

Events have no `id` (except `bash_execution_update`, unused here). The ones
`PiAgent._map_event` understands:

- `message_update` with `"assistantMessageEvent": {"type": "text_delta", "delta": "<text>"}`
  → `EventKind.TEXT_DELTA`. (`text_start`/`text_end`/`toolcall_*` are
  currently ignored; nvsh streams assistant text and tool execution, not the
  raw tool-call argument deltas.)
- `message_update` with `"assistantMessageEvent": {"type": "thinking_delta", "delta": "<text>"}`
  → `EventKind.THINKING` (task t9). Mirrors `text_delta` exactly, except the
  delta is never folded into `PiAgent._said` — the next proposal's rationale
  is what the model *said*, not what it *thought* on the way there.
- `tool_execution_start` with `"toolName"`, `"args"` → `EventKind.TOOL_CALL`.
- `tool_execution_end` with `"toolName"`, `"result"` → `EventKind.TOOL_RESULT`.
- `extension_ui_request` with `"id"`, `"method"` (`select`/`confirm`/`input`/
  `editor` block for a response; `notify`/`setStatus`/`setWidget`/`setTitle`/
  `set_editor_text` are fire-and-forget) → `EventKind.PROPOSAL`. See
  [the approval envelope](#the-approval-envelope) for how the proposed
  command reaches `Proposal.command`.
- `agent_end` → `EventKind.DONE`. (`agent_settled` — the point after which
  no further automatic retry/compaction/queued continuation will run — is
  not currently consumed; `agent_end` is the simpler, earlier signal the
  spec asks for.)
- A malformed line (fails `json.loads`) → `EventKind.ERROR`.
- Per-turn lifecycle and progress bookkeeping — `agent_start`,
  `turn_start`/`turn_end`, `message_start`/`message_end`, `message_final`,
  `agent_settled`, `tool_execution_update`, `queue_update` — → no event at
  all. These only
  say the rpc loop is running; rendering them put `... agent_start`,
  `... turn_start`, `... message_start`, `... message_end` and three
  `... tool_execution_update` lines per tool call into the operator's panel
  (deviation d11), which is noise at a failing prompt. The panel already
  says which tool is running and when it finished.
  `queue_update` joined them in d16: every steer changes pi's steering
  queue, so a single steered instruction printed `... queue_update` twice.
- Anything else — `compaction_start`/`compaction_end`,
  `auto_retry_*`, `extension_error`, etc. — → `EventKind.STATUS` with `text`
  set to the raw `"type"` value, so an event type nvsh has not seen before
  is still never silently dropped.

## The approval envelope

`pi`'s `select` request carries only `{type, id, method, title, options,
timeout}` — there is no room for a custom `"command"` field. So the approval
extension (`nvsh/agent/pi_ext/approval.ts`) passes one machine-readable JSON
envelope as the `title`:

```json
{
  "type": "extension_ui_request",
  "id": "uuid-1",
  "method": "select",
  "title": "{\"nvsh\":\"approval\",\"v\":1,\"tool\":\"bash\",\"command\":\"type ls\",\"reason\":\"\"}",
  "options": ["once", "session", "user", "deny"]
}
```

`command` is the bash tool call's `command` argument **verbatim** — no
prompt text, no prefix, newlines and quoting preserved by JSON — and
`reason` is the model's stated reason when the tool schema carries one
(pi 0.84.2's bash tool takes only `{command, timeout}`, so it is usually
`""`). `PiAgent._proposal_fields` decodes the envelope into
`Proposal.command` / `Proposal.rationale`; nvsh renders its own panel around
that command.

The envelope must never contain human-facing prose. It used to: the
extension built the title as `"nvsh: run this command?\ncommand: <cmd>"` and
`PiAgent` fell back to reading the whole `title` as the command, so the
rendered panel text became `Proposal.command` — pressing Enter would have
run that string, and the panel rendered visibly nested (deviation d8).
`PiAgent` therefore only ever takes a command from a field that holds a
command: the envelope's `command`, or a structured top-level `"command"`
field. A dialog carrying neither yields an empty `Proposal.command`
(its `title`/`message` becomes the rationale), and `run_loop` executes
nothing for a blank command even if the operator approves it.

## What the extension needs in `pi`'s environment

The approval extension decides nothing itself: on **every** `tool_call` it
shells out to `nvsh approve check <command> --json` and relays the answer,
so a pattern the operator widens halfway through a turn applies to the rest
of it. That only works if the extension can find the same `nvsh`, and the
same approval store, the panel writes to. `PiAgent.child_env` therefore
makes those explicit in the environment it spawns `pi` with:

| Variable | Why |
|---|---|
| `NVSH_BIN` | the binary the extension spawns. Taken from the env (`nvsh setup` exports it), else `PATH`, else the console script beside `sys.executable` — which is where a `uv tool` install lives even when its `bin` directory is not on the daemon's `PATH`. |
| `XDG_CONFIG_HOME` | `approved.toml` — the persistent (`user`) store. Defaults to `$HOME/.config`. |
| `XDG_RUNTIME_DIR` | `session-approvals.toml` — the login session's store (d15). Filled in **only** when `/run/user/<uid>` exists; `nvsh.approvals.runtime_dir`'s last resort is a differently named temp directory, and inventing a value here would split the store in two. |
| `XDG_STATE_HOME` | `audit.jsonl`, which the extension appends to through `nvsh approve audit`. |

When `nvsh` cannot be run at all the extension **blocks and says so**
(`nvsh could not check this tool call -- could not run <path>: ...`). It
does not fall through to the dialog: deviation d21 was a daemon-spawned
`pi` whose `PATH` had no `nvsh`, so every `spawnSync` failed with `ENOENT`,
the empty stdout parsed as `"ask"`, and an already-approved pattern raised
the panel again on every single tool call — while `[u]` silently failed to
persist, because the `approve add` spawn failed the same way. An
infrastructure failure the operator's keypress cannot fix must never be
disguised as a question.

## Extension UI sub-protocol

`select`/`confirm`/`input`/`editor` block the agent until an
`extension_ui_response` with the matching `"id"` arrives:

- `{"type": "extension_ui_response", "id": "<id>", "value": "<text>"}` for
  `select`/`input`/`editor`.
- `{"type": "extension_ui_response", "id": "<id>", "confirmed": true|false}`
  for `confirm`.
- `{"type": "extension_ui_response", "id": "<id>", "cancelled": true}` to
  dismiss any of the above.

`PiAgent.respond_ui(request_id, **fields)` writes exactly this.

## The system brief: a launch flag, not a per-turn prompt

There is **no rpc command and no `prompt` field that sets a system prompt**
— `get_state`'s response reports `systemPrompt`, but nothing on the wire
changes it. The system prompt is a *launch* concern: pi's CLI takes
`--system-prompt <text>` (replace) and `--append-system-prompt <text>`
(append, repeatable), both listed in the shipped `docs/usage.md` and in
`pi --help` of the installed pi (0.85.x on this Spark), and both apply to
`--mode rpc` exactly as they do to the TUI.

So `PiAgent.build_argv()` passes nvsh's system brief — who the agent is,
the rules it works under, and the playbook for the detected platform, from
`nvsh.agent.prompt.build_system_prompt` — as one
`--append-system-prompt <brief>` argument, placed after `-e <extension>`
and before `--provider`/`--model`. Consequences worth knowing:

- The brief is sent **once per pi process**, not once per turn: its tokens
  are paid at launch and are carried by every turn of the session,
  including after `new_session` / `switch_session`.
- `PiAgent._prompt_for()` therefore sends only the facts block (the
  failure, the detected platform values with their sources, the captured
  output slice) — never the brief again.
- `--append-system-prompt` *appends* to pi's own coding-assistant prompt
  rather than replacing it, so pi's tool contract (the bash tool the
  approval extension gates) stays intact. A `--system-prompt` replacement
  would drop that, which is why nvsh does not use it.
- The brief is composed from `nvsh.platform.detect()` at argv-build time,
  because argv exists before any request does. Detection failing is not
  fatal: the brief falls back to its generic playbook.

**Extra fields on the response do not reach the extension.** pi's rpc mode
resolves a `select` to `r.value` alone (measured in pi 0.85.1,
`dist/modes/rpc/rpc-mode.js`: the `select` wrapper maps the response to
`"cancelled" in r ? undefined : r.value`), so the `"reason"` nvsh sends
alongside `{"value": "deny"}` when the operator steers is dropped before
`approval.ts` can turn it into a `block` reason. nvsh sends it regardless
— it costs nothing and a backend that does pass the whole response through
gets the operator's words for free — and delivers the same words to the
model where they *are* guaranteed to land: as a steering message on the
same conversation, written before the deny so the turn is provably still
streaming when it is queued.

## Effort → `--thinking`, and `extra_args`/`approval` (task t9)

`PiAgent.__init__` takes `effort: str | None`, `extra_args: list[str] |
None`, and `approval: str = "nvsh"` (the constructor convention shared
across every backend adapter in this wave; task t5 added the matching
`[agents.pi]` config keys of the same names).

- `effort` is passed **verbatim**, never validated (decision c24 — nvsh
  never second-guesses a harness-specific string), as `--thinking <effort>`
  in `build_argv()`, placed after `--provider`/`--model` and before
  `extra_args`. pi 0.85.1's `--help` documents `--thinking <level>` as
  accepting `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` — but
  `PiAgent` does not enforce that set; an invalid value is pi's rejection to
  report, not nvsh's to pre-empt. `effort=None` (the default, and
  `[agents.pi]`'s default when unset) omits the flag entirely.
- `extra_args`, when given, is appended to argv **verbatim and last** —
  after `--thinking` — so an operator can pass a flag `PiAgent` does not
  otherwise know about without waiting on a code change.
- `approval` only affects `PiAgent.capabilities().approval` (default
  `"nvsh"`); it does not change how `pi` itself is launched or how the
  approval extension behaves — the extension already always shells out to
  `nvsh approve check` regardless (see above). Overriding it to `"harness"`
  is for a caller that has separately arranged for `pi`'s own approval UI to
  own the gate instead (e.g. a different extension), which is out of scope
  for this backend as shipped.

`PiAgent.capabilities()` reports `thinking=True` (this backend can stream
`EventKind.THINKING`), `effort=True` (it honors `AgentRequest.target.effort`
via the config/constructor `effort` above — this is a stable ability
declaration, not a report of the value in effect for a given call), and
`path="rpc"` (this adapter is reached by prompting `pi`'s own `--mode rpc`
protocol described above, not a plain HTTP path or a wrapped subcommand).

## What `PiAgent` never sends

No API key ever appears in the argv or on the wire — `pi` reads its own
`models.json`. `PiAgent` passes only `--provider`/`--model` (from
`nvsh.config`'s `[agents.pi]`, defaulting to `nemotron`/`associate`).
