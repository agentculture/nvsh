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
`PiAgent` sends:

- `{"type": "prompt", "message": "<text>"}` — start a turn. Acked with
  `{"type": "response", "command": "prompt", "success": true|false}`;
  `success: false` means the prompt was rejected outright (rare) — a
  rejected/failed turn after acceptance instead shows up in the event
  stream, not a second response.
- `{"type": "abort"}` — cancel the current turn.
- `{"type": "new_session"}` / `{"type": "switch_session", "sessionPath": "<path>"}`
  — start fresh / resume a stored session (`PiAgent.new_session()` /
  `.switch_session()`, pass-throughs the session daemon (t12) will drive).
- `{"type": "extension_ui_response", "id": "<id>", ...}` — answer a pending
  dialog request (see below); sent by `PiAgent.respond_ui()`.

## Events (stdout)

Events have no `id` (except `bash_execution_update`, unused here). The ones
`PiAgent._map_event` understands:

- `message_update` with `"assistantMessageEvent": {"type": "text_delta", "delta": "<text>"}`
  → `EventKind.TEXT_DELTA`. (Other `assistantMessageEvent` sub-types —
  `text_start`/`text_end`, `thinking_*`, `toolcall_*` — are currently
  ignored; nvsh streams assistant text and tool execution, not the raw
  tool-call argument deltas.)
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
  `agent_settled`, `tool_execution_update` — → no event at all. These only
  say the rpc loop is running; rendering them put `... agent_start`,
  `... turn_start`, `... message_start`, `... message_end` and three
  `... tool_execution_update` lines per tool call into the operator's panel
  (deviation d11), which is noise at a failing prompt. The panel already
  says which tool is running and when it finished.
- Anything else — `queue_update`, `compaction_start`/`compaction_end`,
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

## What `PiAgent` never sends

No API key ever appears in the argv or on the wire — `pi` reads its own
`models.json`. `PiAgent` passes only `--provider`/`--model` (from
`nvsh.config`'s `[agents.pi]`, defaulting to `nemotron`/`associate`).
