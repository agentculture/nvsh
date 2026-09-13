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
  `set_editor_text` are fire-and-forget) → `EventKind.PROPOSAL`. The
  approval extension (`nvsh/agent/pi_ext/approval.ts`, written by task t11)
  is expected to carry the proposed command in a `"command"` field on the
  request; `PiAgent` falls back to `"message"` then `"title"` if that field
  is absent, since the exact shape isn't fixed until t11 lands.
- `agent_end` → `EventKind.DONE`. (`agent_settled` — the point after which
  no further automatic retry/compaction/queued continuation will run — is
  not currently consumed; `agent_end` is the simpler, earlier signal the
  spec asks for.)
- A malformed line (fails `json.loads`) → `EventKind.ERROR`.
- Anything else — `queue_update`, `turn_start`/`turn_end`,
  `compaction_start`/`compaction_end`, `auto_retry_*`, `extension_error`,
  etc. — → `EventKind.STATUS` with `text` set to the raw `"type"` value, so
  nothing is silently dropped.

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
