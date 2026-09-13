# The nvsh session daemon

`nvsh.daemon` is the per-user process that owns the warm agent processes and
one conversation per shell. `nvsh.client_transport` is the client the hook
uses to reach it. This file is the protocol reference; the *why* lives in
the spec (claims c43, c51) and in `docs/architecture.md`.

## Lifecycle

- **Never started at shell start.** The hook client starts the daemon lazily
  on the first qualifying failure (`send(..., autostart=True)`), which
  spawns `python -m nvsh.daemon --foreground` detached.
- **Stopped by the last shell.** Bash's `EXIT` trap sends `unregister`; when
  the last registered shell leaves, the daemon closes its agents, unlinks
  the socket and exits.
- **Stopped when idle.** With no request for `--idle-timeout` seconds
  (default 900) the daemon shuts down the same way.
- **Stale sockets are reaped.** At start the daemon tries to connect to an
  existing socket: if the connect fails it unlinks it and binds; if it
  succeeds another daemon owns it and this one refuses to start (exit 2).
- **Never blocks a diagnosis.** If the socket is missing, the connect fails
  or the stream breaks, the client runs the request **one-shot** in its own
  process via `nvsh.agent.registry.choose()`.

## Socket and log

| What | Where | Mode |
| --- | --- | --- |
| Socket | `$XDG_RUNTIME_DIR/nvsh/daemon.sock` (else `<tmp>/nvsh-<uid>/daemon.sock`) | `0600`, directory `0700` |
| Log | `$XDG_STATE_HOME/nvsh/daemon.log` (else the user's local state directory, `nvsh/daemon.log` under `.local/state`) | `0600` |

The daemon never writes to a terminal.

## Wire protocol

JSON lines over `AF_UNIX`, stdlib `socketserver.ThreadingUnixStreamServer`.
The client writes **one** request line, then reads event lines until a
`done` or `error` line; the connection closes after.

Request line:

```json
{
  "shell": "12345",
  "kind": "failure",
  "request": {"kind": "failure", "prompt": "", "command": "ls /nope",
              "exit_code": 2, "failure_id": "f1"},
  "context": {"platform": "dgx-spark", "output": "...", "cwd": "/srv",
              "shell_pid": 12345, "redaction_report": []}
}
```

`shell` is the hook's `$$`; it keys the conversation. `kind` is one of:

| kind | meaning |
| --- | --- |
| `failure` / `slash` / `explicit` | run an agent request (the `request`/`context` objects are required) |
| `register` | this shell is alive |
| `unregister` | this shell exited; the last one stops the daemon |
| `cancel` | abort what this shell is streaming (Ctrl+C) |
| `ui_response` | answer a proposal dialog: adds `request_id` and `fields` |
| `status` / `ping` | one `status` event whose `text` is the state JSON |
| `stop` | stop the daemon |

Response lines mirror `AgentEvent` (`nvsh.agent.base.event_to_dict`), with
default-valued fields omitted:

```json
{"kind": "status", "text": "pi unavailable: pi not on PATH and node missing"}
{"kind": "text_delta", "text": "the path does not exist"}
{"kind": "proposal", "proposal": {"command": "ls /", "rationale": "look",
                                  "kind": "inspect"},
 "args": {"request_id": "ui-1"}}
{"kind": "done"}
```

An unknown `kind` decodes to `status` rather than raising, so a newer daemon
never breaks an older client.

## Conversations and `sessions.max`

Each shell owns a `Conversation` (its pi session path under
`$XDG_STATE_HOME/nvsh/pi-sessions/shell-<id>.jsonl`, plus whether it is
asleep). `sessions.max` (config `[sessions] max`, default `1`) bounds how
many agent **processes** may exist:

- `max = 1` — one process serves everyone. A request from another shell puts
  the active conversation to sleep and resumes the caller's
  (`switch_session`, or `new_session` the first time), so `/ask` from
  terminal A answers from A's conversation only.
- `max = 2` — a second process is started instead of switching; beyond
  `max`, the least recently used process is re-targeted.

Requests are served one at a time (pi steers one-at-a-time, and one shared
process must never interleave two conversations).

An adapter without `new_session`/`switch_session` (the stateless
`openai-compat` client, for instance) is simply used as-is; nothing is
swapped.

## Backends and fallback

With no injected `agent_factory` the daemon picks a backend through
`registry.choose(config)`. When that is not the configured provider, the
first run for each shell is preceded by a `status` event —
`pi unavailable: <reason>` — and `nvsh daemon status` reports the same in
`fallback_notice`.

## CLI

```bash
nvsh daemon status --json     # {running, pid, socket, shells, agents, backend, ...}
nvsh daemon run --foreground  # serve here (what the tests and supervisors use)
nvsh daemon run               # spawn detached, report the pid
nvsh daemon stop              # idempotent
nvsh daemon unregister --shell $$   # the EXIT trap's call
```

## Python API

```python
from nvsh import client_transport
from nvsh.agent.base import AgentContext, AgentRequest, RequestKind

for event in client_transport.send(
    AgentRequest(kind=RequestKind.FAILURE, command="ls /nope", exit_code=2),
    AgentContext(platform="dgx-spark", output=slice_text, cwd=cwd),
    shell_id=shell_pid,          # defaults to os.getpid()
    env=None,                 # defaults to os.environ
    config=None,              # defaults to nvsh.config.load()
    autostart=True,           # start a daemon if none is listening
):
    render(event)
```

Also `register()`, `unregister()`, `cancel()`, `respond_ui()`, `stop()`,
`status()` — all of which return `False`/`{"running": false}` instead of
raising when no daemon is listening.
