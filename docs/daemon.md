# The nvsh session daemon

`nvsh.daemon` is the per-user process that owns the warm agent processes and
one conversation per shell. `nvsh.client_transport` is the client the hook
uses to reach it. This file is the protocol reference; the *why* lives in
the spec (claims c43, c51) and in `docs/architecture.md`.

## Lifecycle

- **Never started at shell start.** The hook client starts the daemon lazily
  on the first qualifying failure (`send(..., autostart=True)`), which
  spawns `python -m nvsh.daemon --foreground` detached — but only when no
  daemon holds the lock (below).
- **An autostart waits for an answer, not for a file.** After spawning, the
  client sends `ping` on a fresh connection until the daemon *answers*, up
  to `$NVSH_DAEMON_START_TIMEOUT` seconds (default 10). A socket file that
  exists, or a `connect()` that succeeds, only proves something bound the
  path; a daemon still inside its imports has both.
- **One daemon per user, enforced by a lock.** The daemon takes a
  non-blocking `flock` on `daemon.lock` next to the socket and holds it for
  its whole life. A daemon that cannot take it exits immediately (exit 2)
  and touches nothing — in particular it never unlinks the live daemon's
  socket, which is how orphan daemons used to pile up against one socket.
  Only the daemon that bound the socket removes it at teardown.
- **Stopped by the last shell.** Bash's `EXIT` trap sends `unregister`; when
  the last registered shell leaves, the daemon closes its agents, unlinks
  the socket and exits.
- **Stopped when idle.** With no request for `--idle-timeout` seconds
  (default 900) the daemon shuts down the same way.
- **A daemon from another nvsh is replaced.** A daemon is long-lived (15
  minutes idle, longer in use), so an upgrade routinely leaves one running
  that predates the client now talking to it. Before every request the
  client asks the listening daemon its version; if it is not the client's
  own, the client stops it and the next step starts a fresh one. See
  "Version handshake" below.
- **A stop takes the harness with it.** Teardown — idle exit, last shell
  out, `nvsh daemon stop`, `nvsh uninstall` — closes every adapter, and
  every adapter closes through one escalation
  (`nvsh.agent._subprocess.escalate_close`): close the child's stdin, wait,
  `terminate()`, `kill()`. No harness child outlives the daemon.
- **Stale sockets are reaped.** Holding the lock, the daemon tries to
  connect to an existing socket: if the connect fails it unlinks it and
  binds; if it succeeds another daemon owns it and this one refuses to
  start (exit 2).
- **Never blocks a diagnosis.** If the daemon genuinely cannot serve the
  request, the client runs it **one-shot** in its own process via
  `nvsh.agent.registry.choose()` — always preceded by a `status` event
  saying why: `daemon did not start within 10s`, `daemon did not answer
  within 10s`, `could not start a daemon: …`, `daemon refused the
  connection`, or `daemon connection lost: …`.

## Socket and log

| What | Where | Mode |
| --- | --- | --- |
| Socket | `$XDG_RUNTIME_DIR/nvsh/daemon.sock` (else `<tmp>/nvsh-<uid>/daemon.sock`) | `0600`, directory `0700` |
| Lock | `daemon.lock` beside the socket | `0600` |
| Log | `$XDG_STATE_HOME/nvsh/daemon.log` (else the user's local state directory, `nvsh/daemon.log` under `.local/state`) | `0600` |

The daemon never writes to a terminal.

## Timeouts

Several different waits, deliberately kept apart — conflating the first two
is what made every cold session pay a timeout and answer one-shot:

| Wait | Default | Override |
| --- | --- | --- |
| `connect()` on an existing socket | 5 s | — |
| Autostart: daemon accepts *and answers* a `ping` | 10 s | `$NVSH_DAEMON_START_TIMEOUT` |
| One read while streaming a request's events | 120 s | `send(..., timeout=…)` |
| One agent turn, wall clock, before the daemon aborts it | 300 s | `$NVSH_TURN_TIMEOUT`, `Daemon(turn_timeout=…)` |
| A queued request re-announces that it is still waiting | every 15 s | — |

The stream timeout has to cover a **cold** backend's first token (pi loading
node, a model warming up), which is far longer than any connect. The turn
cap is longer again, and deliberately finite — see "Busy daemon" below.
Junk or non-positive values of `$NVSH_TURN_TIMEOUT` fall back to the
default rather than disabling the cap.

## Busy daemon

Requests are served **one at a time** (pi steers one-at-a-time, and one
shared process must never interleave two conversations), so a turn that
never ends is a turn that blocks every other shell. Deviation d12 is what
that looks like in the field: one shell's `/ask` turn became the active
conversation and stayed active for nineteen minutes, and four other shells
— including the operator's — each sat on the run lock until their *stream*
timeout fired and answered one-shot, printing `daemon connection lost:
timed out`. Three behaviors make that impossible:

- **A turn whose client is gone is aborted.** While a turn runs, the daemon
  watches the owning client's socket (with `MSG_PEEK`, so nothing is taken
  off the stream) from a separate thread — the thread that owns the turn is
  parked inside `agent.run()` and cannot notice. On EOF or reset it answers
  any approval dialog the turn is parked on with a cancel, calls the
  adapter's `cancel()`, leaves the conversation idle, and releases the
  agent to the next queued request.
- **A turn is capped in wall-clock time.** Past `turn_timeout` (default
  300 s) the turn is aborted the same way and its client receives an
  `error` event saying so and naming `$NVSH_TURN_TIMEOUT`.
- **Waiting is visible.** A request that has to wait receives a `status`
  event *before* it blocks — `waiting for the agent (busy with shell
  3422579)` — and another every 15 s, so a queued client keeps receiving
  bytes instead of hitting its stream timeout in silence. `nvsh daemon
  status --json` reports `active_turn` (`{shell, started, elapsed}`, or
  `null`) and `queued` (`[{shell, started, waiting}]`), and the text form
  prints both.

**Control messages never queue behind a turn.** `status`, `ping`,
`register`, `unregister`, `cancel`, `kill`, `busy_choice`, `kill_active`,
`undo`, `ui_response` and `stop` are
answered without taking the run lock, so `nvsh daemon status` answers
instantly on a busy daemon and an approval dialog can always be answered
while the turn that raised it is still open. (This already held during d12
— it is what let the wedge be diagnosed at all — and is now pinned by a
test.)

An abort is only as good as the adapter's `cancel()`: `NvshAgent.run()`
must respect a pending `cancel()` (the conformance suite checks this). A
harness that ignores it is handled by **`kill`**: the operator's second
Ctrl+C/Esc sends a `kill` control, and the daemon calls the slot's
`force_stop()` (kills the harness's process tree), drops the slot so the
next request builds a fresh agent, and lets the run lock go. `kill` acts
only on the calling shell's own turn; it waits up to 3 s and answers
`killed <shell>` or `stopping <shell>`.

**Busy prompt.** A request from the shell that owns the active turn — or
from any shell when the owner's pid no longer exists — receives a `busy`
event instead of the queue notice:

```json
{"kind": "busy", "text": "...", "args": {"owner": "3422579", "elapsed": 42.0,
 "steerable": true, "choices": ["steer", "replace", "exit"]}}
```

`steer` is listed only for adapters that override `steer()` (pi, codex). The
client answers on a separate connection with a `busy_choice` control
(`choice`: `steer` | `replace` | `exit`). `steer` injects the request into
the running turn (or queues it when the adapter refuses, d16); `replace`
force-stops the turn and runs this request on a fresh agent; `exit` leaves
the turn alone and ends with `done` carrying `args.busy_choice = "exit"`. A
client that never answers (one that predates the prompt) falls back to the
queue after 60 s. A daemon run on a non-default target does not register an
active turn, so it is not visible to `kill` or the busy prompt (plan risk
r6).

**`kill_active`** is for `nvsh doctor --apply`, which never owns the turn:
it force-stops the active turn when its owner pid is gone, or when the
request carries `confirmed: true` after the operator agreed to a prompt
naming the owning shell; otherwise it answers `refused`.

## Wire protocol

JSON lines over `AF_UNIX`, stdlib `socketserver.ThreadingUnixStreamServer`.
The client writes **one** request line, then reads event lines until a
`done` or `error` line; the connection closes after.

Request line:

```json
{
  "shell": "12345",
  "kind": "failure",
  "version": "0.9.2",
  "request": {"kind": "failure", "prompt": "", "command": "ls /nope",
              "exit_code": 2, "failure_id": "f1",
              "target": {"backend": "claude", "model": "opus",
                         "alias": "default"}},
  "context": {"platform": "dgx-spark", "output": "...", "cwd": "/srv",
              "shell_pid": 12345, "redaction_report": []}
}
```

`shell` is the hook's `$$`; it keys the conversation. `version` is the
client's `nvsh.__version__` (see "Version handshake"). `request` is encoded
by `nvsh.agent.base.request_to_dict` and decoded by `request_from_dict` —
one codec, shared with the client, with default-valued fields omitted. Both
`version` and `request.target` are optional: a line written by a client that
predates them parses exactly as before, with `target = None` and no
handshake. `kind` is one of:

| kind | meaning |
| --- | --- |
| `failure` / `slash` / `explicit` | run an agent request (the `request`/`context` objects are required) |
| `register` | this shell is alive |
| `unregister` | this shell exited; the last one stops the daemon |
| `cancel` | ask this shell's running turn to stop (first Ctrl+C/Esc) |
| `kill` | force-stop this shell's running turn and release the agent (second press) |
| `busy_choice` | answer a `busy` prompt: adds `choice` (`steer`/`replace`/`exit`) |
| `kill_active` | force-stop the active turn for `nvsh doctor --apply`: adds `confirmed` |
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

Each shell owns a `Conversation` (the session file its agent is writing,
under `$XDG_STATE_HOME/nvsh/pi-sessions/`, plus whether it is asleep).
**pi names that file, not nvsh**: there is no rpc command that chooses a
session path (see `docs/pi-rpc.md`), so the daemon reads back what pi
reports in `get_state` after `new_session` and remembers it for the later
`switch_session`. Only a backend that reports no session file falls back to
this daemon's own per-shell key, `shell-<id>.jsonl`. A session file the
backend refuses to resume is logged as an error and replaced with a fresh
session rather than failing the request. `sessions.max` (config
`[sessions] max`, default `1`) bounds how many agent **processes** may
exist:

- `max = 1` — one process serves everyone. A request from another shell puts
  the active conversation to sleep and resumes the caller's
  (`switch_session`, or `new_session` the first time), so `/ask` from
  terminal A answers from A's conversation only.
- `max = 2` — a second process is started instead of switching; beyond
  `max`, the least recently used process is re-targeted.

Requests are served one at a time (pi steers one-at-a-time, and one shared
process must never interleave two conversations) — see "Busy daemon" above
for what happens to the requests that have to wait.

An adapter without `new_session`/`switch_session` (the stateless
`openai-compat` client, for instance) is simply used as-is; nothing is
swapped.

Both calls are **synchronous against the backend**: `PiAgent` waits for pi's
acknowledgement before returning, so the prompt the daemon sends next cannot
overtake the session command. It used to, and pi 0.85.1 then answered
neither — the turn produced no event at all until the client's 120 s stream
timeout gave up (deviation d14; see `docs/pi-rpc.md`).

## When the backend breaks

Nothing about a broken backend is silent:

- an agent that will not start, a session command that is never
  acknowledged, or an adapter that raises mid-turn becomes an `error` event
  on the client's stream **and** an `ERROR` line (with traceback) in
  `daemon.log`;
- a pi subprocess that exits or never handshakes within its bound is
  reported with its exit code and the redacted tail of its stderr, e.g.
  `no agent available: pi closed its output before acknowledging get_state
  (pi exited with code 3; stderr tail: pi: cannot find module foo)`;
- the agent process behind a failed start or a failed turn is retired
  (closed and dropped), so the next request builds a fresh one instead of
  talking to a backend in an unknown state.

## Targets: what the warm session is, and what is not

The warm session belongs to **one** target: whatever `default` resolves to.
`Daemon.default_target()` asks `Config.resolve_target("default")`, so
`[aliases] default` decides, falling back to the legacy `[agent] provider`
when there is no such alias. `Daemon._build_agent()` then folds that
target's `model` and `effort` into a copy of the config's
`[agents.<backend>]` table and hands it to the adapter's factory — which is
how the resolved model and effort reach the adapter alongside that table's
own `extra_args` and `approval`, without the daemon knowing which keyword
each adapter takes. `nvsh daemon status --json` reports the result in
`target` (`{backend, model, effort, alias}`).

Anything else — `@claude/opus`, `/ask --agent fast`, any `Target` whose
`alias` is not `default` — runs **one-shot** (decision c25). The client
makes that call itself in `client_transport.send`, because the warm session
holds a conversation with the default backend and answering someone's
`@claude/opus` out of it would quietly answer from the wrong model. A
targeted request that reaches the daemon anyway (an older client, a direct
API caller) is served the same way *inside* the daemon: a fresh adapter,
built `forced` so a missing binary is a loud error rather than a silent
swap for `openai-compat`, and closed when the turn ends. The warm slot is
never touched.

Two consecutive failures from one shell therefore reuse one warm adapter:
`_acquire` returns the slot already serving that shell without rebuilding
or re-activating it. For the ACP and app-server harnesses that is literally
one child process across both turns; for the stream-json harnesses
(`claude`, `agy`, `qwen-p`) the *process* is per turn by the CLI's own
design, and what the warm adapter carries across is the session it resumes.

## Version handshake

A daemon outlives the code that started it: 15 minutes idle by default,
indefinitely while a shell keeps failing. An `nvsh` upgrade (or a `uv sync`
in a checkout) therefore routinely leaves one running that predates the
client now talking to it — different adapters, different wire fields,
different protocol. Rather than guess which differences are survivable,
both ends declare a version:

- every client message carries `version` (`nvsh.__version__`);
- the daemon reports its own in `status`/`ping` (`state()["version"]`);
- before each request the client reads that (`client_transport.daemon_version`)
  and, when it differs, stops the daemon and lets the next step start a
  fresh one — announcing it as a `status` event, `restarted the daemon: it
  was running nvsh 0.9.1, this client is 0.9.2`;
- if one slips through anyway (a daemon that came up between the check and
  the connect), the daemon answers an agent request with a single `error`
  whose `args` carry `{"version_mismatch": true, "daemon_version": …,
  "client_version": …}`; the client restarts it and retries **once**, then
  answers one-shot rather than loop.

Control messages (`status`, `ping`, `stop`, `register`, `unregister`,
`cancel`, `undo`, `ui_response`, `steer`) are never refused on a version
mismatch — `stop` is exactly what the client needs next. A message with no
`version` at all is not a mismatch: a pre-handshake client gets an answer,
not an error it has no code to read.

## Backends and fallback

With no injected `agent_factory` the daemon picks a backend through
`registry.choose(config)`. When that is not the target's backend, the first
run for each shell is preceded by a `status` event —
`pi unavailable: <reason>` — and `nvsh daemon status` reports the same in
`fallback_notice`.

## CLI

```bash
nvsh daemon status --json     # {running, pid, socket, version, target, backend, ...}
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
    autostart=True,           # start a daemon if none holds the lock
    timeout=120.0,            # per-read stream timeout, not the connect wait
):
    render(event)
```

Also `register()`, `unregister()`, `cancel()`, `respond_ui()`, `stop()`,
`status()` — all of which return `False`/`{"running": false}` instead of
raising when no daemon is listening.
