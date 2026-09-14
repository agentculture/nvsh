"""AgyAgent: subprocess adapter over ``agy -p ... --output-format stream-json``.

``agy`` (https://github.com/antigravity-agy/agy -- "jetski" internally, per
its own error messages) is a hosted, tool-using coding agent CLI. Like
``codex``/``qwen``/``claude`` it has no ``--system-prompt`` flag (checked
against the installed 1.2.2 CLI's ``--help``), so the system brief rides at
the head of the prompt text via :func:`nvsh.agent.prompt.build_full_prompt`,
exactly like :class:`nvsh.agent.codex.CodexAgent`.

Two invocation shapes, both requested by the spec:

* **Cold** (default): one ``agy -p <prompt> --output-format stream-json
  --model <m> --effort <e>`` subprocess per turn. Cheap, stateless from
  nvsh's point of view; a conversation can still be resumed across cold
  calls by passing ``--conversation <id>``.
* **Warm** (``warm=True``): one persistent ``agy -p= --output-format
  stream-json --input-format stream-json ...`` subprocess for the whole
  adapter lifetime. Each ``run()`` writes one NDJSON line to its stdin
  (``{"event": "user", "message": {"role": "user", "content": <prompt>}}``)
  and reads the reply off a background-drained queue, exactly the way pi's
  rpc reader works (see ``nvsh/agent/pi.py``) -- a fresh process per line
  would lose the warm session's whole point.

Wire shapes (verified live against the installed ``agy`` 1.2.2, 2026-09-14,
via ``agy -p '<prompt>' --output-format stream-json``)::

    {"event": "init", "conversation_id": "...", "init": {...}}
    {"event": "step_update", "step_update": {
        "step_type": "user_input" | "agent_response" | "tool",
        "state": "ACTIVE" | "DONE" | "ERROR",
        "text_delta": "...",             # agent_response only
        "tool_name": "run_command",      # tool only
        "tool_info": {"name": ..., "parameters": {...}, "output": "..."},
    }}
    {"event": "result", "result": {
        "status": "SUCCESS" | "ERROR", "response": "...", "error": "...",
        "denied_actions": [{"action": "command", ...}],   # optional
    }}

Headless auto-deny: a tool that needs the ``"command"`` permission cannot be
approved interactively in headless/print mode, so agy auto-denies it and
prints one line to stderr (``jetski: no output produced -- a tool required
the "command" permission that headless mode cannot prompt for, so it was
auto-denied...``). That is *not* an adapter failure -- the turn can still
finish successfully (``result.status`` was observed to still be
``"SUCCESS"``, just with an empty ``response`` and a populated
``denied_actions``) -- so it is surfaced as a :attr:`EventKind.STATUS`
event, never :attr:`EventKind.ERROR`. ``--dangerously-skip-permissions``
(the CLI's own escape hatch out of this) is never passed: nvsh always
proposes, never auto-applies (see the repo's ``CLAUDE.md``).
"""

from __future__ import annotations

import json
import queue
import subprocess  # nosec B404 - fixed argv lists below, no shell=True
import threading
from collections import deque
from typing import Iterator, Mapping

from ._env import child_env
from ._subprocess import escalate_close, redacted_tail
from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind, NvshAgent
from .prompt import build_full_prompt

#: How many stderr lines are kept for the tail used in an ERROR/STATUS
#: message. Same bound as ``_subprocess.SubprocessAgent``.
_STDERR_TAIL_LINES = 200

#: How long the drain thread is given to finish once a turn's "result" line
#: has been seen -- long enough for the concurrent stderr line agy prints
#: right before exiting to land, short enough not to stall a clean turn.
_STDERR_SETTLE_SECONDS = 0.5

#: How long close()/cancel() wait for a clean subprocess exit before kill().
_TERMINATE_TIMEOUT_SECONDS = 2.0

#: How often a warm turn's reader loop re-checks process liveness/cancel.
_QUEUE_POLL_SECONDS = 0.2

#: Substring of agy's headless auto-deny stderr line (see module docstring).
#: Matched loosely -- the "jetski:" program name and exact wording are
#: implementation details of the CLI, not part of the contract.
_AUTO_DENY_MARKER = "auto-denied"


def _safe_json_line(line: str) -> dict | None:
    """Parse one NDJSON line, or ``None`` for blank/unparsable input."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _drain_lines(stream, sink: deque) -> None:
    """Read ``stream`` to EOF into ``sink`` (bounded). Never raises."""
    if stream is None:
        return
    try:
        for line in stream:
            sink.append(line)
    except (OSError, ValueError):  # closed underneath us by cancel/terminate
        pass


class AgyAgent(NvshAgent):
    """Drives the ``agy`` CLI, cold (default) or warm (``warm=True``)."""

    binary_default = "agy"

    def __init__(
        self,
        binary: str = "agy",
        model: str | None = None,
        effort: str | None = None,
        extra_args: list[str] | None = None,
        approval: str = "nvsh",
        *,
        warm: bool = False,
        conversation_id: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.binary = binary
        self._model = model
        self._effort = effort
        self._extra_args = list(extra_args or [])
        #: Who nvsh's own loop treats as the approval gate for this backend.
        #: Not sent to agy in any form -- agy's tools are unmediated (see
        #: capabilities()) -- kept purely so every backend's constructor
        #: shares the same shape (t8/t9/t14 agreement).
        self._approval = approval
        self._warm = warm
        self._conversation_id = conversation_id
        self._env = env

        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stdout_queue: "queue.Queue[dict | None]" = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._stderr_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._cancelled = False
        self._closed = False

    # -- argv ---------------------------------------------------------

    def _shared_flags(self) -> list[str]:
        """Flags common to every invocation, cold or warm."""
        flags = ["--output-format", "stream-json"]
        if self._model:
            flags += ["--model", self._model]
        if self._effort:
            flags += ["--effort", self._effort]
        if self._conversation_id:
            flags += ["--conversation", self._conversation_id]
        flags += self._extra_args
        return flags

    def _cold_argv(self, prompt: str) -> list[str]:
        # ``-p`` swallows exactly the next token as its prompt, so the
        # prompt must sit immediately after it -- putting the rest of the
        # flags afterwards (rather than between ``-p`` and the prompt) is
        # what keeps that true regardless of how many flags there are.
        return [self.binary, "-p", prompt] + self._shared_flags()

    def _warm_argv(self) -> list[str]:
        # An empty ``-p=`` keeps one process open reading turns from stdin
        # instead of running a single prompt and exiting.
        return [self.binary, "-p="] + self._shared_flags() + ["--input-format", "stream-json"]

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        self._cancelled = False
        self._closed = False
        if self._warm and (self._proc is None or self._proc.poll() is not None):
            self._spawn_warm()

    def _spawn_warm(self) -> None:
        argv = self._warm_argv()
        self._proc = subprocess.Popen(  # nosec B603 - argv is a fixed list, no shell
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_env(self._env),
        )
        self._stdout_queue = queue.Queue()
        self._stderr_tail.clear()
        self._reader = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader.start()
        self._stderr_thread = threading.Thread(
            target=_drain_lines, args=(self._proc.stderr, self._stderr_tail), daemon=True
        )
        self._stderr_thread.start()

    def _drain_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for raw_line in self._proc.stdout:
                obj = _safe_json_line(raw_line)
                if obj is not None:
                    self._stdout_queue.put(obj)
        except (OSError, ValueError):
            pass
        finally:
            self._stdout_queue.put(None)  # sentinel: stdout closed

    # -- run --------------------------------------------------------------

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        prompt = build_full_prompt(request, context)
        if self._warm:
            yield from self._run_warm(prompt)
        else:
            yield from self._run_cold(prompt)

    def _run_cold(self, prompt: str) -> Iterator[AgentEvent]:
        argv = self._cold_argv(prompt)
        try:
            proc = subprocess.Popen(  # nosec B603 - argv is a fixed list, no shell
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_env(self._env),
            )
        except OSError as exc:
            yield AgentEvent(kind=EventKind.ERROR, error=f"failed to start {argv[0]}: {exc}")
            return

        self._proc = proc
        tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        drain = threading.Thread(target=_drain_lines, args=(proc.stderr, tail), daemon=True)
        drain.start()
        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                if self._cancelled:
                    return
                obj = _safe_json_line(raw_line)
                if obj is None:
                    continue
                for event in self._handle_object(obj, tail, drain):
                    yield event
                if obj.get("event") == "result":
                    return
            if self._cancelled:
                return
            # stdout closed with no "result" line at all -- agy's own
            # description of the headless auto-deny case ("no output
            # produced"). Whatever is in stderr decides STATUS vs ERROR.
            proc.wait()
            drain.join(timeout=_TERMINATE_TIMEOUT_SECONDS)
            yield from self._final_status(proc.returncode, tail)
        finally:
            self._terminate(proc)
            drain.join(timeout=_TERMINATE_TIMEOUT_SECONDS)

    def _run_warm(self, prompt: str) -> Iterator[AgentEvent]:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn_warm()
        assert self._proc is not None and self._proc.stdin is not None
        payload = json.dumps({"event": "user", "message": {"role": "user", "content": prompt}})
        with self._write_lock:
            try:
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                yield AgentEvent(kind=EventKind.ERROR, error=f"agy stdin closed: {exc}")
                return

        while True:
            if self._cancelled:
                return
            try:
                obj = self._stdout_queue.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                if self._proc.poll() is not None:
                    yield from self._final_status(self._proc.returncode or 0, self._stderr_tail)
                    return
                continue
            if obj is None:
                # stdout hit EOF (process exited) without a "result" line.
                rc = self._proc.wait()
                yield from self._final_status(rc, self._stderr_tail)
                return
            for event in self._handle_object(obj, self._stderr_tail, self._stderr_thread):
                yield event
            if obj.get("event") == "result":
                return

    # -- event mapping ------------------------------------------------

    def _handle_object(
        self, obj: dict, tail: deque[str], drain: threading.Thread | None
    ) -> Iterator[AgentEvent]:
        """Map one decoded NDJSON object to zero or more ``AgentEvent``s."""
        kind = obj.get("event")
        if kind == "init":
            conversation_id = (obj.get("init") or {}).get("conversation_id") or obj.get(
                "conversation_id"
            )
            if conversation_id:
                self._conversation_id = str(conversation_id)
            return
        if kind == "step_update":
            event = self._map_step_update(obj.get("step_update") or {})
            if event is not None:
                yield event
            return
        if kind == "result":
            result = obj.get("result") or {}
            conversation_id = result.get("conversation_id")
            if conversation_id:
                self._conversation_id = str(conversation_id)
            # Give the concurrent stderr drain a beat to catch a line agy
            # writes right as it finishes the turn (the auto-deny notice),
            # so it is visible for this same turn instead of the next one.
            if drain is not None:
                drain.join(timeout=_STDERR_SETTLE_SECONDS)
            deny_text = redacted_tail(tail)
            if _AUTO_DENY_MARKER in deny_text:
                yield AgentEvent(kind=EventKind.STATUS, text=deny_text)
            if result.get("status") == "SUCCESS":
                yield AgentEvent(kind=EventKind.DONE, text=str(result.get("response", "")))
            else:
                yield AgentEvent(
                    kind=EventKind.ERROR,
                    error=str(result.get("error") or result.get("response") or "agy error"),
                )
            return
        # Unrecognized event types (a newer agy) are silently ignored rather
        # than surfaced as noise -- there is no catch-all STATUS mapping
        # here the way pi.py has one, because agy's ``event`` vocabulary is
        # small and closed per the verified transcripts.

    def _map_step_update(self, step_update: dict) -> AgentEvent | None:
        step_type = step_update.get("step_type")
        state = step_update.get("state")
        if step_type == "tool":
            tool_info = step_update.get("tool_info") or {}
            tool_name = str(step_update.get("tool_name") or tool_info.get("name") or "")
            if state == "DONE":
                return AgentEvent(
                    kind=EventKind.TOOL_RESULT, tool=tool_name, result=tool_info.get("output")
                )
            if state == "ERROR":
                error = tool_info.get("error") or {}
                message = error.get("message") if isinstance(error, dict) else str(error)
                return AgentEvent(kind=EventKind.TOOL_RESULT, tool=tool_name, result=message)
            return AgentEvent(
                kind=EventKind.TOOL_CALL,
                tool=tool_name,
                args=dict(tool_info.get("parameters") or {}),
            )
        if step_type == "agent_response":
            delta = step_update.get("text_delta")
            if delta:
                return AgentEvent(kind=EventKind.TEXT_DELTA, text=delta)
            return None
        # "user_input" (agy's own echo of the prompt it just received) and
        # any other step type carry nothing the operator needs to see.
        return None

    def _final_status(self, rc: int, tail: deque[str]) -> Iterator[AgentEvent]:
        """Terminal event(s) when stdout closed without a "result" line."""
        text = redacted_tail(tail)
        if _AUTO_DENY_MARKER in text:
            yield AgentEvent(kind=EventKind.STATUS, text=text)
            yield AgentEvent(kind=EventKind.DONE)
            return
        if rc == 0:
            yield AgentEvent(kind=EventKind.DONE)
            return
        yield AgentEvent(kind=EventKind.ERROR, error=text or f"{self.binary} exited {rc}")

    # -- cancel / close -----------------------------------------------

    def steer(self, text: str) -> bool:
        return False

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc is not None and self._proc.poll() is None and not self._warm:
            self._terminate(self._proc)

    def _terminate(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()

    def close(self) -> None:
        # Shared escalation (stdin, wait, terminate, kill) -- deviation d5.
        # ``cancel`` keeps using ``_terminate`` on purpose: it ends one cold
        # turn's child, it does not retire the adapter.
        if self._closed:
            return
        escalate_close(self._proc, wait=_TERMINATE_TIMEOUT_SECONDS)
        self._closed = True

    # -- capabilities ---------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_calling=False,
            cancellation=True,
            persistent_session=self._warm,
            local_model=False,
            thinking=False,
            effort=True,
            path=self.binary,
            approval="none",
            unmediated_file_access=True,
        )
