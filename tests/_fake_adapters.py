"""Every adapter, wired to its fake CLI, plus the cross-adapter case table.

Two layers live here:

**ViaFake factories.** ``<Name>AgentViaFake(script)`` takes a list of
``AgentEvent`` (the conformance fixture contract in
``tests/test_agent_conformance.py``) and returns a real adapter instance
whose subprocess is one of the fakes under ``tests/fakes/``. The adapter is
the real one -- its actual argv building, line parsing and cancellation code
runs -- only the CLI on the other end is fake, so a conformance case proves
the adapter, not a mock of it.

**The case table (:data:`CASES`).** One :class:`AdapterCase` per backend
nvsh ships: ``pi``, ``agy``, ``claude``, ``codex``, ``qwen`` (ACP),
``qwen-p`` (the print-mode fallback) and ``kiro`` (ACP). Each case knows how
to drive its fake's *richest* turn (thinking, a tool call, a permission
request), how to read back everything the child was told when an effort is
configured, and how to point the same adapter at a CLI that rejects that
effort on stderr. ``tests/test_agent_conformance.py`` parametrises over the
table; nothing else needs to know which fake belongs to which adapter.

Two honest limits, both visible in :attr:`AdapterCase.scriptable`:

* Not every fake's wire format can express every event kind. ``agy``'s
  recorded NDJSON vocabulary has no thinking line and no permission request
  at all, and ``tests/fakes/qwen`` serialises only ``status`` /
  ``text_delta`` / ``error`` / ``done``. Where a fake cannot express a case,
  the conformance suite asserts the adapter's *declared* capability instead,
  and its assertion message says so.
* Where an effort string travels in-session rather than on the command line
  (ACP's ``session/set_config_option``), :attr:`AdapterCase.effort_words`
  returns the recorded wire frames alongside argv, so "verbatim" (decision
  c24) is checked against what the child actually received either way.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pytest

from nvsh.agent.acp import build as acp_build
from nvsh.agent.agy import AgyAgent
from nvsh.agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    EventKind,
    NvshAgent,
    Proposal,
    ProposalKind,
    RequestKind,
)
from nvsh.agent.claude import ClaudeAgent
from nvsh.agent.codex import CodexAgent
from nvsh.agent.pi import PiAgent
from nvsh.agent.qwen import QwenAgent

FAKES_DIR = Path(__file__).parent / "fakes"

#: A deliberately meaningless effort string. Decision c24: nvsh never
#: validates or rewrites an effort -- it hands the operator's bytes to the
#: harness and lets the harness complain. A value no real CLI would accept
#: is therefore exactly the right probe for "passed through verbatim".
EFFORT_PROBE = "conformance-effort-42"

#: What a CLI that rejects :data:`EFFORT_PROBE` writes to stderr. The
#: rejecting shim (:func:`rejecting_bin`) prints it and exits 2; the adapter
#: must surface it in an ``ERROR`` event rather than swallowing it.
REJECTED_EFFORT_TAIL = "unsupported reasoning effort conformance-effort-42"


def conformance_request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, command="ls /nope", exit_code=2)


def conformance_context() -> AgentContext:
    return AgentContext(platform="dgx-spark", output="ls: /nope: No such file", cwd="/tmp")


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


def kill_fake_pids(pid_file: Path) -> None:
    """Test teardown: SIGKILL the harness/grandchild a fake recorded in *pid_file*.

    Only a pid whose command line still looks like the fake's (a ``sleep 600``
    grandchild, or a process running a script from ``tests/fakes``) is
    signalled, so a pid the kernel already handed to somebody else is left
    alone. Never raises.
    """
    try:
        pids = json.loads(Path(pid_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for role in ("grandchild", "harness"):
        pid = pids.get(role) if isinstance(pids, dict) else None
        if not isinstance(pid, int) or pid <= 1:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                argv = handle.read().split(b"\0")
        except OSError:
            continue
        ours = (
            argv[:2] == [b"sleep", b"600"]
            if role == "grandchild"
            else any(str(FAKES_DIR).encode() in arg for arg in argv)
        )
        if ours:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


@pytest.fixture
def reap_fake_pids(request):
    """Teardown: kill whatever a fake recorded in ``<tmp_path>/pids.json``.

    ``NVSH_FAKE_GRANDCHILD=1`` fakes start a ``sleep 600``; a test that fails
    (or a harness that respawned and overwrote the pid file) must not leave
    it running for ten minutes after the suite ends.
    """
    yield
    tmp_path = request.node.funcargs.get("tmp_path")
    if tmp_path is not None:
        kill_fake_pids(Path(tmp_path) / "pids.json")


def fake_env(**extra: str) -> dict[str, str]:
    """The process environment with ``tests/fakes`` first on ``PATH``."""
    env = dict(os.environ)
    env["PATH"] = str(FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env.update(extra)
    return env


def _tmp_home() -> dict[str, str]:
    """A throwaway ``HOME``/``XDG_STATE_HOME`` for adapters that use one."""
    home = tempfile.mkdtemp(prefix="nvsh-conformance-home-")
    return {"HOME": home, "XDG_STATE_HOME": os.path.join(home, "state")}


_ACP_BIN_DIR: str | None = None


def acp_bin_dir() -> str:
    """A bin dir where ``qwen`` and ``kiro-cli`` *are* the scripted ACP fake.

    ``nvsh.agent.acp.build()`` owns each harness's argv (``qwen --acp``,
    ``kiro-cli acp``) and offers no binary override -- deliberately, since
    that argv is the contract. Symlinking the two real binary names at
    ``tests/fakes/acp`` is therefore how a test drives ``build()`` itself
    rather than reconstructing an ``AcpAgent`` by hand and losing the
    mode/approval logic ``build()`` applies (decision c53).
    """
    global _ACP_BIN_DIR
    if _ACP_BIN_DIR is None:
        bindir = Path(tempfile.mkdtemp(prefix="nvsh-acp-bin-"))
        for name in ("qwen", "kiro-cli"):
            (bindir / name).symlink_to(FAKES_DIR / "acp")
        _ACP_BIN_DIR = str(bindir)
    return _ACP_BIN_DIR


def acp_env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = acp_bin_dir() + os.pathsep + env.get("PATH", "")
    env.update(extra)
    return env


def rejecting_bin(binary: str) -> str:
    """A throwaway bin dir holding one ``binary`` that fails loudly.

    The shim writes :data:`REJECTED_EFFORT_TAIL` to stderr and exits 2 --
    what a real CLI does when handed an effort level it does not know.
    """
    bindir = Path(tempfile.mkdtemp(prefix="nvsh-reject-effort-"))
    shim = bindir / binary
    shim.write_text(
        "#!/bin/sh\n" "echo 'starting up' >&2\n" f"echo '{REJECTED_EFFORT_TAIL}' >&2\n" "exit 2\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(bindir)


def rejecting_env(binary: str, **extra: str) -> dict[str, str]:
    """An environment whose ``PATH`` finds only the rejecting ``binary``."""
    env = dict(os.environ)
    env["PATH"] = rejecting_bin(binary) + os.pathsep + env.get("PATH", "")
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Generic scripted payload
# ---------------------------------------------------------------------------


def _write_events_file(script: Sequence[AgentEvent]) -> str:
    """Serialise an ``AgentEvent`` script to the JSON file the fakes read.

    Carries ``tool``/``args``/``result`` as well as ``kind``/``text``/
    ``error``, so a fake whose wire format can express a tool call or a
    thinking delta can be scripted generically rather than needing a bespoke
    per-test transcript. Fakes ignore the fields -- and the kinds -- they
    have no shape for; :attr:`AdapterCase.scriptable` records which those are.
    """
    payload = [
        {
            "kind": item.kind.value,
            "text": item.text,
            "error": item.error,
            "tool": item.tool,
            "args": dict(item.args),
            "result": item.result,
        }
        for item in script
    ]
    fd, path = tempfile.mkstemp(suffix=".json", prefix="nvsh-fake-events-")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _fake_env(script: Sequence[AgentEvent]) -> dict[str, str]:
    return fake_env(NVSH_FAKE_EVENTS=_write_events_file(script))


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out: list = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # pragma: no cover - the fakes write valid JSON
            continue
    return out


# ---------------------------------------------------------------------------
# ViaFake factories: an AgentEvent script -> a real adapter on a fake CLI
# ---------------------------------------------------------------------------


def agent_event_to_pi_wire(event: AgentEvent) -> dict:
    """Inverse of ``PiAgent._map_event`` for every kind these tests script.

    An unrecognized wire ``"type"`` maps back to ``STATUS`` with that type as
    its text, which is what lets a plain ``{"type": "one"}`` line stand in
    for ``AgentEvent(kind=STATUS, text="one")`` without a bespoke vocabulary.
    """
    if event.kind == EventKind.STATUS:
        return {"type": event.text}
    if event.kind == EventKind.TEXT_DELTA:
        return {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": event.text},
        }
    if event.kind == EventKind.THINKING:
        return {
            "type": "message_update",
            "assistantMessageEvent": {"type": "thinking_delta", "delta": event.text},
        }
    if event.kind == EventKind.TOOL_CALL:
        return {"type": "tool_execution_start", "toolName": event.tool, "args": dict(event.args)}
    if event.kind == EventKind.PROPOSAL:
        # pi's approval extension carries the command in a JSON envelope in
        # the dialog's title (see nvsh/agent/pi_ext/approval.ts).
        envelope = {
            "nvsh": "approval",
            "v": 1,
            "tool": "bash",
            "command": event.proposal.command if event.proposal else "",
            "reason": event.proposal.rationale if event.proposal else "",
        }
        return {
            "type": "extension_ui_request",
            "id": "ui-conformance-1",
            "method": "select",
            "title": json.dumps(envelope),
            "options": ["once", "session", "user", "deny"],
        }
    if event.kind == EventKind.DONE:
        return {"type": "agent_end"}
    if event.kind == EventKind.ERROR:
        return {"type": "error", "error": event.error}
    raise NotImplementedError(f"no pi wire mapping for {event.kind}")


def PiAgentViaFake(script):  # noqa: N802 - factory name doubles as the pytest id
    """``PiAgent`` replaying ``script`` through ``tests/fakes/pi_scripted``."""
    wire = [agent_event_to_pi_wire(event) for event in script]
    env = fake_env(NVSH_TEST_PI_SCRIPT=json.dumps(wire), **_tmp_home())
    return PiAgent(pi_path="pi_scripted", env=env)


def ClaudeAgentViaFake(script):  # noqa: N802
    return ClaudeAgent({}, env=_fake_env(script))


def CodexAgentViaFake(script):  # noqa: N802
    return CodexAgent({}, env=_fake_env(script))


def QwenAgentViaFake(script):  # noqa: N802
    return QwenAgent({}, env=_fake_env(script))


def _agy_wire(script: Sequence[AgentEvent]) -> dict:
    """``script`` as the recorded agy NDJSON shapes ``tests/fakes/agy`` replays.

    agy's ``event`` vocabulary is small and closed (see ``nvsh/agent/agy.py``):
    there is no catch-all STATUS line, no thinking line and no permission
    request, so those kinds simply have no shape here -- which is why ``agy``
    is not offered as a conformance ``factory`` and declares only
    ``tool_call`` scriptable.
    """
    conversation = "fake-conv-conformance-0001"
    stdout: list[str] = []
    exit_code = 0
    for step, event in enumerate(script, start=1):
        if event.kind == EventKind.TEXT_DELTA:
            stdout.append(
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {
                            "conversation_id": conversation,
                            "step_index": step,
                            "state": "ACTIVE",
                            "step_type": "agent_response",
                            "text_delta": event.text,
                        },
                    }
                )
            )
        elif event.kind == EventKind.TOOL_CALL:
            stdout.append(
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {
                            "conversation_id": conversation,
                            "step_index": step,
                            "state": "ACTIVE",
                            "step_type": "tool",
                            "tool_name": event.tool,
                            "tool_info": {"name": event.tool, "parameters": dict(event.args)},
                        },
                    }
                )
            )
        elif event.kind == EventKind.DONE:
            stdout.append(
                json.dumps(
                    {
                        "event": "result",
                        "result": {
                            "conversation_id": conversation,
                            "status": "SUCCESS",
                            "response": "",
                        },
                    }
                )
            )
        elif event.kind == EventKind.ERROR:
            stdout.append(
                json.dumps(
                    {
                        "event": "result",
                        "result": {
                            "conversation_id": conversation,
                            "status": "ERROR",
                            "error": event.error,
                        },
                    }
                )
            )
            exit_code = 1
    return {"stdout": stdout, "exit_code": exit_code}


def _agy_spec_env(script: Sequence[AgentEvent], **extra: str) -> dict[str, str]:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="nvsh-fake-agy-")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(_agy_wire(script), fh)
    return fake_env(NVSH_FAKE_EVENTS=path, **extra)


def AgyAgentViaFake(script):  # noqa: N802
    return AgyAgent(env=_agy_spec_env(script))


def _acp_directives(script: Sequence[AgentEvent]) -> list[dict]:
    """``script`` as ``tests/fakes/acp`` directives."""
    directives: list[dict] = []
    for event in script:
        if event.kind == EventKind.STATUS:
            directives.append({"update": {"sessionUpdate": event.text}})
        elif event.kind == EventKind.TEXT_DELTA:
            directives.append(
                {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": event.text},
                    }
                }
            )
        elif event.kind == EventKind.THINKING:
            directives.append(
                {
                    "update": {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": event.text},
                    }
                }
            )
        elif event.kind == EventKind.TOOL_CALL:
            directives.append(
                {
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "tool-conformance-1",
                        "status": "pending",
                        "kind": "execute",
                        "rawInput": dict(event.args),
                        "_meta": {"toolName": event.tool},
                    }
                }
            )
        elif event.kind == EventKind.ERROR:
            directives.append({"error": event.error})
        elif event.kind == EventKind.DONE:
            break  # the fake answers the prompt when its script runs out
        else:  # pragma: no cover - no other kind is scripted here
            raise NotImplementedError(f"no fake-ACP directive for {event.kind}")
    return directives


def AcpQwenViaFake(script):  # noqa: N802
    return acp_build("qwen", env=acp_env(NVSH_TEST_ACP_SCRIPT=json.dumps(_acp_directives(script))))


def AcpKiroViaFake(script):  # noqa: N802
    return acp_build("kiro", env=acp_env(NVSH_TEST_ACP_SCRIPT=json.dumps(_acp_directives(script))))


# ---------------------------------------------------------------------------
# Driving a turn
# ---------------------------------------------------------------------------


def answer_proposal(agent: NvshAgent, event: AgentEvent) -> None:
    """Approve a pending PROPOSAL through whichever channel the adapter has.

    Every adapter that mediates approval exposes one of two methods with the
    same meaning: ``respond_ui(request_id, confirmed=True)`` (pi, claude,
    acp) or ``respond_approval(request_id, True)`` (codex).
    """
    request_id = event.args.get("request_id")
    if request_id is None:
        return
    respond_approval = getattr(agent, "respond_approval", None)
    if callable(respond_approval):
        respond_approval(request_id, True)
        return
    respond_ui = getattr(agent, "respond_ui", None)
    if callable(respond_ui):
        respond_ui(str(request_id), confirmed=True)


def drive(agent: NvshAgent, *, approve: bool = True, limit: int = 200) -> list[AgentEvent]:
    """Run one turn to completion, answering any proposal along the way.

    ``limit`` is a runaway guard only: every fake carries its own watchdog,
    so a stuck turn fails the test quickly instead of hanging the suite.
    """
    agent.start()
    events: list[AgentEvent] = []
    try:
        for event in agent.run(conformance_request(), conformance_context()):
            events.append(event)
            if event.kind == EventKind.PROPOSAL and approve:
                answer_proposal(agent, event)
            if len(events) >= limit:  # pragma: no cover - runaway guard
                break
    finally:
        agent.close()
    return events


# ---------------------------------------------------------------------------
# Per-adapter rich turns
# ---------------------------------------------------------------------------
#
# "Rich" means the fullest turn the fake can play: thinking, a tool call and
# (where the adapter mediates approval) a permission request.


RICH_SCRIPT = [
    AgentEvent(kind=EventKind.THINKING, text="the driver node is missing"),
    AgentEvent(kind=EventKind.TEXT_DELTA, text="checking the driver"),
    AgentEvent(kind=EventKind.TOOL_CALL, tool="bash", args={"command": "nvidia-smi"}),
    AgentEvent(
        kind=EventKind.PROPOSAL,
        proposal=Proposal(
            command="modprobe nvidia",
            rationale="the driver module is not loaded",
            kind=ProposalKind.FIX,
        ),
    ),
    AgentEvent(kind=EventKind.DONE),
]


def _pi_rich() -> NvshAgent:
    return PiAgentViaFake(RICH_SCRIPT)


def _claude_rich() -> NvshAgent:
    """Claude's recorded 2.1.270 transcript: thinking, tool_use, can_use_tool."""
    transcript = FAKES_DIR / "claude-2.1.270-thinking-tooluse.jsonl"
    return ClaudeAgent({}, env=fake_env(NVSH_FAKE_TRANSCRIPT=str(transcript)))


def _codex_rich() -> NvshAgent:
    """codex app-server's recorded turn: reasoning deltas, approval, exec item."""
    return CodexAgent({}, binary="codex-app-server", env=fake_env())


def _acp_rich(name: str) -> Callable[[], NvshAgent]:
    def build() -> NvshAgent:
        # No NVSH_TEST_ACP_SCRIPT: the fake plays its recorded qwen 0.23.3
        # turn (thought chunks, a tool call, a request_permission).
        return acp_build(name, env=acp_env())

    return build


def _agy_rich() -> NvshAgent:
    return AgyAgentViaFake(
        [
            AgentEvent(kind=EventKind.TEXT_DELTA, text="listing"),
            AgentEvent(kind=EventKind.TOOL_CALL, tool="run_command", args={"CommandLine": "ls"}),
            AgentEvent(kind=EventKind.DONE),
        ]
    )


def _qwen_print_rich() -> NvshAgent:
    return QwenAgentViaFake(
        [
            AgentEvent(kind=EventKind.TEXT_DELTA, text="looking"),
            AgentEvent(kind=EventKind.DONE),
        ]
    )


# ---------------------------------------------------------------------------
# Per-adapter effort probes
# ---------------------------------------------------------------------------
#
# Each returns every string the child was told, so the suite can assert the
# operator's effort bytes arrive verbatim (decision c24).


def _pi_effort_words(probe: str) -> list[str]:
    agent = PiAgent(pi_path="pi", effort=probe, env=fake_env(**_tmp_home()))
    return list(agent.build_argv())


def _claude_effort_words(probe: str) -> list[str]:
    argv_path = Path(tempfile.mkdtemp(prefix="nvsh-claude-argv-")) / "argv.json"
    env = _fake_env([AgentEvent(kind=EventKind.DONE)])
    env["NVSH_FAKE_ARGV"] = str(argv_path)
    drive(ClaudeAgent({"effort": probe}, env=env))
    return [word for argv in _read_jsonl(argv_path) for word in argv]


def _codex_effort_words(probe: str) -> list[str]:
    return list(CodexAgent({"effort": probe}).app_server_argv())


def _agy_effort_words(probe: str) -> list[str]:
    argv_log = Path(tempfile.mkdtemp(prefix="nvsh-agy-argv-")) / "argv.log"
    env = _agy_spec_env(
        [AgentEvent(kind=EventKind.DONE)],
        NVSH_FAKE_ARGV_LOG=str(argv_log),
    )
    drive(AgyAgent(effort=probe, env=env))
    return [word for argv in _read_jsonl(argv_log) for word in argv]


def _acp_effort_words(name: str) -> Callable[[str], list[str]]:
    def words(probe: str) -> list[str]:
        commands = Path(tempfile.mkdtemp(prefix="nvsh-acp-frames-")) / "frames.jsonl"
        agent = acp_build(name, effort=probe, env=acp_env(NVSH_TEST_ACP_COMMANDS=str(commands)))
        drive(agent)
        # argv *plus* every frame the client sent: an ACP effort travels
        # in-session (session/set_config_option), not on the command line.
        return list(agent.build_argv()) + [json.dumps(frame) for frame in _read_jsonl(commands)]

    return words


def _qwen_print_effort_words(probe: str) -> list[str]:
    # qwen's print mode has no reasoning-effort flag at all, so there is no
    # public argv builder to read: ``_argv`` is what the child would get.
    agent = QwenAgent({"effort": probe}, env=fake_env())
    return list(agent._argv(conformance_request(), conformance_context()))  # noqa: SLF001


# ---------------------------------------------------------------------------
# Per-adapter rejecting CLIs
# ---------------------------------------------------------------------------


def _pi_rejecting() -> NvshAgent:
    return PiAgent(pi_path="pi", effort=EFFORT_PROBE, env=rejecting_env("pi", **_tmp_home()))


def _claude_rejecting() -> NvshAgent:
    return ClaudeAgent({"effort": EFFORT_PROBE}, env=rejecting_env("claude"))


def _codex_rejecting() -> NvshAgent:
    return CodexAgent({"effort": EFFORT_PROBE}, env=rejecting_env("codex"))


def _qwen_rejecting() -> NvshAgent:
    return QwenAgent({}, env=rejecting_env("qwen"))


def _agy_rejecting() -> NvshAgent:
    return AgyAgent(effort=EFFORT_PROBE, env=rejecting_env("agy"))


def _acp_rejecting(name: str, binary: str) -> Callable[[], NvshAgent]:
    def build() -> NvshAgent:
        return acp_build(name, effort=EFFORT_PROBE, env=rejecting_env(binary))

    return build


# ---------------------------------------------------------------------------
# The case table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterCase:
    """One backend under the cross-adapter conformance suite.

    ``scriptable`` names the event kinds this adapter's fake can actually
    produce (``"thinking"``, ``"tool_call"``, ``"proposal"``). A kind that is
    absent is not silently skipped: the suite asserts the adapter's declared
    capability instead.

    ``cli``/``version_range``/``stamp_file`` are the recorded-from provenance
    the version-stamp invariant checks. ``stamp_file`` is the repo-relative
    file carrying that ``# recorded-from:`` header, or ``None`` when nothing
    in the tree records one -- a gap :data:`UNSTAMPED_ADAPTERS` pins rather
    than hides.
    """

    name: str
    build_rich: Callable[[], NvshAgent]
    binary: str
    #: The effort string this case configures, and ``effort_words`` reads
    #: back from whatever the child was told. :data:`EFFORT_PROBE` (a value
    #: no real CLI accepts) everywhere it can be used; see ``qwen``'s entry
    #: for the one adapter whose path filters an unadvertised value out.
    effort_probe: str
    effort_words: Callable[[str], list[str]]
    build_rejecting: Callable[[], NvshAgent]
    scriptable: frozenset
    cli: str
    version_range: tuple
    stamp_file: str | None
    #: Conformance factory for the five original cases, or ``None`` when the
    #: fake's wire format cannot express the STATUS-ordering script they use.
    factory: Callable | None = None

    def __str__(self) -> str:  # pragma: no cover - pytest id helper
        return self.name


#: Adapters whose child is launched during ``start()``, so a CLI that
#: refuses to run surfaces as an exception from ``start()`` rather than an
#: ``ERROR`` from ``run()``. Both paths reach the operator as an ERROR --
#: ``nvsh/daemon.py`` turns a failed ``start()`` into
#: ``ERROR "no agent available: <exc>"`` -- so what the conformance suite
#: requires is that the CLI's stderr tail is in the text either way. Pinned
#: here so a change of path is a deliberate edit.
RAISE_ON_START = frozenset({"pi", "qwen", "kiro"})

#: Adapters whose ``capabilities()`` does not mention
#: ``unmediated_file_access`` at all and so inherits the ``False`` default.
#: ``QwenAgent`` (print mode) reads files with qwen's own tools, which nvsh
#: never sees, so ``False`` is not merely undeclared but wrong -- filed as a
#: deviation rather than patched here (this task owns tests, not adapters).
#: The conformance case xfails on this set, so fixing the adapter turns the
#: xfail into a failure and the entry gets removed with it.
UNDECLARED_FILE_ACCESS: frozenset[str] = frozenset()

#: Adapters tolerated to report a dead child as "no stderr" even though the
#: CLI did write one. Empty since 0.14.3: pi, AcpAgent (qwen, kiro) and the
#: codex app-server all drain stderr on their own thread and used to format
#: the launch failure as soon as they noticed the process was gone, so the
#: CLI's own words were dropped roughly one run in ten for acp (measured,
#: 2026-09-14) and about one whole-suite run in a dozen for pi (issue 27).
#: They now wait for the reader via ``_subprocess.settle_stderr``. The set is
#: kept so that a future adapter with the same race has somewhere to be
#: pinned deliberately, rather than the check being weakened for everyone.
STDERR_TAIL_RACE: frozenset[str] = frozenset()

CASES: list[AdapterCase] = [
    AdapterCase(
        name="pi",
        build_rich=_pi_rich,
        binary="pi",
        effort_probe=EFFORT_PROBE,
        effort_words=_pi_effort_words,
        build_rejecting=_pi_rejecting,
        scriptable=frozenset({"thinking", "tool_call", "proposal"}),
        cli="pi",
        version_range=("0.85", "0.86"),
        stamp_file=None,
        factory=PiAgentViaFake,
    ),
    AdapterCase(
        name="agy",
        build_rich=_agy_rich,
        binary="agy",
        effort_probe=EFFORT_PROBE,
        effort_words=_agy_effort_words,
        build_rejecting=_agy_rejecting,
        scriptable=frozenset({"tool_call"}),
        cli="agy",
        version_range=("1.0", "2.0"),
        stamp_file="tests/test_agent_agy.py",
        factory=None,
    ),
    AdapterCase(
        name="claude",
        build_rich=_claude_rich,
        binary="claude",
        effort_probe=EFFORT_PROBE,
        effort_words=_claude_effort_words,
        build_rejecting=_claude_rejecting,
        scriptable=frozenset({"thinking", "tool_call", "proposal"}),
        cli="claude",
        version_range=("2.1", "2.2"),
        stamp_file="tests/fakes/claude-2.1.270-thinking-tooluse.jsonl",
        factory=ClaudeAgentViaFake,
    ),
    AdapterCase(
        name="codex",
        build_rich=_codex_rich,
        binary="codex",
        effort_probe=EFFORT_PROBE,
        effort_words=_codex_effort_words,
        build_rejecting=_codex_rejecting,
        scriptable=frozenset({"thinking", "tool_call", "proposal"}),
        cli="codex-cli",
        version_range=("0.147", "0.148"),
        stamp_file="tests/fakes/codex-app-server",
        factory=CodexAgentViaFake,
    ),
    AdapterCase(
        name="qwen",
        build_rich=_acp_rich("qwen"),
        binary="qwen",
        # The one adapter that cannot be probed with an arbitrary string:
        # AcpAgent._config_value matches the operator's effort against the
        # options the harness advertises over session/new and sends nothing
        # when it does not match (acp.py: "rather than sending something the
        # harness would reject"). "high" is what the recorded qwen 0.23.3
        # session advertises, so what is proven here is narrower than
        # elsewhere: the bytes that *are* sent are the operator's, unedited.
        effort_probe="high",
        effort_words=_acp_effort_words("qwen"),
        build_rejecting=_acp_rejecting("qwen", "qwen"),
        scriptable=frozenset({"thinking", "tool_call", "proposal"}),
        cli="qwen",
        version_range=("0.23", "0.24"),
        stamp_file="tests/fakes/acp",
        factory=AcpQwenViaFake,
    ),
    AdapterCase(
        name="qwen-p",
        build_rich=_qwen_print_rich,
        binary="qwen",
        effort_probe=EFFORT_PROBE,
        effort_words=_qwen_print_effort_words,
        build_rejecting=_qwen_rejecting,
        scriptable=frozenset(),
        cli="qwen",
        version_range=("0.23", "0.24"),
        stamp_file="tests/fakes/qwen",
        factory=QwenAgentViaFake,
    ),
    AdapterCase(
        name="kiro",
        build_rich=_acp_rich("kiro"),
        binary="kiro-cli",
        effort_probe=EFFORT_PROBE,
        effort_words=_acp_effort_words("kiro"),
        build_rejecting=_acp_rejecting("kiro", "kiro-cli"),
        scriptable=frozenset({"thinking", "tool_call", "proposal"}),
        cli="kiro-cli",
        version_range=("2.0", "3.0"),
        stamp_file=None,
        factory=AcpKiroViaFake,
    ),
]

#: Adapters whose fakes carry no ``# recorded-from:`` provenance anywhere in
#: the tree. Pinned rather than hidden: the version-stamp invariant asserts
#: this set exactly, so adding a stamp (or landing a new unstamped fake) is a
#: deliberate edit, not silent drift.
UNSTAMPED_ADAPTERS = frozenset({"pi", "kiro"})
