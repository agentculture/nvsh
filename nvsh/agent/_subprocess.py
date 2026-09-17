"""Shared subprocess mapping base for thin headless-CLI adapters.

``claude``, ``codex`` and ``qwen`` are all "run a subprocess, read its
stdout line by line, translate each line into an :class:`AgentEvent`"
adapters that differ only in argv and line format. This module factors the
shared plumbing (spawn, cancel-between-yields, exit-code-to-ERROR mapping,
idempotent close) so each concrete adapter only supplies ``_argv`` and
``_parse_line``.
"""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 - fixed argv lists below, no shell=True
import threading
import time
from collections import deque
from typing import IO, Iterator

from ..redact import redact
from ._env import child_env
from .base import AgentContext, AgentEvent, AgentRequest, EventKind, NvshAgent
from .prompt import build_full_prompt as _build_full_prompt
from .prompt import build_prompt as _build_prompt
from .prompt import build_system_prompt as _build_system_prompt

#: Every backend composes its prompt with the *same* function, so the bytes
#: ``nvsh context --show`` prints are the bytes each adapter sends. Composing
#: it twice (once here, once in ``pi``) let the device context be dropped
#: whenever a caller also supplied a prompt -- which the failure client
#: always does -- so the platform block never reached these backends.
build_prompt = _build_prompt

#: The same goes for the system brief (deviation d19): one text, delivered
#: through whichever channel the backend has -- a flag where one exists,
#: the head of the prompt where none does.
build_system_prompt = _build_system_prompt
build_full_prompt = _build_full_prompt


#: How many stderr lines are kept for a non-zero exit. Enough to explain a
#: failure, bounded so a chatty backend cannot grow the daemon's memory.
_STDERR_TAIL_LINES = 200

#: How long the drain thread is given to finish once the child is gone.
_STDERR_JOIN_TIMEOUT = 2.0

#: Default seconds :func:`escalate_close` spends on each rung of its
#: wait -> terminate -> kill escalation.
CLOSE_WAIT_SECONDS = 2.0


def escalate_close(
    proc: subprocess.Popen | None,
    *,
    wait: float | None = None,
    grace: float | None = None,
) -> int | None:
    """Close *proc*'s stdin and see it out: wait, then terminate, then kill.

    One helper for every adapter (deviation d5). Each wave-2 adapter grew
    its own copy of this loop -- ``PiAgent.close``, ``AcpAgent._wait_out``,
    ``CodexAgent._wait_out``, ``AgyAgent._terminate``,
    :meth:`SubprocessAgent.close` -- which is exactly the kind of drift that
    leaves one harness's child running after ``nvsh uninstall`` while the
    others exit cleanly. The rungs, in order:

    1. **Close stdin.** Every long-lived harness nvsh drives (pi's rpc loop,
       ACP, ``codex app-server``, ``claude`` in stream-json input mode)
       exits on EOF, so that is the only rung most closes ever reach.
    2. **Wait** up to ``wait`` seconds for that clean exit.
    3. **``terminate()``**, then wait up to ``grace`` seconds.
    4. **``kill()``**, then wait up to ``grace`` seconds.
    5. **Reap the group.** Whichever rung ended the leader, any member of
       the process group it led that is still alive -- a tool grandchild
       the harness started and did not wait for -- gets SIGTERM, a grace
       period, then SIGKILL (task t23, deviation d14). A harness exiting
       cleanly on EOF is exactly the case that used to orphan them.

    Returns the child's exit status, or ``None`` when it survived even
    ``kill()`` (a process stuck in uninterruptible sleep, or one already
    reaped by somebody else) -- waiting on such a child forever is how
    teardown hangs, so this never does. Never raises: a close on the failure
    path must not itself become the failure.
    """
    if proc is None:
        return None
    # Resolved here, not in the signature's defaults, so the module constant
    # is read at call time (a default argument would freeze it at import).
    wait = CLOSE_WAIT_SECONDS if wait is None else wait
    grace = wait if grace is None else grace
    # Recorded before stdin closes: the EOF wait below reaps a leader that
    # exits cleanly, after which _own_group can no longer ask it.
    pgid = child_group(proc)
    stdin = getattr(proc, "stdin", None)
    if stdin is not None:
        try:
            stdin.close()
        except (OSError, ValueError):  # OSError covers BrokenPipeError
            pass
    try:
        rc = proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        # Terminate and kill through the process group when the child leads
        # its own (task t2), so tool grandchildren the harness started go
        # with it.
        rc = kill_tree(proc, grace=grace)
    except Exception:  # noqa: BLE001
        rc = getattr(proc, "returncode", None)
    reap_group(pgid, grace=grace)
    return rc


def child_group(proc: subprocess.Popen | None) -> int | None:
    """The process group *proc* leads, live or already reaped; ``None`` if unsafe.

    Call it before anything that may reap *proc*, and hand the result to
    :func:`reap_group` afterwards. Never raises.
    """
    if proc is None:
        return None
    try:
        return _own_group(proc) or _reaped_leader_group(proc)
    except Exception:  # noqa: BLE001 - teardown on the failure path must never raise
        return None


def _reaped_leader_group(proc: subprocess.Popen) -> int | None:
    """The group an *already reaped* leader led, when that is still provably it.

    ``kill_tree`` refuses to signal a reaped child's pid, because the pid may
    have been reused. A *group* id is different: Linux will not hand out a
    pid that is still in use as a process-group id, so while any member of
    group ``proc.pid`` lives, the group is the one our child created and its
    members are the child's own descendants. The one way ``proc.pid`` names
    somebody else's group is that every member died, the pid was
    reallocated, and the new process leads a group of its own -- in which
    case a process with that pid is alive again. So: no live process with
    ``proc.pid``, never nvsh's own group, and the group still has members.
    The remaining window (the last member dying and the pid being reused and
    made a group leader between this check and the signal) needs a full pid
    wrap-around in microseconds; ``killpg`` on an empty group is a harmless
    ESRCH.
    """
    if proc.returncode is None:
        return None
    pgid = proc.pid
    if pgid <= 1 or pgid == os.getpgrp():
        return None
    try:
        os.kill(pgid, 0)
    except ProcessLookupError:
        pass  # the leader's pid is free: nobody has reused it
    except OSError:
        return None  # EPERM: a live process (someone else's) holds the pid
    else:
        return None  # a live process holds the pid -- possibly reused
    return pgid if _group_alive(pgid) else None


def _group_alive(pgid: int) -> bool:
    """True while process group *pgid* has at least one member. Never raises."""
    try:
        os.killpg(pgid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def reap_group(pgid: int | None, grace: float = CLOSE_WAIT_SECONDS) -> None:
    """SIGTERM, wait up to *grace*, then SIGKILL whatever is left of *pgid*.

    For a group whose leader is already gone (a harness that exited on EOF
    and left a tool grandchild running). *pgid* must come from
    :func:`_own_group` / :func:`_reaped_leader_group` -- a group this process
    spawned into its own session -- and nvsh's own group is refused here
    too, belt and braces. Members are not our children, so there is nothing
    to ``wait()`` on: the group is polled until it empties. Never raises.
    """
    if pgid is None or pgid <= 1 or pgid == os.getpgrp():
        return
    # Every call below either cannot raise or catches OSError itself.
    if not _group_alive(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return
        time.sleep(0.02)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        return


def _own_group(proc: subprocess.Popen) -> int | None:
    """The process group *proc* leads, or ``None`` when signalling one is unsafe.

    Only a live (or not-yet-reaped zombie) child is asked: until it is
    reaped its pid cannot be reused, so the group id is really its group.
    A child sharing nvsh's own group (spawned without
    ``start_new_session``) returns ``None`` -- signalling that group would
    signal nvsh itself.
    """
    if proc.returncode is not None:
        return None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return None
    if pgid != proc.pid or pgid == os.getpgrp():
        return None
    return pgid


def _signal(proc: subprocess.Popen, pgid: int | None, sig: int) -> None:
    """Send *sig* to *pgid* when known, else to *proc* alone. Never raises."""
    try:
        if pgid is not None:
            os.killpg(pgid, sig)
        elif proc.poll() is None:
            proc.send_signal(sig)
    except (OSError, ValueError):  # already gone, or already reaped
        pass


def kill_tree(proc: subprocess.Popen | None, grace: float = 2.0) -> int | None:
    """Stop *proc* and everything in its process group: SIGTERM, wait, SIGKILL.

    Adapter children are spawned with ``start_new_session=True``, so the
    child leads a process group holding every tool grandchild it started;
    signalling the group is what keeps a stop from orphaning them. A child
    that does not lead its own group (or already shares nvsh's) is
    signalled alone, exactly like ``terminate()``/``kill()``.

    The group gets SIGKILL after the grace period even when the leader
    already exited on SIGTERM, because a grandchild that ignores SIGTERM
    outlives its parent. Only signals are sent -- no harness settings or
    trust file is ever read or written. Returns the child's exit status, or
    ``None`` when it could not be reaped. Never raises.
    """
    if proc is None:
        return None
    try:
        pgid = _own_group(proc)
        _signal(proc, pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        # Reaping the leader does not free the group id while any member
        # lives, so the group SIGKILL still reaches surviving grandchildren.
        _signal(proc, pgid, signal.SIGKILL)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        return proc.poll()
    except Exception:  # noqa: BLE001 - a stop on the failure path must never raise
        return getattr(proc, "returncode", None)


def _drain(stream: IO[str], sink: deque[str]) -> None:
    """Read ``stream`` to EOF into ``sink`` (bounded). Never raises."""
    try:
        for line in stream:
            sink.append(line)
    except (OSError, ValueError):  # closed underneath us by cancel/terminate
        pass


#: Argv fragments that would let a harness approve its own tool calls.
#: "Propose, don't run" is not a policy an operator may configure away
#: through ``extra_args``; every adapter refuses these at construction.
BYPASS_ARGS = frozenset(
    {
        "--dangerously-skip-permissions",
        "--dangerously-bypass-approvals-and-sandbox",
        "--full-auto",
        "--yolo",
        "--trust-all-tools",
        "danger-full-access",
        "bypassPermissions",
    }
)


def reject_bypass_args(extra_args: list[str], harness: str) -> None:
    """Refuse ``extra_args`` that would bypass nvsh's approval gate."""
    for arg in extra_args:
        if arg in BYPASS_ARGS or any(
            token in arg for token in BYPASS_ARGS if token.startswith("--")
        ):
            raise ValueError(f"{harness}: extra_args may not bypass approval ({arg!r})")


def redacted_tail(tail: deque[str] | list[str]) -> str:
    """Join a stderr tail into one string, redacted before anyone sees it.

    Shared by every subprocess-backed adapter (and reusable by any other
    adapter that keeps its own stderr tail, e.g. ``pi``) so "redact the
    stderr before it reaches an ERROR event or a log line" is one choke
    point instead of one per adapter. ``redact`` operates on bytes, so the
    joined text round-trips through UTF-8 with ``surrogateescape`` the same
    way ``nvsh.redact`` itself does.
    """
    text = "".join(tail)
    redacted_bytes = redact(text.encode("utf-8", errors="surrogateescape"))
    return redacted_bytes.decode("utf-8", errors="surrogateescape").strip()


class SubprocessAgent(NvshAgent):
    """Spawns ``binary`` and maps its stdout lines to :class:`AgentEvent`.

    Subclasses set :attr:`binary` and implement :meth:`_argv` and
    :meth:`_parse_line`. ``env`` (test-only override) lets the conformance
    suite point PATH at ``tests/fakes/`` without touching the real PATH.
    """

    binary: str = ""

    def __init__(self, config: dict | None = None, *, env: dict[str, str] | None = None) -> None:
        self._config = dict(config or {})
        self._env = env
        self._proc: subprocess.Popen | None = None
        self._cancelled = False
        self._closed = False
        #: The leader's process group, captured at spawn time (Qodo 6).
        #: ``run()``'s own ``_exit_event`` waits out the leader before the
        #: ``finally`` block runs teardown, so by then ``poll()`` is never
        #: ``None`` and a group captured only there would already be gone
        #: (``_own_group`` refuses a reaped leader). Capturing it here,
        #: before that wait, keeps it reapable either way --
        #: ``child_group`` falls back to ``_reaped_leader_group`` once the
        #: leader is gone.
        self._pgid: int | None = None

    def start(self) -> None:
        self._cancelled = False
        self._closed = False

    def _argv(self, request: AgentRequest, context: AgentContext) -> list[str]:
        raise NotImplementedError

    def _parse_line(self, line: str) -> AgentEvent | None:
        """Return an ``AgentEvent`` for one stdout line, or ``None`` to skip it."""
        raise NotImplementedError

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        # A cancel ends one turn, not the adapter (start() runs once per
        # warm daemon session, not once per turn).
        self._cancelled = False
        argv = self._argv(request, context)
        try:
            self._proc = subprocess.Popen(  # nosec B603 - argv is a fixed list, no shell
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_env(self._env),
                # Its own process group, so a stop can kill the whole tree
                # (kill_tree) and the terminal's SIGINT never reaches it.
                start_new_session=True,
            )
        except OSError as exc:
            yield AgentEvent(kind=EventKind.ERROR, error=f"failed to start {argv[0]}: {exc}")
            return

        # Recorded before anything can reap the leader (``_exit_event``'s
        # own ``wait()`` included, Qodo 6) -- a tool grandchild left in this
        # group must still be reapable in the ``finally`` below even after
        # the leader has already exited normally.
        self._pgid = child_group(self._proc)

        # Drain stderr concurrently. Reading it only once stdout hits EOF
        # deadlocks any backend that writes more than a pipe buffer's worth
        # of warnings while it is still running: the child blocks on stderr,
        # so it never closes stdout, so the hook waits out the daemon's turn
        # timeout instead of showing the diagnosis. Only a bounded tail is
        # kept -- that is all a non-zero exit needs to be explainable.
        tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        drain: threading.Thread | None = None
        if self._proc.stderr is not None:
            drain = threading.Thread(target=_drain, args=(self._proc.stderr, tail), daemon=True)
            drain.start()

        try:
            assert self._proc.stdout is not None
            for raw_line in self._proc.stdout:
                if self._cancelled:
                    return
                event = self._parse_line(raw_line.rstrip("\n"))
                if event is None:
                    continue
                yield event
                if event.kind in (EventKind.ERROR, EventKind.DONE):
                    return
            if self._cancelled:
                return
            yield self._exit_event(argv, tail, drain)
        finally:
            # Terminating closes the child's end of the pipe, so the drain
            # thread sees EOF and returns; join it so no reader outlives the
            # turn (cancellation included).
            self._terminate_if_running()
            if drain is not None:
                drain.join(timeout=_STDERR_JOIN_TIMEOUT)

    def _exit_event(
        self, argv: list[str], tail: deque[str], drain: threading.Thread | None
    ) -> AgentEvent:
        """How the child finished: DONE on 0, else ERROR quoting its stderr.

        The drain thread is joined only on the failure path, and only then,
        because that is the one case whose message needs the whole tail.
        """
        assert self._proc is not None
        rc = self._proc.wait()
        if rc == 0:
            return AgentEvent(kind=EventKind.DONE)
        if drain is not None:
            drain.join(timeout=_STDERR_JOIN_TIMEOUT)
        stderr = redacted_tail(tail)
        return AgentEvent(kind=EventKind.ERROR, error=stderr or f"{argv[0]} exited {rc}")

    def cancel(self) -> None:
        self._cancelled = True
        self._terminate_if_running()

    def _terminate_if_running(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            kill_tree(self._proc, grace=CLOSE_WAIT_SECONDS)
        # Reaps a tool grandchild left in the leader's group even when the
        # leader itself already exited normally -- ``kill_tree`` above only
        # fires while the leader still polls as running, and by the time
        # this runs after ``_exit_event``'s own wait(), it never does
        # (Qodo 6). ``self._pgid`` was captured at spawn time, before that
        # wait could reap the leader out from under ``child_group``.
        reap_group(self._pgid, grace=CLOSE_WAIT_SECONDS)

    def close(self) -> None:
        # Close, not cancel: the stdin rung matters here and must not run
        # mid-turn (``claude`` keeps stdin a live pipe for the whole turn),
        # which is why ``cancel``/``run``'s teardown still go through
        # ``_terminate_if_running`` instead.
        if self._closed:
            return
        self._closed = True
        escalate_close(self._proc)
