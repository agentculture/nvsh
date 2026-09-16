"""The failure client: what the bash hook calls when a command really failed.

``nvsh hook`` classifies the event (:mod:`nvsh.triggers`) and, on ``ask``,
hands off to :func:`handle_failure` here. This module owns everything
between that hand-off and the operator's prompt coming back:

1. **Record the failure** under ``$XDG_STATE_HOME/nvsh/last-failure.json``
   (mode 0600) *before* anything else, so ``/fix`` still finds it if the
   agent is unreachable or the operator presses Ctrl+C.
2. **Hold the rate limit for real.** ``nvsh hook`` runs in a fresh process
   every time, so its in-memory :class:`nvsh.triggers.RateState` always
   looks empty. The persisted state under
   ``$XDG_STATE_HOME/nvsh/rate.json`` is the one that actually enforces the
   window, and it is re-checked here.
3. **Assemble the context** -- the detected platform block (every value with
   its source), the redacted output slice for exactly that command, the cwd
   and the shell pid. Nothing else leaves the process, and
   ``nvsh context --show`` prints exactly these bytes.
4. **Stream the answer into the panel** through
   :mod:`nvsh.client_transport` (daemon, else one-shot), first text on
   screen as soon as it arrives.
5. **Handle proposals.** A read-only ``inspect`` proposal whose command the
   operator has already approved (:mod:`nvsh.approvals`) is run here, with a
   10 s timeout, and its redacted output is fed back as one follow-up
   prompt. Everything else goes to the panel and runs only if the operator
   pressed Enter (run once), ``s`` (run and approve for this login session)
   or ``u`` (run and approve for this user) on that exact command. A
   ``sudo`` command is never auto-run, whatever the approval patterns say,
   and ``s``/``u`` are refused for it with a one-line reason -- only the
   run-once path is ever open to a privileged or destructive command
   (deviation d15). With a backend dialog (pi's approval extension, i.e. a
   proposal carrying a ``request_id``) the answer is forwarded verbatim as
   ``{"value": "once"|"session"|"user"|"deny"}`` and the extension owns
   both the widening and the store write; without one, this module writes
   the pattern through :mod:`nvsh.approvals` and runs the command itself.

Nothing here writes into readline's buffer: nvsh never pre-types a command
into the operator's shell, it only ever prints one.

Stdlib only.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shlex
import subprocess  # nosec B404 - argv is always ["bash", "-c", <approved command>]
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import client_transport
from .agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Proposal,
    ProposalKind,
    RequestKind,
    Target,
)
from .cli._errors import EXIT_DECLINED
from .panel import (
    APPROVE,
    APPROVE_SESSION,
    APPROVE_SESSION_SPECIFIC,
    APPROVE_USER,
    APPROVE_USER_SPECIFIC,
    DETAILS,
    EXPLAIN,
    IGNORE,
    REFUSED,
    REPLACE,
    STEER,
    TELL,
    Panel,
    StreamResult,
)
from .triggers import prose_request

#: The panel choices that mean "run it, and stop asking me about this class".
_SCOPE_CHOICES = (
    APPROVE_SESSION,
    APPROVE_SESSION_SPECIFIC,
    APPROVE_USER,
    APPROVE_USER_SPECIFIC,
)

#: Every choice that means the operator wants the command to run now.
_RUN_CHOICES = (APPROVE,) + _SCOPE_CHOICES

#: The decision recorded when an unprivileged, read-only proposal already
#: matches a stored approval pattern, so nvsh runs it without re-asking.
AUTO_INSPECT = "auto-inspect"

#: Every decision :func:`_approved_command` accepts as "the operator said yes".
_APPROVED_DECISIONS = _RUN_CHOICES + (AUTO_INSPECT,)

#: Control bytes that must never appear in a command nvsh is about to run:
#: C0 minus tab/newline/carriage-return, plus DEL. A proposal carrying an
#: escape sequence or a NUL is malformed or hostile, never approved.
_CONTROL_BYTES_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

#: How long an auto-run read-only inspector may take before it is killed.
INSPECT_TIMEOUT = 10.0

#: How many times the panel re-asks after ``e``/``d`` before giving up.
_MAX_PROPOSAL_ROUNDS = 4

#: What :func:`_decide_proposal` returns when the operator steered the agent
#: instead of deciding: nothing runs, nothing is denied twice, and the
#: conversation carries on with the operator's text in it (deviation d16).
STEERED = "steered"

#: What ``[e]`` asks the agent when the proposal came with no rationale and
#: there is a conversation to ask in (deviation d17). Two sentences, because
#: this answer is read at a failing prompt, not in a report.
EXPLAIN_PROPOSAL_PROMPT = "Explain in two sentences why you propose: {command}"

#: The default prompts. Kept as constants because ``nvsh context --show``
#: has to reproduce the failure prompt byte for byte.
FAILURE_PROMPT = (
    "Diagnose this failure on this machine and propose a fix. "
    "Prefer read-only inspection first. Never assume a package manager or "
    "driver version that is not in the platform block above."
)
FIX_PROMPT = "Propose the smallest fix for this failure, as one command."
EXPLAIN_PROMPT = "Explain this failure: what it means on this machine, and why it happened."


# ---------------------------------------------------------------------------
# state files
# ---------------------------------------------------------------------------


def state_dir(env: Mapping[str, str] | None = None) -> Path:
    """``$XDG_STATE_HOME/nvsh``, falling back to ``~/.local/state/nvsh``."""
    resolved = os.environ if env is None else env
    xdg = resolved.get("XDG_STATE_HOME")
    if xdg:
        base = Path(xdg)
    else:
        home = resolved.get("HOME") or os.path.expanduser("~")
        base = Path(home) / ".local" / "state"
    return base / "nvsh"


def last_failure_path(env: Mapping[str, str] | None = None) -> Path:
    return state_dir(env) / "last-failure.json"


def rate_state_path(env: Mapping[str, str] | None = None) -> Path:
    return state_dir(env) / "rate.json"


def _write_private_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as 0600 JSON, creating the state dir 0700."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:  # pragma: no cover - a shared dir we do not own
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.chmod(path, 0o600)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save_last_failure(
    args: Any, failure_id: str | None = None, env: Mapping[str, str] | None = None
) -> dict:
    """Record the failure so ``/fix`` and ``nvsh context --show`` can find it."""
    state = {
        "failure_id": failure_id or uuid.uuid4().hex,
        "line": str(getattr(args, "line", "") or ""),
        "exit": int(getattr(args, "exit", 0) or 0),
        "pipestatus": str(getattr(args, "pipestatus", "") or ""),
        "cwd": str(getattr(args, "cwd", "") or os.getcwd()),
        "log": str(getattr(args, "log", "") or ""),
        "ts": time.time(),
    }
    _write_private_json(last_failure_path(env), state)
    return state


def load_last_failure(env: Mapping[str, str] | None = None) -> dict | None:
    """The last recorded failure, or ``None`` when there is none."""
    return _read_json(last_failure_path(env))


def _args_from_state(state: Mapping[str, Any]) -> Any:
    @dataclass
    class _Args:
        exit: int
        pipestatus: str
        line: str
        cwd: str
        log: str

    return _Args(
        exit=int(state.get("exit", 0) or 0),
        pipestatus=str(state.get("pipestatus", "") or ""),
        line=str(state.get("line", "") or ""),
        cwd=str(state.get("cwd", "") or os.getcwd()),
        log=str(state.get("log", "") or ""),
    )


# ---------------------------------------------------------------------------
# rate limiting that survives the process
# ---------------------------------------------------------------------------


def _load_rate_state(env: Mapping[str, str] | None):
    from .triggers import RateState

    data = _read_json(rate_state_path(env))
    if not data:
        return RateState()
    last = data.get("last_auto_call")
    return RateState(last_auto_call=float(last) if isinstance(last, (int, float)) else None)


def _last_held_back_notice(env: Mapping[str, str] | None) -> float | None:
    data = _read_json(rate_state_path(env)) or {}
    value = data.get("last_held_back_notice")
    return float(value) if isinstance(value, (int, float)) else None


def _save_rate_state(state, env: Mapping[str, str] | None, notice: float | None = None) -> None:
    """Persist the rate state, carrying the held-back notice flag forward."""
    kept = _last_held_back_notice(env) if notice is None else notice
    payload: dict = {"last_auto_call": state.last_auto_call}
    if kept is not None:
        payload["last_held_back_notice"] = kept
    _write_private_json(rate_state_path(env), payload)


def _note_held_back(config, env: Mapping[str, str] | None, now: float) -> None:
    """Say once per window that an automatic call was held back.

    Deviation d10: a silently suppressed second failure looks exactly like a
    broken nvsh. One stderr line per window says what happened and how to
    ask anyway; the flag that keeps it to one line lives next to the rate
    state, so a burst of failures across separate processes stays quiet.
    """
    from .cli._output import emit_diagnostic

    window = _rate_window(config)
    last_notice = _last_held_back_notice(env)
    if last_notice is not None and now - last_notice < window:
        return
    emit_diagnostic(
        f"nvsh: held back (auto calls limited to 1 per {int(window)}s window); "
        "/fix or Ctrl+G asks now"
    )
    _save_rate_state(_load_rate_state(env), env, notice=now)


def _rate_window(config) -> float:
    from .triggers import DEFAULT_RATE_WINDOW

    raw = (config.triggers or {}).get("rate_window_seconds")
    try:
        return float(raw) if raw is not None else DEFAULT_RATE_WINDOW
    except (TypeError, ValueError):
        return DEFAULT_RATE_WINDOW


def rate_lock_path(env: Mapping[str, str] | None = None) -> Path:
    """The per-user lock that makes the rate decision atomic across shells."""
    return state_dir(env) / "rate.lock"


@contextlib.contextmanager
def _rate_lock(env: Mapping[str, str] | None):
    """Hold an exclusive ``flock`` over the load-decide-save transaction.

    Two shells failing at the same moment used to read the same stale
    timestamp and *both* call the agent, so "one automatic call per window"
    was not actually enforced. The critical section is two small file
    operations, and ``flock`` is released by the kernel when the holder
    exits, so a crashed hook cannot wedge anyone's prompt. If the lock
    cannot be taken at all (a read-only or exotic filesystem), the decision
    still happens -- unsynchronized, as before -- rather than the prompt
    losing its diagnosis.
    """
    fd: int | None = None
    try:
        path = rate_lock_path(env)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
            fd = None
    try:
        yield
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)


def _rate_limited(args: Any, config, env: Mapping[str, str] | None) -> bool:
    """Re-run the trigger decision against the *persisted* rate state.

    The whole load-decide-save transaction runs under :func:`_rate_lock`, so
    concurrent failures in different terminals cannot each authorize a call
    from the same stale state.
    """
    from .triggers import TriggerEvent, decide

    try:
        pipestatus = tuple(int(x) for x in str(getattr(args, "pipestatus", "") or "").split())
    except ValueError:
        pipestatus = ()
    with _rate_lock(env):
        decision = decide(
            TriggerEvent(
                command=str(getattr(args, "line", "") or ""),
                exit_code=int(getattr(args, "exit", 0) or 0),
                pipestatus=pipestatus,
                now=time.time(),
                rate_window=_rate_window(config),
                rate_state=_load_rate_state(env),
            )
        )
        _save_rate_state(decision.rate_state, env)
    return decision.action != "ask"


# ---------------------------------------------------------------------------
# context assembly
# ---------------------------------------------------------------------------


def _platform_block() -> str:
    """The detected platform, every value with its source. Never raises."""
    try:
        from .platform import detect

        return detect().render_block()
    except Exception as exc:  # noqa: BLE001 - a diagnosis without platform beats none
        return f"platform: unknown (detection failed: {exc})"


def build_context(args: Any, env: Mapping[str, str] | None = None) -> AgentContext:
    """Assemble everything nvsh is willing to send for one request.

    The output slice comes from :func:`nvsh.capture.last_slice`, which has
    already bounded, stripped and redacted it; the rules that fired are
    carried along as ``redaction_report`` so the operator can see what was
    scrubbed.
    """
    resolved = dict(os.environ if env is None else env)
    log = str(getattr(args, "log", "") or "")
    output = "no capture"
    rules: tuple[str, ...] = ()
    if log:
        from .capture import last_slice

        result = last_slice(Path(log))
        output = result.text if result.text else result.status
        rules = tuple(result.redaction_rules)
    return AgentContext(
        platform=_platform_block(),
        output=output,
        cwd=str(getattr(args, "cwd", "") or os.getcwd()),
        shell_pid=_shell_pid(resolved),
        redaction_report=rules,
    )


def _shell_pid(env: Mapping[str, str]) -> int:
    raw = env.get("NVSH_SHELL_PID")
    try:
        return int(raw) if raw else os.getppid()
    except ValueError:
        return os.getppid()


def build_request(kind: RequestKind, args: Any, prompt: str | None = None) -> AgentRequest:
    """Build one :class:`AgentRequest` from hook-shaped ``args``."""
    return AgentRequest(
        kind=kind,
        prompt=prompt or "",
        command=str(getattr(args, "line", "") or ""),
        exit_code=(
            int(getattr(args, "exit", 0)) if getattr(args, "exit", None) is not None else None
        ),
        failure_id=str(getattr(args, "failure_id", "") or ""),
    )


def _ask_header(panel: Panel, state: Mapping[str, Any], question: str) -> None:
    """Print the "this was a question" header, however the panel spells it.

    ``Panel.header`` is owned elsewhere and is growing an ``ask`` form of
    its own; this call site uses it as soon as its signature accepts an
    ``ask`` keyword and prints the plain line until then, so the two changes
    can land in either order without one breaking the other.
    """
    import inspect

    try:
        accepts_ask = "ask" in inspect.signature(panel.header).parameters
    except (TypeError, ValueError):  # pragma: no cover - a non-introspectable panel
        accepts_ask = False
    if accepts_ask:
        panel.header(str(state.get("line", "") or ""), int(state.get("exit", 0) or 0), ask=question)
        return
    panel.line(f"nvsh: asking the agent: {question}")


def resolve_target(config, name: str) -> Target | None:
    """Resolve one ``@mark`` / ``--agent`` name to a :class:`Target`.

    Everything the operator may type goes through
    :meth:`Config.resolve_target` -- an alias (``fast``, ``default``), a
    literal ``backend/model/effort`` string, with or without a leading
    ``@``. A bare harness name that is *not* an alias (``@qwen``) is not a
    target string at all, so it is resolved here against the registry
    instead, picking up that harness's configured model. ``None`` when the
    name means nothing; the caller turns that into the refusal line.
    """
    try:
        backend, model, effort, by_alias = config.resolve_target(name)
    except Exception:  # noqa: BLE001 - not an alias and not a target string
        return _bare_backend_target(config, name)
    return Target(backend=backend, model=model, effort=effort, alias=name if by_alias else None)


def _bare_backend_target(config, name: str) -> Target | None:
    """``@qwen``: a registered harness named on its own, with its own model."""
    from .agent import registry

    bare = name[1:] if name.startswith("@") else name
    if bare not in registry.ADAPTERS:
        return None
    try:
        model = str((config.agents.get(bare) or {}).get("model") or "") or None
    except Exception:  # noqa: BLE001 - a config too broken to read names no model
        model = None
    return Target(backend=bare, model=model)


def default_target(config) -> Target | None:
    """What this machine's ``default`` resolves to, for the panel header."""
    from .config import DEFAULT_ALIAS

    return resolve_target(config, DEFAULT_ALIAS)


def _is_one_shot(target) -> bool:
    """c25: a named target runs one-shot -- except ``default`` itself, which
    *is* the warm daemon target and must stay in its conversation."""
    from .config import DEFAULT_ALIAS

    return target is not None and getattr(target, "alias", None) != DEFAULT_ALIAS


def agent_override(config, name: str):
    """Point *config* at one target for a single request (d23, decision c25).

    Returns ``(config, target, "")`` when ``name`` resolves to a registered,
    installed harness -- a copy of *config* whose provider is that backend
    and whose ``[agents.<backend>]`` carries the target's model/effort, so
    :func:`nvsh.agent.registry.choose` picks it and :func:`backend_label`
    names it -- and ``(None, None, line)`` otherwise, where ``line`` is the
    single line the operator gets and the whole of nvsh's answer. A harness
    that is not there must never be silently swapped for the default: the
    operator asked ``@qwen`` on purpose.
    """
    from .agent import registry

    target = resolve_target(config, name)
    if target is None:
        known = ", ".join(registry.ADAPTERS)
        return None, None, f"nvsh: @{name} is not available: unknown harness (known: {known})"
    if target.backend not in registry.ADAPTERS:
        known = ", ".join(registry.ADAPTERS)
        return (
            None,
            None,
            (
                f"nvsh: @{name} is not available: unknown harness "
                f"{target.backend!r} (known: {known})"
            ),
        )
    try:
        ok = registry.installed(target.backend)
    except Exception as exc:  # noqa: BLE001 - a broken probe is a plain refusal
        return None, None, f"nvsh: @{name} is not available: {exc}"
    if not ok:
        binary = registry.ADAPTERS[target.backend].binary or target.backend
        return None, None, f"nvsh: @{name} is not available: '{binary}' is not on PATH"
    try:
        return client_transport.targeted_config(config, target), target, ""
    except Exception as exc:  # noqa: BLE001
        return None, None, f"nvsh: @{name} is not available: {exc}"


def _prose_request(state: Mapping[str, Any], question: str) -> AgentRequest:
    """The request for a plain-language question typed at the prompt (d20).

    ``command``/``exit_code`` are deliberately left empty: there is no
    failed command to diagnose, so the prompt the adapters compose carries
    the operator's sentence and the detected machine facts, and nothing
    about ``what: command not found``.
    """
    return AgentRequest(
        kind=RequestKind.EXPLICIT,
        prompt=question,
        command="",
        exit_code=None,
        failure_id=str(state.get("failure_id", "") or ""),
        ask=question,
    )


def _failure_request(state: Mapping[str, Any], prompt: str = FAILURE_PROMPT) -> AgentRequest:
    return AgentRequest(
        kind=RequestKind.FAILURE,
        prompt=prompt,
        command=str(state.get("line", "") or ""),
        exit_code=int(state.get("exit", 0) or 0) if state else None,
        failure_id=str(state.get("failure_id", "") or ""),
    )


# ---------------------------------------------------------------------------
# running commands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """The outcome of one command nvsh ran on the operator's behalf."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""


class UnapprovedCommandError(RuntimeError):
    """A command tried to reach the executor without a decided approval behind it."""


def _approved_command(proposal: Proposal, decision: str) -> str:
    """The trust boundary: the *only* way a string may reach :func:`_run_command`.

    nvsh runs the exact line the operator approved -- the spec requires it,
    so the argv stays ``["bash", "-c", <line>]`` and is never rewritten. What
    makes that safe is where the line comes from, and this function is the
    one place that is checked:

    * it must be the ``command`` of a :class:`~nvsh.agent.base.Proposal` --
      the object the panel rendered and the operator saw, never a string
      taken from ``argv``, an environment variable or an agent's raw output;
    * *decision* must be one of the tokens :func:`_decide_proposal` returns
      for an explicit keypress (``[y]``/``[s]``/``[u]``) or the
      :data:`AUTO_INSPECT` token, which is only reachable for a read-only,
      unprivileged proposal that already matches a stored approval pattern
      (see :mod:`nvsh.approvals`);
    * the line must have a body and must not carry control bytes -- a NUL or
      an escape sequence in a proposal is a malformed or hostile proposal,
      never something the operator meant to approve, and the panel could not
      have shown it faithfully either.

    Every run is recorded by the audit log at its call sites. A refusal
    raises :class:`UnapprovedCommandError` rather than running anything.
    """
    if not isinstance(proposal, Proposal):
        raise UnapprovedCommandError("only a Proposal the operator saw may be run")
    if decision not in _APPROVED_DECISIONS:
        raise UnapprovedCommandError(f"command not approved (decision: {decision!r})")
    command = proposal.command
    if not isinstance(command, str) or not command.strip():
        raise UnapprovedCommandError("an approved proposal must carry a command")
    if _CONTROL_BYTES_RE.search(command):
        raise UnapprovedCommandError("refusing a command carrying control bytes")
    return command


def _run_approved(
    proposal: Proposal,
    decision: str,
    *,
    timeout: float | None = None,
    login: bool = False,
) -> RunResult:
    """Run the command of an approved *proposal*. The only caller of the executor."""
    return _run_command(_approved_command(proposal, decision), timeout=timeout, login=login)


def _run_command(command: str, timeout: float | None = None, login: bool = False) -> RunResult:
    """Run an already-approved ``command`` through bash, capturing its output.

    Private, and reached only through :func:`_run_approved` /
    :func:`_approved_command`: the argv is always
    ``["bash", "-c"/"-lc", command]`` -- never ``shell=True``, never an
    environment variable, and never anything the operator has not approved
    with a keypress at the panel.
    """
    argv = ["bash", "-lc" if login else "-c", command]
    try:
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell=True
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return RunResult(exit_code=124, stderr=f"timed out after {timeout}s")
    except OSError as exc:
        return RunResult(exit_code=127, stderr=str(exc))
    return RunResult(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def _is_privileged(command: str) -> bool:
    """Does ``command`` escalate privileges anywhere in it?"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    return any(token in ("sudo", "doas", "pkexec") for token in tokens)


def _redact(text: str) -> str:
    from .redact import redact_report

    cleaned, _rules = redact_report(text.encode("utf-8", errors="surrogateescape"))
    return cleaned.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# streaming one request into the panel
# ---------------------------------------------------------------------------


def _autostart(env: Mapping[str, str]) -> bool:
    """``NVSH_NO_DAEMON`` keeps the client one-shot (used by tests and by
    operators who do not want a background process)."""
    return not env.get("NVSH_NO_DAEMON")


def _send(
    request: AgentRequest,
    context: AgentContext,
    *,
    env: Mapping[str, str],
    shell_id: int,
    config,
    responder=None,
    one_shot: bool = False,
) -> Iterable[AgentEvent]:
    # A one-request harness override (``@name`` / ``/ask --agent``) always
    # runs one-shot: the warm daemon holds a session for the *configured*
    # harness and its own config, so asking it would answer from the default
    # backend and quietly ignore the operator's choice (deviation d23).
    if one_shot:
        return client_transport.one_shot(request, context, config=config, responder=responder)
    return client_transport.send(
        request,
        context,
        shell_id=shell_id,
        env=env,
        config=config,
        autostart=_autostart(env),
        responder=responder,
    )


def _load_config():
    from .config import Config, load

    try:
        return load()
    except Exception:  # noqa: BLE001 - a broken config must never block a diagnosis
        return Config()


def _audit(env: Mapping[str, str]):
    from .agent.audit import AuditLog

    try:
        return AuditLog(env=env)
    except OSError:  # pragma: no cover - unwritable state dir
        return None


def scope_patterns(command: str, scope: str, chosen: Sequence[int] | None = None) -> list[str]:
    """The globs ``scope`` would store for ``command`` -- one per stage (d24).

    Delegates to :func:`nvsh.approvals.patterns_for`, the single pure helper
    the panel's scope line, this store write, the details view and (through
    ``nvsh approve add --scope``) the pi approval extension all share, so
    the four can never describe or persist different things.

    ``chosen`` is d26's stage pick: the 1-based stage numbers the operator
    typed at the panel's ``stages [all,1,2]: `` prompt, or ``None`` for
    every stage.
    """
    from .approvals import patterns_for

    return patterns_for(command, scope, chosen)


def stage_tokens(command: str) -> list[str]:
    """One rendered token per stage, for the panel's numbered ``stages:`` line.

    Deviation d26: d24's ``[s] each stage exactly`` never showed the
    operator what the stages *were*. An approvable stage is quoted as it
    will be matched; a stage no pattern may ever cover (a ``sudo``/``rm``
    stage) reads ``(not approvable)`` rather than being quoted as though a
    key could store it. An opaque line is one stage by construction
    (:func:`nvsh.approvals.stages`), so it keeps d24's single-stage
    rendering and its whole-line refusal.
    """
    from .approvals import command_refusal_reason, stages

    return [
        "(not approvable)" if command_refusal_reason(stage) else f"'{stage}'"
        for stage in stages(command)
    ]


def scope_stage_patterns(command: str) -> dict[str, list[str]]:
    """Per scope, the quoted pattern it would store for *each* stage (d26).

    Positional and undeduplicated, so entry ``N - 1`` is always stage ``N``
    -- the numbers the panel prints are the numbers it reads back.
    """
    from .approvals import stage_patterns

    return {
        scope: [f"'{pattern}'" for pattern in stage_patterns(command, scope)]
        for scope in _SCOPE_CHOICES
    }


def scope_pattern(command: str, scope: str) -> str:
    """:func:`scope_patterns` as one string, joined by ``" | "``.

    A single-stage command -- the overwhelmingly common case -- reads
    exactly as it always did (``"ssh *"``); a pipeline reads as its stages.
    """
    return " | ".join(scope_patterns(command, scope))


def scope_refusal(command: str, scope: str) -> str | None:
    """Why ``command`` may not be approved for ``scope``, or ``None``.

    Two independent rules, both of which have to hold: a command that
    escalates privilege is never approved in advance whatever the pattern
    would look like, and the resulting pattern still has to survive
    :func:`nvsh.approvals.refusal_reason` (no ``sudo``, no ``rm``, no bare
    ``*``). The run-once path is unaffected -- the operator can still press
    Enter and watch it happen.
    """
    from .approvals import command_refusal_reason, refusal_reason

    if _is_privileged(command):
        return "a command that escalates privilege is only ever run once, never pre-approved"
    # d24: every stage has to survive the policy, not just the first word --
    # and an opaque line (a subshell, `$(...)`) survives none of it.
    reason = command_refusal_reason(command)
    if reason is not None:
        return reason
    for pattern in scope_patterns(command, scope):
        refused = refusal_reason(pattern)
        if refused is not None:
            return refused
    return None


def _quoted(command: str, scope: str) -> str:
    """The patterns ``scope`` would store, each in single quotes."""
    return " ".join(f"'{pattern}'" for pattern in scope_patterns(command, scope))


def scope_patterns_text(command: str) -> dict[str, str]:
    """Per scope, the quoted patterns it would store -- what an ack names."""
    return {scope: _quoted(command, scope) for scope in _SCOPE_CHOICES}


def scope_descriptions(command: str) -> dict[str, str]:
    """How to describe, in one phrase each, what the four scope keys store.

    An operator reading the d15 keys took ``[u]`` to approve "that exact
    argument", when what it stores is ``<first word> *`` -- every argument
    of that command, forever. d24 is the same complaint from the Spark:
    ``ssh *`` offered for ``ssh orin "ps ... | head"`` is far too broad, so
    ``[S]``/``[U]`` keep the first argument. The phrases are computed from
    the same :func:`scope_patterns` that does the storing, and the details
    view quotes the same patterns.

    When the line has no second word the specific key would store exactly
    what its plain twin stores, and the phrase says so rather than
    repeating the pattern as though it were different.
    """
    from .approvals import stages

    per_stage = scope_stage_patterns(command)
    if len(stages(command)) > 1:
        # d26: the stages are numbered on their own line just above, so each
        # family names the pattern it would store *per stage*, in the same
        # order and separated the way the stages are -- `[u] 'ls *' | 'grep
        # *'` against `stages: 1 'ls ...'  2 'grep ...'`. `[s]` reads
        # `exact` because its per-stage pattern is the stage itself, which
        # the stages line already shows verbatim.
        return {
            APPROVE_SESSION: "exact",
            APPROVE_SESSION_SPECIFIC: " | ".join(per_stage[APPROVE_SESSION_SPECIFIC]),
            APPROVE_USER: " | ".join(per_stage[APPROVE_USER]),
            APPROVE_USER_SPECIFIC: " | ".join(per_stage[APPROVE_USER_SPECIFIC]),
        }
    described = {
        APPROVE_SESSION: "this exact line",
        APPROVE_SESSION_SPECIFIC: _quoted(command, APPROVE_SESSION_SPECIFIC),
        APPROVE_USER: _quoted(command, APPROVE_USER),
        APPROVE_USER_SPECIFIC: _quoted(command, APPROVE_USER_SPECIFIC),
    }
    for plain, specific, key in (
        (APPROVE_SESSION, APPROVE_SESSION_SPECIFIC, "s"),
        (APPROVE_USER, APPROVE_USER_SPECIFIC, "u"),
    ):
        if scope_patterns(command, specific) == scope_patterns(command, plain):
            described[specific] = f"same as [{key}] (no second word)"
    return described


def _decide_proposal(
    panel: Panel,
    proposal: Proposal,
    *,
    details: Mapping[str, str] | None = None,
    inject=None,
) -> str:
    """Ask the operator, re-showing the proposal after ``e``/``d``/``t``/a refusal.

    ``inject`` is how this proposal's conversation can be talked to: it
    takes one line of text, denies the pending dialog with it as the reason
    and puts the text into the same conversation (mid-turn where the
    backend has a channel, as the next request where it does not). It is
    ``None`` when there is no conversation to talk to, and then ``[t]`` and
    an empty-rationale ``[e]`` degrade to saying so rather than pretending.
    """

    def guard(scope: str) -> str | None:
        return scope_refusal(proposal.command, scope)

    scopes = scope_descriptions(proposal.command)
    patterns = scope_patterns_text(proposal.command)
    tokens = stage_tokens(proposal.command)
    per_stage = scope_stage_patterns(proposal.command)
    for _ in range(_MAX_PROPOSAL_ROUNDS):
        choice = panel.show_proposal(
            proposal,
            guard=guard,
            scopes=scopes,
            patterns=patterns,
            stages=tokens,
            stage_patterns=per_stage,
        )
        if choice == TELL:
            decided = _tell_decision(panel, inject)
        elif choice == EXPLAIN:
            decided = _explain_decision(panel, proposal, inject)
        elif choice == DETAILS:
            panel.detail_proposal(proposal, details)
            decided = None
        elif choice == REFUSED:
            # The panel already printed the guard's reason; ask again with
            # run-once still on the table.
            decided = None
        else:
            decided = choice
        if decided is not None:
            return decided
    return IGNORE


def _tell_decision(panel: Panel, inject) -> str | None:
    """``[t]``: one line at the ``nvsh> `` prompt, into this conversation (d16).

    Returns the decision, or ``None`` to re-show the proposal -- an empty
    line or a Ctrl+C at that prompt has decided nothing.
    """
    text = panel.read_tell()
    if not text:
        return None
    if inject is None:
        panel.note("nvsh: no conversation to steer; ignoring it instead")
        return IGNORE
    inject(text)
    return STEERED


def _explain_decision(panel: Panel, proposal: Proposal, inject) -> str | None:
    """``[e]``: the rationale, or the agent's own answer when there is none.

    Returns ``None`` (re-show the proposal) once the rationale has been
    printed. d17: never print "(no rationale given)" when the agent that
    made the proposal is right there and can be asked -- then the question
    goes into the conversation and the decision is ``STEERED``.
    """
    if proposal.rationale or inject is None:
        panel.explain_proposal(proposal)
        return None
    panel.note("... asking the agent why")
    inject(EXPLAIN_PROPOSAL_PROMPT.format(command=proposal.command))
    return STEERED


def backend_label(config) -> str:
    """``<harness>/<model>`` -- who the panel is about to wait on (d22).

    An operator watching a silent panel could not tell whether nvsh was
    talking to the local pi, a remote model, or nothing at all. The harness
    comes from the same :func:`nvsh.agent.registry.choose` the transport
    will use; the model from that harness's own config block (pi's
    ``[agents.pi] model``, openai-compat's ``model``). A harness with no
    configured model is named alone rather than with a guess.
    """
    try:
        from .agent import registry

        name, _reason = registry.choose(config)
    except Exception:  # noqa: BLE001 - the header must never block a diagnosis
        name = str(getattr(config, "agent_provider", "") or "")
    if not name:
        return ""
    try:
        model = str((config.agents.get(name) or {}).get("model") or "")
    except Exception:  # noqa: BLE001
        model = ""
    return f"{name}/{model}" if model else name


def _conversation_label(responder, env: Mapping[str, str]) -> str:
    """``daemon``/``one-shot`` plus the shell id, from what the client knows."""
    shell = _shell_pid(env)
    if getattr(responder, "agent", None) is not None:
        return f"one-shot, shell {shell}"
    if not client_transport.daemon_socket_path(env).exists():
        return f"one-shot, shell {shell}"
    return f"daemon, shell {shell}"


def proposal_details(
    proposal: Proposal,
    *,
    approvals,
    context: AgentContext,
    config,
    responder,
    env: Mapping[str, str],
) -> dict[str, str]:
    """Everything ``[d]`` should show and the panel cannot know (d18).

    The old details view printed the kind, the command and the rationale --
    all three already on screen in the box above it. What an operator
    actually needs before pressing a key is whether this command is already
    approved and under which pattern, the exact patterns ``[s]``/``[u]``
    would store (and why one of them may be refused), which backend and
    conversation proposed it, and how much of their terminal output went
    out with the question.
    """
    command = proposal.command
    scope, pattern = approvals.matches(command) if command else ("ask", None)
    approved = "no" if scope == "ask" or not pattern else f"{scope} pattern '{pattern}'"
    details = {
        "approved": approved,
        "session pattern": scope_pattern(command, APPROVE_SESSION),
        "session-specific pattern": scope_pattern(command, APPROVE_SESSION_SPECIFIC),
        "user pattern": scope_pattern(command, APPROVE_USER),
        "user-specific pattern": scope_pattern(command, APPROVE_USER_SPECIFIC),
    }
    refusals = [
        f"[{key}] {reason}"
        for key, scope_name in (
            ("s", APPROVE_SESSION),
            ("S", APPROVE_SESSION_SPECIFIC),
            ("u", APPROVE_USER),
            ("U", APPROVE_USER_SPECIFIC),
        )
        for reason in [scope_refusal(command, scope_name)]
        if reason
    ]
    if refusals:
        details["refused"] = "; ".join(refusals)
    details.update(_stage_details(command, approvals))
    details["backend"] = backend_label(config) or "unknown"
    details["conversation"] = _conversation_label(responder, env)
    sent = len(context.output.encode("utf-8", errors="replace")) if context.output else 0
    details["output"] = f"{sent} bytes of redacted output sent to the model"
    return details


def _stage_details(command: str, approvals) -> dict[str, str]:
    """One ``stage N`` row per stage for the ``[d]`` view (deviation d26).

    A multi-stage line is now approvable stage by stage, so the details
    view has to say, for each numbered stage, what the three pattern forms
    would store and who (if anyone) already approves it. A single-stage
    line adds nothing here: the four whole-line pattern rows above already
    say all of it.
    """
    from .approvals import command_refusal_reason, pattern_for, stages

    stage_list = stages(command)
    if len(stage_list) < 2:
        return {}
    rows: dict[str, str] = {}
    for number, stage in enumerate(stage_list, 1):
        reason = command_refusal_reason(stage)
        if reason:
            rows[f"stage {number}"] = f"{stage} -- not approvable ({reason})"
            continue
        scope, pattern = approvals.match_stage(stage)
        approver = "none" if scope == "ask" or not pattern else f"{scope} '{pattern}'"
        rows[f"stage {number}"] = (
            f"exact '{pattern_for(stage, APPROVE_SESSION)}'  "
            f"specific '{pattern_for(stage, APPROVE_USER_SPECIFIC)}'  "
            f"broad '{pattern_for(stage, APPROVE_USER)}'  "
            f"approved-by {approver}"
        )
    return rows


def _proposal_handler(
    panel: Panel,
    *,
    approvals,
    inspections: list[tuple[str, RunResult]],
    responder,
    audit,
    steers: list[str] | None = None,
    context: AgentContext | None = None,
    config=None,
    env: Mapping[str, str] | None = None,
    turn: "_Turn | None" = None,
):
    """Answer one proposal, sending the answer wherever the dialog came from.

    ``responder`` is the d11 fix: a proposal carrying a ``request_id`` was
    raised by the backend and only that backend can unblock the turn. The
    handler no longer assumes a daemon is listening -- it hands the answer
    to :class:`nvsh.client_transport.Responder`, which is pointed at the
    daemon socket or at the one-shot, in-process agent as appropriate.
    """

    resolved_env = dict(os.environ if env is None else env)

    def handle(proposal: Proposal, event: AgentEvent) -> None:
        request_id = (event.args or {}).get("request_id")
        if audit is not None:
            audit.record(event="proposal", proposal=proposal)

        if _auto_inspected(
            panel,
            proposal,
            approvals=approvals,
            inspections=inspections,
            responder=responder,
            audit=audit,
            request_id=request_id,
        ):
            return

        inject = _injector(panel, responder, steers, request_id)
        details = None
        if context is not None:
            details = proposal_details(
                proposal,
                approvals=approvals,
                context=context,
                config=config,
                responder=responder,
                env=resolved_env,
            )
        choice = _decide_proposal(panel, proposal, details=details, inject=inject)
        if audit is not None:
            audit.record(event="decision", proposal=proposal, decision=choice)
        if choice == IGNORE and turn is not None:
            # Esc/ignore at a proposal declines the agent (c9): exit 3.
            turn.decline("ignored proposal")
        _apply_decision(
            panel,
            proposal,
            choice,
            approvals=approvals,
            responder=responder,
            audit=audit,
            request_id=request_id,
        )

    return handle


def _auto_inspected(
    panel: Panel,
    proposal: Proposal,
    *,
    approvals,
    inspections: list[tuple[str, RunResult]],
    responder,
    audit,
    request_id,
) -> bool:
    """Run an already-approved, unprivileged inspection without asking.

    Returns ``True`` when it did (the proposal is then fully handled: the
    output is queued for the follow-up prompt and the backend's dialog, if
    there was one, is answered ``once``), ``False`` when the operator still
    has to decide.
    """
    command = proposal.command
    auto = (
        proposal.kind is ProposalKind.INSPECT
        and not _is_privileged(command)
        and approvals.decide(command) in ("user", "session")
    )
    if not auto:
        return False
    panel.running(command)
    result = _run_approved(proposal, AUTO_INSPECT, timeout=INSPECT_TIMEOUT)
    panel.finished(result.exit_code)
    inspections.append((command, result))
    if audit is not None:
        audit.record(event="decision", proposal=proposal, decision=AUTO_INSPECT)
        audit.record(event="outcome", proposal=proposal, outcome=result.exit_code)
    if request_id:
        responder.respond(request_id, {"value": "once"})
    return True


def _injector(panel: Panel, responder, steers: list[str] | None, request_id):
    """Build the ``inject`` callback ``_decide_proposal`` steers with.

    The steer goes out first and the deny second, in that order for two
    reasons: while the dialog is open the backend is provably still
    streaming (so a mid-turn steer is accepted rather than starting a fresh
    turn), and the deny is an unacknowledged write -- writing it first
    would leave a command in flight behind an un-acked one, which is
    exactly what d14 forbids.

    The reason rides on the deny as well as in the steer. pi 0.85.1 reduces
    an ``extension_ui_response`` to its ``value`` before the extension sees
    it (``docs/pi-rpc.md``), so today only the steer reaches the model --
    but a backend that does pass the whole response gets the operator's
    words as the block reason for free.
    """

    def inject(text: str) -> None:
        """Put ``text`` into *this* conversation (deviation d16)."""
        delivered = responder.steer(text)
        if request_id:
            responder.respond(request_id, {"value": "deny", "reason": text})
        if delivered:
            panel.note("nvsh: steering the agent ...")
        elif steers is not None:
            steers.append(text)
            panel.note("nvsh: no mid-turn channel; asking next instead ...")
        else:
            panel.note("nvsh: the agent could not be steered")

    return inject


def _apply_decision(
    panel: Panel,
    proposal: Proposal,
    choice: str,
    *,
    approvals,
    responder,
    audit,
    request_id,
) -> None:
    """Act on the operator's answer: relay it, run it, or say it was not run."""
    command = proposal.command
    if choice == STEERED:
        # The dialog is already answered and the conversation carries the
        # operator's words; nothing to run and nothing more to say.
        return
    if choice not in _RUN_CHOICES:
        if request_id:
            responder.respond(request_id, {"value": "deny"})
        panel.note("nvsh: not run")
        return
    if request_id:
        # The backend owns execution (pi's approval extension); nvsh only
        # relays the operator's answer, and must not run it a second time.
        # "session"/"user" are the extension's own choice tokens, so the
        # widening and the store write happen there, once.
        answer = "once" if choice == APPROVE else choice
        if choice in _SCOPE_CHOICES:
            answer = encode_choice(choice, getattr(panel, "stage_choice", None), command)
        responder.respond(request_id, {"value": answer})
        return
    if choice in _SCOPE_CHOICES:
        # No dialog: this adapter (openai-compat and friends) has nvsh run
        # the command itself, so nvsh also owns the store write. The guard
        # in _decide_proposal has already cleared the pattern, but add()
        # is the authority and may still refuse -- a refusal must cost the
        # operator the approval, never the command they asked for.
        _approve_scope(panel, approvals, command, choice, getattr(panel, "stage_choice", None))
    result = _run_approved(proposal, choice)
    if audit is not None:
        audit.record(event="outcome", proposal=proposal, decision=choice, outcome=result.exit_code)
    _print_run(panel, command, result)


def encode_choice(choice: str, chosen: Sequence[int] | None, command: str) -> str:
    """The scope answer a backend dialog receives, stages included (d26).

    pi's ``ctx.ui.select`` carries exactly one value string back to the
    extension (``docs/pi-rpc.md``: pi 0.85.1 reduces an
    ``extension_ui_response`` to its ``value``), and there is no second
    field to put a stage list in. So a *partial* pick rides in the value
    itself as ``<scope>:<comma-separated stages>`` --
    ``session-specific:1,2`` -- which ``nvsh/agent/pi_ext/approval.ts``
    splits on the first colon and forwards as
    ``nvsh approve add <cmd> --scope <scope> --stages 1,2``. A pick that
    covers every stage stays the bare token every pre-d26 reader expects.
    """
    from .approvals import stages

    if not chosen or len(chosen) >= len(stages(command)):
        return choice
    return f"{choice}:{','.join(str(n) for n in chosen)}"


def _approve_scope(
    panel: Panel, approvals, command: str, scope: str, chosen: Sequence[int] | None = None
) -> None:
    """Persist the operator's scope-key choice into the approval store.

    d24: one pattern *per stage* of the command line, so approving
    ``ps ... | head ...`` approves ``ps *`` and ``head *`` and nothing
    wider. A refusal on any stage costs the operator the whole approval
    (never the command they asked for, which still runs once).

    d26: ``chosen`` narrows that to the stages the operator picked at the
    ``stages [all,1,2]: `` prompt. Only those stages reach the store -- the
    command itself still runs once either way, because the keypress
    approved *this* execution and the store write is about future turns.
    """
    from .approvals import ApprovalError, base_scope

    where = base_scope(scope)
    patterns = scope_patterns(command, scope, chosen)
    if not patterns:
        panel.note(f"nvsh: nothing to approve for this {where}")
        return
    for pattern in patterns:
        try:
            approvals.add(pattern, scope=scope)
        except ApprovalError as exc:
            panel.note(f"nvsh: not approved for this {where}: {exc}")
            return
    if where == "user":
        try:
            approvals.save()
        except OSError as exc:  # pragma: no cover - unwritable config dir
            panel.note(f"nvsh: could not save the approval: {exc}")
            return
    panel.note(f"nvsh: approved for this {where}: {' '.join(patterns)}")


def _print_run(panel: Panel, command: str, result: RunResult) -> None:
    if result.stdout:
        panel.write(result.stdout)
    if result.stderr:
        panel.write(result.stderr)
    panel.note(f"nvsh: {command} -> exit {result.exit_code}")


def _target_path(target: Target) -> str:
    """The protocol path (``acp``, ``stream-json``, ...) *target* speaks."""
    try:
        from .agent import registry

        return registry.ADAPTERS[target.backend].path
    except Exception:  # noqa: BLE001 - an unregistered backend has no known path
        return ""


class _TargetedAudit:
    """An audit log that stamps every entry with the resolved target (t16).

    A thin wrapper rather than a ``target=`` keyword threaded through
    ``_proposal_handler``, ``_auto_inspected`` and ``_apply_decision``: the
    answer is the same for every line of one turn, and a call site that
    forgets to pass it is exactly the drift this is meant to prevent.
    """

    def __init__(self, audit, target: Target | None) -> None:
        self._audit = audit
        self._target = target

    def record(self, *args, **kwargs):
        kwargs.setdefault("target", self._target)
        return self._audit.record(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._audit, name)


def _targeted_audit(audit, target: Target | None):
    if audit is None or target is None:
        return audit
    return _TargetedAudit(audit, target)


class _StopTarget:
    """Where the panel's two presses go for one request (t16).

    The first press calls :meth:`cancel`, the second :meth:`force_stop`.
    Which process they reach is decided *at press time*, not up front: the
    request starts out aimed at the daemon, but :func:`client_transport.one_shot`
    -- whether chosen outright or fallen back to mid-call -- binds the
    in-process adapter onto ``responder``. That adapter's own ``cancel()`` /
    ``force_stop()`` is then the only thing that can stop it, since the
    harnesses that spawn in their own session no longer see the terminal's
    SIGINT. With no bound agent the daemon gets ``cancel`` then ``kill``.
    The first press also marks ``responder.stopping``, so a one-shot turn
    that winds down politely is still torn down with ``force_stop()`` and
    leaves no harness grandchild behind.
    Both are called from the panel's main thread while ``run()`` may be
    blocked on the feeder thread, which every adapter's stop methods allow.
    """

    def __init__(self, responder, *, shell_id: int, env: Mapping[str, str]) -> None:
        self._responder = responder
        self._shell_id = shell_id
        self._env = env

    def cancel(self) -> object:
        self._responder.stopping = True
        agent = self._responder.agent
        if agent is not None:
            return agent.cancel()
        return client_transport.cancel(shell_id=self._shell_id, env=self._env)

    def force_stop(self) -> object:
        agent = self._responder.agent
        if agent is not None:
            return agent.force_stop()
        return client_transport.kill(shell_id=self._shell_id, env=self._env)


class _Turn:
    """One request's stop lifecycle: its audit trail and whether it was declined (t17).

    ``declined`` is shared across a turn and its follow-up (the caller owns
    the list), so any decline in either makes the entry point exit
    :data:`~nvsh.cli._errors.EXIT_DECLINED`. ``ended`` asks the event source
    to stop yielding -- set when a busy exit could not reach the daemon, so
    the request would otherwise sit in the daemon's queue with the operator
    already gone.
    """

    def __init__(
        self, audit, *, shell_id: int, target: Target | None, declined: list[str] | None
    ) -> None:
        self._audit = audit
        self._shell_id = shell_id
        self._target = target
        self._started = time.monotonic()
        self.declined = declined if declined is not None else []
        self.ended = False

    def elapsed(self) -> float:
        return round(time.monotonic() - self._started, 3)

    def record(self, kind: str, outcome: object, elapsed: float | None = None) -> None:
        """Append one ``stop`` audit line; an unwritable log never breaks the stop."""
        if self._audit is None:
            return
        try:
            self._audit.record_stop(
                kind,
                self._shell_id,
                self._target,
                self.elapsed() if elapsed is None else elapsed,
                outcome,
            )
        except OSError:  # pragma: no cover - state dir vanished mid-turn
            pass

    def decline(self, outcome: str) -> None:
        self.declined.append(outcome)
        self.record("declined", outcome)

    def audited(self, kind: str, action):
        """Wrap one stop press so it is recorded exactly once, with its result."""

        def call() -> object:
            try:
                result = action()
            except Exception:
                self.record(kind, "error")
                raise
            self.record(kind, "failed" if result is False else "sent")
            return result

        return call

    def until_ended(self, events: Iterable[AgentEvent]):
        iterator = iter(events)
        try:
            for event in iterator:
                yield event
                if self.ended:
                    return
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


#: The daemon's answer to one busy choice, as the audit line names it.
_BUSY_OUTCOME = {True: "accepted", False: "no busy prompt open"}


def _busy_choice_within(choice: str, timeout: float, *, shell_id: int, env) -> bool:
    """Send one busy choice; ``False`` if refused or not answered within ``timeout``."""
    answer: list[bool] = []

    def send() -> None:
        try:
            answer.append(bool(client_transport.busy_choice(choice, shell_id=shell_id, env=env)))
        except Exception:  # noqa: BLE001 - an unreachable daemon is a refusal
            answer.append(False)

    worker = threading.Thread(target=send, daemon=True, name="nvsh-busy-choice")
    worker.start()
    worker.join(timeout)
    return bool(answer and answer[0])


def _busy_handler(panel: Panel, turn: _Turn, *, shell_id: int, env: Mapping[str, str]):
    """Answer the daemon's ``busy`` prompt: steer, replace or exit (t17).

    The daemon sends it when this shell already owns a running turn (or the
    owner is gone). Steer is offered only when the daemon says the harness
    has a mid-turn channel; the steer counts as heard once the daemon
    accepts it, and when it does not within the panel's silence window the
    panel re-offers replace/exit. A choice the daemon no longer takes (its
    busy prompt timed out into the queue) is said so, never raised.
    """

    def handle(event: AgentEvent) -> None:
        args = event.args or {}
        owner = str(args.get("owner") or "agent")
        try:
            elapsed = float(args.get("elapsed") or 0.0)
        except (TypeError, ValueError):
            elapsed = 0.0
        choices = args.get("choices")
        steerable = bool(args.get("steerable")) and (
            not isinstance(choices, list) or "steer" in choices
        )

        def await_steer(timeout: float) -> bool:
            heard = _busy_choice_within("steer", timeout, shell_id=shell_id, env=env)
            turn.record("steer", "accepted" if heard else "silent", elapsed)
            return heard

        choice = panel.show_busy(
            owner, elapsed, steerable, await_event=await_steer if steerable else None
        )
        if choice == STEER:
            return
        wire = "replace" if choice == REPLACE else "exit"
        accepted = _busy_choice_within(wire, 10.0, shell_id=shell_id, env=env)
        if choice == REPLACE:
            turn.record("replace", _BUSY_OUTCOME[accepted], elapsed)
            if not accepted:
                panel.note("nvsh: the busy prompt already closed; this request is queued")
            return
        # Exit: leave the running turn alone and decline this request.
        turn.record("busy_exit", _BUSY_OUTCOME[accepted], elapsed)
        turn.decline("busy exit")
        if not accepted:
            turn.ended = True

    return handle


def _stream_request(
    panel: Panel,
    request: AgentRequest,
    context: AgentContext,
    *,
    env: Mapping[str, str],
    shell_id: int,
    config,
    approvals=None,
    inspections: list[tuple[str, RunResult]] | None = None,
    audit=None,
    steers: list[str] | None = None,
    one_shot: bool = False,
    target: Target | None = None,
    declined: list[str] | None = None,
) -> StreamResult:
    # One resolution, used three ways: the adapter routes on it, the panel
    # header names it, and every audit line is stamped with it (t16/t18).
    # ``target=None`` means the operator named nothing, so the request goes
    # to whatever ``default`` resolves to -- which is also what the warm
    # daemon session holds (decision c25).
    warm = not one_shot and target is None
    resolved = target if target is not None else default_target(config)
    if resolved is not None:
        panel.set_target(resolved, _target_path(resolved), warm)
    if target is not None:
        request = replace(request, target=target)
    turn = _Turn(audit, shell_id=shell_id, target=resolved, declined=declined)
    audit = _targeted_audit(audit, resolved)
    responder = client_transport.Responder(shell_id=shell_id, env=env)
    on_proposal = None
    if approvals is not None and inspections is not None:
        on_proposal = _proposal_handler(
            panel,
            approvals=approvals,
            inspections=inspections,
            responder=responder,
            audit=audit,
            steers=steers,
            context=context,
            config=config,
            env=env,
            turn=turn,
        )
    stop = _StopTarget(responder, shell_id=shell_id, env=env)
    return panel.stream(
        turn.until_ended(
            _send(
                request,
                context,
                env=env,
                shell_id=shell_id,
                config=config,
                responder=responder,
                one_shot=one_shot,
            )
        ),
        on_proposal=on_proposal,
        cancel=turn.audited("cancel", stop.cancel),
        force_stop=turn.audited("force_kill", stop.force_stop),
        on_busy=_busy_handler(panel, turn, shell_id=shell_id, env=env),
    )


def _follow_up_prompt(inspections: list[tuple[str, RunResult]], steers: list[str]) -> str:
    """The one prompt that carries everything the last turn produced back.

    An inspection's output and the operator's steer can both come out of a
    single turn (``[t]`` after an auto-run inspector), and they belong in
    one follow-up: two requests would be two turns, and the second would
    have lost the first's answer.
    """
    parts = []
    if inspections:
        parts.append(_inspection_prompt(inspections))
    parts.extend(steers)
    return "\n\n".join(parts)


def _inspection_prompt(inspections: list[tuple[str, RunResult]]) -> str:
    blocks = ["Results of the inspection commands you proposed:"]
    for command, result in inspections:
        body = _redact((result.stdout + result.stderr).strip())
        blocks.append(f"$ {command}\nexit {result.exit_code}\n{body}")
    blocks.append("Use these results to finish the diagnosis and propose a fix.")
    return "\n\n".join(blocks)


def _load_approvals():
    from .approvals import ApprovalError, Approvals

    try:
        return Approvals.load()
    except ApprovalError:
        return Approvals.default()


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def _panel_for(panel: Panel | None, env: Mapping[str, str]) -> Panel:
    return panel if panel is not None else Panel(env=env)


def _failed_line(state: Mapping[str, Any]) -> tuple[str, int]:
    """The failed line and its status, as :func:`prose_request` wants them.

    The state file is written by the hook and read back as JSON, so both
    fields may be missing, ``null`` or the wrong type; neither may reach
    the classifier as anything but a string and an int.
    """
    return str(state.get("line", "") or ""), int(state.get("exit", 0) or 0)


def _apply_agent_mark(panel: Panel, config, marked) -> tuple[Any, Target | None]:
    """Apply an ``@name`` mark's one-request target override (d23/c25).

    Returns ``(config, target)``; ``target`` is ``None`` when the operator
    marked nothing, which is what keeps the request on the warm daemon
    session. A refused override (an unknown or uninstalled harness) returns
    ``(None, None)`` having printed the one-line refusal: the caller must
    then do nothing else at all.
    """
    if marked is None or not marked.agent:
        return config, None
    config, target, refusal = agent_override(config, marked.agent)
    if config is None:
        panel.line(refusal)
        return None, None
    return config, target


def _opening_request(panel: Panel, state: Mapping[str, Any], marked, config) -> AgentRequest:
    """Print the panel's first line and build the turn's first request.

    A marked or guessed question (d20/d23) goes out under the "asking"
    header carrying the operator's own words; anything else is an ordinary
    failure under the "failed (exit N)" header.
    """
    label = backend_label(config)
    if marked is not None:
        panel.header(state["line"], state["exit"], backend_label=label, ask=marked.question)
        return _prose_request(state, marked.question)
    panel.header(state["line"], state["exit"], backend_label=label)
    return _failure_request(state)


def handle_failure(
    args: Any, *, panel: Panel | None = None, env: Mapping[str, str] | None = None
) -> int:
    """Diagnose one failed command. Returns the exit code ``nvsh hook`` reports.

    ``0`` in every normal case (the operator's own command already reported
    its status); ``130`` when the operator pressed Ctrl+C during the stream;
    :data:`~nvsh.cli._errors.EXIT_DECLINED` (3) when the operator declined the
    agent -- exit at a busy prompt, or Esc/ignore at a proposal (t17).
    """
    resolved = dict(os.environ if env is None else env)
    panel = _panel_for(panel, resolved)
    config = _load_config()

    state = save_last_failure(args, env=resolved)

    # A sentence typed at the prompt ("what are the memory levels?") is a
    # question, not a failed command: bash's "command not found" is an
    # artefact of where it was typed, and diagnosing `what` helps nobody.
    # It goes to the agent as an explicit request carrying the operator's
    # own words, under its own header (deviation d20). A line the operator
    # *marked* -- `? ...` or `@name ...` (d23) -- says so outright, and like
    # Ctrl+G it is an explicit call: it is never held back by the auto-call
    # window, and never consumes it either. Only an unmarked line (an
    # ordinary failure, or the d20 guess) is rate-limited, so the check has
    # to come after the classification rather than before it.
    line, exit_code = _failed_line(state)
    marked = prose_request(line, exit_code)
    explicit = marked is not None and marked.explicit
    if not explicit and _rate_limited(args, config, resolved):
        _note_held_back(config, resolved, time.time())
        return 0

    config, target = _apply_agent_mark(panel, config, marked)
    if config is None:
        return 0
    # c25: a named target never rides the warm daemon session -- that
    # session belongs to ``default``, and answering `@claude/opus` out of it
    # would quietly answer from the default model instead.
    one_shot = _is_one_shot(target)

    shell_id = _shell_pid(resolved)
    context = build_context(args, resolved)

    request = _opening_request(panel, state, marked, config)
    approvals = _load_approvals()
    inspections: list[tuple[str, RunResult]] = []
    steers: list[str] = []
    declined: list[str] = []
    audit = _audit(resolved)
    result = _stream_request(
        panel,
        request,
        context,
        env=resolved,
        shell_id=shell_id,
        config=config,
        approvals=approvals,
        inspections=inspections,
        audit=audit,
        steers=steers,
        one_shot=one_shot,
        target=target,
        declined=declined,
    )
    if result.interrupted:
        return 130

    if inspections or steers:
        follow_up = _failure_request(state, prompt=_follow_up_prompt(inspections, steers))
        follow = _stream_request(
            panel,
            follow_up,
            context,
            env=resolved,
            shell_id=shell_id,
            config=config,
            approvals=approvals,
            inspections=[],
            audit=audit,
            one_shot=one_shot,
            target=target,
            declined=declined,
        )
        if follow.interrupted:
            return 130
    return EXIT_DECLINED if declined else 0


def ask(
    prompt: str,
    *,
    draft: str | None = None,
    panel: Panel | None = None,
    env: Mapping[str, str] | None = None,
    kind: RequestKind = RequestKind.EXPLICIT,
    agent: str | None = None,
) -> int:
    """``/ask`` and ``Ctrl+G``: a free-form question with the machine's context.

    ``kind`` defaults to :attr:`RequestKind.EXPLICIT` (a direct call, e.g.
    ``Ctrl+G``); :mod:`nvsh.slash` passes :attr:`RequestKind.SLASH` when the
    operator typed ``/ask`` at the prompt, so the two entry points stay
    distinguishable on the wire without duplicating this function.

    ``agent`` names one harness to answer *this* request (``/ask --agent
    qwen ...``, which is what the ``@qwen`` mark is rewritten to). An
    unavailable one is a single refusal line and exit 1 -- never a silent
    fall back to the default (deviation d23).

    Returns ``130`` after Ctrl+C/Esc and :data:`~nvsh.cli._errors.EXIT_DECLINED`
    when the operator declined the agent (t17). A proposal is answered on
    the panel like the hook's, so ignoring one is a decline too.
    """
    resolved = dict(os.environ if env is None else env)
    panel = _panel_for(panel, resolved)
    config = _load_config()
    target: Target | None = None
    if agent:
        config, target, refusal = agent_override(config, agent)
        if config is None:
            panel.line(refusal)
            return 1
    text = prompt
    if draft:
        text = f"{prompt}\n\nThe operator was in the middle of typing: {draft}"
    question = (prompt or "").strip()
    if question:
        # Same first line as the hook's question path (d20/d22), so `? ...`
        # reads the same whichever route carried it.
        panel.header("", 0, backend_label=backend_label(config), ask=question)
    state = load_last_failure(resolved) or {}
    args = _args_from_state(state) if state else _args_from_state({"cwd": os.getcwd()})
    request = AgentRequest(
        kind=kind,
        prompt=text,
        command=str(state.get("line", "") or ""),
        failure_id=str(state.get("failure_id", "") or ""),
        ask=question,
    )
    context = build_context(args, resolved)
    shell_id = _shell_pid(resolved)
    approvals = _load_approvals()
    inspections: list[tuple[str, RunResult]] = []
    steers: list[str] = []
    declined: list[str] = []
    audit = _audit(resolved)
    result = _stream_request(
        panel,
        request,
        context,
        env=resolved,
        shell_id=shell_id,
        config=config,
        approvals=approvals,
        inspections=inspections,
        audit=audit,
        steers=steers,
        one_shot=_is_one_shot(target),
        target=target,
        declined=declined,
    )
    if result.interrupted:
        return 130
    if inspections or steers:
        follow = _stream_request(
            panel,
            replace(request, prompt=_follow_up_prompt(inspections, steers)),
            context,
            env=resolved,
            shell_id=shell_id,
            config=config,
            approvals=approvals,
            inspections=[],
            audit=audit,
            one_shot=_is_one_shot(target),
            target=target,
            declined=declined,
        )
        if follow.interrupted:
            return 130
    return EXIT_DECLINED if declined else 0


def _on_last_failure(prompt: str, panel: Panel | None, env: Mapping[str, str] | None) -> int:
    resolved = dict(os.environ if env is None else env)
    panel = _panel_for(panel, resolved)
    state = load_last_failure(resolved)
    if not state:
        panel.line("nvsh: no recorded failure yet")
        return 1
    config = _load_config()
    context = build_context(_args_from_state(state), resolved)
    shell_id = _shell_pid(resolved)
    approvals = _load_approvals()
    audit = _audit(resolved)
    steers: list[str] = []
    declined: list[str] = []
    result = _stream_request(
        panel,
        _failure_request(state, prompt=prompt),
        context,
        env=resolved,
        shell_id=shell_id,
        config=config,
        approvals=approvals,
        inspections=[],
        audit=audit,
        steers=steers,
        declined=declined,
    )
    if result.interrupted:
        return 130
    if steers:
        follow = _stream_request(
            panel,
            _failure_request(state, prompt=_follow_up_prompt([], steers)),
            context,
            env=resolved,
            shell_id=shell_id,
            config=config,
            approvals=approvals,
            inspections=[],
            audit=audit,
            declined=declined,
        )
        if follow.interrupted:
            return 130
    return EXIT_DECLINED if declined else 0


def steer(text: str, *, panel: Panel | None = None, env: Mapping[str, str] | None = None) -> int:
    """``/steer <text>``: tell the agent something without a proposal on screen.

    The same move the proposal prompt's ``[t]`` makes, from the shell
    prompt instead: if a turn is running (this shell's, in the daemon) the
    text is injected into it mid-turn and this returns at once; otherwise
    it becomes the next request in the same conversation, carrying the last
    recorded failure as its context, and the answer streams into the panel
    as usual (deviation d16).
    """
    resolved = dict(os.environ if env is None else env)
    panel = _panel_for(panel, resolved)
    message = (text or "").strip()
    if not message:
        panel.line("nvsh: /steer <text> (tell the agent what to do instead)")
        return 1
    if client_transport.steer(message, shell_id=_shell_pid(resolved), env=resolved):
        panel.note("nvsh: steered the running turn")
        return 0
    return _on_last_failure(message, panel, resolved)


def fix(*, panel: Panel | None = None, env: Mapping[str, str] | None = None) -> int:
    """``/fix``: ask for the smallest fix for the last recorded failure."""
    return _on_last_failure(FIX_PROMPT, panel, env)


def explain(*, panel: Panel | None = None, env: Mapping[str, str] | None = None) -> int:
    """``/explain``: ask what the last failure means on this machine."""
    return _on_last_failure(EXPLAIN_PROMPT, panel, env)


def verify(command: str, exit_code: int, panel: Panel) -> int:
    """Report the status of a re-run. Returns ``exit_code`` unchanged."""
    verdict = "worked" if exit_code == 0 else "still failing"
    panel.line(f"nvsh: verify: {command} -> exit {exit_code} ({verdict})")
    return exit_code


def _run_in(directory: str, action):
    """Run ``action()`` with the process cwd set to ``directory``.

    An empty ``directory`` (an old state file with no recorded cwd) runs
    ``action`` where we are. The previous directory is always restored, so a
    failed run never leaves the client somewhere unexpected. This client is
    a short-lived, single-threaded process: one chdir and back is honest
    here, and it keeps the ``_run_command`` argv contract untouched.
    """
    if not directory:
        return action()
    previous = os.getcwd()
    os.chdir(directory)
    try:
        return action()
    finally:
        with contextlib.suppress(OSError):
            os.chdir(previous)


def retry(
    last: Mapping[str, Any] | None = None,
    *,
    panel: Panel | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """``/retry``: re-run the failed line, but only after the operator's Enter.

    The command is shown exactly as it was typed, confirmed through the
    panel, re-run with ``bash -lc`` (so the login environment matches an
    interactive shell) **in the directory it failed in**, and its new status
    is reported by :func:`verify`. Re-running a command with relative paths
    somewhere else is a different command: if the operator has since ``cd``'d
    and the recorded directory is gone or unusable, nvsh says so and runs
    nothing rather than guessing.
    """
    resolved = dict(os.environ if env is None else env)
    panel = _panel_for(panel, resolved)
    state = last if last is not None else load_last_failure(resolved)
    if not state or not state.get("line"):
        panel.line("nvsh: no recorded failure to retry")
        return 1
    command = str(state["line"])
    recorded_cwd = str(state.get("cwd") or "")
    if recorded_cwd and not os.path.isdir(recorded_cwd):
        panel.line(f"nvsh: not re-run: the directory it failed in is gone ({recorded_cwd})")
        return 1
    proposal = Proposal(
        command=command,
        rationale=f"re-run the command that failed (exit {state.get('exit')})",
        kind=ProposalKind.RETRY,
    )
    choice = _decide_proposal(panel, proposal)
    if choice not in _RUN_CHOICES:
        panel.note("nvsh: not re-run")
        return 0
    if choice in _SCOPE_CHOICES:
        _approve_scope(
            panel, _load_approvals(), command, choice, getattr(panel, "stage_choice", None)
        )
    try:
        result = _run_in(recorded_cwd, lambda: _run_approved(proposal, choice, login=True))
    except OSError as exc:
        panel.line(f"nvsh: not re-run: cannot enter the directory it failed in ({exc})")
        return 1
    if result.stdout:
        panel.write(result.stdout)
    if result.stderr:
        panel.write(result.stderr)
    return verify(command, result.exit_code, panel)


def context_show(
    json_mode: bool = False,
    *,
    env: Mapping[str, str] | None = None,
    out=None,
) -> int:
    """``nvsh context --show``: print exactly the bytes that would be sent.

    The bytes are the system brief
    (:func:`nvsh.agent.prompt.build_system_prompt` -- who the agent is, the
    rules it works under, and the playbook for the detected platform) and
    then the prompt text :func:`nvsh.agent.pi.build_prompt` builds from the
    request and context -- the same two functions every adapter's ``run()``
    feeds its backend, one through a system-prompt channel and one as the
    turn's prompt -- so what is printed is what the model sees, not a
    summary of it. Both halves are already redacted: the brief is a
    constant plus the detected platform block, and the failure context went
    through :mod:`nvsh.redact` in :func:`build_context`.
    """
    from .agent.pi import build_prompt
    from .agent.prompt import build_full_prompt, build_system_prompt
    from .cli._output import emit_result

    resolved = dict(os.environ if env is None else env)
    state = load_last_failure(resolved) or {}
    args = _args_from_state(state)
    context = build_context(args, resolved)
    request = _failure_request(state) if state else _failure_request({})
    prompt = build_prompt(request, context)
    system_prompt = build_system_prompt(context)
    if json_mode:
        emit_result(
            {
                "command": request.command,
                "exit_code": request.exit_code,
                "failure_id": request.failure_id,
                "platform": context.platform,
                "cwd": context.cwd,
                "shell_pid": context.shell_pid,
                "output": context.output,
                "redaction_rules": list(context.redaction_report),
                "system_prompt": system_prompt,
                "prompt": prompt,
            },
            json_mode=True,
            stream=out,
        )
    else:
        emit_result(build_full_prompt(request, context), json_mode=False, stream=out)
    return 0


def handle_slash(
    line: str,
    draft: str | None = None,
    *,
    panel: Panel | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Route one ``/verb ...`` line to the right entry point.

    ``nvsh slash`` (task t14's registry) calls this with the operator's
    original line, ``/`` included.
    """
    resolved = dict(os.environ if env is None else env)
    text = (line or "").strip()
    if text.startswith("/"):
        text = text[1:]
    verb, _, rest = text.partition(" ")
    rest = rest.strip()
    verb = verb.lower()

    if verb == "ask":
        return ask(rest, draft=draft, panel=panel, env=resolved)
    if verb == "fix":
        return fix(panel=panel, env=resolved)
    if verb == "explain":
        return explain(panel=panel, env=resolved)
    if verb == "retry":
        return retry(panel=panel, env=resolved)
    if verb == "steer":
        return steer(rest, panel=panel, env=resolved)
    if verb == "context":
        return context_show(json_mode=False, env=resolved)

    _panel_for(panel, resolved).line(
        f"nvsh: unknown slash command '/{verb}' "
        "(try /ask, /fix, /explain, /retry, /steer, /context)"
    )
    return 1
