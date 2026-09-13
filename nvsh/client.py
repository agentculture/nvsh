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
   pressed Enter on that exact command. A ``sudo`` command is never
   auto-run, whatever the approval patterns say.

Nothing here writes into readline's buffer: nvsh never pre-types a command
into the operator's shell, it only ever prints one.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess  # nosec B404 - argv is always ["bash", "-c", <approved command>]
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import client_transport
from .agent.base import AgentContext, AgentEvent, AgentRequest, Proposal, ProposalKind, RequestKind
from .panel import APPROVE, DETAILS, EXPLAIN, Panel, StreamResult

#: How long an auto-run read-only inspector may take before it is killed.
INSPECT_TIMEOUT = 10.0

#: How many times the panel re-asks after ``e``/``d`` before giving up.
_MAX_PROPOSAL_ROUNDS = 4

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


def _save_rate_state(state, env: Mapping[str, str] | None) -> None:
    _write_private_json(rate_state_path(env), {"last_auto_call": state.last_auto_call})


def _rate_window(config) -> float:
    from .triggers import DEFAULT_RATE_WINDOW

    raw = (config.triggers or {}).get("rate_window_seconds")
    try:
        return float(raw) if raw is not None else DEFAULT_RATE_WINDOW
    except (TypeError, ValueError):
        return DEFAULT_RATE_WINDOW


def _rate_limited(args: Any, config, env: Mapping[str, str] | None) -> bool:
    """Re-run the trigger decision against the *persisted* rate state."""
    from .triggers import TriggerEvent, decide

    try:
        pipestatus = tuple(int(x) for x in str(getattr(args, "pipestatus", "") or "").split())
    except ValueError:
        pipestatus = ()
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


def _run_command(command: str, timeout: float | None = None, login: bool = False) -> RunResult:
    """Run ``command`` through bash, capturing its output.

    The argv is always ``["bash", "-c"/"-lc", command]`` -- never
    ``shell=True``, never an environment variable, and never anything the
    operator has not approved.
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
) -> Iterable[AgentEvent]:
    return client_transport.send(
        request,
        context,
        shell_id=shell_id,
        env=env,
        config=config,
        autostart=_autostart(env),
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


def _decide_proposal(panel: Panel, proposal: Proposal) -> str:
    """Ask the operator, re-showing the proposal after ``e``/``d``."""
    for _ in range(_MAX_PROPOSAL_ROUNDS):
        choice = panel.show_proposal(proposal)
        if choice == EXPLAIN:
            panel.explain_proposal(proposal)
            continue
        if choice == DETAILS:
            panel.detail_proposal(proposal)
            continue
        return choice
    return "ignore"


def _proposal_handler(
    panel: Panel,
    *,
    approvals,
    inspections: list[tuple[str, RunResult]],
    env: Mapping[str, str],
    shell_id: int,
    audit,
):
    def handle(proposal: Proposal, event: AgentEvent) -> None:
        command = proposal.command
        request_id = (event.args or {}).get("request_id")
        if audit is not None:
            audit.record(event="proposal", proposal=proposal)

        auto = (
            proposal.kind is ProposalKind.INSPECT
            and not _is_privileged(command)
            and approvals.decide(command) in ("user", "session")
        )
        if auto:
            panel.note(f"... running {command}")
            result = _run_command(command, timeout=INSPECT_TIMEOUT)
            inspections.append((command, result))
            if audit is not None:
                audit.record(event="decision", proposal=proposal, decision="auto-inspect")
                audit.record(event="outcome", proposal=proposal, outcome=result.exit_code)
            if request_id:
                client_transport.respond_ui(
                    request_id, {"value": "once"}, shell_id=shell_id, env=env
                )
            return

        choice = _decide_proposal(panel, proposal)
        if audit is not None:
            audit.record(event="decision", proposal=proposal, decision=choice)
        if choice != APPROVE:
            if request_id:
                client_transport.respond_ui(
                    request_id, {"value": "deny"}, shell_id=shell_id, env=env
                )
            panel.note("nvsh: not run")
            return
        if request_id:
            # The backend owns execution (pi's approval extension); nvsh only
            # relays the operator's answer, and must not run it a second time.
            client_transport.respond_ui(request_id, {"value": "once"}, shell_id=shell_id, env=env)
            return
        result = _run_command(command)
        if audit is not None:
            audit.record(
                event="outcome", proposal=proposal, decision=choice, outcome=result.exit_code
            )
        _print_run(panel, command, result)

    return handle


def _print_run(panel: Panel, command: str, result: RunResult) -> None:
    if result.stdout:
        panel.write(result.stdout)
    if result.stderr:
        panel.write(result.stderr)
    panel.note(f"nvsh: {command} -> exit {result.exit_code}")


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
) -> StreamResult:
    on_proposal = None
    if approvals is not None and inspections is not None:
        on_proposal = _proposal_handler(
            panel,
            approvals=approvals,
            inspections=inspections,
            env=env,
            shell_id=shell_id,
            audit=audit,
        )
    return panel.stream(
        _send(request, context, env=env, shell_id=shell_id, config=config),
        on_proposal=on_proposal,
        cancel=lambda: client_transport.cancel(shell_id=shell_id, env=env),
    )


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


def handle_failure(
    args: Any, *, panel: Panel | None = None, env: Mapping[str, str] | None = None
) -> int:
    """Diagnose one failed command. Returns the exit code ``nvsh hook`` reports.

    ``0`` in every normal case (the operator's own command already reported
    its status); ``130`` when the operator pressed Ctrl+C during the stream.
    """
    resolved = dict(os.environ if env is None else env)
    panel = _panel_for(panel, resolved)
    config = _load_config()

    state = save_last_failure(args, env=resolved)
    if _rate_limited(args, config, resolved):
        return 0

    shell_id = _shell_pid(resolved)
    context = build_context(args, resolved)
    request = _failure_request(state)

    panel.header(state["line"], state["exit"])
    approvals = _load_approvals()
    inspections: list[tuple[str, RunResult]] = []
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
    )
    if result.interrupted:
        return 130

    if inspections:
        follow_up = _failure_request(state, prompt=_inspection_prompt(inspections))
        follow = _stream_request(
            panel, follow_up, context, env=resolved, shell_id=shell_id, config=config
        )
        if follow.interrupted:
            return 130
    return 0


def ask(
    prompt: str,
    *,
    draft: str | None = None,
    panel: Panel | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """``/ask`` and ``Ctrl+G``: a free-form question with the machine's context."""
    resolved = dict(os.environ if env is None else env)
    panel = _panel_for(panel, resolved)
    text = prompt
    if draft:
        text = f"{prompt}\n\nThe operator was in the middle of typing: {draft}"
    state = load_last_failure(resolved) or {}
    args = _args_from_state(state) if state else _args_from_state({"cwd": os.getcwd()})
    request = AgentRequest(
        kind=RequestKind.EXPLICIT,
        prompt=text,
        command=str(state.get("line", "") or ""),
        failure_id=str(state.get("failure_id", "") or ""),
    )
    result = _stream_request(
        panel,
        request,
        build_context(args, resolved),
        env=resolved,
        shell_id=_shell_pid(resolved),
        config=_load_config(),
    )
    return 130 if result.interrupted else 0


def _on_last_failure(prompt: str, panel: Panel | None, env: Mapping[str, str] | None) -> int:
    resolved = dict(os.environ if env is None else env)
    panel = _panel_for(panel, resolved)
    state = load_last_failure(resolved)
    if not state:
        panel.line("nvsh: no recorded failure yet")
        return 1
    result = _stream_request(
        panel,
        _failure_request(state, prompt=prompt),
        build_context(_args_from_state(state), resolved),
        env=resolved,
        shell_id=_shell_pid(resolved),
        config=_load_config(),
        approvals=_load_approvals(),
        inspections=[],
        audit=_audit(resolved),
    )
    return 130 if result.interrupted else 0


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


def retry(
    last: Mapping[str, Any] | None = None,
    *,
    panel: Panel | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """``/retry``: re-run the failed line, but only after the operator's Enter.

    The command is shown exactly as it was typed, confirmed through the
    panel, re-run with ``bash -lc`` (so the login environment matches an
    interactive shell), and its new status is reported by :func:`verify`.
    """
    resolved = dict(os.environ if env is None else env)
    panel = _panel_for(panel, resolved)
    state = last if last is not None else load_last_failure(resolved)
    if not state or not state.get("line"):
        panel.line("nvsh: no recorded failure to retry")
        return 1
    command = str(state["line"])
    proposal = Proposal(
        command=command,
        rationale=f"re-run the command that failed (exit {state.get('exit')})",
        kind=ProposalKind.RETRY,
    )
    if _decide_proposal(panel, proposal) != APPROVE:
        panel.note("nvsh: not re-run")
        return 0
    result = _run_command(command, login=True)
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

    The bytes are the prompt text :func:`nvsh.agent.pi.build_prompt` builds
    from the request and context -- the same function every adapter's
    ``run()`` feeds its backend -- so what is printed is what the model
    sees, not a summary of it.
    """
    from .agent.pi import build_prompt
    from .cli._output import emit_result

    resolved = dict(os.environ if env is None else env)
    state = load_last_failure(resolved) or {}
    args = _args_from_state(state)
    context = build_context(args, resolved)
    request = _failure_request(state) if state else _failure_request({})
    prompt = build_prompt(request, context)
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
                "prompt": prompt,
            },
            json_mode=True,
            stream=out,
        )
    else:
        emit_result(prompt, json_mode=False, stream=out)
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
    if verb == "context":
        return context_show(json_mode=False, env=resolved)

    _panel_for(panel, resolved).line(
        f"nvsh: unknown slash command '/{verb}' " "(try /ask, /fix, /explain, /retry, /context)"
    )
    return 1
