"""Detect and (with explicit confirmation) install nvsh's helper tools.

Deviation d1 on task t21: ``nvsh setup`` picks an agent backend
(:mod:`nvsh.agent.registry`) but until this module existed it only *printed*
the ``pi`` install command. Some machines in the fleet are missing more than
that -- the Jetson AGX Orin has no ``uv`` and no ``tmux`` at all -- so
``setup`` now detects every missing helper tool and, with the operator's
explicit go-ahead, installs it.

Nothing here ever runs a command on its own initiative:

* :func:`missing_tools` only inspects ``PATH`` (via an injectable ``which``).
* :func:`plan_installs` only *computes* the exact argv nvsh would run; it
  never calls ``subprocess``.
* :func:`run_install` is the only function that can execute anything, and
  only after its ``confirm`` callback returns ``True`` -- the caller decides
  what that means (an interactive ``[y/N]`` prompt, or ``--yes``). A step
  whose ``executable`` flag is ``False`` (a curl-pipe-sh installer with no
  package-manager alternative) is never executed, confirmed or not: only its
  command text is ever shown to the operator.

Every attempt -- run or declined -- is written to the audit log
(:mod:`nvsh.agent.audit`) with the same ``proposal``/``decision``/``outcome``
shape :mod:`nvsh.agent.loop` uses, so an install leaves the same kind of
trail a proposed shell fix does.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - fixed argv below, no shell=True
from dataclasses import dataclass
from typing import Callable, Optional

from .agent.audit import AuditLog
from .agent.registry import ADAPTERS, PI_INSTALL_CMD

WhichFn = Callable[[str], Optional[str]]
ConfirmFn = Callable[[str], bool]
RunFn = Callable[..., "subprocess.CompletedProcess[str]"]

#: uv's official installer. Printed only -- see module docstring; never run,
#: even with ``--yes`` (a curl-pipe-sh has no place running unattended).
_UV_CURL_INSTALLER = "curl -LsSf https://astral.sh/uv/install.sh | sh"

#: npm package names for the harnesses nvsh knows how to ``npm install -g``.
#: Kept here (not in ``nvsh/agent/registry.py``) so this module stays
#: file-disjoint from the registry task: the registry owns *what* backends
#: exist, this module owns *how to install* them. ``pi`` is not in this
#: dict -- it keeps its own install command
#: (:data:`nvsh.agent.registry.PI_INSTALL_CMD`) via :func:`_pi_install_commands`.
#: A harness not listed here (``agy``, ``kiro``, ``openai-compat``) has no
#: known installer at all -- see :func:`harness_install_step`.
NPM_PACKAGES: dict[str, str] = {
    "claude": "@anthropic-ai/claude-code",
    "codex": "@openai/codex",
    "qwen": "@qwen-code/qwen-code",
    "qwen-p": "@qwen-code/qwen-code",
}


@dataclass(frozen=True)
class ToolSpec:
    """One helper tool nvsh knows how to detect and (maybe) install."""

    name: str
    purpose: str
    binary: str
    install_commands: Callable[[WhichFn], "InstallStep"]
    needs_sudo: bool = False


@dataclass(frozen=True)
class InstallStep:
    """One planned install: the exact command, or why there isn't one.

    ``argv`` is ``None`` when there is nothing this machine can run right
    now -- either no known installer exists for it (no package manager
    detected), or the only known installer is a curl-pipe-sh nvsh refuses to
    run unattended. ``executable`` is ``False`` in both of those cases;
    ``shell_line`` still carries a human-readable command (or explanation)
    to print either way.
    """

    tool: str
    argv: tuple[str, ...] | None
    shell_line: str
    needs_sudo: bool
    executable: bool
    requires_confirmation: bool = True


@dataclass(frozen=True)
class InstallResult:
    """What actually happened when :func:`run_install` processed one step."""

    tool: str
    ran: bool
    returncode: int | None
    output_tail: str = ""


def _apt_step(tool: str, argv: tuple[str, ...], which: WhichFn) -> InstallStep:
    shell_line = " ".join(argv)
    if which("apt-get") is None:
        return InstallStep(
            tool=tool,
            argv=None,
            shell_line=f"no known installer for {tool} on this machine (apt-get not found)",
            needs_sudo=True,
            executable=False,
        )
    return InstallStep(
        tool=tool, argv=argv, shell_line=shell_line, needs_sudo=True, executable=True
    )


def _pi_install_commands(which: WhichFn) -> InstallStep:
    if which("npm") is None:
        return InstallStep(
            tool="pi",
            argv=None,
            shell_line="npm not found; install node first (see the 'node' offer)",
            needs_sudo=False,
            executable=False,
        )
    argv = tuple(PI_INSTALL_CMD.split())
    return InstallStep(
        tool="pi", argv=argv, shell_line=PI_INSTALL_CMD, needs_sudo=False, executable=True
    )


def harness_install_step(name: str, which: WhichFn = shutil.which) -> InstallStep:
    """The install step for agent backend ``name`` (an :data:`ADAPTERS` key).

    ``pi`` keeps its existing dedicated command
    (:data:`nvsh.agent.registry.PI_INSTALL_CMD`). ``claude``, ``codex`` and
    ``qwen``/``qwen-p`` get an executable ``npm install -g <package>`` step
    (:data:`NPM_PACKAGES`) when ``npm`` is on ``PATH``, and the same
    'npm not found' non-executable step ``pi`` gets otherwise. A harness
    with no known installer at all (``agy``, ``kiro``, ``openai-compat``)
    gets a non-executable step saying so -- never a curl-pipe-sh, never
    anything executed.
    """
    if name == "pi":
        return _pi_install_commands(which)

    package = NPM_PACKAGES.get(name)
    if package is None:
        return InstallStep(
            tool=name,
            argv=None,
            shell_line=f"no known installer for {name} (install it manually)",
            needs_sudo=False,
            executable=False,
        )

    if which("npm") is None:
        return InstallStep(
            tool=name,
            argv=None,
            shell_line="npm not found; install node first (see the 'node' offer)",
            needs_sudo=False,
            executable=False,
        )

    argv = ("npm", "install", "-g", package)
    return InstallStep(
        tool=name, argv=argv, shell_line=" ".join(argv), needs_sudo=False, executable=True
    )


def _node_install_commands(which: WhichFn) -> InstallStep:
    return _apt_step("node", ("sudo", "apt-get", "install", "-y", "nodejs", "npm"), which)


def _tmux_install_commands(which: WhichFn) -> InstallStep:
    return _apt_step("tmux", ("sudo", "apt-get", "install", "-y", "tmux"), which)


def _uv_install_commands(which: WhichFn) -> InstallStep:
    if which("snap") is not None:
        argv = ("sudo", "snap", "install", "astral-uv", "--classic")
        return InstallStep(
            tool="uv", argv=argv, shell_line=" ".join(argv), needs_sudo=True, executable=True
        )
    if which("curl") is not None:
        return InstallStep(
            tool="uv",
            argv=None,
            shell_line=_UV_CURL_INSTALLER,
            needs_sudo=False,
            executable=False,
        )
    return InstallStep(
        tool="uv",
        argv=None,
        shell_line="no known installer for uv on this machine (no snap, no curl)",
        needs_sudo=True,
        executable=False,
    )


#: Dependency order matters: node before pi, since pi's only known installer
#: (npm) needs node first. uv and tmux have no dependents.
TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="node",
        purpose="JavaScript runtime pi (and other npm-based backends) needs",
        binary="node",
        install_commands=_node_install_commands,
        needs_sudo=True,
    ),
    ToolSpec(
        name="pi",
        purpose="agent backend nvsh's failure client talks to by default",
        binary="pi",
        install_commands=_pi_install_commands,
        needs_sudo=False,
    ),
    ToolSpec(
        name="uv",
        purpose="Python package/dependency manager",
        binary="uv",
        install_commands=_uv_install_commands,
        needs_sudo=True,
    ),
    ToolSpec(
        name="tmux",
        purpose="terminal multiplexer nvsh's inline panel and daemon target",
        binary="tmux",
        install_commands=_tmux_install_commands,
        needs_sudo=True,
    ),
)


def _node_or_pi_applies(name: str, chosen: str | None, which: WhichFn) -> bool:
    """Should the ``node``/``pi`` offer be scoped in given the picked harness?

    ``chosen is None`` preserves today's behaviour (offer both regardless of
    the pick -- callers that never pass ``chosen``, like the existing
    ``setup`` flow, see no change). Once a harness is picked:

    * ``pi`` is only offered when ``chosen == 'pi'``.
    * ``node`` is offered when ``chosen == 'pi'`` (pi's own prerequisite), or
      when the chosen harness both needs node and is not itself already on
      ``PATH`` and has no ``npm`` to install it with. A harness that's
      already installed, or that doesn't need node at all, has no use for
      the node offer.
    """
    if chosen is None:
        return True
    if name == "pi":
        return chosen == "pi"
    # name == "node"
    if chosen == "pi":
        return True
    spec = ADAPTERS.get(chosen)
    if spec is None or not spec.needs_node:
        return False
    if spec.binary is not None and which(spec.binary) is not None:
        return False
    return which("npm") is None


def missing_tools(which: WhichFn = shutil.which, chosen: str | None = None) -> list[ToolSpec]:
    """Which of :data:`TOOLS` have no binary on ``PATH`` right now.

    ``chosen`` (the harness name ``nvsh setup`` picked, if any) scopes the
    ``node``/``pi`` offers to what that pick actually needs -- see
    :func:`_node_or_pi_applies`. ``uv`` and ``tmux`` are offered regardless
    of ``chosen``, since every backend needs them. ``chosen=None`` (the
    default) behaves exactly as before this parameter existed.
    """
    return [
        tool
        for tool in TOOLS
        if which(tool.binary) is None
        and (tool.name not in ("pi", "node") or _node_or_pi_applies(tool.name, chosen, which))
    ]


def plan_installs(missing: list[ToolSpec], which: WhichFn = shutil.which) -> list[InstallStep]:
    """Compute the exact :class:`InstallStep` for each of ``missing``.

    Pure computation -- never touches ``subprocess``. A tool with no viable
    installer on this machine still gets a step (``executable=False``) so
    the caller has something to print.
    """
    return [tool.install_commands(which) for tool in missing]


def _default_confirm(prompt: str) -> bool:
    answer = input(prompt)  # nosec B322 - plain y/N prompt, no eval of input
    return answer.strip().lower() == "y"


def run_install(
    step: InstallStep,
    run: RunFn | None = None,
    confirm: ConfirmFn = _default_confirm,
    audit: AuditLog | None = None,
) -> InstallResult:
    """Run one planned install, but only after ``confirm`` approves it.

    A non-executable step (no known installer, or a curl-pipe-sh) is never
    run -- ``confirm`` is not even called, since there is nothing it could
    approve. A sudo command is executed with ``sudo`` as ``argv[0]`` in list
    form (a shell is never spawned) and its output is left attached to the
    terminal (no ``capture_output``) so the operator sees and can answer the
    password prompt; a non-sudo command's output is captured so it can be
    logged and reported. Every attempt -- run, declined, or not executable
    -- is appended to the audit log.
    """
    log = audit if audit is not None else AuditLog()
    run = run if run is not None else subprocess.run

    if not step.executable:
        log.record(
            event="install",
            proposal=step.shell_line,
            decision=None,
            outcome={"ran": False, "reason": "not executable"},
        )
        return InstallResult(tool=step.tool, ran=False, returncode=None)

    decision = confirm(f"install {step.tool}? [y/N] ")
    if not decision:
        log.record(event="install", proposal=step.shell_line, decision=False, outcome=None)
        return InstallResult(tool=step.tool, ran=False, returncode=None)

    argv = list(step.argv)  # type: ignore[arg-type]
    if step.needs_sudo:
        # No capture: sudo needs the real terminal attached so the operator
        # can type their password. We still get a returncode.
        completed = run(argv, check=False)  # nosec B603 - fixed argv, no shell
        output_tail = ""
    else:
        completed = run(argv, check=False, capture_output=True, text=True)  # nosec B603
        stdout = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
        output_tail = (stdout + stderr)[-2000:]

    outcome = {"returncode": completed.returncode, "output_tail": output_tail}
    log.record(event="install", proposal=step.shell_line, decision=True, outcome=outcome)
    return InstallResult(
        tool=step.tool, ran=True, returncode=completed.returncode, output_tail=output_tail
    )
