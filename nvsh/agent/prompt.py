"""The one prompt composer every backend adapter shares.

Why this is its own module: the prompt is the contract between
``nvsh context --show`` (what the operator is told will be sent) and what an
adapter actually sends. When two adapters compose it separately they drift
-- and they did: the subprocess/openai-compat composer returned
``request.prompt`` alone whenever a prompt was supplied, which is exactly
what the failure client always does, so the detected platform block and the
captured output slice never reached those backends at all (spec c38 wants
the opposite). One function, imported by all of them, makes that class of
drift impossible.

Imports nothing but the dataclasses in :mod:`nvsh.agent.base`.
"""

from __future__ import annotations

from .base import AgentContext, AgentRequest
from .playbooks import playbook_for

#: The agent's background: who it is, what it may do, and how to answer.
#: Shipped in code because an agent with no background investigates its own
#: harness -- deviation d19, where the real associate model answered
#: "whats memory levels are now?" with ``which pi && pi --help``.
SYSTEM_BRIEF = """\
You are nvsh's on-failure assistant on an NVIDIA machine (Jetson, DGX Spark
or an RTX workstation). nvsh is a shell first and an agent second: the
operator works at a normal bash prompt and you are called only when a
command fails or when they ask you something directly.

Rules:
- Diagnose from the context you are given first. The detected facts below
  were read from this machine; trust them over any assumption, and never
  assume a package, driver or CUDA version that is not in that block.
- Propose ONE command at a time through the bash tool, then wait for its
  output before proposing the next. Never a list, never a script.
- Prefer read-only inspection. Never propose `sudo` unless the operator
  asked for a change that cannot be made without it, and say why.
- Never investigate nvsh, pi, your own harness, model or tools. `which pi`,
  `pi --help`, `ls /usr/local/bin` and the like answer nothing about the
  operator's problem. Look at the machine.
- When the operator asks a question or gives an instruction in plain
  language, do the work yourself: propose the inspection commands one at a
  time via the bash tool, read their output, and answer with the concrete
  numbers and a one-line conclusion. Never answer a question with a list of
  commands for the operator to run.
- Be terse. State facts with their source (the file or command they came
  from). If something is unknown, say it is unknown.

Answer format: a one-line diagnosis, then the proposal (or, once you have
the numbers, the answer).
"""


def platform_kind(context: AgentContext) -> str:
    """The detected platform kind carried by ``context.platform``.

    ``Platform.render_block()`` opens with ``platform: <kind>``; that one
    line is the whole coupling between detection and the playbooks, so this
    module still imports nothing but :mod:`nvsh.agent.base` and the
    playbooks themselves. Anything unparseable is ``generic``.
    """
    first = (context.platform or "").strip().splitlines()[:1]
    if not first:
        return "generic"
    label, sep, kind = first[0].partition(":")
    if label.strip().lower() != "platform" or not sep:
        return "generic"
    kind = kind.strip().lower().replace("_", "-")
    return kind or "generic"


def build_system_prompt(context: AgentContext) -> str:
    """The brief every adapter puts where its backend expects a system prompt.

    Depends only on the *detected kind*, never on the failure, so a backend
    that takes a session-scoped system prompt (pi, claude, qwen) can send it
    once per session instead of once per turn.
    """
    return f"{SYSTEM_BRIEF}\n{playbook_for(platform_kind(context))}".rstrip() + "\n"


def build_full_prompt(request: AgentRequest, context: AgentContext) -> str:
    """Brief first, then the detected facts and the failure.

    For backends with no system-prompt channel of their own (``codex
    exec``): the brief has to ride in the prompt, and it has to come first.
    """
    return f"{build_system_prompt(context)}\n{build_prompt(request, context)}"


def build_prompt(request: AgentRequest, context: AgentContext) -> str:
    """Compose the exact prompt text for one request, folding in the context.

    Order is fixed (command, exit code, platform, cwd, output, then the
    caller's own prompt) so the bytes are reproducible: ``nvsh context
    --show`` prints this and nothing else.
    """
    lines: list[str] = []
    if request.command:
        lines.append(f"Command: {request.command}")
    if request.exit_code is not None:
        lines.append(f"Exit code: {request.exit_code}")
    if context.platform:
        lines.append(f"Platform: {context.platform}")
    if context.cwd:
        lines.append(f"cwd: {context.cwd}")
    if context.output:
        lines.append("Output:")
        lines.append(context.output)
    if request.prompt:
        lines.append(request.prompt)
    return "\n".join(lines) if lines else request.prompt
