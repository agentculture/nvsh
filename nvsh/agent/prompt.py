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
