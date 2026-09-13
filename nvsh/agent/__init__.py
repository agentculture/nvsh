"""The nvsh agent contract: a small, backend-agnostic surface adapters plug into.

Public surface: :class:`NvshAgent` (the abstract adapter contract),
:class:`FakeAgent` (a scripted adapter for tests), :class:`PiAgent` (the
``pi --mode rpc`` adapter), :func:`run_loop` (the UI-free
approve/execute/verify loop) and :class:`AuditLog` (its JSONL audit trail),
plus the dataclasses/enums they share. ``base.py`` itself imports nothing
concrete (see its own docstring); this package's ``__init__`` is simply
where every adapter is re-exported for callers.
"""

from __future__ import annotations

from .audit import AuditLog, default_audit_path
from .base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Capabilities,
    EventKind,
    NvshAgent,
    Proposal,
    ProposalKind,
    RequestKind,
)
from .fake import FakeAgent
from .loop import ExecResult, run_loop
from .pi import PiAgent, build_prompt, default_approval_extension_path, default_session_dir

__all__ = [
    "AgentContext",
    "AgentEvent",
    "AgentRequest",
    "AuditLog",
    "Capabilities",
    "EventKind",
    "ExecResult",
    "FakeAgent",
    "NvshAgent",
    "PiAgent",
    "Proposal",
    "ProposalKind",
    "RequestKind",
    "build_prompt",
    "default_approval_extension_path",
    "default_audit_path",
    "default_session_dir",
    "run_loop",
]
