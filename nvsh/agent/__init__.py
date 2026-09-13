"""The nvsh agent contract: a small, backend-agnostic surface adapters plug into.

Public surface: :class:`NvshAgent` (the abstract adapter contract),
:class:`FakeAgent` (a scripted adapter for tests), :func:`run_loop` (the
UI-free approve/execute/verify loop) and :class:`AuditLog` (its JSONL audit
trail), plus the dataclasses/enums they share. Nothing here imports pi or
any other concrete backend.
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
    "Proposal",
    "ProposalKind",
    "RequestKind",
    "default_audit_path",
    "run_loop",
]
