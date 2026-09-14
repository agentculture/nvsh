"""Scrubbed environment for subprocess-backed agent adapters.

nvsh's own harness invocation runs inside Claude Code sometimes (the nvsh
repo itself is developed with Claude Code, and daemon/tests can run under
it too). Claude Code marks its own process tree with ``CLAUDECODE`` and a
family of ``CLAUDE_CODE_*`` variables (SSE port, entrypoint, ...). Handing
those to a spawned backend CLI -- ``claude``, ``codex``, ``qwen``, a future
adapter -- would let it believe it is itself running nested inside Claude
Code, which is never true: the child is a plain subprocess started by
nvsh, not a Claude Code session. :func:`child_env` drops that whole family
before any adapter builds its subprocess environment.
"""

from __future__ import annotations

import os
from typing import Mapping

#: Exact key dropped outright (no trailing underscore -- it is not a prefix
#: match, it is this literal name).
_EXACT_DROP = {"CLAUDECODE"}

#: Prefix dropped: every ``CLAUDE_CODE_*`` variable Claude Code sets.
_PREFIX_DROP = "CLAUDE_CODE_"


def child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of ``base`` (default: ``os.environ``) with Claude-Code-
    in-the-parent-process markers removed.

    Never mutates ``base`` (or ``os.environ``) -- callers get a fresh dict
    safe to pass straight to ``subprocess.Popen(..., env=...)``.
    """
    source: Mapping[str, str] = base if base is not None else os.environ
    return {
        key: value
        for key, value in source.items()
        if key not in _EXACT_DROP and not key.startswith(_PREFIX_DROP)
    }
