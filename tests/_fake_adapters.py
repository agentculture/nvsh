"""Factories wiring claude/codex/qwen adapters to the fake CLIs on PATH.

Used by ``tests/test_agent_conformance.py`` to register the real
subprocess-based adapters (``ClaudeAgent``, ``CodexAgent``, ``QwenAgent``)
into the shared conformance suite, exactly like ``FakeAgent`` but exercised
through a real subprocess -- the fakes under ``tests/fakes/`` -- so the
suite proves the adapters' actual line-parsing and cancellation code, not a
mock of it.

Each factory takes ``script`` (a list of ``AgentEvent``, per the conformance
fixture contract), serializes it to a small JSON events file, and returns an
adapter instance whose subprocess ``env`` prepends ``tests/fakes`` to
``PATH`` and points ``NVSH_FAKE_EVENTS`` at that file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from nvsh.agent.claude import ClaudeAgent
from nvsh.agent.codex import CodexAgent
from nvsh.agent.qwen import QwenAgent

FAKES_DIR = Path(__file__).parent / "fakes"


def _write_events_file(script) -> str:
    payload = [{"kind": item.kind.value, "text": item.text, "error": item.error} for item in script]
    fd, path = tempfile.mkstemp(suffix=".json", prefix="nvsh-fake-events-")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _fake_env(script) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env["NVSH_FAKE_EVENTS"] = _write_events_file(script)
    return env


def ClaudeAgentViaFake(script):  # noqa: N802 - factory name doubles as pytest id
    return ClaudeAgent({}, env=_fake_env(script))


def CodexAgentViaFake(script):  # noqa: N802
    return CodexAgent({}, env=_fake_env(script))


def QwenAgentViaFake(script):  # noqa: N802
    return QwenAgent({}, env=_fake_env(script))
