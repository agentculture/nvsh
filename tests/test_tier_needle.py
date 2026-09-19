"""Tests for the Needle3 child-process tier (task t9).

Covers spec targets c32 (the local engine runs in a child process, never in
the daemon), h27 (a killed or hung child yields a decline and the next
request gets a fresh child), c12/h11 (offline: telemetry and hub access are
switched off before the engine is imported) and c9/h8 (a missing engine is
reported, not raised).

Nothing here needs ``cactus-needle``: the worker command is injectable and
every process test drives ``tests/fakes/needle_worker``.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import time
from pathlib import Path

import pytest

from nvsh.agent.base import AgentContext, AgentRequest, RequestKind
from nvsh.ops import table as ops_table
from nvsh.tiers import needle_worker
from nvsh.tiers.base import Decline, DeclineReason, TierDecision
from nvsh.tiers.memfloor import FloorResult
from nvsh.tiers.needle import MAX_FRAME_BYTES, NeedleTier, pack_frame

NVSH_ROOT = Path(__file__).resolve().parent.parent / "nvsh"
FAKE_WORKER = Path(__file__).resolve().parent / "fakes" / "needle_worker"

#: A call the operation table accepts, so ``decide()`` reaches a decision.
GOOD_CALL = {"name": "disk_stats", "arguments": {}}


# -- helpers (never inside a test body: tests carry no try/except) --


def _request(text: str = "am I out of disk?") -> AgentRequest:
    return AgentRequest(kind=RequestKind.EXPLICIT, prompt=text)


def _context() -> AgentContext:
    return AgentContext(platform="test", cwd="/tmp")


def _tier(tmp_path: Path, script: list, **kwargs) -> NeedleTier:
    """A tier wired to the fake worker, with the given per-launch script."""
    pid_file = tmp_path / "pids"
    env = dict(os.environ)
    env["NVSH_TEST_NEEDLE_PIDS"] = str(pid_file)
    env["NVSH_TEST_NEEDLE_SCRIPT"] = json.dumps(script)
    kwargs.setdefault("timeout", 10.0)
    tier = NeedleTier(
        weights_path=str(tmp_path / "weights.cact"),
        worker_argv=[sys.executable, str(FAKE_WORKER)],
        availability=lambda: None,
        env=env,
        **kwargs,
    )
    tier.pid_file = pid_file  # type: ignore[attr-defined]
    return tier


def _pids(tier: NeedleTier) -> list[int]:
    path: Path = tier.pid_file  # type: ignore[attr-defined]
    if not path.exists():
        return []
    return [int(line) for line in path.read_text(encoding="utf-8").split()]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.02)
    return not _alive(pid)


def _python_sources() -> list[Path]:
    return sorted(NVSH_ROOT.rglob("*.py"))


@pytest.fixture(name="closing")
def _closing():
    """Close every tier a test built, however the test ends."""
    tiers: list[NeedleTier] = []
    yield tiers.append
    for tier in tiers:
        tier.close()


# -- criterion 1: a child that dies mid-request --


def test_dead_child_declines(tmp_path, closing):
    """The child exits without answering -> a TIER_ERROR decline, no exception."""
    tier = _tier(tmp_path, [["die"], [{"calls": [GOOD_CALL]}]])
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline) and result.reason is DeclineReason.TIER_ERROR


def test_next_request_restarts_the_child(tmp_path, closing):
    """After the death the next select() gets a decision from a fresh child."""
    tier = _tier(tmp_path, [["die"], [{"calls": [GOOD_CALL], "confidence": 0.9}]])
    closing(tier)
    tier.select(_request(), _context())
    result = tier.select(_request(), _context())
    assert result == TierDecision(operation="disk_stats", args={}, confidence=0.9, read_only=True)


def test_restart_is_a_new_process(tmp_path, closing):
    """Two launches, two distinct pids -- the child really was replaced."""
    tier = _tier(tmp_path, [["die"], [{"calls": [GOOD_CALL]}]])
    closing(tier)
    tier.select(_request(), _context())
    tier.select(_request(), _context())
    assert len(set(_pids(tier))) == 2


# -- criterion 1: a hung child --


def test_hung_child_declines_after_timeout(tmp_path, closing):
    """A child that never answers is declined once the deadline passes."""
    tier = _tier(tmp_path, [["hang"]], timeout=0.3)
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline) and result.reason is DeclineReason.TIER_ERROR


def test_hung_child_is_killed(tmp_path, closing):
    """The timed-out child is not left running."""
    tier = _tier(tmp_path, [["hang"]], timeout=0.3)
    closing(tier)
    tier.select(_request(), _context())
    (pid,) = _pids(tier)
    assert _wait_gone(pid)


def test_garbled_frame_declines(tmp_path, closing):
    """A well-framed payload that is not JSON is a decline, not a crash."""
    tier = _tier(tmp_path, [["garbage"]], timeout=2.0)
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline) and result.reason is DeclineReason.TIER_ERROR


def test_oversized_frame_declines(tmp_path, closing):
    """A length header past the 1 MiB bound is refused before any read."""
    tier = _tier(tmp_path, [["huge"]], timeout=2.0)
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline) and "too large" in result.detail


def test_worker_error_reply_declines(tmp_path, closing):
    """The worker reporting its own failure becomes a TIER_ERROR decline."""
    tier = _tier(tmp_path, [[{"error": "needle_complete failed"}]])
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline) and "needle_complete failed" in result.detail


def test_close_kills_the_child(tmp_path):
    """close() leaves no worker behind."""
    tier = _tier(tmp_path, [[{"calls": [GOOD_CALL]}]])
    tier.select(_request(), _context())
    (pid,) = _pids(tier)
    tier.close()
    assert _wait_gone(pid)


def test_untrusted_output_goes_through_decide(tmp_path, closing):
    """A raw shell escape hatch from the worker is declined by decide()."""
    tier = _tier(tmp_path, [[{"calls": [{"name": "bash", "arguments": {"cmd": "rm -rf /"}}]}]])
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline) and result.reason is DeclineReason.RAW_SHELL


def test_warm_child_is_reused(tmp_path, closing):
    """Two successful selects share one worker process."""
    tier = _tier(tmp_path, [[{"calls": [GOOD_CALL]}]])
    closing(tier)
    tier.select(_request(), _context())
    tier.select(_request(), _context())
    assert len(_pids(tier)) == 1


def test_missing_worker_command_declines(tmp_path, closing):
    """A worker binary that does not exist is unavailable, not an exception."""
    tier = NeedleTier(
        weights_path=str(tmp_path / "weights.cact"),
        worker_argv=[str(tmp_path / "no-such-worker")],
        availability=lambda: None,
    )
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline) and result.reason is DeclineReason.TIER_UNAVAILABLE


# -- criterion 3: cactus-needle absent --


def test_unavailable_tier_declines(tmp_path, closing):
    """No engine -> TIER_UNAVAILABLE, nothing raised, no child spawned."""
    tier = NeedleTier(availability=lambda: "cactus-needle is not installed")
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline) and result.reason is DeclineReason.TIER_UNAVAILABLE


def test_unavailable_status_is_one_line():
    """status() is a single human-readable line naming the reason."""
    tier = NeedleTier(availability=lambda: "cactus-needle is not installed\nsecond line")
    line = tier.status()
    assert "\n" not in line and "cactus-needle is not installed" in line


def test_ready_status_is_one_line():
    """A tier with nothing wrong still answers with exactly one line."""
    tier = NeedleTier(availability=lambda: None)
    assert "\n" not in tier.status()


def test_default_availability_reports_missing_weights(tmp_path):
    """The stock probe reports a missing weights file instead of raising."""
    tier = NeedleTier(weights_path=str(tmp_path / "absent.cact"))
    assert "absent.cact" in (tier.status() or "")


def test_memory_floor_declines(tmp_path, closing):
    """An injected floor check that says no is a MEMORY_FLOOR decline."""
    floor = FloorResult(ok=False, available_mb=100, status="too little memory")
    tier = _tier(tmp_path, [[{"calls": [GOOD_CALL]}]], floor_check=lambda: floor)
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline) and result.reason is DeclineReason.MEMORY_FLOOR


# -- framing --


def test_pack_frame_is_length_prefixed():
    """A frame is a 4-byte big-endian length followed by its UTF-8 JSON."""
    frame = pack_frame({"op": "select"})
    assert int.from_bytes(frame[:4], "big") == len(frame) - 4


def test_pack_frame_refuses_an_oversized_message():
    """Nothing beyond the bound is ever written to the pipe."""
    with pytest.raises(ValueError):
        pack_frame({"text": "x" * (MAX_FRAME_BYTES + 1)})


# -- criterion 2: the worker's offline environment and tool schemas --


def test_worker_hardens_env_before_importing_needle(monkeypatch):
    """The three offline variables are set in os.environ *before* the import."""
    monkeypatch.setenv("NEEDLE_TELEMETRY", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    seen: dict[str, str] = {}

    def importer():
        seen.update(os.environ)
        return _FakeNeedleModule()

    needle_worker.build_engine(None, importer=importer)
    keys = ("NEEDLE_TELEMETRY", "DO_NOT_TRACK", "HF_HUB_OFFLINE")
    assert {key: seen.get(key) for key in keys} == {
        "NEEDLE_TELEMETRY": "0",
        "DO_NOT_TRACK": "1",
        "HF_HUB_OFFLINE": "1",
    }


class _FakeNeedle:
    """Stands in for ``needle.Needle``: records what it was given."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.completed: list[str] = []
        self.resets = 0

    def complete(self, text: str) -> dict:
        self.completed.append(text)
        return {"type": "call", "function_calls": [GOOD_CALL], "confidence": 0.75}

    def reset(self) -> None:
        self.resets += 1


class _FakeNeedleModule:
    """Stands in for the ``needle`` module."""

    def __init__(self) -> None:
        self.instances: list[_FakeNeedle] = []

    def Needle(self, **kwargs: object) -> _FakeNeedle:  # noqa: N802 - mirrors the real name
        engine = _FakeNeedle(**kwargs)
        self.instances.append(engine)
        return engine


def test_worker_passes_the_weights_path(tmp_path):
    """The verified local weights file is what the engine is built from."""
    module = _FakeNeedleModule()
    weights = tmp_path / "needle3.cact"
    needle_worker.build_engine(str(weights), importer=lambda: module)
    assert module.instances[0].kwargs["weights"] == str(weights)


def test_worker_builds_one_tool_per_operation():
    """Tool schemas come from the operation table, never a hand-written list."""
    names = [schema["name"] for schema in needle_worker.tool_schemas()]
    assert names == list(ops_table.names())


def test_choice_argument_becomes_an_enum():
    """A choice argument is offered to the model as its exact allowed values."""
    schemas = {schema["name"]: schema for schema in needle_worker.tool_schemas()}
    power_set = schemas["power_set"]["parameters"]["properties"]["mode"]
    assert power_set == {"type": "string", "enum": ["max_performance", "balanced", "low_power"]}


def test_tool_functions_do_nothing():
    """The callables handed to needle have empty bodies -- selection only."""
    results = [function() for function in needle_worker.tool_functions()]
    assert results == [None] * len(ops_table.OPERATIONS)


def test_tool_functions_carry_their_schema():
    """needle reads the pre-built schema off each callable rather than guessing."""
    functions = needle_worker.tool_functions()
    assert [fn._needle_tool["name"] for fn in functions] == list(ops_table.names())


def test_engine_session_resets_between_requests():
    """reset() is called before every request after the first."""
    module = _FakeNeedleModule()
    session = needle_worker.EngineSession(builder=lambda weights: module.Needle(weights=weights))
    session.select(None, "first")
    session.select(None, "second")
    assert module.instances[0].resets == 1


def test_engine_session_returns_calls_and_confidence():
    """The worker forwards the engine's raw calls plus its confidence, unvalidated."""
    module = _FakeNeedleModule()
    session = needle_worker.EngineSession(builder=lambda weights: module.Needle(weights=weights))
    assert session.select(None, "text") == ([GOOD_CALL], 0.75)


@pytest.mark.parametrize(
    "envelope,expected",
    [
        ({"type": "call", "function_calls": [GOOD_CALL], "confidence": 0.5}, ([GOOD_CALL], 0.5)),
        ({"type": "text", "content": "hello"}, ([], None)),
        ({"function_calls": None, "confidence": None}, ([], None)),
        ("not an envelope", ([], None)),
    ],
)
def test_extract_selection_shapes(envelope, expected):
    """Every envelope shape the engine may return reduces to (calls, confidence)."""
    assert needle_worker.extract_selection(envelope) == expected


# -- criterion 2: nvsh never calls Needle.run --


def test_nvsh_never_calls_needle_run():
    """``run()`` executes the tool bodies; nvsh must only ever ``complete()``."""
    forbidden = re.compile(r"\bNeedle\s*\.\s*run\b|\b(needle|engine|_engine)\s*\.\s*run\s*\(")
    offenders = [
        f"{path}:{number}"
        for path in _python_sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if forbidden.search(line)
    ]
    assert offenders == []


def test_worker_does_not_import_needle_at_module_level():
    """Importing the worker module must not pull in cactus-needle."""
    source = (NVSH_ROOT / "tiers" / "needle_worker.py").read_text(encoding="utf-8")
    # The relative ``from .needle import`` is nvsh's own module, not the package.
    absolute = re.compile(r"^(import|from)\s+needle\b")
    assert not [line for line in source.splitlines() if absolute.match(line)]


def test_fake_worker_is_executable():
    """The fake is run through the interpreter, but stays a runnable script."""
    assert stat.S_IMODE(FAKE_WORKER.stat().st_mode) & stat.S_IXUSR
