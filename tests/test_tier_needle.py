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
import threading
import time
import zipfile
from pathlib import Path

import pytest

from nvsh.agent.base import AgentContext, AgentRequest, RequestKind
from nvsh.ops import table as ops_table
from nvsh.tiers import needle, needle_home, needle_worker
from nvsh.tiers.base import Decline, DeclineReason, TierDecision
from nvsh.tiers.fetch import FetchProblem
from nvsh.tiers.memfloor import FloorResult
from nvsh.tiers.needle import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_FRAME_BYTES,
    MAX_PROMPT_CHARS,
    NeedleTier,
    pack_frame,
)

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
    assert isinstance(result, Decline)
    assert result.reason is DeclineReason.TIER_ERROR


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
    assert isinstance(result, Decline)
    assert result.reason is DeclineReason.TIER_ERROR


def test_hung_child_is_killed(tmp_path, closing):
    """The timed-out child is not left running."""
    tier = _tier(tmp_path, [["hang"]], timeout=0.3)
    closing(tier)
    tier.select(_request(), _context())
    (pid,) = _pids(tier)
    assert _wait_gone(pid)


def test_dead_child_kill_uses_a_short_grace(tmp_path, closing, monkeypatch):
    """A child killed after the deadline gets a short grace, not kill_tree's default.

    Qodo #4053821262: ``_dead_child`` used to call ``_shutdown()`` with no
    grace, which meant a resistant child could add two more full
    ``kill_tree`` grace periods (SIGTERM wait, then SIGKILL wait) on top of
    the timeout the operator already sat through.
    """
    tier = _tier(tmp_path, [["hang"]], timeout=0.3)
    closing(tier)
    seen_grace: list[float] = []
    real_kill_tree = needle.kill_tree

    def recording_kill_tree(proc, grace=2.0):
        seen_grace.append(grace)
        return real_kill_tree(proc, grace=grace)

    monkeypatch.setattr(needle, "kill_tree", recording_kill_tree)
    tier.select(_request(), _context())
    assert seen_grace == [needle._DEAD_CHILD_KILL_GRACE]


def test_garbled_frame_declines(tmp_path, closing):
    """A well-framed payload that is not JSON is a decline, not a crash."""
    tier = _tier(tmp_path, [["garbage"]], timeout=2.0)
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline)
    assert result.reason is DeclineReason.TIER_ERROR


def test_oversized_frame_declines(tmp_path, closing):
    """A length header past the 1 MiB bound is refused before any read."""
    tier = _tier(tmp_path, [["huge"]], timeout=2.0)
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline)
    assert "too large" in result.detail


def test_worker_error_reply_declines(tmp_path, closing):
    """The worker reporting its own failure becomes a TIER_ERROR decline."""
    tier = _tier(tmp_path, [[{"error": "needle_complete failed"}]])
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline)
    assert "needle_complete failed" in result.detail


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
    assert isinstance(result, Decline)
    assert result.reason is DeclineReason.RAW_SHELL


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
    assert isinstance(result, Decline)
    assert result.reason is DeclineReason.TIER_UNAVAILABLE


# -- hardening: one turn at a time, matched replies, bounded prompts --


def test_concurrent_selects_do_not_interleave(tmp_path, closing):
    """Two threads share one tier (as the daemon does) and both get decisions."""
    tier = _tier(tmp_path, [[{"calls": [GOOD_CALL], "delay": 0.2}]], timeout=5.0)
    closing(tier)
    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(tier.select(_request(), _context())))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert [isinstance(result, TierDecision) for result in results] == [True, True]


def test_mismatched_reply_id_declines(tmp_path, closing):
    """A reply that answers another request means the stream is out of step."""
    tier = _tier(tmp_path, [["wrong_id"]], timeout=2.0)
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline)
    assert result.reason is DeclineReason.TIER_ERROR


def test_prompt_is_clamped_on_the_wire(tmp_path, closing):
    """A huge prompt is cut to the bound before the frame is built."""
    tier = _tier(tmp_path, [["echo"]], timeout=5.0)
    closing(tier)
    result = tier.select(_request("x" * (MAX_PROMPT_CHARS * 3)), _context())
    assert result.args == {"service": str(MAX_PROMPT_CHARS)}


def test_default_timeout_keeps_the_prompt_responsive():
    """Tier 1 exists to be fast: a broken child must not hold the prompt for long."""
    assert DEFAULT_TIMEOUT_SECONDS <= 10.0


# -- criterion 3: cactus-needle absent --


def test_unavailable_tier_declines(tmp_path, closing):
    """No engine -> TIER_UNAVAILABLE, nothing raised, no child spawned."""
    tier = NeedleTier(availability=lambda: "cactus-needle is not installed")
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline)
    assert result.reason is DeclineReason.TIER_UNAVAILABLE


def test_unavailable_status_is_one_line():
    """status() is a single human-readable line naming the reason."""
    tier = NeedleTier(availability=lambda: "cactus-needle is not installed\nsecond line")
    line = tier.status()
    assert "\n" not in line
    assert "cactus-needle is not installed" in line


def test_ready_status_is_one_line():
    """A tier with nothing wrong still answers with exactly one line."""
    tier = NeedleTier(availability=lambda: None)
    assert "\n" not in tier.status()


def test_default_availability_reports_missing_weights(tmp_path):
    """The stock probe reports a missing weights file instead of raising."""
    tier = NeedleTier(weights_path=str(tmp_path / "absent.cact"))
    assert "absent.cact" in (tier.status() or "")


def test_default_availability_reports_an_unstaged_cache(tmp_path, monkeypatch):
    """With nothing prefetched, the stock probe says so and never downloads."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    tier = NeedleTier()
    assert "needle engine files unusable" in tier.status()


def test_memory_floor_declines(tmp_path, closing):
    """An injected floor check that says no is a MEMORY_FLOOR decline."""
    floor = FloorResult(ok=False, available_mb=100, status="too little memory")
    tier = _tier(tmp_path, [[{"calls": [GOOD_CALL]}]], floor_check=lambda: floor)
    closing(tier)
    result = tier.select(_request(), _context())
    assert isinstance(result, Decline)
    assert result.reason is DeclineReason.MEMORY_FLOOR


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


def _env_at_import(monkeypatch, spec: needle_worker.EngineSpec) -> dict[str, str]:
    """os.environ as the worker had it at the moment ``needle`` was imported."""
    monkeypatch.setenv("NEEDLE_TELEMETRY", "1")
    # Recorded through monkeypatch so the real HOME comes back at teardown,
    # even though the worker sets it through os.environ itself.
    monkeypatch.setenv("HOME", os.environ.get("HOME", "/nonexistent"))
    for key in ("DO_NOT_TRACK", "HF_HUB_OFFLINE", "NEEDLE3_LIB_PATH"):
        monkeypatch.delenv(key, raising=False)
    seen: dict[str, str] = {}

    def importer():
        seen.update(os.environ)
        return _FakeNeedleModule()

    needle_worker.build_engine(spec, importer=importer, linker=lambda _module, path: path)
    return seen


def test_worker_hardens_env_before_importing_needle(monkeypatch):
    """The three offline variables are set in os.environ *before* the import."""
    seen = _env_at_import(monkeypatch, needle_worker.EngineSpec())
    keys = ("NEEDLE_TELEMETRY", "DO_NOT_TRACK", "HF_HUB_OFFLINE")
    assert {key: seen.get(key) for key in keys} == {
        "NEEDLE_TELEMETRY": "0",
        "DO_NOT_TRACK": "1",
        "HF_HUB_OFFLINE": "1",
    }


def test_worker_points_home_and_lib_at_the_staged_files(monkeypatch, tmp_path):
    """$HOME and NEEDLE3_LIB_PATH are in place before the library can look."""
    spec = needle_worker.EngineSpec(lib=str(tmp_path / "libneedle3.so"), home=str(tmp_path))
    seen = _env_at_import(monkeypatch, spec)
    assert (seen.get("HOME"), seen.get("NEEDLE3_LIB_PATH")) == (spec.home, spec.lib)


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


class _FakeFetch:
    """Stands in for ``needle.agent.fetch``: where the base archive lives."""

    def __init__(self, cache: Path) -> None:
        self._cache = cache

    def cache_dir(self, _generation: int) -> str:
        return str(self._cache)

    def base_weights(self, _generation: int) -> str:
        return "needle3.cact"


class _FakeAgent:
    def __init__(self, fetch: _FakeFetch) -> None:
        self.fetch = fetch


class _FakeNeedleModule:
    """Stands in for the ``needle`` module.

    ``broken=True`` drops the ``agent.fetch`` layout entirely and names a
    module that cannot be imported, standing in for a future cactus-needle
    whose internals moved.
    """

    def __init__(self, cache_dir: Path | None = None, broken: bool = False) -> None:
        self.instances: list[_FakeNeedle] = []
        self.__name__ = "nvsh_not_a_real_module" if broken else "needle"
        if not broken:
            self.agent = _FakeAgent(_FakeFetch(cache_dir or Path("/nonexistent")))

    def Needle(self, **kwargs: object) -> _FakeNeedle:  # noqa: N802 - mirrors the real name
        engine = _FakeNeedle(**kwargs)
        self.instances.append(engine)
        return engine


def test_stock_weights_are_linked_not_passed(monkeypatch, tmp_path):
    """Stock weights go in through the library's own cache, so confidence survives.

    Passing ``weights=`` would mark the model "tuned": the library then
    reports ``confidence: None`` and starts a nested subprocess of its own.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    module = _FakeNeedleModule()
    spec = needle_worker.EngineSpec(weights=str(tmp_path / "needle3.cact"), home=str(tmp_path))
    needle_worker.build_engine(spec, importer=lambda: module, linker=lambda _m, path: path)
    assert "weights" not in module.instances[0].kwargs


def test_tuned_weights_are_passed_through(monkeypatch, tmp_path):
    """A fine-tune is handed to the library as ``weights=``, as it expects."""
    monkeypatch.setenv("HOME", str(tmp_path))
    module = _FakeNeedleModule()
    weights = tmp_path / "tuned.cact"
    spec = needle_worker.EngineSpec(weights=str(weights), tuned=True)
    needle_worker.build_engine(spec, importer=lambda: module)
    assert module.instances[0].kwargs["weights"] == str(weights)


def test_base_weights_are_symlinked_into_the_library_cache(tmp_path):
    """The verified file becomes the base archive, without a copy or a download."""
    weights = tmp_path / "needle3.cact"
    weights.write_bytes(b"pinned bytes")
    module = _FakeNeedleModule(cache_dir=tmp_path / "cache" / "v3")
    linked = needle_worker.link_base_weights(module, str(weights))
    assert Path(linked).resolve() == weights.resolve()


def test_link_replaces_a_stale_link(tmp_path):
    """A link left pointing at other weights is repointed, not trusted."""
    cache = tmp_path / "cache" / "v3"
    cache.mkdir(parents=True)
    stale = cache / "needle3.cact"
    stale.symlink_to(tmp_path / "somewhere-else.cact")
    weights = tmp_path / "needle3.cact"
    weights.write_bytes(b"pinned bytes")
    module = _FakeNeedleModule(cache_dir=cache)
    needle_worker.link_base_weights(module, str(weights))
    assert stale.resolve() == weights.resolve()


def test_link_failure_is_a_clean_error(tmp_path):
    """A library whose layout moved raises, so the request fails with an error reply."""
    module = _FakeNeedleModule(broken=True)
    with pytest.raises(RuntimeError):
        needle_worker.link_base_weights(module, str(tmp_path / "needle3.cact"))


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


def _session(module: _FakeNeedleModule) -> needle_worker.EngineSession:
    return needle_worker.EngineSession(builder=lambda spec: module.Needle(spec=spec))


def test_engine_session_resets_between_requests():
    """reset() is called before every request after the first."""
    module = _FakeNeedleModule()
    session = _session(module)
    session.select(needle_worker.EngineSpec(), "first")
    session.select(needle_worker.EngineSpec(), "second")
    assert module.instances[0].resets == 1


def test_engine_session_returns_calls_and_confidence():
    """The worker forwards the engine's raw calls plus its confidence, unvalidated."""
    session = _session(_FakeNeedleModule())
    assert session.select(needle_worker.EngineSpec(), "text") == ([GOOD_CALL], 0.75)


def test_engine_spec_is_read_off_the_request_frame():
    """The child takes its paths from the frame, coercing whatever arrives."""
    spec = needle_worker.EngineSpec.from_request(
        {"lib": "/lib.so", "weights": "/w.cact", "home": "/home", "tuned": 1}
    )
    assert spec == needle_worker.EngineSpec(
        lib="/lib.so", weights="/w.cact", home="/home", tuned=True
    )


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


def test_extract_selection_ignores_calls_on_a_non_call_envelope():
    """A "text" envelope carrying a call-shaped field is not a selection.

    Qodo #4053821266: forwarding ``function_calls`` regardless of ``type``
    would let a malformed or future text response reach the operator as a
    proposal instead of being declined.
    """
    envelope = {"type": "text", "function_calls": [GOOD_CALL], "confidence": 0.9}
    assert needle_worker.extract_selection(envelope) == ([], None)


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


# -- staging the pinned engine into a private home (nvsh/tiers/needle_home.py) --


def _staged_cache(tmp_path: Path, members: dict[str, bytes]) -> Path:
    """A tiers cache holding a pinned weights file and a wheel of *members*."""
    cache = tmp_path / "nvsh" / "tiers"
    cache.mkdir(parents=True)
    (cache / "needle3.cact").write_bytes(b"pinned weights")
    wheel = cache / "cactus_needle.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return cache


def _stage(cache: Path, monkeypatch) -> object:
    """Run stage() with fetch.resolve() answering from *cache*, never the network."""

    def resolve(kind, **_kwargs):
        return cache / ("needle3.cact" if kind == "weights" else "cactus_needle.whl")

    monkeypatch.setattr(needle_home.fetch, "resolve", resolve)
    return needle_home.stage(cache)


def test_stage_extracts_the_engine(tmp_path, monkeypatch):
    """The one needle/libneedle*.so member lands in nvsh's own cache."""
    cache = _staged_cache(tmp_path, {"needle/libneedle3.so": b"ELF-ish", "x/RECORD": b""})
    home = _stage(cache, monkeypatch)
    assert home.lib.read_bytes() == b"ELF-ish"


def test_stage_reports_a_home_directory(tmp_path, monkeypatch):
    """The worker gets a $HOME inside the cache, created and private."""
    cache = _staged_cache(tmp_path, {"needle/libneedle3.so": b"ELF-ish"})
    home = _stage(cache, monkeypatch)
    assert home.home.is_dir()


def test_stage_is_idempotent(tmp_path, monkeypatch):
    """An unchanged wheel is not extracted twice."""
    cache = _staged_cache(tmp_path, {"needle/libneedle3.so": b"ELF-ish"})
    first = _stage(cache, monkeypatch)
    stamped = first.lib.stat().st_mtime_ns
    second = _stage(cache, monkeypatch)
    assert second.lib.stat().st_mtime_ns == stamped


def test_stage_re_extracts_when_the_wheel_changes(tmp_path, monkeypatch):
    """A different wheel under the same name is staged again, not reused."""
    cache = _staged_cache(tmp_path, {"needle/libneedle3.so": b"ELF-ish"})
    _stage(cache, monkeypatch)
    replaced = _staged_cache(tmp_path / "second", {"needle/libneedle3.so": b"NEWER"})
    (cache / "cactus_needle.whl").write_bytes((replaced / "cactus_needle.whl").read_bytes())
    home = _stage(cache, monkeypatch)
    assert home.lib.read_bytes() == b"NEWER"


def test_stage_refuses_a_wheel_without_an_engine(tmp_path, monkeypatch):
    """No engine member -> a problem, not a guess at which file to use."""
    cache = _staged_cache(tmp_path, {"needle/tools.py": b"print()"})
    assert isinstance(_stage(cache, monkeypatch), FetchProblem)


def test_stage_refuses_two_engines(tmp_path, monkeypatch):
    """Two candidate members -> a problem; nvsh never picks one arbitrarily."""
    members = {"needle/libneedle3.so": b"a", "needle/libneedle2.so": b"b"}
    cache = _staged_cache(tmp_path, members)
    assert isinstance(_stage(cache, monkeypatch), FetchProblem)


def test_stage_ignores_a_traversing_member(tmp_path, monkeypatch):
    """A member reaching outside the package never matches the engine pattern."""
    members = {"needle/../../evil.so": b"nope", "needle/libneedle3.so": b"ELF-ish"}
    cache = _staged_cache(tmp_path, members)
    home = _stage(cache, monkeypatch)
    assert home.lib.name == "libneedle3.so"


def test_stage_refuses_an_oversized_engine(tmp_path, monkeypatch):
    """A member past the size bound is refused before it fills the cache."""
    monkeypatch.setattr(needle_home, "MAX_LIB_BYTES", 8)
    cache = _staged_cache(tmp_path, {"needle/libneedle3.so": b"much longer than eight"})
    assert isinstance(_stage(cache, monkeypatch), FetchProblem)


def test_stage_passes_through_a_missing_pin(tmp_path, monkeypatch):
    """Nothing prefetched -> fetch's own problem, unchanged and unraised."""
    cache = _staged_cache(tmp_path, {"needle/libneedle3.so": b"ELF-ish"})
    problem = FetchProblem(item="weights", code="missing", message="needle3.cact not cached")
    monkeypatch.setattr(needle_home.fetch, "resolve", lambda kind, **_kw: problem)
    assert needle_home.stage(cache) is problem


def test_stage_never_opens_a_socket(tmp_path, monkeypatch):
    """Staging is zipfile and os over already-verified files, nothing else."""
    source = (NVSH_ROOT / "tiers" / "needle_home.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    networking = ("urllib", "socket", "http", "requests")
    assert not [line for line in imports if any(word in line for word in networking)]
