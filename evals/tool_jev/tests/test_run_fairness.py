"""Fair sharing of one provider's sync slots across its models (full run 2026-09-26).

build.nvidia.com's queued work listed every gemma-4-31b call first, and each
gemma timeout paused the whole provider, so kimi-k3, glm-5.3 and both
nemotrons sent nothing for hours.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from evals.tool_jev import run as runner
from evals.tool_jev.providers.errors import Classification, Outcome


def _model(label: str, kind: str = "nvidia"):
    return SimpleNamespace(label=label, kind=kind)


def test_interleave_takes_one_call_per_model_in_turn():
    a, b, c = _model("nvidia/a"), _model("nvidia/b"), _model("local/c", "local")
    work = [(a, "a1"), (a, "a2"), (a, "a3"), (b, "b1"), (b, "b2"), (c, "c1")]
    order = [item for _model_, item in runner.interleave(work)]
    assert order == ["a1", "b1", "c1", "a2", "b2", "a3"]


def test_interleave_keeps_every_item_once():
    a, b = _model("x/a"), _model("x/b")
    work = [(a, i) for i in range(5)] + [(b, i) for i in range(2)]
    assert sorted(map(repr, runner.interleave(work))) == sorted(map(repr, work))


def _bare_runner(now: float = 1000.0):
    r = object.__new__(runner.Runner)
    r.paused, r.paused_until = {}, {}
    r.model_stops, r.provider_stops = {}, {}
    r._lock = threading.RLock()
    r.clock = lambda: now
    return r


def test_a_timeout_pauses_only_the_model_that_timed_out():
    r = _bare_runner()
    slow, other = _model("nvidia/slow"), _model("nvidia/other")
    r._pause_for(
        slow,
        Classification(Outcome.PENDING, "timeout", stop=True, retryable=True),
        "slow timed out",
    )
    assert r.blocked(slow)
    assert not r.blocked(other)


def test_a_rate_limit_still_pauses_the_whole_provider():
    r = _bare_runner()
    one, two = _model("nvidia/one"), _model("nvidia/two")
    r._pause_for(
        one, Classification(Outcome.PENDING, "rate_limited", stop=True, retryable=True), "429"
    )
    assert r.blocked(one) and r.blocked(two)
