"""Timing budget for the two numbers task t19's acceptance criteria name.

Both are measured, not asserted from theory, and both print their measured
value unconditionally so the number lands in the pytest output (and from
there into ``docs/verification.md`` next to the target):

``test_success_path_overhead_under_5ms``
    Drives a real interactive bash on a pty for N successful commands with
    and without ``nvsh/shell/hook.bash`` sourced, and compares the median
    inter-prompt interval. The hook must add **< 5 ms** per prompt. The
    agent is never called: ``NVSH_AUTO=0`` plus a ``NVSH_BIN`` pointed at
    ``/bin/true`` means even a mis-classified failure would fork nothing
    interesting, and ``NVSH_CAPTURE=0`` keeps the session out of
    ``script(1)``.

``test_time_to_first_agent_text_under_2s``
    Measures, end to end from process spawn, how long it takes for the
    first agent *text* to reach the panel with :class:`nvsh.agent.fake.
    FakeAgent` behind the transport. Spawning a fresh interpreter is
    deliberate: Python startup plus ``nvsh.client``'s imports are the real
    cost on the failure path, and an in-process measurement would hide
    them. The budget is **< 2 s**.

Both skip cleanly when there is no usable bash/pty.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess  # nosec B404 - fixed argv, never shell=True
import sys
import time
from pathlib import Path

import pytest

from tests.test_hook_bash import HOOK, _base_env, _prompt_intervals, _run_bash, _source

#: Target from the t19 acceptance criteria, overridable for a loaded box.
SUCCESS_BUDGET_MS = float(os.environ.get("NVSH_TIMING_SUCCESS_MS", "5"))
FIRST_TEXT_BUDGET_S = float(os.environ.get("NVSH_TIMING_FIRST_TEXT_S", "2"))

#: Successful commands per pty session (a).
SUCCESS_RUNS = int(os.environ.get("NVSH_TIMING_SUCCESS_RUNS", "200"))
#: Repeats of the spawn-to-first-text measurement (b).
FIRST_TEXT_RUNS = int(os.environ.get("NVSH_TIMING_FIRST_TEXT_RUNS", "5"))

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The marker the child prints as its first TEXT_DELTA.
_MARKER = "NVSH-FIRST-TEXT"

_CHILD = '''\
"""Stream one FakeAgent answer into a Panel on stdout, as ``nvsh hook`` would."""
import io
import sys
import types

from nvsh import client as client_mod
from nvsh import client_transport
from nvsh import panel as panel_mod
from nvsh.agent.base import AgentEvent, EventKind
from nvsh.agent.fake import FakeAgent

MARKER = {marker!r}

agent = FakeAgent(
    [
        AgentEvent(kind=EventKind.TEXT_DELTA, text=MARKER),
        AgentEvent(kind=EventKind.DONE),
    ]
)


def _send(request, context=None, **kwargs):
    agent.start()
    yield from agent.run(request, context)


client_transport.send = _send
client_mod._platform_block = lambda: "platform: fixture"

args = types.SimpleNamespace(
    exit=2, pipestatus="2", line="ls /nope", cwd=".", log="", json=False
)
panel = panel_mod.Panel(out=sys.stdout, in_=io.StringIO(), env={{}}, isatty=False)
raise SystemExit(client_mod.handle_failure(args, panel=panel))
'''


bash_required = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available for a pty session"
)


@pytest.fixture()
def no_agent_env(tmp_path):
    """Hook env with the agent disabled outright and no session capture."""
    return _base_env(
        tmp_path,
        NVSH_AUTO="0",
        NVSH_BIN="/bin/true",
        NVSH_CAPTURE="0",
    )


@bash_required
def test_success_path_overhead_under_5ms(tmp_path, no_agent_env, capsys):
    """N successful commands cost < 5 ms more per prompt with the hook."""
    try:
        baseline = _run_bash([":", *(["true"] * SUCCESS_RUNS)], no_agent_env, timeout=300.0)
        hooked = _run_bash([_source(), *(["true"] * SUCCESS_RUNS)], no_agent_env, timeout=300.0)
    except OSError as exc:  # pragma: no cover - no pty on this box
        pytest.skip(f"no usable pty: {exc}")

    base = _prompt_intervals(baseline)
    hook = _prompt_intervals(hooked)
    if len(base) < SUCCESS_RUNS // 2 or len(hook) < SUCCESS_RUNS // 2:  # pragma: no cover
        pytest.skip("pty session produced too few prompt timestamps to measure")

    base_ms = statistics.median(base) * 1000.0
    hook_ms = statistics.median(hook) * 1000.0
    delta_ms = hook_ms - base_ms
    with capsys.disabled():
        print(
            f"\n[t19] success-path overhead: baseline median {base_ms:.3f} ms, "
            f"hooked median {hook_ms:.3f} ms, delta {delta_ms:.3f} ms "
            f"(target < {SUCCESS_BUDGET_MS} ms, n={len(hook)})"
        )
    assert HOOK.is_file()
    assert delta_ms < SUCCESS_BUDGET_MS


def _first_text_seconds(script: Path, env: dict) -> float:
    """Seconds from spawning the interpreter to the marker reaching stdout."""
    started = time.monotonic()
    proc = subprocess.Popen(  # nosec B603 - fixed argv
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
        env=env,
    )
    seen = bytearray()
    elapsed = None
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            seen.extend(chunk)
            if _MARKER.encode() in seen:
                elapsed = time.monotonic() - started
                break
    finally:
        proc.kill()
        proc.wait(timeout=10)
        errors = proc.stderr.read() if proc.stderr is not None else b""
    if elapsed is None:  # pragma: no cover - a broken child
        raise AssertionError(
            f"marker never reached stdout; rc={proc.returncode} "
            f"stdout={bytes(seen)!r} stderr={errors!r}"
        )
    return elapsed


def test_time_to_first_agent_text_under_2s(tmp_path, capsys):
    """Spawn to first FakeAgent text on the panel is < 2 s, end to end."""
    script = tmp_path / "first_text_child.py"
    script.write_text(_CHILD.format(marker=_MARKER), encoding="utf-8")

    samples = []
    for run in range(FIRST_TEXT_RUNS):
        # A fresh state dir per sample: the persisted rate limit would
        # otherwise silence every failure after the first (by design), and
        # a rate-limited run measures nothing.
        env = dict(os.environ)
        env.update(
            HOME=str(tmp_path),
            XDG_STATE_HOME=str(tmp_path / f"state-{run}"),
            XDG_CONFIG_HOME=str(tmp_path / "config"),
            XDG_RUNTIME_DIR=str(tmp_path / "run"),
            NVSH_NO_DAEMON="1",
            NO_COLOR="1",
            PYTHONPATH=str(REPO_ROOT),
        )
        samples.append(_first_text_seconds(script, env))
    median_s = statistics.median(samples)
    with capsys.disabled():
        print(
            f"\n[t19] time to first agent text (fake agent, cold interpreter): "
            f"median {median_s * 1000:.1f} ms, min {min(samples) * 1000:.1f} ms, "
            f"max {max(samples) * 1000:.1f} ms "
            f"(target < {FIRST_TEXT_BUDGET_S} s, n={FIRST_TEXT_RUNS})"
        )
    assert median_s < FIRST_TEXT_BUDGET_S
