"""Timing test: ``nvsh setup``'s rc block must not add meaningfully to prompt
startup latency, even against a heavy fake rc.

Builds a 300+-line fake rc (aliases, functions, a loop) and measures the
median of ``bash --rcfile <rc> -ic true`` over several runs before and after
``nvsh setup`` has inserted its block, with ``NVSH_CAPTURE=0`` (no
``script(1)`` re-exec) and ``NVSH_BIN`` pointed at ``/bin/true`` so the one
``nvsh complete --json`` call the readline layer makes at source time
degrades instantly to no palette instead of paying a real Python startup.
Both medians are printed unconditionally so a slow CI box's numbers are
visible even when the assertion is relaxed via
``NVSH_LATENCY_TOLERANCE_PCT``/``NVSH_LATENCY_FLOOR_MS``.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import statistics
import subprocess  # nosec B404 - fixed argv below, no shell=True
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from nvsh.cli._commands import setup as setup_cmd
from nvsh.rcfile import find_insert_point

BASH = shutil.which("bash") or "/bin/bash"
RUNS = int(os.environ.get("NVSH_TIMING_RUNS", "10"))
TOLERANCE_PCT = float(os.environ.get("NVSH_LATENCY_TOLERANCE_PCT", "10"))
FLOOR_MS = float(os.environ.get("NVSH_LATENCY_FLOOR_MS", "5"))


def _heavy_fake_rc() -> str:
    lines = [
        "# If not running interactively, don't do anything",
        "case $- in",
        "    *i*) ;;",
        "      *) return;;",
        "esac",
        "",
    ]
    for i in range(150):
        lines.append(f"alias nvsh_fake_alias_{i}='echo {i}'")
    for i in range(120):
        lines.append(f"nvsh_fake_fn_{i}() {{ echo fn-{i}; }}")
    lines.append("for _nvsh_i in $(seq 1 30); do : $((_nvsh_i * 2)); done")
    return "\n".join(lines) + "\n"


def _median_bash_startup(rc_path: Path, env: dict, runs: int) -> float:
    samples = []
    for _ in range(runs):
        start = time.monotonic()
        subprocess.run(  # nosec B603
            [BASH, "--rcfile", str(rc_path), "-ic", "true"],
            env=env,
            capture_output=True,
            timeout=30,
        )
        samples.append((time.monotonic() - start) * 1000.0)
    return statistics.median(samples)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_setup_does_not_meaningfully_slow_down_prompt_startup(tmp_path, capsys, monkeypatch):
    # cmd_setup bakes its own resolved NVSH_BIN into the rendered rc block
    # (so a uv-tool install works without a PATH edit); point that resolution
    # at /bin/true too, so the block's `nvsh complete --json` call at
    # readline-source time degrades instantly instead of paying a real
    # Python startup, matching the design guidance's timing-test note.
    monkeypatch.setattr(setup_cmd.render, "resolve_nvsh_bin", lambda: "/bin/true")
    home = tmp_path / "home"
    home.mkdir()
    rc = home / "fakerc"
    rc.write_text(_heavy_fake_rc())
    assert find_insert_point(rc.read_text()) > 0  # sanity: guard was found

    base_env = dict(os.environ)
    base_env.update(
        HOME=str(home),
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        TERM="dumb",
        NVSH_CAPTURE="0",
        NVSH_BIN="/bin/true",
        HISTFILE=str(tmp_path / ".bash_history"),
    )

    before_median = _median_bash_startup(rc, base_env, RUNS)

    xdg_data = tmp_path / "xdg-data"
    setup_env = dict(base_env, XDG_DATA_HOME=str(xdg_data), XDG_RUNTIME_DIR=str(tmp_path / "run"))

    with redirect_stdout(io.StringIO()):
        code = _run_setup(rc, setup_env)
    assert code == 0

    after_env = dict(setup_env)
    after_median = _median_bash_startup(rc, after_env, RUNS)

    delta_pct = ((after_median - before_median) / before_median) * 100 if before_median else 0.0
    with capsys.disabled():
        print(
            f"\nbash startup: before {before_median:.3f} ms, after {after_median:.3f} ms, "
            f"delta {after_median - before_median:.3f} ms ({delta_pct:.1f}%), "
            f"tolerance {TOLERANCE_PCT}% + {FLOOR_MS} ms floor, n={RUNS}"
        )

    allowed = before_median * (TOLERANCE_PCT / 100.0) + FLOOR_MS
    assert (after_median - before_median) < allowed


def _run_setup(rc: Path, env: dict) -> int:
    old_environ = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        args = argparse.Namespace(rc=str(rc), json=True, yes=False)
        return setup_cmd.cmd_setup(args)
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
