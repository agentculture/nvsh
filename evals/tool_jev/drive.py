"""The autonomous driver loop (deviation d2): ``continue`` until the gate run is done.

Deviation d2 packages this loop in a docker compose service (a separate task);
this module is the loop itself, runnable anywhere:

- each step is one :func:`evals.tool_jev.run.step` pass under the run lock
  (the manifest is re-read every step, so an operator's fix is picked up
  without a restart);
- a **money stop** on one provider leaves the others going; the runner
  re-checks it every ``recheck_seconds`` (default 30 minutes) with exactly
  one probe call and keeps the provider blocked until that probe's answer is
  recorded; the stop and probe state live in ``run.json``, so they survive a
  restart;
- submitted batches keep being polled every ``poll_seconds``;
- a rejected request, failed batches or a truncation stop stop only that
  model; the loop keeps polling, so a manifest fix resumes it;
- **stop and ask** (a batch lookup that cannot be resolved) exits non-zero
  with the message and never resubmits;
- any other error in a step (a configuration mistake, a bug, an adapter
  surprise) is logged and the loop backs off (2^n x ``poll_seconds``, at
  most 30 minutes) and tries again: it never crash-loops;
- one line per step is appended to ``<run_dir>/drive.log``;
- SIGTERM / SIGINT set a flag and the loop exits cleanly between steps.

The clock and sleep are injectable so tests never wait for real.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Mapping

from . import run as runner

DEFAULT_POLL_SECONDS = runner.DEFAULT_POLL_SECONDS
DEFAULT_RECHECK_SECONDS = runner.DEFAULT_RECHECK_SECONDS
LOG_FILE = "drive.log"


def _log(run_dir: Path, clock: Callable[[], float], text: str) -> None:
    line = json.dumps({"t": round(clock(), 3), "msg": text})
    with open(Path(run_dir) / LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def drive(
    run_dir: Path,
    manifest_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    factory: runner.ProviderFactory | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] | None = None,
    stop: threading.Event | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
    max_steps: int | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Run :func:`run.step` until the run completes, asks or is told to stop; the exit code."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stop = stop or threading.Event()

    def wait(seconds: float) -> None:
        if sleep is not None:
            sleep(seconds)
        else:
            stop.wait(seconds)

    steps = 0
    errors = 0
    code = runner.EXIT_OK
    while not stop.is_set():
        steps += 1
        try:
            outcome = runner.step(
                run_dir,
                manifest_path,
                env=env,
                factory=factory,
                retry_money=False,
                out=out,
                clock=clock,
                poll_seconds=poll_seconds,
                recheck_seconds=recheck_seconds,
            )
        except runner.StopAndAsk as exc:
            _log(run_dir, clock, f"stop and ask: {exc}")
            out(f"stop and ask: {exc}")
            return runner.EXIT_ASK
        except Exception as exc:  # noqa: BLE001 -- logged; the loop backs off, never crash-loops
            errors += 1
            delay = min(poll_seconds * 2**errors, runner.MAX_BACKOFF_SECONDS)
            text = f"step {steps}: error {type(exc).__name__}: {exc}; retrying in {delay:.0f} s"
            _log(run_dir, clock, text)
            out(text)
            code = runner.EXIT_ENV if isinstance(exc, runner.EnvError) else runner.EXIT_USER
            if max_steps is not None and steps >= max_steps:
                return code
            wait(delay)
            continue
        errors = 0
        _log(run_dir, clock, f"step {steps}: {outcome.status}; " + " | ".join(outcome.messages))
        if outcome.status == runner.STATUS_COMPLETE:
            return runner.EXIT_OK
        code = outcome.exit_code
        if max_steps is not None and steps >= max_steps:
            return code
        wait(poll_seconds)
    _log(run_dir, clock, "signal received: exiting cleanly between steps")
    return runner.EXIT_OK
