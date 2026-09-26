"""The autonomous driver loop (deviation d2): ``continue`` until the gate run is done.

Deviation d2 packages this loop in a docker compose service (a separate task);
this module is the loop itself, runnable anywhere:

- each step is one :func:`evals.tool_jev.run.step` pass (the manifest is
  re-read every step, so an operator's fix is picked up without a restart);
- a **money stop** on one provider leaves the others going; the stopped
  provider is re-checked every ``recheck_seconds`` (default 30 minutes) with
  exactly one probe call; the stop times survive a restart in
  ``<run_dir>/drive.json``;
- submitted batches keep being polled every ``poll_seconds``;
- a rejected request or a truncation stop stops only that model; the loop
  keeps polling, so a manifest fix resumes it;
- **stop and ask** (a batch lookup that cannot be resolved) exits non-zero
  with the message and never resubmits;
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

DEFAULT_POLL_SECONDS = 60.0
DEFAULT_RECHECK_SECONDS = 30 * 60.0
LOG_FILE = "drive.log"
STATE_FILE = "drive.json"


def _log(run_dir: Path, clock: Callable[[], float], text: str) -> None:
    line = json.dumps({"t": round(clock(), 3), "msg": text})
    with open(run_dir / LOG_FILE, "a", encoding="utf-8") as handle:
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
    smoke_cases: int | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Run :func:`run.step` until the run completes, asks or is told to stop; the exit code."""
    run_dir = Path(run_dir)
    stop = stop or threading.Event()

    def wait(seconds: float) -> None:
        if sleep is not None:
            sleep(seconds)
        else:
            stop.wait(seconds)

    state_path = run_dir / STATE_FILE
    saved = runner._read_json(state_path, {}) or {}
    money_since: dict[str, float] = dict(saved.get("money_since", {}))
    steps = 0
    while not stop.is_set():
        now = clock()
        probe = {kind for kind, since in money_since.items() if now - since >= recheck_seconds}
        try:
            outcome = runner.step(
                run_dir,
                manifest_path,
                env=env,
                factory=factory,
                retry_money=False,
                probe=probe,
                smoke_cases=smoke_cases,
                out=out,
            )
        except runner.StopAndAsk as exc:
            _log(run_dir, clock, f"stop and ask: {exc}")
            out(f"stop and ask: {exc}")
            return runner.EXIT_ASK
        except runner.RunError as exc:
            _log(run_dir, clock, f"error: {exc}")
            out(f"error: {exc}")
            return runner.EXIT_ENV if isinstance(exc, runner.EnvError) else runner.EXIT_USER
        steps += 1
        for kind in list(money_since):
            if kind not in outcome.money_stopped:
                money_since.pop(kind)
                _log(run_dir, clock, f"{kind}: money stop cleared")
        for kind in outcome.money_stopped:
            if kind in probe or kind not in money_since:
                money_since[kind] = clock()
        runner._write_json(state_path, {"money_since": money_since})
        probed = f" probed {sorted(probe)}" if probe else ""
        _log(
            run_dir,
            clock,
            f"step {steps}: {outcome.status}{probed}; " + " | ".join(outcome.messages),
        )
        if outcome.status == runner.STATUS_COMPLETE:
            return runner.EXIT_OK
        if max_steps is not None and steps >= max_steps:
            return outcome.exit_code
        wait(poll_seconds)
    _log(run_dir, clock, "signal received: exiting cleanly between steps")
    return runner.EXIT_OK
