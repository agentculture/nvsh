"""``python -m evals.tool_jev run|continue|status|smoke|drive`` (issue #64, t17).

Usage::

    uv run --group evals python -m evals.tool_jev run      --manifest M --run-dir D
        [--run-id ID] [--date YYYY-MM-DD] [--expand]
    uv run --group evals python -m evals.tool_jev continue --manifest M --run-dir D
        [--retry-rejected]
    uv run --group evals python -m evals.tool_jev status   --run-dir D [--json]
    uv run --group evals python -m evals.tool_jev smoke    --manifest M --run-dir D [--cases 10]
    uv run --group evals python -m evals.tool_jev drive    --manifest M --run-dir D
        [--poll-seconds 60] [--recheck-minutes 30]

A smoke run dir stays a smoke run under ``continue`` and ``drive``; ``run
--expand`` turns it into the full run (the smoke answers are reused).

``--manifest`` defaults to ``$NVSH_EVALS_MANIFEST`` and ``--run-dir`` to
``$NVSH_EVALS_RUN_DIR``; the run dir must be outside every git worktree.
Case sets, the ground snapshot and relative prediction paths resolve under
``$NVSH_EVALS_PRIVATE_ROOT``. Provider keys are read from the environment
variables the manifest names, at call time.

Exit codes: 0 complete; 1 configuration error; 2 environment error; 3 stop
and ask (a batch that cannot be found or ruled out; never resubmitted);
4 stopped with calls pending (money, rejected, truncation, transient);
5 waiting on submitted batches; 130 interrupted (Ctrl+C; ``continue``
resumes).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Callable, Sequence

from . import drive as drive_mod
from . import run as runner
from .manifest import ENV_MANIFEST_PATH

ENV_RUN_DIR = "NVSH_EVALS_RUN_DIR"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.tool_jev", description="Tool-Jev DeepEval release gate runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(name: str, help_text: str, *, manifest: bool = True) -> argparse.ArgumentParser:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--run-dir", default=os.environ.get(ENV_RUN_DIR))
        if manifest:
            cmd.add_argument("--manifest", default=os.environ.get(ENV_MANIFEST_PATH))
        cmd.add_argument("--json", action="store_true", help="print a JSON document")
        return cmd

    start = common("run", "start a gate run and take the first pass")
    start.add_argument("--run-id")
    start.add_argument("--date")
    start.add_argument(
        "--expand", action="store_true", help="expand a smoke run dir into the full run"
    )
    resume = common("continue", "resume a run: re-attach batches, send what is pending")
    resume.add_argument(
        "--retry-rejected",
        action="store_true",
        help="retry models stopped by a rejected request (e.g. after fixing a key)",
    )
    common("status", "per-provider counts and spend so far", manifest=False)
    smoke = common("smoke", "N test cases, both interfaces, one judge pass, cost projection")
    smoke.add_argument("--cases", type=int, default=10)
    loop = common("drive", "repeat `continue` until done (deviation d2)")
    loop.add_argument("--poll-seconds", type=float, default=drive_mod.DEFAULT_POLL_SECONDS)
    loop.add_argument(
        "--recheck-minutes", type=float, default=drive_mod.DEFAULT_RECHECK_SECONDS / 60
    )
    return parser


def _require(value: str | None, flag: str, env_name: str) -> Path:
    if not value:
        raise runner.RunError(f"{flag} is required (or set {env_name})")
    return Path(value)


def main(
    argv: Sequence[str] | None = None,
    *,
    factory: runner.ProviderFactory | None = None,
    env=None,
    out: Callable[[str], None] | None = None,
    stop: threading.Event | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    lines: list[str] = []
    emit = out or (lambda text: print(text, flush=True))
    say = lines.append if args.json else emit
    try:
        run_dir = _require(args.run_dir, "--run-dir", ENV_RUN_DIR)
        if args.command == "status":
            doc = runner.status(run_dir)
            if args.json:
                emit(json.dumps(doc, indent=2, sort_keys=True))
            else:
                for line in runner.render_status(doc):
                    emit(line)
            return runner.EXIT_OK
        manifest = _require(args.manifest, "--manifest", ENV_MANIFEST_PATH)
        kwargs = {"env": env, "factory": factory, "out": say}
        if clock is not None:
            kwargs["clock"] = clock
        if args.command == "run":
            outcome = runner.start(
                run_dir, manifest, run_id=args.run_id, date=args.date, expand=args.expand, **kwargs
            )
        elif args.command == "continue":
            outcome = runner.step(run_dir, manifest, retry_rejected=args.retry_rejected, **kwargs)
        elif args.command == "smoke":
            if args.cases < 1:
                raise runner.RunError("--cases must be at least 1")
            outcome = runner.start(run_dir, manifest, smoke_cases=args.cases, **kwargs)
        else:
            stop = stop or threading.Event()
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGTERM, signal.SIGINT):
                    signal.signal(signum, lambda *_: stop.set())
            return drive_mod.drive(
                run_dir,
                manifest,
                stop=stop,
                sleep=sleep,
                poll_seconds=args.poll_seconds,
                recheck_seconds=args.recheck_minutes * 60,
                **kwargs,
            )
        if args.json:
            emit(
                json.dumps(
                    {"status": outcome.status, "exit_code": outcome.exit_code, "messages": lines},
                    indent=2,
                )
            )
        return outcome.exit_code
    except runner.StopAndAsk as exc:
        emit(f"stop and ask: {exc}")
        return runner.EXIT_ASK
    except runner.EnvError as exc:
        emit(f"error: {exc}")
        return runner.EXIT_ENV
    except runner.RunError as exc:
        emit(f"error: {exc}")
        return runner.EXIT_USER
    except KeyboardInterrupt:
        emit("interrupted: the ledger is consistent; `continue` resumes where this stopped")
        return runner.EXIT_INTERRUPTED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
