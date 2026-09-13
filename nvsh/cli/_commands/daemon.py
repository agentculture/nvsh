"""``nvsh daemon`` — run, inspect and stop the per-user session daemon.

The daemon (:mod:`nvsh.daemon`) owns the warm agent processes and one
conversation per shell. The hook client starts it lazily on the first
qualifying failure, so these verbs are for operators and for the bash
integration's ``EXIT`` trap:

* ``nvsh daemon run [--foreground]`` — serve (in this process, or detached)
* ``nvsh daemon status [--json]``    — report shells, agents and backend
* ``nvsh daemon stop``              — stop a running daemon
* ``nvsh daemon unregister --shell <pid>`` — this shell exited
"""

from __future__ import annotations

import argparse

from nvsh import client_transport
from nvsh import daemon as daemon_mod
from nvsh.cli._errors import EXIT_ENV_ERROR, CliError
from nvsh.cli._output import emit_result


def _json(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False))


def cmd_daemon_run(args: argparse.Namespace) -> int:
    from nvsh.config import ConfigError, load

    try:
        config = load()
    except ConfigError as err:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"config is unusable: {err}",
            remediation="fix $XDG_CONFIG_HOME/nvsh/config.toml, or delete it to use defaults",
        ) from err

    idle_timeout = float(args.idle_timeout)
    if not args.foreground:
        pid = daemon_mod.spawn(idle_timeout=idle_timeout)
        result = {"started": True, "pid": pid, "socket": str(daemon_mod.socket_path())}
        emit_result(result if _json(args) else f"daemon started (pid {pid})", json_mode=_json(args))
        return 0

    daemon = daemon_mod.Daemon(config, idle_timeout=idle_timeout)
    try:
        daemon.serve_forever()
    except OSError as err:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"could not serve on {daemon_mod.socket_path()}: {err}",
            remediation="run 'nvsh daemon stop', then try again",
        ) from err
    return 0


def cmd_daemon_status(args: argparse.Namespace) -> int:
    state = client_transport.status()
    if _json(args):
        emit_result(state, json_mode=True)
        return 0
    if not state.get("running"):
        emit_result(f"daemon: not running ({state.get('socket', '')})", json_mode=False)
        return 0
    lines = [
        f"daemon: running (pid {state.get('pid')})",
        f"socket: {state.get('socket')}",
        f"backend: {state.get('backend')} — {state.get('backend_reason', '')}".rstrip(" —"),
        f"shells: {', '.join(state.get('shells', [])) or 'none'}",
        f"agents: {state.get('agents')}",
    ]
    notice = state.get("fallback_notice")
    if notice:
        lines.append(f"note: {notice}")
    emit_result("\n".join(lines), json_mode=False)
    return 0


def cmd_daemon_stop(args: argparse.Namespace) -> int:
    stopped = client_transport.stop()
    if _json(args):
        emit_result({"stopped": stopped}, json_mode=True)
    else:
        emit_result("daemon stopped" if stopped else "daemon: not running", json_mode=False)
    return 0


def cmd_daemon_unregister(args: argparse.Namespace) -> int:
    running = client_transport.unregister(shell_id=args.shell)
    if _json(args):
        emit_result({"running": running, "shell": str(args.shell)}, json_mode=True)
    else:
        emit_result(
            f"unregistered {args.shell}" if running else "daemon: not running",
            json_mode=False,
        )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "daemon",
        help="Run, inspect or stop the per-user session daemon " "(see 'nvsh explain daemon').",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_daemon_status, json=False)
    noun_sub = p.add_subparsers(dest="daemon_command", parser_class=type(p))

    run = noun_sub.add_parser("run", help="Serve the session daemon.")
    run.add_argument(
        "--foreground", action="store_true", help="Serve in this process instead of detaching."
    )
    run.add_argument(
        "--idle-timeout",
        type=float,
        default=daemon_mod.DEFAULT_IDLE_TIMEOUT,
        help="Seconds of inactivity after which the daemon exits.",
    )
    run.add_argument("--json", action="store_true", help="Emit structured JSON.")
    run.set_defaults(func=cmd_daemon_run)

    status = noun_sub.add_parser("status", help="Report the daemon's shells, agents and backend.")
    status.add_argument("--json", action="store_true", help="Emit structured JSON.")
    status.set_defaults(func=cmd_daemon_status)

    stop = noun_sub.add_parser("stop", help="Stop a running daemon (idempotent).")
    stop.add_argument("--json", action="store_true", help="Emit structured JSON.")
    stop.set_defaults(func=cmd_daemon_stop)

    unregister = noun_sub.add_parser(
        "unregister", help="Tell the daemon a shell exited; the last one stops it."
    )
    unregister.add_argument("--shell", required=True, help="The shell's pid (bash's $$).")
    unregister.add_argument("--json", action="store_true", help="Emit structured JSON.")
    unregister.set_defaults(func=cmd_daemon_unregister)
