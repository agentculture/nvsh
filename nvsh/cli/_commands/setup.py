"""``nvsh setup`` / ``uninstall`` / ``on`` / ``off`` / ``hook`` — the rc-editor,
kill-switch and hook-handoff verbs.

``setup`` renders ``nvsh/shell/*.bash`` into the user's data dir
(:mod:`nvsh.shell.render`) and inserts one small marked block into the rc
file (:mod:`nvsh.rcfile`) right after the distro's interactive guard.
``uninstall`` reverses it. ``on``/``off`` print the bash needed to rebind or
unbind the hook in the *current* shell (a subprocess cannot rebind its
parent's readline, so the marked rc block also defines a ``nvsh()`` shell
function that captures ``on``/``off`` and ``eval``s this verb's ``--shell``
output). ``hook`` is the thin verb the bash hook calls on a qualifying
failure: it classifies the event with :mod:`nvsh.triggers` and, on ``ask``,
hands off to task t13's failure client.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess  # nosec B404 - fixed argv below, no shell=True
import time
from pathlib import Path

from nvsh import __version__
from nvsh import config as nvsh_config
from nvsh import installers, rcfile, runtimedir
from nvsh.agent import registry
from nvsh.cli._errors import EXIT_USER_ERROR, CliError
from nvsh.cli._output import emit_diagnostic, emit_result
from nvsh.shell import render
from nvsh.triggers import TriggerEvent, decide

#: Help text every ``--json`` flag in this verb group shares.
_JSON_HELP = "Emit structured JSON."

# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _default_rc() -> Path:
    return Path.home() / ".bashrc"


def _rc_path(args: argparse.Namespace) -> rcfile.RcPath:
    """The validated rc file this invocation may touch.

    ``--rc`` is operator input that ends up in ``open()``/``write_text()``,
    so it never reaches the filesystem as a bare string: every read, write
    and backup below goes through the :class:`nvsh.rcfile.RcPath` returned
    here, which has already rejected ``..`` traversal, symlinks escaping
    ``$HOME``, foreign-owned files outside ``$HOME`` and non-regular files.
    """
    rc = getattr(args, "rc", None)
    try:
        return rcfile.RcPath.validate(rc if rc else _default_rc())
    except rcfile.RcPathError as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=str(exc),
            remediation=(
                "pass --rc with a regular file you own, inside your home "
                "directory and with no '..' components"
            ),
        ) from exc


def _runtime_dir() -> Path:
    """``$XDG_RUNTIME_DIR/nvsh``, else the per-uid ``<tmp>/nvsh-<uid>`` fallback.

    Shared with :mod:`nvsh.capture` and ``nvsh/shell/hook.bash`` through
    :mod:`nvsh.runtimedir`, which also owns the ownership/mode check that
    makes a world-writable temp dir safe to fall back to.
    """
    return runtimedir.runtime_dir(os.environ)


def _build_block(shell_dir: Path, nvsh_bin: str) -> str:
    body_lines = [
        f'export NVSH_HOOK_VERSION="{__version__}"',
        f'export NVSH_BIN="{nvsh_bin}"',
        # Exactly the condition the hook's own kill switch uses: a `0` means
        # the off switch is off, so the hook still loads. Skipping on any
        # non-empty value left `NVSH_DISABLE=0` shells with no hook at all.
        "[[ -n $NVSH_DISABLE && $NVSH_DISABLE != 0 ]] || "
        f'{{ source "{shell_dir}/hook.bash"; source "{shell_dir}/readline.bash"; }}',
        'nvsh() { case $1 in on|off) eval "$(command nvsh "$@" --shell)";; '
        '*) command nvsh "$@";; esac; }',
    ]
    return rcfile.build_block(body_lines)


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


def _install_step_dict(
    step: installers.InstallStep, result: installers.InstallResult | None
) -> dict:
    return {
        "tool": step.tool,
        "purpose": next((t.purpose for t in installers.TOOLS if t.name == step.tool), ""),
        "command": step.shell_line,
        "needs_sudo": step.needs_sudo,
        "executable": step.executable,
        "ran": bool(result.ran) if result is not None else False,
        "returncode": result.returncode if result is not None else None,
    }


def _process_installs(
    missing: list[installers.ToolSpec], *, offer_only: bool, confirm, which=None
) -> tuple[list[dict], bool]:
    """Turn the missing tools into JSON-able rows, running each unless ``offer_only``.

    Each step is computed **immediately before it runs**, in
    :data:`nvsh.installers.TOOLS` dependency order. Planning the whole list
    up front froze pi's step at "npm not found" on a machine with no Node,
    even though the node step installed npm seconds later, so a fresh system
    never got the agent backend installed at all.

    Returns ``(rows, any_ran)`` -- ``any_ran`` tells the caller whether it is
    worth re-running :func:`nvsh.agent.registry.choose` (a freshly installed
    ``pi`` only gets picked up on the next PATH lookup).
    """
    # Resolved at call time, never as a def-time default: `shutil.which` is
    # what the tests (and a future caller) substitute.
    lookup = shutil.which if which is None else which
    rows: list[dict] = []
    any_ran = False
    for tool in missing:
        step = tool.install_commands(lookup)
        if offer_only:
            rows.append(_install_step_dict(step, None))
            continue
        kwargs = {} if confirm is None else {"confirm": confirm}
        result = installers.run_install(step, **kwargs)
        any_ran = any_ran or result.ran
        rows.append(_install_step_dict(step, result))
    return rows, any_ran


def _agent_key_hint(chosen: str, cfg) -> str | None:
    """For ``openai-compat``, say where its bearer comes from (deviation d10).

    An operator who finished ``nvsh setup`` and then met an HTTP 401 had no
    way to know nvsh was looking only at an environment variable nothing
    exports. This line names the key file -- in placeholder spelling, never
    a resolved path -- or, when a key already resolves, where it came from.
    """
    if chosen != "openai-compat":
        return None
    outcome = nvsh_config.resolve_bearer(cfg.agents.get("openai-compat", {}))
    if outcome.diagnostic:
        return outcome.diagnostic
    if outcome.source:
        return f"bearer from {outcome.source}"
    return (
        f"no bearer resolved: put the gateway's key in "
        f"{nvsh_config.DEFAULT_KEY_FILE_DISPLAY} (mode 0600), or set "
        "api_key_file / api_key_env under [agents.openai-compat]"
    )


def _write_rc_block(rc_path: rcfile.RcPath, block: str) -> tuple[bool, bool, bool, Path | None]:
    """Insert *block* into the rc file, backing the original up when it changes.

    Returns ``(changed, was_present, was_edited, backup_path)``. Writing
    nothing when the rendered text is byte-identical is what makes ``nvsh
    setup`` idempotent.
    """
    original_text = rc_path.read_text()
    base_text, was_present, was_edited = rcfile.remove_block(original_text)
    new_text = rcfile.insert_block(base_text, block)

    changed = new_text != original_text
    backup_path: Path | None = None
    if changed:
        if rc_path.exists():
            backup_path = rc_path.write_backup(original_text)
        rc_path.write_text(new_text)
    return changed, was_present, was_edited, backup_path


def _install_mode(args: argparse.Namespace) -> tuple[bool, object]:
    """``(offer_only, confirm)`` for :func:`_process_installs`, from the flags."""
    if bool(getattr(args, "no_install", False)):
        return True, None
    if bool(getattr(args, "yes", False)):
        return False, lambda _msg: True
    if bool(getattr(args, "json", False)):
        # --json is non-interactive by construction: there is no terminal
        # to prompt on, so list the offers and run nothing without --yes.
        return True, None
    return False, None  # run_install's own input() prompt


def _setup_lines(result: dict, install_rows: list[dict], offer_only: bool) -> list[str]:
    """The text-mode rendering of :func:`cmd_setup`'s result."""
    lines = [
        f"rc: {result['rc']}",
        f"block inserted: {result['block_inserted']}",
        f"backup: {result['backup']}",
        f"shell files: {result['shell_dir']}",
        f"nvsh bin: {result['nvsh_bin']}",
        f"agent: {result['agent']['name']} ({result['agent']['reason']})",
    ]
    key_hint = result["agent"]["key_hint"]
    if key_hint:
        lines.append(f"  {key_hint}")
    for row in install_rows:
        lines.append(f"{row['tool']} ({row['purpose']}): {row['command']}")
        if not offer_only:
            lines.append(f"  ran: {row['ran']} (returncode: {row['returncode']})")
    return lines


def cmd_setup(args: argparse.Namespace) -> int:
    rc_path = _rc_path(args)

    shell_dir = render.render_shell_files()
    nvsh_bin = render.resolve_nvsh_bin()
    block = _build_block(shell_dir, nvsh_bin)

    changed, was_present, was_edited, backup_path = _write_rc_block(rc_path, block)

    cfg = nvsh_config.load()
    chosen, reason = registry.choose(cfg)

    offer_only, confirm = _install_mode(args)
    missing = installers.missing_tools(which=shutil.which)
    install_rows, any_ran = _process_installs(missing, offer_only=offer_only, confirm=confirm)
    if any_ran:
        chosen, reason = registry.choose(cfg)

    # Persist the chosen backend as `[aliases].default` -- what `'default'`
    # (and a bare `--agent`) resolves to from here on -- without touching
    # `[agent] provider`, so a fallback pick (e.g. openai-compat because the
    # configured provider isn't on PATH yet) never silently overwrites the
    # operator's own intent in `[agent]`. Only written when it would change,
    # so a repeat `nvsh setup` on an already-current config stays idempotent.
    # An explicit default alias the operator already wrote (possibly with a
    # model and effort, e.g. "claude/opus/high") is kept whenever its backend
    # is still usable; setup only fills in a missing or unusable default.
    existing = cfg.aliases.get(nvsh_config.DEFAULT_ALIAS)
    keep_existing = False
    if existing:
        try:
            existing_backend, _m, _e, _a = cfg.resolve_target(nvsh_config.DEFAULT_ALIAS)
            keep_existing = existing_backend in registry.ADAPTERS and registry.installed(
                existing_backend, shutil.which
            )
        except nvsh_config.ConfigError:
            keep_existing = False
    if keep_existing:
        chosen, reason = existing.split("/", 1)[0], f"[aliases].default = {existing!r} kept"
    default_alias_written = not keep_existing and existing != chosen
    if default_alias_written:
        cfg.aliases[nvsh_config.DEFAULT_ALIAS] = chosen
        nvsh_config.save(cfg)

    result = {
        "rc": str(rc_path),
        "block_inserted": changed,
        "block_was_present": was_present,
        "block_was_edited": was_edited,
        "backup": str(backup_path) if backup_path else None,
        "shell_dir": str(shell_dir),
        "nvsh_bin": nvsh_bin,
        "agent": {
            "name": chosen,
            "reason": reason,
            "key_hint": _agent_key_hint(chosen, cfg),
            "default_alias_written": default_alias_written,
        },
        "installs": install_rows,
    }

    if bool(getattr(args, "json", False)):
        emit_result(result, json_mode=True)
    else:
        emit_result("\n".join(_setup_lines(result, install_rows, offer_only)), json_mode=False)
    return 0


# --------------------------------------------------------------------------
# uninstall
# --------------------------------------------------------------------------


def _stop_daemon(nvsh_bin: str) -> bool:
    """Stop a running daemon if the (parallel) t12 daemon verb exists.

    Detected with ``importlib.util.find_spec`` — this checks the module is
    importable *without* importing it (``nvsh.daemon`` is not this task's to
    depend on). When it is not present, the socket is unlinked directly.

    Returns whether the daemon actually reported itself stopped: ignoring
    ``daemon stop``'s exit status made uninstall claim a clean shutdown
    while an orphaned agent process was still running.
    """
    if importlib.util.find_spec("nvsh.daemon") is None:
        return False
    try:
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell
            [nvsh_bin, "daemon", "stop"], check=False, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _restore_rc(rc_path: rcfile.RcPath) -> tuple[bool, bool, str | None]:
    """Strip the block out of the rc file. Returns ``(removed, edited, restored_from)``.

    A block that was hand-edited since ``nvsh setup`` wrote it cannot be
    removed by text surgery alone without risking the operator's own edits,
    so the newest backup is restored wholesale when there is one.
    """
    original_text = rc_path.read_text()
    stripped_text, removed, edited = rcfile.remove_block(original_text)
    if not removed:
        return False, edited, None

    restored_from = None
    new_text = stripped_text
    if edited:
        backup = rc_path.newest_backup()
        if backup is not None:
            new_text = backup.read_text(encoding="utf-8")
            restored_from = str(backup)
    if new_text != original_text:
        rc_path.write_text(new_text)
    return True, edited, restored_from


def _remove_shell_files() -> list[str]:
    """Delete the rendered ``hook.bash``/``readline.bash``; return what went."""
    shell_dir = render.data_dir() / "shell"
    removed_files: list[str] = []
    if not shell_dir.exists():
        return removed_files
    for name in render.SHELL_FILES:
        f = shell_dir / name
        if f.exists():
            f.unlink()
            removed_files.append(str(f))
    return removed_files


def _remove_runtime_files() -> list[str]:
    """Delete this user's capture logs, notices and daemon socket."""
    runtime_dir = _runtime_dir()
    removed: list[str] = []
    if not runtime_dir.exists():
        return removed
    for pattern in ("*.log", "*.notice"):
        for path in runtime_dir.glob(pattern):
            path.unlink()
            removed.append(str(path))
    sock = runtime_dir / "daemon.sock"
    if sock.exists():
        sock.unlink()
        removed.append(str(sock))
    return removed


def cmd_uninstall(args: argparse.Namespace) -> int:
    rc_path = _rc_path(args)
    removed, edited, restored_from_backup = _restore_rc(rc_path)

    removed_files = _remove_shell_files()

    # Stop the daemon *before* touching its socket: unlinking first leaves a
    # live daemon bound to a pathname nothing can reach any more, so
    # `daemon stop` cannot connect and the orphan survives to its own timeout.
    daemon_stopped = _stop_daemon(render.resolve_nvsh_bin())

    removed_runtime = _remove_runtime_files()

    result = {
        "rc": str(rc_path),
        "block_removed": removed,
        "block_was_edited": edited,
        "restored_from_backup": restored_from_backup,
        "removed_files": removed_files,
        "removed_runtime": removed_runtime,
        "daemon_stopped": daemon_stopped,
    }

    if bool(getattr(args, "json", False)):
        emit_result(result, json_mode=True)
    else:
        lines = [
            f"rc: {result['rc']}",
            f"block removed: {removed}",
            f"restored from backup: {restored_from_backup}",
            f"removed files: {len(removed_files) + len(removed_runtime)}",
            f"daemon stopped: {daemon_stopped}",
        ]
        emit_result("\n".join(lines), json_mode=False)
    return 0


# --------------------------------------------------------------------------
# on / off
# --------------------------------------------------------------------------


def _off_bash() -> str:
    return "__nvsh_hook_unload; __nvsh_readline_unbind; export NVSH_DISABLE=1"


def _on_bash() -> str:
    shell_dir = render.data_dir() / "shell"
    return (
        "unset NVSH_DISABLE; "
        f'source "{shell_dir}/hook.bash"; '
        f'source "{shell_dir}/readline.bash"'
    )


def _print_toggle(args: argparse.Namespace, action: str, bash: str) -> None:
    """Print the toggle's bash in whichever of the three shapes was asked for."""
    if getattr(args, "shell", False):
        emit_result(bash, json_mode=False)
        return
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result({"action": action, "eval": bash}, json_mode=True)
    else:
        emit_result(f'run: eval "$(nvsh {action})"\n{bash}', json_mode=False)


def cmd_off(args: argparse.Namespace) -> int:
    _print_toggle(args, "off", _off_bash())
    return 0


def cmd_on(args: argparse.Namespace) -> int:
    _print_toggle(args, "on", _on_bash())
    return 0


# --------------------------------------------------------------------------
# hook
# --------------------------------------------------------------------------


def _notice_path() -> Path:
    runtime_dir = _runtime_dir()
    shell_pid = os.environ.get("NVSH_SHELL_PID") or str(os.getppid())
    return runtime_dir / f"{shell_pid}.notice"


def _maybe_print_refresh_notice() -> None:
    hook_version = os.environ.get("NVSH_HOOK_VERSION")
    if not hook_version or hook_version == __version__:
        return
    notice_file = _notice_path()
    if notice_file.exists():
        return
    try:
        runtimedir.ensure_private(notice_file.parent)
        notice_file.write_text("1", encoding="utf-8")
    except (OSError, runtimedir.RuntimeDirError):
        pass
    emit_diagnostic(f"nvsh: hook files are from {hook_version}, run 'nvsh setup' to refresh")


def cmd_hook(args: argparse.Namespace) -> int:
    _maybe_print_refresh_notice()

    pipestatus_raw = getattr(args, "pipestatus", "") or ""
    try:
        pipestatus = tuple(int(x) for x in pipestatus_raw.split())
    except ValueError:
        pipestatus = ()

    event = TriggerEvent(
        command=args.line,
        exit_code=args.exit,
        pipestatus=pipestatus,
        now=time.time(),
    )
    decision = decide(event)
    if decision.action == "skip":
        return 0

    try:
        from nvsh.client import handle_failure
    except ImportError:
        emit_diagnostic(f"nvsh: {args.line} failed (exit {args.exit}); agent client not installed")
        return 0
    return handle_failure(args)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    setup_p = sub.add_parser(
        "setup",
        help="Render the bash hook files and insert the rc block (see 'nvsh explain setup').",
    )
    setup_p.add_argument("--rc", default=None, help="rc file to edit (default: ~/.bashrc).")
    setup_p.add_argument("--json", action="store_true", help=_JSON_HELP)
    setup_p.add_argument(
        "--yes",
        action="store_true",
        help="Install every detected missing helper tool (pi, node, uv, tmux) without prompting.",
    )
    setup_p.add_argument(
        "--no-install",
        action="store_true",
        help="List missing helper tools and their install commands, but install nothing.",
    )
    setup_p.set_defaults(func=cmd_setup)

    uninstall_p = sub.add_parser(
        "uninstall",
        help="Remove the rc block, rendered files, sockets, logs and daemon "
        "(see 'nvsh explain uninstall').",
    )
    uninstall_p.add_argument("--rc", default=None, help="rc file to edit (default: ~/.bashrc).")
    uninstall_p.add_argument("--json", action="store_true", help=_JSON_HELP)
    uninstall_p.set_defaults(func=cmd_uninstall)

    off_p = sub.add_parser(
        "off",
        help="Print the bash that unbinds the hook in the current shell (see 'nvsh explain off').",
    )
    off_p.add_argument(
        "--shell", action="store_true", help="Print raw bash only (for eval), no JSON/text wrap."
    )
    off_p.add_argument("--json", action="store_true", help=_JSON_HELP)
    off_p.set_defaults(func=cmd_off)

    on_p = sub.add_parser(
        "on",
        help="Print the bash that rebinds the hook in the current shell (see 'nvsh explain on').",
    )
    on_p.add_argument(
        "--shell", action="store_true", help="Print raw bash only (for eval), no JSON/text wrap."
    )
    on_p.add_argument("--json", action="store_true", help=_JSON_HELP)
    on_p.set_defaults(func=cmd_on)

    hook_p = sub.add_parser(
        "hook",
        help="Internal: called by the bash hook on a qualifying failure "
        "(see 'nvsh explain hook').",
    )
    hook_p.add_argument("--exit", type=int, required=True, help="The failed command's exit code.")
    hook_p.add_argument("--pipestatus", default="", help="Space-separated PIPESTATUS values.")
    hook_p.add_argument("--line", required=True, help="The command line that failed.")
    hook_p.add_argument("--cwd", required=True, help="The shell's working directory.")
    hook_p.add_argument("--log", default="", help="Path of the session capture log, if any.")
    hook_p.add_argument("--json", action="store_true", help=_JSON_HELP)
    hook_p.set_defaults(func=cmd_hook)
