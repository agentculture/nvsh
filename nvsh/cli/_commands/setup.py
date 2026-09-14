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
import platform
import shutil
import subprocess  # nosec B404 - fixed argv below, no shell=True
import sys
import time
from pathlib import Path

from nvsh import __version__
from nvsh import config as nvsh_config
from nvsh import doctor_checks, installers, rcfile, runtimedir
from nvsh.agent import registry
from nvsh.cli._errors import EXIT_USER_ERROR, CliError
from nvsh.cli._output import emit_diagnostic, emit_result
from nvsh.shell import render
from nvsh.triggers import TriggerEvent, decide


#: The reachability call, injected so tests can stub it and so a hung or
#: exception-raising probe never takes ``setup`` down with it. Bound at
#: 2 seconds: ``setup`` must not hang on a dead endpoint (``check_agent_reachable``
#: takes a ``timeout`` -- the ``base_url``/openai-compat probe -- and a
#: separate ``cli_timeout`` for the ``<binary> --version``/auth probes used
#: by the CLI harnesses; both are pinned to 2.0s here).
def _check_agent_reachable(cfg) -> dict:
    return doctor_checks.check_agent_reachable(cfg, timeout=2.0, cli_timeout=2.0)


#: Exact text nvsh shows when the operator's pick is a hosted backend --
#: substituted with the picked name, no backticks in the real output.
_HOSTED_LINE = (
    "{name} is hosted: on a failure the redacted command, output and "
    "device context leave this machine"
)

#: Exact warning text for an untested platform/shell combination (issue #11).
_MACOS_ZSH_WARNING = "nvsh is not tested on macOS/zsh yet (see issue #11)"


def _agent_reachable(cfg) -> dict:
    """Call :func:`_check_agent_reachable` for the pick, never letting a
    failure or exception change ``setup``'s own outcome."""
    try:
        check = _check_agent_reachable(cfg)
        return {"passed": bool(check.get("passed")), "message": str(check.get("message", ""))}
    except Exception as exc:  # noqa: BLE001 - a probe must never fail setup
        return {"passed": False, "message": str(exc)}


def _setup_warnings() -> list[str]:
    """macOS/zsh is untested (issue #11); warn without changing behavior."""
    is_darwin = platform.system() == "Darwin"
    is_zsh = os.environ.get("SHELL", "").endswith("zsh")
    return [_MACOS_ZSH_WARNING] if (is_darwin or is_zsh) else []


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


#: How a probed harness with no approval channel is labelled in the pick
#: list, so the operator sees *before* choosing that it cannot run commands.
PLAN_MODE_LABEL = "read-only / plan mode"


def _prompt_input(prompt: str) -> str:
    """The default pick prompt. Injected (like ``run_install``'s ``confirm``)
    so tests never read stdin."""
    return input(prompt)  # nosec B322 - plain numbered menu, no eval of input


def _is_interactive() -> bool:
    """Whether there is a terminal to ask the operator on."""
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _pick_prompt(rows: list[dict]) -> str:
    """The numbered menu shown when several harnesses are installed."""
    lines = ["several agent harnesses are installed; which should nvsh use?"]
    for index, row in enumerate(rows, start=1):
        suffix = "" if row["tool_calling"] else f"  [{PLAN_MODE_LABEL}]"
        lines.append(f"  {index}) {row['name']}{suffix}")
    lines.append(f"choose 1-{len(rows)} [1]: ")
    return "\n".join(lines)


def _answer_to_name(answer: str, rows: list[dict]) -> str:
    """Map one menu answer to a harness name, defaulting to the first row.

    An empty answer takes the default (the first row, which the ordering in
    :func:`nvsh.agent.registry.probe` already made the tool-calling one).
    A number selects by position, a word selects by name; anything else
    falls back to the default rather than looping -- setup must finish.
    """
    answer = answer.strip()
    if not answer:
        return rows[0]["name"]
    if answer.isdigit():
        index = int(answer)
        if 1 <= index <= len(rows):
            return rows[index - 1]["name"]
        return rows[0]["name"]
    for row in rows:
        if row["name"] == answer:
            return answer
    return rows[0]["name"]


def _pick_from_probe(rows: list[dict], *, json_mode: bool, prompt) -> tuple[str, str]:
    """Turn a non-empty probe into ``(name, reason)``.

    One installed harness is the default silently. Several on a terminal
    always ask -- ``--yes`` answers *install* prompts, never this one, so a
    machine with both claude and qwen never gets a harness chosen for the
    operator behind their back. Without a terminal (``--json``, a pipe, a
    provisioning script) the first row of the ordered list wins.
    """
    names = ", ".join(row["name"] for row in rows)
    if len(rows) == 1:
        return rows[0]["name"], f"{rows[0]['name']} is the only agent harness on PATH"
    if json_mode or not _is_interactive():
        return (
            rows[0]["name"],
            f"installed: {names}; no terminal to ask on, picked {rows[0]['name']}",
        )
    chosen = _answer_to_name(prompt(_pick_prompt(rows)), rows)
    return chosen, f"installed: {names}; operator picked {chosen}"


def _resolve_agent(args: argparse.Namespace, cfg, prompt) -> tuple[str, str, str, list[dict], bool]:
    """Pick the harness ``setup`` will make the default.

    Returns ``(backend, reason, target, probe_rows, forced)``. ``target`` is
    the literal string written to ``[aliases].default`` -- for ``--agent
    codex/gpt-5/high`` that is the whole target, model and effort included,
    not just the backend name.

    ``--agent`` skips the probe entirely and goes through
    :func:`nvsh.agent.registry.choose`, which fails loudly (naming the
    missing binary) instead of falling back the way the unforced path does.
    """
    forced = getattr(args, "agent", None)
    if forced:
        backend, reason = registry.choose(cfg, shutil.which, forced=forced)
        return backend, reason, _canonical_target(cfg, forced), [], True

    rows = registry.probe(shutil.which, cfg)
    if not rows:
        # Nothing installed: today's path -- the configured provider if it
        # somehow resolves, else openai-compat with its key hint.
        # `which` is passed explicitly: registry.choose's default argument
        # was bound to the real shutil.which at import time, so a caller
        # (or a test) that swaps shutil.which out would otherwise be
        # ignored on exactly this fallback path.
        backend, reason = registry.choose(cfg, shutil.which)
        return backend, reason, backend, rows, False

    backend, reason = _pick_from_probe(
        rows, json_mode=bool(getattr(args, "json", False)), prompt=prompt
    )
    return backend, reason, backend, rows, False


def _canonical_target(cfg, forced: str) -> str:
    """The ``backend[/model[/effort]]`` string a forced ``--agent`` persists as.

    :meth:`nvsh.config.Config.resolve_target` neither follows an alias whose
    value is another alias nor strips a leading ``@`` from a *stored* value,
    so writing the raw ``--agent`` text (``reviewer``, ``default``,
    ``@claude``) into ``[aliases].default`` left a default that later
    resolved to an unknown backend -- or to itself. The precedence mirrors
    :func:`nvsh.agent.registry._forced_backend`: an alias spelled exactly
    wins, then a bare adapter name, then an ``@``-prefixed alias, then a
    literal target. A model that exists only as ``[agents.<backend>].model``
    is never baked in: a bare name stays bare.
    """
    bare = forced[1:] if forced.startswith("@") else forced
    if forced in cfg.aliases:
        key = forced
    elif bare in registry.ADAPTERS:
        return bare
    elif bare in cfg.aliases:
        key = bare
    else:
        return bare  # a literal 'backend/model[/effort]', already validated by choose()
    backend, model, effort, _alias = cfg.resolve_target(key)
    stored = cfg.aliases[key]
    stored = stored[1:] if stored.startswith("@") else stored
    if "/" not in stored:
        model = None  # only the [agents.<backend>].model fallback: keep it bare
    return "/".join(part for part in (backend, model, effort) if part)


def _pick_agent(
    args: argparse.Namespace, cfg, prompt
) -> tuple[str, str, str, list[dict], bool, bool]:
    """Decide the harness, keeping a usable existing default *before* asking.

    Returns ``(backend, reason, target, probe_rows, forced, kept)``. The
    keep-or-replace decision on ``[aliases].default`` runs ahead of the pick
    menu, so an operator is never asked a question whose answer would then
    be discarded in favour of the default they already had.
    """
    if not getattr(args, "agent", None):
        rows = registry.probe(shutil.which, cfg)
        existing = cfg.aliases.get(nvsh_config.DEFAULT_ALIAS)
        if existing and _keep_existing_default(cfg, rows):
            backend = existing.split("/", 1)[0]
            return backend, f"[aliases].default = {existing!r} kept", existing, rows, False, True
    return (*_resolve_agent(args, cfg, prompt), False)


def _harness_installed(install_rows: list[dict]) -> bool:
    """Whether an install that ran could have put a new agent harness on PATH.

    Only a successful install of a tool that is itself a registered adapter
    (today ``pi``) can change the probe; ``uv``/``tmux``/``node`` cannot, so
    they never re-ask the pick.
    """
    return any(
        row["ran"] and row["returncode"] == 0 and row["tool"] in registry.ADAPTERS
        for row in install_rows
    )


def _keep_existing_default(cfg, probe_rows: list[dict]) -> bool:
    """Whether an operator's own ``[aliases].default`` survives this setup.

    A default whose backend is still installed is kept -- setup only fills
    in a missing or unusable one. The exception is the sticky fallback: a
    previous run on a bare machine wrote ``openai-compat``, whose "binary"
    is always "installed", so every later run kept it even once a real
    harness appeared. When that default is openai-compat with no
    ``base_url`` configured (i.e. nvsh chose it, the operator did not point
    it anywhere) and the probe now finds a harness, it is re-probed.
    """
    try:
        backend, _model, _effort, _alias = cfg.resolve_target(nvsh_config.DEFAULT_ALIAS)
    except nvsh_config.ConfigError:
        return False
    if backend not in registry.ADAPTERS or not registry.installed(backend, shutil.which):
        return False
    if backend == "openai-compat" and probe_rows:
        if not cfg.agents.get("openai-compat", {}).get("base_url"):
            return False
    return True


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
        f"default target: {result['agent']['target']}",
        f"daemon stopped: {result['daemon_stopped']}",
    ]
    if result["agent"]["hosted"]:
        lines.append(_HOSTED_LINE.format(name=result["agent"]["name"]))
    key_hint = result["agent"]["key_hint"]
    if key_hint:
        lines.append(f"  {key_hint}")
    reachable = result["agent"]["reachable"]
    lines.append(f"agent reachable: {reachable['passed']} ({reachable['message']})")
    for row in install_rows:
        lines.append(f"{row['tool']} ({row['purpose']}): {row['command']}")
        if not offer_only:
            lines.append(f"  ran: {row['ran']} (returncode: {row['returncode']})")
    return lines


def _refuse_demo_default(args: argparse.Namespace) -> None:
    """Refuse ``--agent demo`` before any side effect (rc edit, install
    probe, daemon restart): 'demo' is a scripted fixture (see
    registry.DEMO_DEFAULT_MESSAGE), never a persisted default. A literal
    '--agent demo'/'--agent @demo' names the adapter directly, so this
    catches it without waiting on _pick_agent's alias resolution -- an
    alias whose *target* happens to be demo is unaffected, it is still a
    per-request pick until something asks for it as the default."""
    if getattr(args, "agent", None) in ("demo", "@demo"):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=registry.DEMO_DEFAULT_MESSAGE,
            remediation=registry.DEMO_DEFAULT_HINT,
        )


def cmd_setup(args: argparse.Namespace, prompt=None) -> int:
    _refuse_demo_default(args)
    rc_path = _rc_path(args)

    shell_dir = render.render_shell_files()
    nvsh_bin = render.resolve_nvsh_bin()
    block = _build_block(shell_dir, nvsh_bin)

    changed, was_present, was_edited, backup_path = _write_rc_block(rc_path, block)

    cfg = nvsh_config.load()
    prompt = _prompt_input if prompt is None else prompt
    chosen, reason, target, probe_rows, forced, keep_existing = _pick_agent(args, cfg, prompt)

    offer_only, confirm = _install_mode(args)
    # Scope the offers to the pick -- except on a bare machine, where the
    # pick is the openai-compat fallback and nothing is installed yet: there
    # the unscoped list is what lets an operator bootstrap a harness at all.
    scoped = chosen if (forced or probe_rows) else None
    missing = installers.missing_tools(shutil.which, chosen=scoped)
    install_rows, _any_ran = _process_installs(missing, offer_only=offer_only, confirm=confirm)
    if not forced and _harness_installed(install_rows):
        # A harness that just got installed only shows up on the next PATH
        # lookup, so re-probe rather than keep the pre-install pick. Any
        # other install (uv, tmux, node) cannot change the probe, so the
        # operator is not asked the same question twice.
        chosen, reason, target, probe_rows, forced, keep_existing = _pick_agent(args, cfg, prompt)

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
    default_alias_written = not keep_existing and existing != target
    if default_alias_written:
        cfg.aliases[nvsh_config.DEFAULT_ALIAS] = target
        nvsh_config.save(cfg)

    # A warm daemon holds the *previous* default's session, so it has to go
    # before the new default can take effect on the next failure.
    daemon_stopped = False
    if default_alias_written or forced:
        daemon_stopped = _stop_daemon(nvsh_bin)

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
            "target": target,
            "reason": reason,
            "key_hint": _agent_key_hint(chosen, cfg),
            "default_alias_written": default_alias_written,
            "probe": probe_rows,
            "hosted": (
                bool(registry.ADAPTERS[chosen].hosted) if chosen in registry.ADAPTERS else False
            ),
            "reachable": _agent_reachable(cfg),
        },
        "installs": install_rows,
        "daemon_stopped": daemon_stopped,
        "warnings": _setup_warnings(),
    }

    if bool(getattr(args, "json", False)):
        emit_result(result, json_mode=True)
    else:
        emit_result("\n".join(_setup_lines(result, install_rows, offer_only)), json_mode=False)
        # Warnings are diagnostics: stderr, never mixed into the stdout result.
        for warning in result["warnings"]:
            emit_diagnostic(f"warning: {warning}")
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
            # Quiet: the child's "daemon: not running" line would otherwise
            # land on *this* verb's stdout, ahead of its --json payload.
            [nvsh_bin, "daemon", "stop"],
            check=False,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
        "--agent",
        default=None,
        metavar="TARGET",
        help=(
            "Harness target to make the default: an alias or "
            "backend[/model[/effort]]; skips the probe."
        ),
    )
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
