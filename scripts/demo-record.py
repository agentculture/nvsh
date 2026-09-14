#!/usr/bin/env python3
"""Re-record the README demo in a sandbox, scrubbed, one ``.cast`` per device.

The recording has to be *nvsh*, not a mock of it: a real interactive bash
with the real hook in its ``PROMPT_COMMAND``, a real failure, the real
failure client, the real daemon and the real panel. What it must not be is a
recording of the operator's machine. So this driver builds a throwaway
``$HOME`` under a temp directory, points every XDG variable at it, and runs
the whole session in there:

* ``XDG_CONFIG_HOME`` -- a ``config.toml`` whose ``[aliases].default`` is
  ``demo``, the fixture adapter (``nvsh/agent/demo.py``). No harness, no key,
  no network, same answer on every box.
* ``XDG_DATA_HOME`` -- ``nvsh.shell.render`` renders ``hook.bash`` and
  ``readline.bash`` here, exactly as ``nvsh setup`` would. The sandbox rc
  file sources those rendered copies and mirrors the two exports the marked
  rc block carries (``NVSH_HOOK_VERSION``, ``NVSH_BIN``); ``nvsh setup``
  itself is never run, because it would edit a real rc file.
* ``XDG_STATE_HOME`` -- fresh per run, so the persisted auto-call rate limit
  (``rate.json``) cannot hold back the second of two recordings.
* ``XDG_RUNTIME_DIR`` -- the daemon socket lives here, and :func:`stop_daemon`
  stops it before the driver exits, so nothing of the recording outlives it.

The operator's own ``~/.bashrc``, config, data, state and runtime
directories are never read and never written: the only rc file involved is
the one written into the sandbox, and bash is started with
``--rcfile <sandbox>/.bashrc``.

Everything that leaves the pty is scrubbed by ``record-cast.py
--replace-from`` before it is written, so the machine's hostname, the
operator's user name and its LAN addresses never reach the ``.cast`` file --
not even transiently, and not in this process's argv either.

    python3 scripts/demo-record.py docs/demos/failure.cast

Stdlib only, and no third-party recorder: ``scripts/record-cast.py`` is the
pty driver (see its module docstring for why asciinema is not assumed).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import socket
import subprocess  # nosec B404 - fixed argv below, never shell=True
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # `python3 scripts/demo-record.py` from anywhere
    sys.path.insert(0, str(REPO_ROOT))

from nvsh import __version__  # noqa: E402  (after the sys.path bootstrap)
from nvsh.shell import render  # noqa: E402

#: The pty recorder this driver drives.
RECORD_CAST = Path(__file__).resolve().with_name("record-cast.py")

#: The script the demo plants without its execute bit. ``./`` matters: it is
#: what ``nvsh.agent.demo.script_from_command`` reads off the failing line.
SCRIPT_NAME = "run-model.sh"

#: What the script prints once the proposal has been approved and it is run
#: again -- the line that makes the fix visible in the recording.
SUCCESS_LINE = "model loaded, 42 tokens/s"

#: The sandbox prompt. Short, and with no host or path in it, so the
#: recording carries nothing to scrub in the first place.
PROMPT = "nvsh$ "

#: Terminal geometry of the recording. 100x34 fits the panel without wrapping.
COLS, ROWS = 100, 34

#: Replacement values. RFC 5737 / RFC 2606 style placeholders, so a reader of
#: the cast can tell at a glance that these are not real.
HOST_PLACEHOLDER = "demo-box"
USER_PLACEHOLDER = "operator"
IPV4_PLACEHOLDER = "192.0.2.10"

#: A scrub rule shorter than this is refused: replacing a two-character user
#: name everywhere would shred the recording rather than redact it.
MIN_SCRUB_LEN = 3

#: Seconds the feed waits after each step. Generous on purpose: ``--feed``
#: takes fixed waits, and a Jetson under load pays Python startup, the
#: daemon's first spawn and the panel's stream on the first one.
WAIT_PANEL = 10.0
WAIT_APPROVE = 6.0
WAIT_RETRY = 6.0

_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


# ---------------------------------------------------------------------------
# scrubbing
# ---------------------------------------------------------------------------


def local_ipv4s(output: str | None = None) -> list[str]:
    """Every non-loopback IPv4 this machine answers on (``hostname -I``)."""
    if output is None:
        try:
            proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
                ["hostname", "-I"], capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return []
        output = proc.stdout
    found = []
    for token in (output or "").split():
        if _IPV4_RE.match(token) and not token.startswith("127.") and token not in found:
            found.append(token)
    return found


def protected_words() -> tuple[str, ...]:
    """Words the scrub must leave alone: the detected platform kind.

    The demo reply names the platform (``dgx-spark``, ``jetson``); a user or
    host name that is a substring of it must not be rewritten there.
    """
    try:
        from nvsh.platform import detect

        return (str(detect().kind),)
    except Exception:  # noqa: BLE001 - detection failing must not stop a recording
        return ()


def scrub_rules(host: str, user: str, ips: list[str], protect: tuple[str, ...] = ()) -> list[str]:
    """``OLD=NEW`` lines for ``record-cast.py --replace-from``.

    Longest first: the hostname often *contains* the user name
    (``spark-f8a9`` / ``spark``), and a shorter rule applied first would
    leave a half-replaced hostname behind. Tokens shorter than
    :data:`MIN_SCRUB_LEN` are dropped rather than applied blindly, and so
    is any token that occurs inside a *protect* string: the operator on a
    DGX Spark is often called ``spark``, and blindly scrubbing that turns
    the platform kind ``dgx-spark`` in the demo's own reply into
    ``dgx-operator``. The protected words are the platform kind and the
    fixture's placeholders' hosts -- text the recording must keep.
    """
    pairs = [(host, HOST_PLACEHOLDER), (user, USER_PLACEHOLDER)]
    pairs += [(ip, IPV4_PLACEHOLDER) for ip in ips]
    rules = []
    seen = set()
    for old, new in sorted(pairs, key=lambda pair: len(pair[0]), reverse=True):
        old = (old or "").strip()
        if len(old) < MIN_SCRUB_LEN or old in seen or "=" in old or "\n" in old:
            continue
        if any(old in word for word in protect):
            print(
                f"demo-record: not scrubbing {old!r}: part of {[w for w in protect if old in w]}",
                file=sys.stderr,
            )
            continue
        seen.add(old)
        rules.append(f"{old}={new}")
    return rules


# ---------------------------------------------------------------------------
# the sandbox
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sandbox:
    """Every path the recording is allowed to touch."""

    root: Path
    home: Path
    config: Path
    data: Path
    state: Path
    runtime: Path
    rc: Path
    nvsh_bin: str

    @property
    def script(self) -> Path:
        return self.home / SCRIPT_NAME


def resolve_nvsh_bin(sandbox_bin_dir: Path) -> str:
    """An ``nvsh`` entrypoint that runs *this* nvsh on this device.

    ``shutil.which("nvsh")`` is preferred, but only when it really is the
    package this driver imported: a box can easily carry an older ``uv tool
    install nvsh`` on ``PATH`` (the dev box does), and recording a demo with
    an nvsh that predates the ``demo`` adapter would fail in a way that looks
    like nvsh is broken. When the versions disagree -- or there is no
    ``nvsh`` on ``PATH`` at all -- a three-line wrapper in the sandbox runs
    ``<this interpreter> -m nvsh`` with the imported package on
    ``PYTHONPATH`` instead.
    """
    found = shutil.which("nvsh")
    if found and _reports_version(found):
        return str(Path(found).resolve())

    package_root = Path(render.__file__).resolve().parent.parent.parent
    wrapper = sandbox_bin_dir / "nvsh"
    sandbox_bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/bin/sh\n"
        f'PYTHONPATH="{package_root}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f'exec "{sys.executable}" -m nvsh "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return str(wrapper)


def _reports_version(path: str) -> bool:
    try:
        proc = subprocess.run(  # nosec B603 - argv is a resolved executable path
            [path, "--version"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.stdout.strip().endswith(__version__)


def config_toml() -> str:
    """The sandbox's ``config.toml``: the demo adapter and nothing else."""
    return (
        "# nvsh demo sandbox -- written by scripts/demo-record.py.\n"
        "[agent]\n"
        'provider = "demo"\n'
        "\n"
        "[aliases]\n"
        'default = "demo"\n'
        "\n"
        "[sessions]\n"
        "max = 1\n"
    )


def rc_text(shell_dir: Path, nvsh_bin: str) -> str:
    """The sandbox rc file: the marked block's body, plus a bare prompt.

    The three middle lines are ``nvsh/cli/_commands/setup.py``'s
    ``_build_block`` verbatim in shape (the two exports, the kill-switch
    guard around the two ``source`` calls, and the ``nvsh`` shell function
    that captures ``on``/``off``). ``nvsh setup`` is not run: it edits a real
    rc file, and the point of this driver is that it does not.
    """
    return f"""\
# nvsh demo sandbox rc -- written by scripts/demo-record.py, never sourced
# from a real ~/.bashrc. Interactive guard first, exactly as a distro rc has
# it, then the body of the block `nvsh setup` would insert.
case $- in
    *i*) ;;
    *) return ;;
esac

cd "$HOME" || true
PS1='{PROMPT}'
PS2='> '

export NVSH_HOOK_VERSION="{__version__}"
export NVSH_BIN="{nvsh_bin}"
[[ -n $NVSH_DISABLE && $NVSH_DISABLE != 0 ]] || \
{{ source "{shell_dir}/hook.bash"; source "{shell_dir}/readline.bash"; }}
nvsh() {{ case $1 in on|off) eval "$("${{NVSH_BIN:-nvsh}}" "$@" --shell)";; \
*) "${{NVSH_BIN:-nvsh}}" "$@";; esac; }}
"""


def script_text() -> str:
    """The planted script: it works, it just is not executable yet."""
    return f'#!/bin/sh\necho "{SUCCESS_LINE}"\n'


def build_sandbox(root: Path) -> Sandbox:
    """Lay out a throwaway ``$HOME`` under *root* and return its paths."""
    home = root / "home"
    sandbox = Sandbox(
        root=root,
        home=home,
        config=home / ".config",
        data=home / ".local" / "share",
        state=home / ".local" / "state",
        runtime=root / "run",
        rc=home / ".bashrc",
        nvsh_bin="",
    )
    for path in (sandbox.home, sandbox.config, sandbox.data, sandbox.state):
        path.mkdir(parents=True, exist_ok=True)
    # The daemon refuses a runtime dir that is not a 0700 directory it owns
    # (nvsh.runtimedir.ensure_private), so create it that way up front.
    sandbox.runtime.mkdir(parents=True, exist_ok=True)
    sandbox.runtime.chmod(0o700)

    nvsh_bin = resolve_nvsh_bin(root / "bin")
    sandbox = replace(sandbox, nvsh_bin=nvsh_bin)

    config_dir = sandbox.config / "nvsh"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(config_toml(), encoding="utf-8")

    shell_dir = render.render_shell_files(sandbox.data / "nvsh")
    sandbox.rc.write_text(rc_text(shell_dir, nvsh_bin), encoding="utf-8")

    sandbox.script.write_text(script_text(), encoding="utf-8")
    sandbox.script.chmod(0o644)  # the failure: found, but not executable

    # Ubuntu's /etc/bash.bashrc prints its four-line sudo lecture into every
    # interactive shell whose $HOME has no marker file. A real operator's
    # home has one; a fresh sandbox would open the recording with it.
    (sandbox.home / ".sudo_as_admin_successful").touch()
    return sandbox


def sandbox_env(sandbox: Sandbox) -> dict[str, str]:
    """The environment the recorded bash starts from.

    ``record-cast.py --clean-env`` has already reduced the parent env to an
    allowlist; these override every variable that could still point at the
    operator's own directories, and pin the knobs the recording depends on.
    ``NVSH_CAPTURE=0`` keeps the session out of ``script(1)``: the capture
    path re-``exec``s bash inside the recording pty, and the demo's answer
    comes from a committed fixture, so the captured output slice would
    change nothing on screen.
    """
    return {
        "HOME": str(sandbox.home),
        "XDG_CONFIG_HOME": str(sandbox.config),
        "XDG_DATA_HOME": str(sandbox.data),
        "XDG_STATE_HOME": str(sandbox.state),
        "XDG_RUNTIME_DIR": str(sandbox.runtime),
        "XDG_CACHE_HOME": str(sandbox.home / ".cache"),
        "TERM": "xterm-256color",
        "NVSH_BIN": sandbox.nvsh_bin,
        "NVSH_DISABLE": "0",
        "NVSH_CAPTURE": "0",
        "NVSH_AUTO": "1",
        "NVSH_WRAPPED": "",
        "NVSH_LOG": "",
        "NVSH_DEBUG": "",
        "NVSH_HOOK_DEBUG_FILE": "",
        "NVSH_NO_DAEMON": "",  # the recording must go through the real daemon
    }


# ---------------------------------------------------------------------------
# driving the recording
# ---------------------------------------------------------------------------


def feed_steps(waits: tuple[float, float, float]) -> list[str]:
    """The three scripted steps: fail, approve, retry.

    nvsh proposes and the operator approves; the *fix* runs, and then the
    operator runs the original command again. nvsh never re-runs it for
    them, so the retry is a third typed line, not something Enter did.
    """
    panel, approve, retry = waits
    return [
        f"./{SCRIPT_NAME}\\r|{panel}",
        f"\\r|{approve}",
        f"./{SCRIPT_NAME}\\r|{retry}",
    ]


def record_command(
    sandbox: Sandbox, out: Path, rules_file: Path, waits: tuple[float, float, float]
) -> list[str]:
    """The full ``record-cast.py record`` argv."""
    argv = [
        sys.executable,
        str(RECORD_CAST),
        "record",
        str(out),
        "--clean-env",
        "--cols",
        str(COLS),
        "--rows",
        str(ROWS),
        "--timestamp",
        "0",
        "--title",
        "nvsh: a failing command, diagnosed and fixed",
        "--replace-from",
        str(rules_file),
    ]
    for name, value in sandbox_env(sandbox).items():
        argv += ["--env", f"{name}={value}"]
    for step in feed_steps(waits):
        argv += ["--feed", step]
    argv += ["--", "bash", "--rcfile", str(sandbox.rc), "-i"]
    return argv


def stop_daemon(sandbox: Sandbox) -> bool:
    """Stop the session daemon the client auto-started inside the sandbox.

    Runs ``nvsh daemon stop`` with the sandbox's own ``XDG_RUNTIME_DIR``, so
    it can only ever reach the daemon this recording started. Never raises:
    a driver that failed to stop a daemon must still report the recording it
    made.
    """
    env = dict(os.environ)
    env.update(sandbox_env(sandbox))
    try:
        proc = subprocess.run(  # nosec B603 - argv from resolve_nvsh_bin, no shell
            [sandbox.nvsh_bin, "daemon", "stop", "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"demo-record: could not stop the daemon: {exc}", file=sys.stderr)
        return False
    return proc.returncode == 0


def record(out: Path, waits: tuple[float, float, float], keep: bool = False) -> Path:
    """Record one scrubbed ``.cast`` at *out*. Returns *out*."""
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="nvsh-demo-"))
    sandbox = build_sandbox(root)
    rules = scrub_rules(
        socket.gethostname(), getpass.getuser(), local_ipv4s(), protect=protected_words()
    )
    rules_file = root / "scrub.rules"
    rules_file.write_text("\n".join(rules) + "\n", encoding="utf-8")

    try:
        subprocess.run(  # nosec B603 - argv built above, no shell
            record_command(sandbox, out, rules_file, waits), check=True, timeout=600
        )
    finally:
        stop_daemon(sandbox)
        if keep:
            print(f"demo-record: sandbox kept at {root}", file=sys.stderr)
        else:
            shutil.rmtree(root, ignore_errors=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo-record.py",
        description="Record the nvsh demo in a throwaway HOME, scrubbed.",
    )
    parser.add_argument("out", help="output .cast path")
    parser.add_argument(
        "--wait-panel", type=float, default=WAIT_PANEL, help="seconds to wait for the panel"
    )
    parser.add_argument(
        "--wait-approve", type=float, default=WAIT_APPROVE, help="seconds to wait after Enter"
    )
    parser.add_argument(
        "--wait-retry", type=float, default=WAIT_RETRY, help="seconds to wait after the retry"
    )
    parser.add_argument(
        "--keep-sandbox", action="store_true", help="leave the temp HOME in place for inspection"
    )
    return parser


#: What a complete recording must show, in order of appearance: the
#: failure, the panel with its proposal, the approved fix having run, and
#: the retry succeeding. A cast missing any of them is not published.
REQUIRED_MARKERS = (
    "Permission denied",
    "chmod +x",
    "[Enter] run",
    "-> exit 0",
    SUCCESS_LINE,
)


def missing_markers(cast: Path) -> list[str]:
    """The :data:`REQUIRED_MARKERS` absent from *cast*'s output events."""
    text = []
    for line in cast.read_text(encoding="utf-8").splitlines()[1:]:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, list) and len(event) == 3 and event[1] == "o":
            text.append(str(event[2]))
    plain = "".join(text)
    return [marker for marker in REQUIRED_MARKERS if marker not in plain]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    waits = (args.wait_panel, args.wait_approve, args.wait_retry)
    out = record(Path(args.out), waits, keep=args.keep_sandbox)
    missing = missing_markers(out)
    if missing:
        out.unlink(missing_ok=True)
        print(
            f"demo-record: incomplete recording, not written: missing {missing} "
            "(raise --wait-panel/--wait-approve/--wait-retry on a slow device)",
            file=sys.stderr,
        )
        return 1
    print(f"demo-record: wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
