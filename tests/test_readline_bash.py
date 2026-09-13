"""t7: the readline layer, nvsh/shell/readline.bash (covers c5, h28, c6, h5, c39, h32).

Every test drives a real ``bash --norc --noprofile -i`` on a pty (stdlib
``pty``, the same thing ``script(1)`` gives you) and feeds CR-terminated
keystrokes, because the behaviour under test *is* readline's: a ``\\r``
keystroke is what a terminal sends for Enter and what the ``\\C-m`` macro
binds, and a plain ``\\n`` would bypass it entirely.

The slash-command palette and argument candidates come from a fake ``nvsh``
executable (``tests/fakes/nvsh``) placed first on ``PATH``; it answers
``complete --json`` from a fixture list and records every ``slash <line>``
dispatch to ``$NVSH_FAKE_RECORD``, so the whole bash-side contract is
exercised before the real verbs exist (plan task t14).
"""

from __future__ import annotations

import fcntl
import os
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READLINE_BASH = os.path.join(REPO_ROOT, "nvsh", "shell", "readline.bash")
HOOK_BASH = os.path.join(REPO_ROOT, "nvsh", "shell", "hook.bash")
FAKE_DIR = os.path.join(REPO_ROOT, "tests", "fakes")

PROMPT = "@nvsh@ "
KEYMAPS = ("emacs", "vi-insert", "vi-command")

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("script") is None or not hasattr(os, "forkpty"),
    reason="needs a pty (script(1) / os.forkpty) to drive interactive readline",
)


class BashPty:
    """An interactive bash on a pty, fed raw keystrokes."""

    def __init__(self, tmp_path, env_extra=None):
        env = {
            "PATH": FAKE_DIR + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "TERM": "xterm",
            "INPUTRC": "/dev/null",
            "LC_ALL": "C.UTF-8",
            "PS1": PROMPT,
            "HISTFILE": str(tmp_path / "bash_history"),
            "NVSH_FAKE_RECORD": str(tmp_path / "record"),
        }
        env.update(env_extra or {})
        self.record = tmp_path / "record"
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 60, 200, 0, 0))

        def _setup():  # pragma: no cover - runs in the forked child
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        self.proc = subprocess.Popen(
            ["bash", "--norc", "--noprofile", "-i"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=str(tmp_path),
            preexec_fn=_setup,
            close_fds=True,
        )
        os.close(slave)
        self.buf = ""
        self.expect(PROMPT)

    # -- plumbing ---------------------------------------------------------
    def _drain(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.master], [], [], 0.05)
            if not ready:
                continue
            try:
                chunk = os.read(self.master, 65536)
            except OSError:
                return
            if not chunk:
                return
            self.buf += chunk.decode("utf-8", "replace")
            return

    def expect(self, needle, timeout=8.0):
        deadline = time.monotonic() + timeout
        while needle not in self.buf:
            if time.monotonic() > deadline:
                raise AssertionError("timeout waiting for %r in:\n%s" % (needle, self.buf))
            self._drain(0.2)
        return self.buf

    def send(self, data, settle=0.35):
        os.write(self.master, data.encode("utf-8"))
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            self._drain(0.05)

    def mark(self):
        return len(self.buf)

    def since(self, mark, settle=0.0):
        if settle:
            deadline = time.monotonic() + settle
            while time.monotonic() < deadline:
                self._drain(0.05)
        return self.buf[mark:]

    def run(self, line, settle=0.6):
        """Type ``line`` and press Enter (a real CR keystroke)."""
        mark = self.mark()
        self.send(line + "\r", settle=settle)
        return self.since(mark)

    def records(self):
        if not self.record.exists():
            return []
        return [ln for ln in self.record.read_text().splitlines() if ln]

    def close(self):
        try:
            os.write(self.master, b"\x03\rexit\r")
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait(timeout=3)
        os.close(self.master)


def start(tmp_path, env_extra=None, source=True, pre=(), post=()):
    sh = BashPty(tmp_path, env_extra)
    for line in pre:
        sh.run(line, settle=0.25)
    if source:
        sh.run("source %s" % READLINE_BASH, settle=0.5)
    sh.run("bind 'set show-all-if-ambiguous on'", settle=0.25)
    for line in post:
        sh.run(line, settle=0.25)
    return sh


def shim(tmp_path, *, doctor_exit=0):
    """A louder fake ``nvsh``, first on PATH ahead of ``tests/fakes/nvsh``.

    It logs every invocation to ``$NVSH_SHIM_LOG`` (so "was the hook's client
    path invoked?" is answerable), prints a marker on a ``slash`` dispatch (so
    "did the panel reach the tty?" is answerable), and can make the
    ``/doctor`` dispatch exit non-zero — the d5 case, where nvsh's own hidden
    dispatch reports a failure. Everything else is delegated to the shared
    fake, which still answers ``complete --json``.
    """
    bindir = tmp_path / "shimbin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "shim.log"
    log.write_text("")
    script = bindir / "nvsh"
    body = (
        "#!/bin/sh\n"
        'printf "CALL %s\\n" "$*" >> "$NVSH_SHIM_LOG"\n'
        'case "$1" in\n'
        "slash)\n"
        '    printf "NVSH-PANEL draft=[%s] line=[%s]\\n" "${NVSH_DRAFT:-}" "$2"\n'
        '    case "$2" in "/doctor") exit __DOCTOR_EXIT__ ;; esac\n'
        "    exit 0 ;;\n"
        "hook)\n"
        "    exit 0 ;;\n"
        "esac\n"
        'exec __FAKE_DIR__/nvsh "$@"\n'
    )
    body = body.replace("__DOCTOR_EXIT__", str(doctor_exit)).replace("__FAKE_DIR__", FAKE_DIR)
    script.write_text(body)
    script.chmod(0o755)
    env = {
        "PATH": os.pathsep.join([str(bindir), FAKE_DIR, os.environ.get("PATH", "/usr/bin:/bin")]),
        "NVSH_SHIM_LOG": str(log),
    }
    return env, log


@pytest.fixture
def sh(tmp_path):
    shell = start(tmp_path)
    yield shell
    shell.close()


# -- criterion 1: Enter routing, both keymaps ----------------------------


@pytest.mark.parametrize("mode", ["emacs", "vi"])
def test_enter_routes_slash_commands(tmp_path, mode):
    sh = start(tmp_path, post=(["set -o vi"] if mode == "vi" else []))
    try:
        sh.run("/doctor")
        sh.run("/ask why is memory high?")
        assert sh.records() == ["slash /doctor", "slash /ask why is memory high?"]
        mark = sh.mark()
        sh.send("history 6\r", settle=0.6)
        out = sh.since(mark)
        assert "/doctor" in out
        assert "/ask why is memory high?" in out
        assert "nvsh slash" not in out
    finally:
        sh.close()


@pytest.mark.parametrize("mode", ["emacs", "vi"])
def test_non_slash_lines_pass_through(tmp_path, mode):
    sh = start(tmp_path, post=(["set -o vi"] if mode == "vi" else []))
    try:
        out = sh.run("/notacmd")
        assert "No such file" in out or "not found" in out
        out = sh.run("ls -d /tmp")
        assert "/tmp" in out
        mark = sh.mark()
        sh.send('echo "one\r', settle=0.4)
        sh.send('two"\r', settle=0.6)
        out = sh.since(mark)
        assert "one" in out and "two" in out
        assert sh.records() == []
    finally:
        sh.close()


@pytest.mark.parametrize("mode", ["emacs", "vi"])
def test_ctrl_g_dispatches_with_the_draft(tmp_path, mode):
    sh = start(tmp_path, post=(["set -o vi"] if mode == "vi" else []))
    try:
        mark = sh.mark()
        sh.send("echo restored", settle=0.2)
        sh.send("\x07", settle=0.8)
        out = sh.since(mark)
        assert "nvsh (Ctrl+G)" in out
        assert any(r.startswith("slash /ask") for r in sh.records())
        assert "draft echo restored" in sh.records()
        # d6: the draft is not lost -- it is pushed with `history -s`, so the
        # operator can recall it after the panel.
        mark = sh.mark()
        sh.send("history 3\r", settle=0.6)
        assert "echo restored" in sh.since(mark)
    finally:
        sh.close()


@pytest.mark.parametrize("mode", ["emacs", "vi"])
def test_ctrl_g_streams_the_answer_on_the_tty(tmp_path, mode):
    """d6: Ctrl+G must deliver the agent's answer, not swallow it.

    The dispatch has to leave the bind -x callback (readline owns the tty
    there) and run as a real command line, exactly as the Enter macro does
    for a typed slash line.
    """
    env, log = shim(tmp_path)
    sh = start(tmp_path, env_extra=env, post=(["set -o vi"] if mode == "vi" else []))
    try:
        mark = sh.mark()
        sh.send("echo restored", settle=0.2)
        sh.send("\x07", settle=1.0)
        out = sh.since(mark, settle=0.5)
        assert "nvsh (Ctrl+G)" in out
        assert "NVSH-PANEL" in out, out
        assert "draft=[echo restored]" in out, out
        assert "line=[/ask]" in out, out
        # the shell is still usable afterwards
        assert "still-alive" in sh.run("echo still-alive")
        # and the dispatch line itself stays out of history
        mark = sh.mark()
        sh.send("history 5\r", settle=0.6)
        assert "nvsh slash" not in sh.since(mark)
    finally:
        sh.close()


def test_hidden_slash_dispatch_never_auto_triggers(tmp_path):
    """d5: a failing hidden dispatch must not make nvsh answer its own panel."""
    env, log = shim(tmp_path, doctor_exit=1)
    debug = tmp_path / "hook-debug"
    env = dict(env, NVSH_CAPTURE="0", NVSH_HOOK_DEBUG_FILE=str(debug))
    sh = start(tmp_path, env_extra=env, pre=["source %s" % HOOK_BASH])
    try:
        sh.run("/doctor", settle=1.0)
        calls = log.read_text()
        assert "CALL slash /doctor" in calls, calls
        assert "CALL hook" not in calls, calls
        # the hook really did run for that prompt (so the assertion above is
        # not vacuous): it recorded the failing status.
        assert debug.exists()
        assert any(ln.startswith("1\t") for ln in debug.read_text().splitlines()), debug.read_text()
        # ...and a genuine failure still reaches the client.
        sh.run("nvsh-definitely-not-a-command", settle=1.0)
        assert "CALL hook" in log.read_text()
    finally:
        sh.close()


# -- criterion 2: completion ---------------------------------------------


def test_slash_tab_lists_palette_merged_with_paths(sh):
    mark = sh.mark()
    sh.send("/\t", settle=0.8)
    out = sh.since(mark)
    # readline lists candidates with the common "/" prefix stripped, so the
    # palette shows as bare words next to the real directories of /.
    assert "doctor" in out and "retry" in out and "undo" in out
    assert "usr/" in out or "etc/" in out
    sh.send("\x03", settle=0.2)


def test_prefix_completes_slash_command(sh):
    mark = sh.mark()
    sh.send("/do\t", settle=0.8)
    assert "/doctor" in sh.since(mark)
    sh.send("\x03", settle=0.2)


def test_prefix_still_completes_real_paths(sh):
    mark = sh.mark()
    sh.send("/tm\t", settle=0.8)
    assert "/tmp/" in sh.since(mark)
    sh.send("\x03", settle=0.2)


def test_argument_completion_comes_from_nvsh_complete(sh):
    mark = sh.mark()
    sh.send("/doctor \t", settle=0.8)
    out = sh.since(mark)
    assert "--json" in out and "--strict" in out
    sh.send("\x03", settle=0.2)


def test_ordinary_word_completion_still_works(sh):
    mark = sh.mark()
    sh.send("ec\t", settle=0.8)
    assert "echo" in sh.since(mark)
    sh.send("\x03", settle=0.2)


def test_registers_complete_I_and_leaves_the_D_loader_alone(tmp_path):
    sh = start(tmp_path, pre=["_fake_loader() { return 124; }", "complete -D -F _fake_loader"])
    try:
        mark = sh.mark()
        sh.send("complete -p -I; complete -p -D\r", settle=0.6)
        out = sh.since(mark)
        assert "__nvsh_complete_initial" in out
        assert "_fake_loader" in out
    finally:
        sh.close()


# -- criterion 3: degrade, never lock ------------------------------------


def test_erroring_hook_still_accepts_the_line(tmp_path):
    sh = start(tmp_path, post=["__nvsh_enter() { __nvsh_definitely_not_a_command; }"])
    try:
        out = sh.run("echo still-alive")
        assert "still-alive" in out
        out = sh.run("echo again")
        assert "again" in out
    finally:
        sh.close()


# -- bindings, keymaps, kill switch, idempotence -------------------------


def test_slash_dispatch_exports_every_bind_section(sh, tmp_path):
    """`bind -p` lists neither bind -x functions nor macros, so the payload
    doctor reads carries all three sections behind a known marker."""
    out = tmp_path / "bind-p.txt"
    sh.run(
        "READLINE_LINE=/doctor; __nvsh_enter; printf '%%s\\n' \"$NVSH_BIND_P\" > %s" % out,
        settle=0.8,
    )
    text = out.read_text()
    assert "# nvsh: bind -s/-X follow" in text
    assert "__nvsh_enter" in text, text
    assert "__nvsh_ctrl_g" in text, text
    assert r"\C-m" in text
    # The real payload must satisfy doctor's check, not just a hand-written one
    # (d4 parsed the dumps, d6 changed Ctrl+G into a macro; both seams meet here).
    from nvsh import doctor_checks

    check = doctor_checks.check_bindings_present(text, "emacs")
    assert check["passed"], check


def test_binds_every_keymap(sh):
    for keymap in KEYMAPS:
        mark = sh.mark()
        sh.send("bind -m %s -X\r" % keymap, settle=0.5)
        out = sh.since(mark)
        assert "__nvsh_enter" in out, (keymap, out)
        assert "__nvsh_ctrl_g" in out, (keymap, out)
        mark = sh.mark()
        sh.send("bind -m %s -s\r" % keymap, settle=0.5)
        assert "C-x" in sh.since(mark), keymap


def test_histcontrol_gains_ignorespace(tmp_path):
    sh = start(tmp_path, env_extra={"HISTCONTROL": "erasedups"})
    try:
        mark = sh.mark()
        sh.send('echo "HC=$HISTCONTROL"\r', settle=0.5)
        out = sh.since(mark)
        assert "HC=erasedups:ignorespace" in out
    finally:
        sh.close()


def test_nvsh_disable_makes_sourcing_a_no_op(tmp_path):
    sh = start(tmp_path, env_extra={"NVSH_DISABLE": "1"})
    try:
        mark = sh.mark()
        sh.send("bind -p | grep '\"\\\\C-m\"'; complete -p -I\r", settle=0.6)
        out = sh.since(mark)
        assert "__nvsh" not in out
        sh.run("/doctor")
        assert sh.records() == []
    finally:
        sh.close()


def test_unbind_restores_plain_bash(sh):
    sh.run("__nvsh_readline_unbind")
    sh.run("/doctor")
    assert sh.records() == []
    mark = sh.mark()
    sh.send("complete -p -I\r", settle=0.5)
    assert "__nvsh_complete_initial" not in sh.since(mark)


def test_sourcing_twice_is_idempotent(sh):
    mark = sh.mark()
    sh.send("source %s; echo LOADED=$__NVSH_READLINE_LOADED\r" % READLINE_BASH, settle=0.6)
    assert "LOADED=1" in sh.since(mark)
    sh.run("/doctor")
    assert sh.records() == ["slash /doctor"]


# -- packaging ------------------------------------------------------------


def test_bash_file_ships_in_the_wheel():
    pyproject = open(os.path.join(REPO_ROOT, "pyproject.toml"), encoding="utf-8").read()
    wheel = pyproject.split("[tool.hatch.build.targets.wheel]", 1)[1].split("\n[", 1)[0]
    assert "nvsh/shell/*.bash" in wheel


def test_shell_is_a_package():
    assert os.path.isfile(os.path.join(REPO_ROOT, "nvsh", "shell", "__init__.py"))
    assert os.path.isfile(READLINE_BASH)


def test_no_static_command_list_in_the_bash_file():
    text = open(READLINE_BASH, encoding="utf-8").read()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    for command in ("/doctor", "/retry", "/approve"):
        assert command not in code, "command list must come from 'nvsh complete'"
