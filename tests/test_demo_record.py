"""Tests for scripts/demo-record.py (task t5).

Acceptance criteria covered, in order:

* ``scripts/demo-record.py <out.cast>`` builds a temp ``$HOME`` carrying
  ``XDG_CONFIG_HOME``/``XDG_DATA_HOME``/``XDG_STATE_HOME``/``XDG_RUNTIME_DIR``,
  writes a ``config.toml`` whose ``[aliases].default`` is ``demo``, renders
  ``hook.bash``/``readline.bash`` into it, writes an rc file that sources
  both with ``PS1='nvsh$ '``, plants a non-executable ``./run-model.sh`` and
  drives ``record-cast.py --feed`` (run it, wait for the panel, Enter, wait
  for the retry).
* Every scrub token -- this machine's hostname, this user's name and every
  non-loopback IPv4 from ``hostname -I`` -- is absent from the recording.
* The operator's ``.bashrc`` and XDG directories are byte-identical before
  and after: the end-to-end test hands the driver a *fake* ``$HOME`` full of
  sentinel files, hashes the whole tree either side of the run, and so also
  proves the driver ignores the ambient ``$HOME``/``XDG_*`` entirely.
* Two consecutive runs both contain the panel -- the second starts well
  inside the 60 s auto-call rate window, and only its fresh
  ``XDG_STATE_HOME`` (a new ``rate.json``) keeps it from being held back.
* Nothing is left running: the second run keeps its sandbox, and its
  ``XDG_RUNTIME_DIR/nvsh`` holds no daemon socket afterwards.

The end-to-end test runs the real hook in a real interactive bash on a real
pty, with the real daemon and the real ``demo`` adapter behind it. Each
recording takes roughly half a minute, so both runs live in one test
function (an xdist worker must not repeat them), and every subprocess call
carries an explicit timeout rather than relying on a plugin.
"""

from __future__ import annotations

import getpass
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from nvsh.panel import LEGEND

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "demo-record.py"

#: Ceiling for one ``demo-record.py`` subprocess. The driver's own feed waits
#: add up to well under a minute; this is the "something hung" bound.
RUN_TIMEOUT = 300

#: Shorter than the driver's defaults, so the suite does not pay twice for
#: waits tuned for a loaded Jetson, but still generous under ``-n auto``.
TEST_WAITS = ("--wait-panel", "12", "--wait-approve", "6", "--wait-retry", "6")

pytestmark = pytest.mark.skipif(
    sys.platform not in ("linux", "darwin") or shutil.which("bash") is None,
    reason="demo-record.py needs bash on a real pty",
)


def _module():
    """Import the hyphenated script as a module for its pure helpers."""
    spec = importlib.util.spec_from_file_location("nvsh_demo_record", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `@dataclass` resolves annotations through ``sys.modules[cls.__module__]``,
    # so the module has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo_record = _module()


# --- pure helpers ----------------------------------------------------------


def test_scrub_rules_put_the_longest_token_first():
    # A hostname that contains the user name is the common case on a
    # single-user box; replacing the short token first would leave a
    # half-scrubbed hostname behind.
    rules = demo_record.scrub_rules("spark-f8a9", "spark", ["10.1.2.3"])
    assert rules.index(f"spark-f8a9={demo_record.HOST_PLACEHOLDER}") < rules.index(
        f"spark={demo_record.USER_PLACEHOLDER}"
    )
    assert f"10.1.2.3={demo_record.IPV4_PLACEHOLDER}" in rules


def test_scrub_rules_refuse_a_token_too_short_to_replace_safely():
    assert demo_record.scrub_rules("hostname", "ab", []) == [
        f"hostname={demo_record.HOST_PLACEHOLDER}"
    ]


def test_local_ipv4s_drops_loopback_and_ipv6():
    parsed = demo_record.local_ipv4s("192.168.1.157 127.0.0.1 fd7a:115c:a1e0::ea33 10.0.0.4\n")
    assert parsed == ["192.168.1.157", "10.0.0.4"]


def test_config_toml_points_the_default_alias_at_the_demo_adapter():
    data = tomllib.loads(demo_record.config_toml())
    assert data["aliases"]["default"] == "demo"
    assert data["agent"]["provider"] == "demo"


def test_rc_text_sources_both_rendered_files_and_sets_the_demo_prompt():
    rc = demo_record.rc_text(Path("/sandbox/share/nvsh/shell"), "/sandbox/bin/nvsh")
    assert "PS1='nvsh$ '" in rc
    assert 'source "/sandbox/share/nvsh/shell/hook.bash"' in rc
    assert 'source "/sandbox/share/nvsh/shell/readline.bash"' in rc
    assert 'export NVSH_BIN="/sandbox/bin/nvsh"' in rc
    assert "export NVSH_HOOK_VERSION=" in rc
    # The interactive guard comes first, exactly as a distro rc has it.
    assert rc.index("case $- in") < rc.index("source ")


def test_feed_drives_run_then_enter_then_retry():
    steps = demo_record.feed_steps((11.0, 5.0, 4.0))
    assert steps == [
        r"./run-model.sh\r|11.0",
        r"\r|5.0",
        r"./run-model.sh\r|4.0",
    ]


def test_sandbox_env_redirects_every_xdg_variable_into_the_sandbox(tmp_path):
    sandbox = demo_record.build_sandbox(tmp_path)
    env = demo_record.sandbox_env(sandbox)
    for name in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"):
        assert Path(env[name]).is_relative_to(tmp_path)
    assert env["NVSH_BIN"] == sandbox.nvsh_bin
    assert "NO_COLOR" not in env  # the recording is in colour


def test_build_sandbox_plants_a_script_that_is_readable_but_not_executable(tmp_path):
    sandbox = demo_record.build_sandbox(tmp_path)
    assert sandbox.script.exists()
    assert not os.access(sandbox.script, os.X_OK)
    assert demo_record.SUCCESS_LINE in sandbox.script.read_text(encoding="utf-8")
    shell_dir = sandbox.data / "nvsh" / "shell"
    assert (shell_dir / "hook.bash").is_file()
    assert (shell_dir / "readline.bash").is_file()
    assert (sandbox.config / "nvsh" / "config.toml").is_file()
    assert sandbox.runtime.is_dir()


# --- the recording itself --------------------------------------------------


def _plain_text(cast: Path) -> str:
    """The cast's output with OSC and SGR sequences stripped."""
    chunks = []
    for line in cast.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            _, kind, data = json.loads(line)
            if kind == "o":
                chunks.append(data)
    text = "".join(chunks)
    text = re.sub(r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
    return text.replace("\r", "")


def _tree_hash(root: Path) -> str:
    """A hash over every path under *root*: name, mode, and file bytes."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(str(oct(path.stat().st_mode)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _fake_home(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """An operator-shaped ``$HOME`` the driver must not touch."""
    home = tmp_path / "operator-home"
    dirs = {
        "XDG_CONFIG_HOME": home / ".config",
        "XDG_DATA_HOME": home / ".local" / "share",
        "XDG_STATE_HOME": home / ".local" / "state",
        "XDG_RUNTIME_DIR": home / "run",
    }
    for path in dirs.values():
        (path / "nvsh").mkdir(parents=True, exist_ok=True)
        (path / "nvsh" / "sentinel").write_text("do not touch\n", encoding="utf-8")
    (home / ".bashrc").write_text("# the operator's own rc\nalias ll='ls -l'\n", encoding="utf-8")
    # The driver's own sandbox is an mkdtemp under $TMPDIR; point that at a
    # directory of ours (outside the hashed tree) so "it cleaned up after
    # itself" is an observation rather than a guess.
    driver_tmp = tmp_path / "driver-tmp"
    driver_tmp.mkdir(exist_ok=True)
    env = dict(os.environ, HOME=str(home), TMPDIR=str(driver_tmp))
    env.update({name: str(path) for name, path in dirs.items()})
    return home, env


def _run(out: Path, env: dict[str, str], *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, str(SCRIPT), str(out), *TEST_WAITS, *extra],
        env=env,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
        check=True,
    )


def test_demo_record_end_to_end(tmp_path):
    """One driver invocation, twice, against the real hook/daemon/panel."""
    home, env = _fake_home(tmp_path)
    before = _tree_hash(home)

    driver_tmp = tmp_path / "driver-tmp"
    first = tmp_path / "first.cast"
    second = tmp_path / "second.cast"
    plain = _run(first, env)
    # the default run leaves no sandbox behind at all
    assert _kept_sandbox(plain.stderr) is None
    assert list(driver_tmp.iterdir()) == []
    # No sleep: the second run starts about half a minute after the first,
    # comfortably inside the 60 s auto-call window. Only its fresh
    # XDG_STATE_HOME (and so a fresh rate.json) lets the panel open again.
    proc = _run(second, env, "--keep-sandbox")

    text = _plain_text(first)
    # the failure, as bash reported it
    assert "permission denied" in text.lower()
    # nvsh took it to the agent, and the agent proposed the fix
    assert "forwarding to demo" in text
    assert "chmod +x ./run-model.sh" in text
    assert LEGEND in text
    # the operator's Enter ran it, and the retry succeeded
    assert "nvsh: chmod +x ./run-model.sh -> exit 0" in text
    assert demo_record.SUCCESS_LINE in text
    assert "nvsh$ " in text

    # nothing identifying this machine survived into either recording -- the
    # scrub tokens the driver actually applied (a user called "spark" inside
    # the platform word "dgx-spark" is deliberately kept; see scrub_rules)
    rules = demo_record.scrub_rules(
        socket.gethostname(),
        getpass.getuser(),
        demo_record.local_ipv4s(),
        protect=demo_record.protected_words(),
    )
    tokens = [rule.partition("=")[0] for rule in rules]
    assert tokens, "expected at least one scrub token on this box"
    for cast in (first, second):
        raw = cast.read_text(encoding="utf-8")
        for token in tokens:
            assert token not in raw, f"{token!r} leaked into {cast.name}"

    # the second run opened the panel too: the rate limit did not hold it back
    assert LEGEND in _plain_text(second)

    # the operator's rc and XDG directories are byte-identical
    assert _tree_hash(home) == before

    # and the daemon the second run started was stopped with it
    kept = _kept_sandbox(proc.stderr)
    assert kept is not None
    assert kept.is_dir()
    sockets = list((kept / "run" / "nvsh").glob("*.sock"))
    assert sockets == [], f"daemon socket left behind: {sockets}"
    shutil.rmtree(kept, ignore_errors=True)


def _kept_sandbox(stderr: str) -> Path | None:
    match = re.search(r"sandbox kept at (\S+)", stderr)
    return Path(match.group(1)) if match else None


def test_scrub_rules_leave_a_token_inside_a_protected_word_alone(capsys):
    # The operator on a DGX Spark is often called "spark"; the demo reply
    # names the platform "dgx-spark", which the scrub must not rewrite.
    rules = demo_record.scrub_rules("spark-f8a9", "spark", [], protect=("dgx-spark",))
    assert rules == [f"spark-f8a9={demo_record.HOST_PLACEHOLDER}"]
    assert "not scrubbing 'spark'" in capsys.readouterr().err


def test_protected_words_is_the_detected_platform_kind():
    words = demo_record.protected_words()
    assert len(words) == 1
    assert words[0]


def test_missing_markers_names_what_an_incomplete_cast_lacks(tmp_path):
    # Qodo 9 (PR #14): an incomplete recording is not published.
    cast = tmp_path / "partial.cast"
    cast.write_text(
        '{"version": 2}\n[0.1, "o", "bash: ./run-model.sh: Permission denied\\r\\n"]\n',
        encoding="utf-8",
    )
    missing = demo_record.missing_markers(cast)
    assert "Permission denied" not in missing
    assert demo_record.SUCCESS_LINE in missing
    assert "[Enter] run" in missing


def test_sandbox_env_pins_cache_dir_and_daemon_knob():
    sandbox = demo_record.build_sandbox(Path("/tmp/x"))
    env = demo_record.sandbox_env(sandbox)
    assert env["XDG_CACHE_HOME"].startswith("/tmp/x")
    assert env["NVSH_NO_DAEMON"] == ""
