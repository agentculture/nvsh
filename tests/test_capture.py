"""Tests for nvsh.capture — session log, OSC 133 slicing, tmux pipe-pane, bounding.

Acceptance criteria covered (task t8):

* ``open_session_log`` creates ``$XDG_RUNTIME_DIR/nvsh/<shell-pid>.log`` with
  mode 0600, refuses to start when already under script/tmux or when
  ``NVSH_WRAPPED`` is set, and ``cleanup`` removes the log; a killed
  ``script(1)`` (partial log, no D marker; or a missing log) never raises.
* ``last_slice(log)`` returns exactly the bytes between the last OSC 133 C
  and D markers (fixture from the scratchpad experiment: ``hello-out`` +
  ``ls: cannot access``), capped at 64 KB with head+tail and a truncation
  marker, escape sequences stripped, invalid UTF-8 replaced; the log path
  never appears in the returned context.
* Inside tmux (``$TMUX`` set) the source is ``tmux pipe-pane -o`` to the
  same log; a test proves only the slice, never the whole log, reaches the
  redactor.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from nvsh import capture, redact

FIXTURES = Path(__file__).parent / "fixtures" / "capture"


# --- session_log_path -------------------------------------------------------


def test_session_log_path_uses_xdg_runtime_dir(monkeypatch, tmp_path):
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}
    path = capture.session_log_path(env, 4242)
    assert path == tmp_path / "nvsh" / "4242.log"


def test_session_log_path_falls_back_to_tmp_uid(monkeypatch):
    env: dict[str, str] = {}
    path = capture.session_log_path(env, 99)
    assert path == Path(f"/tmp/nvsh-{os.getuid()}/99.log")


# --- open_session_log --------------------------------------------------------


def test_open_session_log_creates_dir_and_file_with_correct_perms(tmp_path):
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}
    handle = capture.open_session_log(env, 1234)
    assert handle is not None
    log_dir = handle.path.parent
    assert log_dir.is_dir()
    assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700
    assert handle.path.is_file()
    assert stat.S_IMODE(handle.path.stat().st_mode) == 0o600


def test_open_session_log_refuses_when_wrapped(tmp_path):
    env = {"XDG_RUNTIME_DIR": str(tmp_path), "NVSH_WRAPPED": "1"}
    handle = capture.open_session_log(env, 1234)
    assert handle is None


def test_open_session_log_refuses_under_tmux(tmp_path):
    env = {"XDG_RUNTIME_DIR": str(tmp_path), "TMUX": "/tmp/tmux-1000/default,123,0"}
    handle = capture.open_session_log(env, 1234)
    assert handle is None


def test_open_session_log_refuses_under_screen(tmp_path):
    env = {"XDG_RUNTIME_DIR": str(tmp_path), "STY": "1234.pts-0.host"}
    handle = capture.open_session_log(env, 1234)
    assert handle is None


def test_open_session_log_refuses_when_script_missing(tmp_path, monkeypatch):
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}
    monkeypatch.setattr(capture.shutil, "which", lambda name: None)
    handle = capture.open_session_log(env, 1234)
    assert handle is None


def test_cleanup_removes_log(tmp_path):
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}
    handle = capture.open_session_log(env, 5555)
    assert handle is not None
    assert handle.path.exists()
    capture.cleanup(handle)
    assert not handle.path.exists()


def test_cleanup_missing_log_does_not_raise(tmp_path):
    handle = capture.LogHandle(path=tmp_path / "nvsh" / "9999.log", pid=9999)
    capture.cleanup(handle)  # must not raise


# --- wrapper_command / tmux_pipe_command / capture_source -------------------


def test_wrapper_command_argv_and_bash_line(tmp_path):
    log = tmp_path / "nvsh" / "42.log"
    wrapper = capture.wrapper_command({}, log)
    assert wrapper.argv == ["script", "-qfc", "$BASH", str(log)]
    assert "NVSH_WRAPPED=1" in wrapper.bash_line
    assert f'script -qfc "$BASH" "{log}"' in wrapper.bash_line
    assert "exec" in wrapper.bash_line


def test_tmux_pipe_command(tmp_path):
    log = tmp_path / "nvsh" / "42.log"
    cmd = capture.tmux_pipe_command(log)
    assert cmd == f"tmux pipe-pane -o \"cat >> '{log}'\""


def test_capture_source_tmux():
    assert capture.capture_source({"TMUX": "/tmp/tmux-1000/default,1,0"}) == "tmux"


def test_capture_source_script_when_wrapped():
    assert capture.capture_source({"NVSH_WRAPPED": "1"}) == "script"


def test_capture_source_none_by_default():
    assert capture.capture_source({}) == "none"


def test_capture_source_tmux_takes_precedence_over_wrapped():
    assert capture.capture_source({"TMUX": "x", "NVSH_WRAPPED": "1"}) == "tmux"


# --- last_slice: real fixtures -----------------------------------------------


def test_last_slice_script_fixture_returns_last_command_output():
    log = FIXTURES / "ts.log"
    result = capture.last_slice(log)
    assert result.status == "ok"
    assert result.source == "script"
    assert "hello-out" in result.text
    assert "ls: cannot access" in result.text
    # No raw escape bytes should survive.
    assert "\x1b" not in result.text
    # The log path itself must never leak into the returned context.
    assert str(log) not in result.text
    assert log.name not in result.text


def test_last_slice_tmux_fixture_returns_last_command_output():
    log = FIXTURES / "tm.log"
    result = capture.last_slice(log, source="tmux")
    assert result.status == "ok"
    assert result.source == "tmux"
    assert "hello-out" in result.text
    assert "ls: cannot access" in result.text
    assert "\x1b" not in result.text
    assert str(log) not in result.text


def test_last_slice_only_bounded_slice_reaches_redactor(monkeypatch):
    log = FIXTURES / "ts.log"
    raw = log.read_bytes()
    received: list[bytes] = []

    def fake_redact_report(data: bytes):
        received.append(data)
        return redact.redact_report(data)

    monkeypatch.setattr(capture, "redact_report", fake_redact_report)
    result = capture.last_slice(log)
    assert result.status == "ok"
    assert len(received) == 1
    sent = received[0]
    # The redactor must never see the whole log, and never anything before
    # the last C marker (which includes the OSC-133-P prompt/cwd noise).
    assert len(sent) < len(raw)
    assert b"kitty-shell-cwd" not in sent
    assert b"Script started on" not in sent


# --- last_slice: missing / unreadable log -----------------------------------


def test_last_slice_missing_log_returns_no_capture(tmp_path):
    log = tmp_path / "nvsh" / "does-not-exist.log"
    result = capture.last_slice(log)
    assert result.status == "no capture"
    assert result.text == ""
    assert result.bytes_total == 0


def test_last_slice_unreadable_log_never_raises(tmp_path, monkeypatch):
    log = tmp_path / "nvsh" / "unreadable.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"junk")

    def boom(*a, **kw):
        raise OSError("simulated permission error")

    monkeypatch.setattr(capture.Path, "read_bytes", boom)
    result = capture.last_slice(log)
    assert result.status == "no capture"
    assert result.text == ""


# --- last_slice: killed script(1) mid-session (C, no D) ---------------------


def test_last_slice_partial_log_no_d_marker(tmp_path):
    log = tmp_path / "nvsh" / "partial.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"\x1b]133;C;\x07partial output, script got killed here")
    result = capture.last_slice(log)
    assert result.status == "partial"
    assert "partial output, script got killed here" in result.text


def test_last_slice_no_markers_at_all_is_no_capture(tmp_path):
    log = tmp_path / "nvsh" / "nomarkers.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"just some bytes with no OSC 133 markers at all")
    result = capture.last_slice(log)
    assert result.status == "no capture"


# --- last_slice: bounding / truncation ---------------------------------------


def test_last_slice_truncates_large_output_head_and_tail(tmp_path):
    log = tmp_path / "nvsh" / "big.log"
    log.parent.mkdir(parents=True)
    head = b"HEAD-" + b"a" * 100
    tail = b"TAIL-" + b"z" * 100
    middle = b"m" * (200 * 1024)
    body = head + middle + tail
    log.write_bytes(b"\x1b]133;C;\x07" + body + b"\x1b]133;D;0;\x07")
    result = capture.last_slice(log, limit=64 * 1024)
    assert result.status == "truncated"
    assert "HEAD-" in result.text
    assert "TAIL-" in result.text
    assert "truncated" in result.text
    assert len(result.text.encode("utf-8", errors="replace")) < len(body)
    assert result.bytes_total == len(body)


def test_last_slice_under_limit_is_not_truncated(tmp_path):
    log = tmp_path / "nvsh" / "small.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"\x1b]133;C;\x07small output\x1b]133;D;0;\x07")
    result = capture.last_slice(log, limit=64 * 1024)
    assert result.status == "ok"
    assert "small output" in result.text


# --- last_slice: invalid UTF-8 -----------------------------------------------


def test_last_slice_replaces_invalid_utf8(tmp_path):
    log = tmp_path / "nvsh" / "badutf8.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"\x1b]133;C;\x07good \xff\xfe bytes\x1b]133;D;0;\x07")
    result = capture.last_slice(log)
    assert result.status == "ok"
    assert "good" in result.text
    assert "bytes" in result.text
    # Must not raise UnicodeDecodeError and must not contain the raw invalid bytes.
    result.text.encode("utf-8")


# --- last_slice: open vs closed region selection (deviation d3) --------------
#
# Under Ghostty the *terminal* owns OSC 133 and its ``__ghostty_hook`` runs
# after ``__nvsh_hook`` in ``PROMPT_COMMAND``, so when the hook calls the
# client the failing command's region is still open: the log holds
# ``C <prev output> D ... C <this command's output>`` with no closing ``D``
# yet. The open region is the command we are diagnosing.


def test_last_slice_prefers_open_region_over_previous_closed_region(tmp_path):
    log = tmp_path / "nvsh" / "ghostty.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(
        b"\x1b]133;A;\x07prompt$ echo hi\r\n"
        b"\x1b]133;C;\x07previous command output\r\n"
        b"\x1b]133;D;0;\x07"
        b"\x1b]133;A;\x07prompt$ ls /nope\r\n"
        b"\x1b]133;C;\x07ls: cannot access '/nope': No such file or directory\r\n"
    )
    result = capture.last_slice(log)
    assert "ls: cannot access" in result.text
    assert "previous command output" not in result.text
    assert result.status == "partial"


def test_last_slice_falls_back_to_last_closed_region_when_none_open(tmp_path):
    log = tmp_path / "nvsh" / "closed.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(
        b"\x1b]133;C;\x07first output\x1b]133;D;0;\x07"
        b"\x1b]133;C;\x07second output\x1b]133;D;1;\x07"
        b"\x1b]133;A;\x07prompt$ "
    )
    result = capture.last_slice(log)
    assert result.status == "ok"
    assert "second output" in result.text
    assert "first output" not in result.text


def test_last_slice_open_region_ends_at_next_prompt_marker(tmp_path):
    """A ``133;A`` after the last ``C`` is an implicit close: a new prompt began."""
    log = tmp_path / "nvsh" / "implicit-close.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(
        b"\x1b]133;C;\x07previous command output\x1b]133;D;0;\x07"
        b"\x1b]133;C;\x07current command output\r\n"
        b"\x1b]133;A;\x07prompt$ trailing prompt noise"
    )
    result = capture.last_slice(log)
    assert result.status == "ok"
    assert "current command output" in result.text
    assert "previous command output" not in result.text
    assert "trailing prompt noise" not in result.text


def test_last_slice_ignores_trailing_exit_region_of_a_finished_script_log(tmp_path):
    """``script(1)``'s own epilogue means the session ended, not a live command."""
    log = tmp_path / "nvsh" / "finished.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(
        b"\x1b]133;C;\x07ls: cannot access '/nope': No such file or directory\r\n"
        b"\x1b]133;D;2;\x07"
        b"\x1b]133;A;\x07bash-5.2$ exit\r\n"
        b"\x1b]133;C;\x07exit\r\n"
        b'\nScript done on 2026-09-13 13:43:11+03:00 [COMMAND_EXIT_CODE="1"]\n'
    )
    result = capture.last_slice(log)
    assert result.status == "ok"
    assert "ls: cannot access" in result.text
    assert "Script done on" not in result.text


def test_last_slice_empty_log_is_no_capture(tmp_path):
    log = tmp_path / "nvsh" / "empty.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"")
    result = capture.last_slice(log)
    assert result.status == "no capture"
    assert result.text == ""


# --- Slice dataclass shape ----------------------------------------------------


def test_slice_has_expected_fields():
    s = capture.Slice(text="hi", status="ok", source="script", redaction_rules=[], bytes_total=2)
    assert s.text == "hi"
    assert s.status == "ok"
    assert s.source == "script"
    assert s.redaction_rules == []
    assert s.bytes_total == 2
