"""Security tests for the shared runtime-directory helper (SonarCloud S5443).

When ``XDG_RUNTIME_DIR`` is unset nvsh falls back to a per-uid directory under
the system temp dir. A publicly writable directory is only safe if the
directory nvsh actually uses is its own: owned by the calling uid, mode 0700,
and not a symlink someone else planted. :mod:`nvsh.runtimedir` is the one
helper that proves that, and capture/setup/daemon all go through it.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pytest

from nvsh import capture, runtimedir
from nvsh.cli._commands import setup as setup_cmd

# --- where the fallback lives ---------------------------------------------


def test_fallback_dir_is_per_uid_under_the_temp_dir():
    expected = Path(tempfile.gettempdir()) / f"nvsh-{os.getuid()}"
    assert runtimedir.fallback_dir() == expected


def test_runtime_dir_prefers_xdg(tmp_path):
    assert runtimedir.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)}) == tmp_path / "nvsh"


def test_runtime_dir_falls_back_per_uid():
    assert runtimedir.runtime_dir({}) == runtimedir.fallback_dir()


def test_capture_and_setup_share_the_same_fallback(monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert capture.session_log_path({}, 7).parent == runtimedir.fallback_dir()
    assert setup_cmd._runtime_dir() == runtimedir.fallback_dir()


# --- ensure_private -------------------------------------------------------


def test_ensure_private_creates_the_dir_0700(tmp_path):
    target = tmp_path / "run" / "nvsh"
    assert runtimedir.ensure_private(target) == target
    assert stat.S_IMODE(os.lstat(target).st_mode) == 0o700


def test_ensure_private_is_idempotent(tmp_path):
    target = tmp_path / "nvsh"
    runtimedir.ensure_private(target)
    runtimedir.ensure_private(target)
    assert stat.S_IMODE(os.lstat(target).st_mode) == 0o700


def test_ensure_private_refuses_a_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(runtimedir.RuntimeDirError) as exc:
        runtimedir.ensure_private(link)
    assert "symlink" in str(exc.value).lower()


def test_ensure_private_refuses_a_non_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.write_text("x", encoding="utf-8")
    with pytest.raises(runtimedir.RuntimeDirError):
        runtimedir.ensure_private(plain)


def test_ensure_private_refuses_a_foreign_owner(tmp_path, monkeypatch):
    target = tmp_path / "nvsh"
    target.mkdir(mode=0o700)
    other_uid = os.getuid() + 1
    monkeypatch.setattr(runtimedir.os, "getuid", lambda: other_uid)
    with pytest.raises(runtimedir.RuntimeDirError) as exc:
        runtimedir.ensure_private(target)
    assert "own" in str(exc.value).lower()


def test_ensure_private_tightens_a_loose_mode(tmp_path):
    target = tmp_path / "nvsh"
    target.mkdir(mode=0o777)
    runtimedir.ensure_private(target)
    assert stat.S_IMODE(os.lstat(target).st_mode) == 0o700


def test_ensure_private_refuses_when_it_cannot_tighten(tmp_path, monkeypatch):
    target = tmp_path / "nvsh"
    target.mkdir(mode=0o777)
    monkeypatch.setattr(runtimedir.os, "chmod", lambda *a, **k: None)
    with pytest.raises(runtimedir.RuntimeDirError) as exc:
        runtimedir.ensure_private(target)
    assert "0700" in str(exc.value) or "permissions" in str(exc.value).lower()


# --- the callers ----------------------------------------------------------


def test_open_session_log_refuses_a_hijacked_runtime_dir(tmp_path, monkeypatch):
    if __import__("shutil").which("script") is None:
        pytest.skip("script(1) not installed")
    hijacked = tmp_path / "hijack"
    hijacked.mkdir(mode=0o700)
    link = tmp_path / "run" / "nvsh"
    link.parent.mkdir()
    link.symlink_to(hijacked)
    env = {"XDG_RUNTIME_DIR": str(tmp_path / "run")}
    with pytest.raises(runtimedir.RuntimeDirError):
        capture.open_session_log(env, 4242)
