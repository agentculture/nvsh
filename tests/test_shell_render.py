"""Tests for :mod:`nvsh.shell.render` — copying the bash files into the data dir."""

from __future__ import annotations

from pathlib import Path

from nvsh import __version__
from nvsh.shell import render


def test_data_dir_honours_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert render.data_dir() == tmp_path / "xdg" / "nvsh"


def test_data_dir_defaults_to_local_share(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert render.data_dir() == tmp_path / ".local" / "share" / "nvsh"


def test_render_shell_files_copies_both_files(tmp_path):
    shell_dir = render.render_shell_files(tmp_path)
    assert shell_dir == tmp_path / "shell"
    assert (shell_dir / "hook.bash").is_file()
    assert (shell_dir / "readline.bash").is_file()


def test_render_shell_files_stamps_version_header(tmp_path):
    shell_dir = render.render_shell_files(tmp_path)
    hook_text = (shell_dir / "hook.bash").read_text()
    assert hook_text.startswith(f"# nvsh hook version {__version__}\n")
    assert f"export NVSH_HOOK_VERSION={__version__}\n" in hook_text.splitlines(keepends=True)[1]


def test_render_shell_files_preserves_original_content(tmp_path):
    shell_dir = render.render_shell_files(tmp_path)
    original = (Path(__file__).resolve().parent.parent / "nvsh" / "shell" / "hook.bash").read_text()
    rendered = (shell_dir / "hook.bash").read_text()
    assert original in rendered


def test_render_is_idempotent(tmp_path):
    first = render.render_shell_files(tmp_path)
    second = render.render_shell_files(tmp_path)
    assert (first / "hook.bash").read_text() == (second / "hook.bash").read_text()


def test_resolve_nvsh_bin_prefers_which_of_nvsh(monkeypatch, tmp_path):
    fake = tmp_path / "nvsh"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(render.shutil, "which", lambda name: str(fake) if name == "nvsh" else None)
    assert render.resolve_nvsh_bin() == str(fake.resolve())


def test_resolve_nvsh_bin_falls_back_to_literal_nvsh(monkeypatch):
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    monkeypatch.setattr(render.sys, "argv", ["nvsh", "setup"])
    assert render.resolve_nvsh_bin() == "nvsh"
