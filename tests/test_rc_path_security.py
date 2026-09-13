"""Security tests for the validated ``--rc`` path (SonarCloud S2083, path traversal).

Every filesystem touch ``nvsh setup`` / ``nvsh uninstall`` make on the rc file
goes through :class:`nvsh.rcfile.RcPath`, which is the single validation choke
point. These tests pin what it accepts and what it refuses; no real ``$HOME``
is ever touched (``HOME`` is monkeypatched to a ``tmp_path`` sandbox).
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from nvsh import rcfile
from nvsh.cli import main as cli_main


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    return h


# --- accepted -------------------------------------------------------------


def test_default_home_bashrc_is_accepted(home):
    rc = rcfile.RcPath.validate(home / ".bashrc")
    assert rc.path == home / ".bashrc"
    assert str(rc) == str(home / ".bashrc")


def test_existing_home_bashrc_is_accepted_and_readable(home):
    (home / ".bashrc").write_text("alias ll='ls -la'\n", encoding="utf-8")
    rc = rcfile.RcPath.validate("~/.bashrc")
    assert rc.exists()
    assert rc.read_text() == "alias ll='ls -la'\n"


def test_rc_inside_a_temp_home_is_accepted(home):
    target = home / "custom-rc"
    rc = rcfile.RcPath.validate(target)
    rc.write_text("# hi\n")
    assert target.read_text(encoding="utf-8") == "# hi\n"


def test_path_outside_home_is_refused(home, tmp_path):
    # The rc file must sit directly under $HOME (spec c36); a path elsewhere
    # is refused even when the caller owns it, so no operator-supplied
    # directory ever reaches the writes.
    outside = tmp_path / "elsewhere" / "fakerc"
    outside.parent.mkdir()
    outside.write_text("# rc\n", encoding="utf-8")
    with pytest.raises(rcfile.RcPathError, match="directly under"):
        rcfile.RcPath.validate(outside)


def test_nested_path_under_home_is_refused(home):
    nested = home / "dotfiles" / ".bashrc"
    nested.parent.mkdir()
    nested.write_text("# rc\n", encoding="utf-8")
    with pytest.raises(rcfile.RcPathError, match="directly under"):
        rcfile.RcPath.validate(nested)


# --- refused --------------------------------------------------------------


def test_dotdot_component_is_rejected(home):
    with pytest.raises(rcfile.RcPathError) as exc:
        rcfile.RcPath.validate(str(home / ".." / ".." / "etc" / "passwd"))
    assert ".." in str(exc.value)


def test_relative_traversal_is_rejected(home):
    with pytest.raises(rcfile.RcPathError):
        rcfile.RcPath.validate("../../etc/passwd")


def test_symlink_escaping_home_is_rejected(home, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target").write_text("# elsewhere\n", encoding="utf-8")
    link = home / ".bashrc"
    link.symlink_to(outside / "target")
    with pytest.raises(rcfile.RcPathError) as exc:
        rcfile.RcPath.validate(link)
    assert "symlink" in str(exc.value).lower()


def test_symlink_inside_home_is_accepted(home):
    real = home / "real-rc"
    real.write_text("# inside\n", encoding="utf-8")
    link = home / ".bashrc"
    link.symlink_to(real)
    assert rcfile.RcPath.validate(link).read_text() == "# inside\n"


def test_empty_and_nul_paths_are_rejected(home):
    with pytest.raises(rcfile.RcPathError):
        rcfile.RcPath.validate("")
    with pytest.raises(rcfile.RcPathError):
        rcfile.RcPath.validate("rc\x00.bashrc")


def test_directory_is_rejected(home):
    (home / "adir").mkdir()
    with pytest.raises(rcfile.RcPathError):
        rcfile.RcPath.validate(home / "adir")


@pytest.mark.skipif(os.getuid() == 0, reason="root owns everything")
def test_foreign_owned_path_outside_home_is_rejected(home):
    with pytest.raises(rcfile.RcPathError) as exc:
        rcfile.RcPath.validate("/etc/passwd")
    assert "own" in str(exc.value).lower()


# --- the CLI surface ------------------------------------------------------


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


def test_setup_refuses_a_traversing_rc_as_a_user_error(home):
    code, out, err = _run(["setup", "--rc", "../../etc/passwd", "--json", "--no-install"])
    assert code == 1
    payload = json.loads(out or err)
    assert "error" in payload or "message" in json.dumps(payload)


def test_uninstall_refuses_a_traversing_rc(home):
    code, _out, _err = _run(["uninstall", "--rc", "../../etc/passwd", "--json"])
    assert code == 1


def test_setup_still_works_on_a_temp_home_rc(home):
    rc = home / ".bashrc"
    rc.write_text("# rc\n", encoding="utf-8")
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["rc"] == str(rc)
    assert rcfile.has_block(rc.read_text(encoding="utf-8"))


def test_backup_of_a_symlink_escape_never_happens(home, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim"
    victim.write_text("untouched\n", encoding="utf-8")
    link = home / ".bashrc"
    link.symlink_to(victim)
    code, _out, _err = _run(["setup", "--rc", str(link), "--json", "--no-install"])
    assert code == 1
    assert victim.read_text(encoding="utf-8") == "untouched\n"
    assert list(Path(outside).glob("*.nvsh-backup-*")) == []
