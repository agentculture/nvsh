"""Pure-function tests for :mod:`nvsh.rcfile` — the rc-editor half of ``nvsh setup``.

No real ``$HOME`` is ever touched here: every test builds its own rc text in
memory (or under ``tmp_path``) and passes it straight to the pure functions.
"""

from __future__ import annotations

import pytest

from nvsh import rcfile

UBUNTU_GUARD = """\
# ~/.bashrc: executed by bash(1) for non-login shells.

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# some more rc below
HISTCONTROL=ignoredups
"""

SINGLE_LINE_GUARD = """\
#!/bin/bash
[[ $- != *i* ]] && return

alias ll='ls -la'
"""

NO_GUARD = """\
export PATH=$PATH:/opt/bin
alias ll='ls -la'
"""


def _block():
    return rcfile.build_block(
        [
            'export NVSH_HOOK_VERSION="0.9.2"',
            'export NVSH_BIN="/usr/local/bin/nvsh"',
            '[[ -n $NVSH_DISABLE ]] || { source "/data/shell/hook.bash"; '
            'source "/data/shell/readline.bash"; }',
            'nvsh() { case $1 in on|off) eval "$(command nvsh "$@" --shell)";; '
            '*) command nvsh "$@";; esac; }',
        ]
    )


# --------------------------------------------------------------------------
# find_insert_point
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _home_is_tmp(tmp_path, monkeypatch):
    """RcPath only edits files directly under $HOME; point HOME at the sandbox."""
    monkeypatch.setenv("HOME", str(tmp_path))


def test_finds_point_after_case_dash_guard():
    lines = UBUNTU_GUARD.splitlines(keepends=True)
    idx = rcfile.find_insert_point(UBUNTU_GUARD)
    assert "".join(lines[:idx]).rstrip().endswith("esac")
    assert lines[idx].strip() == "" or lines[idx].startswith("# some more")


def test_finds_point_after_single_line_dash_test():
    idx = rcfile.find_insert_point(SINGLE_LINE_GUARD)
    lines = SINGLE_LINE_GUARD.splitlines(keepends=True)
    assert "$-" in lines[idx - 1]


def test_fallback_is_top_of_file_when_no_guard():
    assert rcfile.find_insert_point(NO_GUARD) == 0


def test_fallback_on_empty_text():
    assert rcfile.find_insert_point("") == 0


# --------------------------------------------------------------------------
# has_block / build_block
# --------------------------------------------------------------------------


def test_has_block_false_on_plain_text():
    assert rcfile.has_block(UBUNTU_GUARD) is False


def test_build_block_has_markers_and_hash():
    block = _block()
    assert block.startswith(rcfile.MARK_START_PREFIX)
    assert "sha256:" in block.splitlines()[0]
    assert block.rstrip("\n").endswith(rcfile.MARK_END)
    assert "~/" not in block


def test_build_block_stays_short():
    block = _block()
    assert len(block.splitlines()) <= 10


# --------------------------------------------------------------------------
# insert_block / remove_block round-trip
# --------------------------------------------------------------------------


def test_insert_then_remove_round_trips_exactly():
    block = _block()
    inserted = rcfile.insert_block(UBUNTU_GUARD, block)
    assert rcfile.has_block(inserted)
    restored, removed, edited = rcfile.remove_block(inserted)
    assert removed is True
    assert edited is False
    assert restored == UBUNTU_GUARD


def test_insert_places_block_right_after_guard():
    block = _block()
    inserted = rcfile.insert_block(UBUNTU_GUARD, block)
    before_block = inserted.split(rcfile.MARK_START_PREFIX, 1)[0]
    assert before_block.rstrip().endswith("esac")


def test_insert_on_no_guard_file_goes_to_top():
    block = _block()
    inserted = rcfile.insert_block(NO_GUARD, block)
    assert inserted.startswith(rcfile.MARK_START_PREFIX)


def test_remove_block_detects_edit():
    block = _block()
    inserted = rcfile.insert_block(UBUNTU_GUARD, block)
    tampered = inserted.replace(
        'export NVSH_BIN="/usr/local/bin/nvsh"', 'export NVSH_BIN="/tmp/evil"'
    )
    restored, removed, edited = rcfile.remove_block(tampered)
    assert removed is True
    assert edited is True


def test_remove_block_on_text_without_block_is_noop():
    text, removed, edited = rcfile.remove_block(UBUNTU_GUARD)
    assert text == UBUNTU_GUARD
    assert removed is False
    assert edited is False


def test_double_insert_is_idempotent_when_reapplied_over_stripped_text():
    block = _block()
    once = rcfile.insert_block(UBUNTU_GUARD, block)
    stripped, _, _ = rcfile.remove_block(once)
    twice = rcfile.insert_block(stripped, block)
    assert once == twice


# --------------------------------------------------------------------------
# backups
# --------------------------------------------------------------------------


def test_write_backup_creates_timestamped_sibling(tmp_path):
    rc = tmp_path / ".bashrc"
    rc.write_text(UBUNTU_GUARD)
    backup = rcfile.write_backup(rc, UBUNTU_GUARD)
    assert backup.parent == tmp_path
    assert backup.name.startswith(".bashrc.nvsh-backup-")
    assert backup.read_text() == UBUNTU_GUARD


def test_newest_backup_picks_the_latest(tmp_path):
    rc = tmp_path / ".bashrc"
    rc.write_text(UBUNTU_GUARD)
    first = rcfile.write_backup(rc, "one")
    second = rcfile.write_backup(rc, "two")
    assert rcfile.newest_backup(rc) in (first, second)
    # whichever sorts last lexically (== chronologically) wins
    assert rcfile.newest_backup(rc) == sorted([first, second])[-1]


def test_newest_backup_none_when_absent(tmp_path):
    rc = tmp_path / ".bashrc"
    rc.write_text(UBUNTU_GUARD)
    assert rcfile.newest_backup(rc) is None


def test_backup_path_never_collides(tmp_path, monkeypatch):
    rc = tmp_path / ".bashrc"
    rc.write_text(UBUNTU_GUARD)
    # Force the same timestamp twice to exercise the collision-avoidance path.
    import datetime as _dt

    class _FixedDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 13, 12, 0, 0)

    monkeypatch.setattr(rcfile, "datetime", _FixedDatetime)
    first = rcfile.write_backup(rc, "one")
    second = rcfile.write_backup(rc, "two")
    assert first != second
    assert first.read_text() == "one"
    assert second.read_text() == "two"
