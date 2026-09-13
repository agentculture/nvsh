"""Tests for nvsh.approvals — the glob-pattern approval store.

Acceptance criterion covered (task t5):
approved.toml (0600) holds glob patterns; matches(cmd) uses fnmatch on the
full command line; add() refuses 'sudo *', 'rm *', bare '*' and patterns
starting with 'sudo'/'rm'; the default list contains 'nvidia-smi *',
'docker ps*', 'journalctl *', 'systemctl status *', 'df *', 'free *',
'nvsh *'; a session list is scoped to the login session (d15: it is held in
the runtime dir, not in one process's memory, and never in approved.toml).
"""

from __future__ import annotations

import stat

import pytest

from nvsh.approvals import DEFAULT_PATTERNS, Approvals


@pytest.fixture()
def xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Session approvals live under the runtime dir (d15); every test points
    # it at a private directory so the operator's real one is never touched.
    runtime = tmp_path / "run"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    return tmp_path


def test_default_patterns_present(xdg_home):
    approvals = Approvals.default()
    for pattern in (
        "nvidia-smi *",
        "docker ps*",
        "journalctl *",
        "systemctl status *",
        "df *",
        "free *",
        "nvsh *",
    ):
        assert pattern in approvals.user_patterns
    assert set(DEFAULT_PATTERNS) <= set(approvals.user_patterns)


def test_decide_user_pattern_matches(xdg_home):
    approvals = Approvals.default()
    assert approvals.decide("nvidia-smi -q") == "user"
    assert approvals.decide("docker ps -a") == "user"


def test_decide_no_match_is_ask(xdg_home):
    approvals = Approvals.default()
    assert approvals.decide("rm -rf /") == "ask"
    assert approvals.decide("curl http://example.com") == "ask"


def test_decide_session_pattern_matches(xdg_home):
    approvals = Approvals.default()
    approvals.add("pip install *", scope="session")
    assert approvals.decide("pip install requests") == "session"


def test_decide_prefers_user_over_session(xdg_home):
    approvals = Approvals.default()
    approvals.add("echo *", scope="session")
    approvals.add("echo *", scope="user")
    assert approvals.decide("echo hi") == "user"


def test_matches_full_command_line_not_just_argv0(xdg_home):
    approvals = Approvals.default()
    approvals.add("git status*", scope="user")
    assert approvals.decide("git status --short") == "user"
    assert approvals.decide("git status") == "user"
    assert approvals.decide("git log") == "ask"


def test_matches_normalizes_whitespace(xdg_home):
    approvals = Approvals.default()
    approvals.add("git   status*", scope="user")
    assert approvals.decide("git status --short") == "user"


@pytest.mark.parametrize(
    "pattern",
    ["sudo *", "rm *", "*", "sudo", "sudo reboot", "rm", "rm -rf /"],
)
def test_add_refuses_dangerous_patterns(xdg_home, pattern):
    approvals = Approvals.default()
    with pytest.raises(ValueError):
        approvals.add(pattern, scope="user")
    with pytest.raises(ValueError):
        approvals.add(pattern, scope="session")


def test_add_refusal_leaves_store_unchanged(xdg_home):
    approvals = Approvals.default()
    before = list(approvals.user_patterns)
    with pytest.raises(ValueError):
        approvals.add("sudo *", scope="user")
    assert approvals.user_patterns == before


def test_session_patterns_never_reach_approved_toml(xdg_home):
    approvals = Approvals.default()
    approvals.add("kubectl get *", scope="session")
    approvals.save()
    reloaded = Approvals.load()
    assert "kubectl get *" not in reloaded.user_patterns
    assert "kubectl get *" not in (xdg_home / "nvsh" / "approved.toml").read_text(encoding="utf-8")


# --- d15: session approvals survive the process, die with the session ----


def test_session_pattern_survives_into_a_second_process(xdg_home):
    """A session approval is useless if it dies with the CLI process that made it.

    ``nvsh approve add <cmd> --session`` runs in a throwaway subprocess, so
    the pattern has to land somewhere a later process can read it — the
    runtime dir, which the login session owns.
    """
    Approvals.load().add("apt install foo", scope="session")
    fresh = Approvals.load()  # a whole new process
    assert fresh.decide("apt install foo") == "session"
    assert "apt install foo" in fresh.session_patterns


def test_session_store_lives_under_the_runtime_dir_at_0600(xdg_home):
    Approvals.load().add("apt install foo", scope="session")
    path = xdg_home / "run" / "nvsh" / "session-approvals.toml"
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_session_patterns_die_with_the_runtime_dir(xdg_home, monkeypatch):
    Approvals.load().add("apt install foo", scope="session")
    fresh_runtime = xdg_home / "run2"
    fresh_runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(fresh_runtime))
    assert Approvals.load().decide("apt install foo") == "ask"


def test_session_remove_is_persisted_too(xdg_home):
    Approvals.load().add("apt install foo", scope="session")
    approvals = Approvals.load()
    approvals.remove("apt install foo")
    approvals.save()
    assert Approvals.load().decide("apt install foo") == "ask"


def test_refusal_reason_is_public(xdg_home):
    from nvsh.approvals import refusal_reason

    assert refusal_reason("nvidia-smi *") is None
    assert "rm" in (refusal_reason("rm -rf /x") or "")
    assert refusal_reason("*") is not None
    assert "sudo" in (refusal_reason("sudo nvpmodel -m 0") or "")


def test_save_writes_0600(xdg_home):
    approvals = Approvals.default()
    approvals.add("kubectl get *", scope="user")
    approvals.save()
    path = xdg_home / "nvsh" / "approved.toml"
    assert path.is_file()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_load_reads_saved_patterns(xdg_home):
    approvals = Approvals.default()
    approvals.add("kubectl get *", scope="user")
    approvals.save()
    reloaded = Approvals.load()
    assert "kubectl get *" in reloaded.user_patterns


def test_load_missing_file_uses_defaults(xdg_home):
    approvals = Approvals.load()
    assert "nvsh *" in approvals.user_patterns


def test_remove_pattern(xdg_home):
    approvals = Approvals.default()
    approvals.add("kubectl get *", scope="user")
    approvals.remove("kubectl get *")
    assert "kubectl get *" not in approvals.user_patterns


def test_decide_returns_pattern_via_matches_helper(xdg_home):
    approvals = Approvals.default()
    scope, pattern = approvals.matches("nvidia-smi -q")
    assert scope == "user"
    assert pattern == "nvidia-smi *"


# --- PR #8 review: a wildcard must not smuggle sudo/rm past the policy ------


@pytest.mark.parametrize(
    "pattern",
    ["sudo*", "sudo *", "su[d]o *", "rm*", "r?", "*", "* -rf /", "?udo *"],
)
def test_add_refuses_globbed_privileged_patterns(xdg_home, pattern):
    """A pattern whose executable token can *match* sudo/rm is refused too.

    ``refusal_reason`` used to compare the first word literally, so ``sudo*``
    slipped through and then auto-approved ``sudo rm -rf /`` at match time.
    """
    from nvsh.approvals import ApprovalError

    approvals = Approvals.default()
    with pytest.raises(ApprovalError):
        approvals.add(pattern, scope="user")


def test_stored_wildcard_never_auto_approves_sudo(xdg_home):
    """Defence in depth: even a pattern already on disk cannot approve sudo."""
    approvals = Approvals.default()
    approvals.user_patterns.append("sudo*")  # as if hand-edited into approved.toml
    assert approvals.decide("sudo rm -rf /") == "ask"
    assert approvals.matches("sudo rm -rf /") == ("ask", None)


def test_stored_wildcard_never_auto_approves_rm(xdg_home):
    approvals = Approvals.default()
    approvals.session_patterns.append("*")
    assert approvals.decide("rm -rf /etc") == "ask"


def test_ordinary_commands_still_match_their_patterns(xdg_home):
    approvals = Approvals.default()
    assert approvals.decide("nvidia-smi -q") == "user"


def test_default_patterns_are_all_acceptable(xdg_home):
    from nvsh.approvals import refusal_reason

    for pattern in DEFAULT_PATTERNS:
        assert refusal_reason(pattern) is None, pattern


# --- PR #8 review: removal normalizes the way add() does -------------------


def test_remove_normalizes_whitespace(xdg_home):
    approvals = Approvals.default()
    approvals.add("kubectl   get *", scope="user")
    assert approvals.remove("kubectl get  *") is True
    assert not [p for p in approvals.user_patterns if p.startswith("kubectl")]


def test_remove_reports_whether_anything_went(xdg_home):
    approvals = Approvals.default()
    assert approvals.remove("nothing like this *") is False
    assert approvals.remove("nvidia-smi *") is True


# --- d24: the specific form, and one pattern per pipeline stage -----------


@pytest.mark.parametrize(
    "command,scope,expected",
    [
        ("ssh orin ps", "session", "ssh orin ps"),
        ("ssh orin ps", "user", "ssh *"),
        ("ssh orin ps", "session-specific", "ssh orin *"),
        ("ssh orin ps", "user-specific", "ssh orin *"),
        ("docker ps -a", "user-specific", "docker ps *"),
        ("git status --short", "user-specific", "git status *"),
        # A quoted first argument keeps its quotes: the pattern is matched
        # against the raw (whitespace-normalized) line, not a shlex-split one.
        ('ssh "my host" ps', "user-specific", 'ssh "my host" *'),
        ('ssh orin "ps | head"', "user-specific", "ssh orin *"),
        # No second word: the specific keys fall back to the plain form.
        ("htop", "user-specific", "htop *"),
        ("htop", "session-specific", "htop"),
        ("htop", "session", "htop"),
        ("htop", "user", "htop *"),
        # Whitespace is normalized before anything else.
        ("  ssh   orin   ps  ", "user-specific", "ssh orin *"),
    ],
)
def test_pattern_for_covers_every_scope(command, scope, expected):
    from nvsh.approvals import pattern_for

    assert pattern_for(command, scope) == expected


def test_pattern_for_rejects_an_unknown_scope():
    from nvsh.approvals import ApprovalError, pattern_for

    with pytest.raises(ApprovalError):
        pattern_for("ls", "forever")


@pytest.mark.parametrize(
    "command,expected",
    [
        ("ls -l", ["ls -l"]),
        ("ps -eo pid | head -n 20", ["ps -eo pid", "head -n 20"]),
        ("make && make install", ["make", "make install"]),
        ("make || echo failed", ["make", "echo failed"]),
        ("cd /tmp; ls", ["cd /tmp", "ls"]),
        ("sleep 1 & wait", ["sleep 1", "wait"]),
        ("ls\nps", ["ls", "ps"]),
        # Quoted separators are data, not separators: this is ONE stage.
        ('ssh orin "ps | head"', ['ssh orin "ps | head"']),
        ("echo 'a && b'", ["echo 'a && b'"]),
        ("grep -e 'x;y' file", ["grep -e 'x;y' file"]),
        # Trailing/duplicate separators never produce an empty stage.
        ("ls ;", ["ls"]),
        ("", []),
        # A subshell or command substitution is one opaque stage.
        ("(cd /tmp && ls)", ["(cd /tmp && ls)"]),
        ("echo $(rm -rf /tmp/x)", ["echo $(rm -rf /tmp/x)"]),
        ("echo `id`", ["echo `id`"]),
    ],
)
def test_stages_splits_on_separators_outside_quotes(command, expected):
    from nvsh.approvals import stages

    assert stages(command) == expected


def test_patterns_for_gives_one_pattern_per_stage():
    from nvsh.approvals import patterns_for

    assert patterns_for("ps -eo pid | head -n 20", "user") == ["ps *", "head *"]
    assert patterns_for("ps -eo pid | head -n 20", "user-specific") == ["ps -eo *", "head -n *"]
    assert patterns_for("ps -eo pid | head -n 20", "session") == ["ps -eo pid", "head -n 20"]


def test_patterns_for_dedupes_identical_stage_patterns():
    from nvsh.approvals import patterns_for

    assert patterns_for("ls /a | ls /b", "user") == ["ls *"]


def test_an_opaque_command_is_never_auto_approved(xdg_home):
    from nvsh.approvals import command_refusal_reason

    approvals = Approvals.default()
    approvals.add("echo *")
    assert approvals.decide("echo $(id)") == "ask"
    assert command_refusal_reason("echo $(id)") is not None


def test_every_stage_must_match_an_approved_pattern(xdg_home):
    approvals = Approvals.default()
    approvals.add("ps *")
    assert approvals.decide("ps -eo pid") == "user"
    assert approvals.decide("ps -eo pid | head -n 20") == "ask"
    approvals.add("head *")
    assert approvals.decide("ps -eo pid | head -n 20") == "user"


def test_a_broad_first_stage_pattern_never_covers_a_privileged_second_stage(xdg_home):
    """The whole point of d24's stage rule: `ls *` must not authorize `sudo x`."""
    approvals = Approvals.default()
    approvals.add("ls *")
    assert approvals.decide("ls -l") == "user"
    assert approvals.decide("ls | sudo x") == "ask"
    assert approvals.decide("ls -l ; rm -rf /tmp/x") == "ask"


def test_matches_reports_the_session_scope_when_any_stage_is_session_only(xdg_home):
    approvals = Approvals.default()
    approvals.add("ps *")
    approvals.add("head -n 20", scope="session")
    scope, pattern = approvals.matches("ps -eo pid | head -n 20")
    assert scope == "session"
    assert "ps *" in pattern and "head -n 20" in pattern


def test_unapproved_stage_names_the_stage_that_failed(xdg_home):
    approvals = Approvals.default()
    approvals.add("ps *")
    assert approvals.unapproved_stage("ps -eo pid | head -n 20") == "head -n 20"
    assert approvals.unapproved_stage("ps -eo pid") is None
    assert approvals.unapproved_stage("ls | sudo rmmod x") == "sudo rmmod x"


def test_add_accepts_the_specific_scope_tokens(xdg_home):
    from nvsh.approvals import base_scope

    approvals = Approvals.default()
    approvals.add("ssh orin *", scope="user-specific")
    assert "ssh orin *" in approvals.user_patterns
    approvals.add("ssh thor *", scope="session-specific")
    assert "ssh thor *" in approvals.session_patterns
    assert base_scope("user-specific") == "user"
    assert base_scope("session-specific") == "session"


# --- d26: per-stage pattern lists and the stage selection parser ----------


def test_stage_patterns_keeps_one_entry_per_stage_without_deduplicating():
    """The picker indexes by stage number, so position N must stay stage N."""
    from nvsh.approvals import patterns_for, stage_patterns

    assert stage_patterns("ls /a | ls /b", "user") == ["ls *", "ls *"]
    assert patterns_for("ls /a | ls /b", "user") == ["ls *"]
    assert stage_patterns("ls /tmp/git | grep -i orin", "user-specific") == [
        "ls /tmp/git *",
        "grep -i *",
    ]


def test_match_stage_is_public_so_the_details_view_can_name_the_approver(xdg_home):
    approvals = Approvals.default()
    approvals.add("ps *")
    assert approvals.match_stage("ps -eo pid") == ("user", "ps *")
    assert approvals.match_stage("head -n 20") == ("ask", None)


@pytest.mark.parametrize(
    "typed,expected",
    [
        ("all", [1, 2]),
        ("", [1, 2]),
        ("  ", [1, 2]),
        ("1", [1]),
        ("2", [2]),
        ("1,2", [1, 2]),
        ("1 2", [1, 2]),
        ("2,1", [1, 2]),
        ("2, 2", [2]),
        ("junk", None),
        ("3", None),
        ("0", None),
        ("1,junk", None),
        ("-1", None),
    ],
)
def test_parse_stages_table(typed, expected):
    from nvsh.approvals import parse_stages

    assert parse_stages(typed, 2) == expected
