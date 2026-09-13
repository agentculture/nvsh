"""Tests for nvsh.approvals — the glob-pattern approval store.

Acceptance criterion covered (task t5):
approved.toml (0600) holds glob patterns; matches(cmd) uses fnmatch on the
full command line; add() refuses 'sudo *', 'rm *', bare '*' and patterns
starting with 'sudo'/'rm'; the default list contains 'nvidia-smi *',
'docker ps*', 'journalctl *', 'systemctl status *', 'df *', 'free *',
'nvsh *'; a session list lives in memory only.
"""

from __future__ import annotations

import stat

import pytest

from nvsh.approvals import DEFAULT_PATTERNS, Approvals


@pytest.fixture()
def xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
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


def test_session_patterns_not_persisted(xdg_home):
    approvals = Approvals.default()
    approvals.add("kubectl get *", scope="session")
    approvals.save()
    reloaded = Approvals.load()
    assert "kubectl get *" not in reloaded.user_patterns
    assert reloaded.session_patterns == []


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
