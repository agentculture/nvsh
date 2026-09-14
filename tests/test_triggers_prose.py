"""``prose_request``: a sentence typed at the prompt is a question, not a failure (d20).

The operator types ``what are the memory levels?`` at a bash prompt. Bash
reports ``command not found`` and exit 127. Treating that as a failed
command asks the agent to diagnose ``what``; treating it as prose asks the
agent the question the operator actually asked.
"""

from __future__ import annotations

import pytest

from nvsh.triggers import prose_request

_POSITIVES = [
    "what are the memory levels?",
    "why is the gpu slow",
    "show me disk usage please",
    "how much free memory is there",
    "check whether docker is running",
    "is the nvidia driver loaded?",
    "tell me the cuda version",
]

_NEGATIVES = [
    "ls /nope",
    "git sttus",
    "make -j8",
    "python3 train.py",
    "docker ps",
    "./configure --prefix=/usr",
    "cat a b c",
    "foo bar baz",
    "grep -r needle .",
    "what",
    "why?",
    "what is this | grep x",
    "echo what is the memory > /tmp/x",
]


@pytest.mark.parametrize("line", _POSITIVES)
def test_prose_is_recognized(line):
    found = prose_request(line, 127)
    assert found is not None
    assert found.question == line.strip()
    assert found.agent is None
    # An unmarked sentence is a guess, not an explicit call: it still pays
    # the auto-call rate limit.
    assert found.explicit is False


@pytest.mark.parametrize("line", _NEGATIVES)
def test_commands_are_not_prose(line):
    assert prose_request(line, 127) is None


def test_only_command_not_found_counts():
    assert prose_request("what are the memory levels?", 2) is None
    assert prose_request("what are the memory levels?", 1) is None


def test_empty_line_is_not_prose():
    assert prose_request("", 127) is None
    assert prose_request("   ", 127) is None


# --- d23: the ? and @name marks -------------------------------------------

_MARKED = [
    ("? what are the ram memory levels?", "what are the ram memory levels?"),
    ("?what are the ram memory levels?", "what are the ram memory levels?"),
    ("? ram", "ram"),
    ("?ram?", "ram?"),
    ("? why is /dev/nvme0n1 full", "why is /dev/nvme0n1 full"),
    ("?  rebuild the container  ", "rebuild the container"),
]


@pytest.mark.parametrize("line,question", _MARKED)
def test_question_mark_is_an_explicit_request(line, question):
    found = prose_request(line, 127)
    assert found is not None
    assert found.question == question
    assert found.agent is None
    assert found.explicit is True


_NOT_MARKED = [
    "?",
    "?*.txt",
    "? ",
    "?1x",
    "?.config",
    "?foo",
    "@",
    "@ ",
    "@pi",
    "@foo.bar hello",
    "@notaharness what is up",
    "mail @foo.bar",
    "echo user@example.com",
]


@pytest.mark.parametrize("line", _NOT_MARKED)
def test_non_marks_are_not_requests(line):
    assert prose_request(line, 127) is None


@pytest.mark.parametrize("name", ["pi", "qwen", "claude", "codex", "openai-compat"])
def test_at_name_picks_that_harness(name):
    found = prose_request(f"@{name} how much ram is free?", 127)
    assert found is not None
    assert found.question == "how much ram is free?"
    assert found.agent == name
    assert found.explicit is True


def test_marks_still_need_command_not_found():
    assert prose_request("? how much ram", 0) is None
    assert prose_request("@pi how much ram", 2) is None


# --- task t6: @target grammar (backend[/model[/effort]] and aliases) ------


@pytest.fixture
def aliased_config(tmp_path, monkeypatch):
    """A config.toml with [aliases] reviewer/local/default, for the @target tests."""
    config_home = tmp_path / "config"
    (config_home / "nvsh").mkdir(parents=True)
    (config_home / "nvsh" / "config.toml").write_text(
        '[aliases]\ndefault = "pi"\nreviewer = "claude/opus/high"\nlocal = "pi"\n'
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return config_home


_TARGET_POSITIVES = [
    ("@claude/sonnet/medium why is the gpu slow", "why is the gpu slow", "claude/sonnet/medium"),
    ("@reviewer why is the gpu slow", "why is the gpu slow", "reviewer"),
    ("@local why is the gpu slow", "why is the gpu slow", "local"),
    ("@default why is the gpu slow", "why is the gpu slow", "default"),
    ("@claude/sonnet why is the gpu slow", "why is the gpu slow", "claude/sonnet"),
]


@pytest.mark.parametrize("line,question,agent", _TARGET_POSITIVES)
def test_at_target_grammar_is_recognized(aliased_config, line, question, agent):
    found = prose_request(line, 127)
    assert found is not None
    assert found.question == question
    assert found.agent == agent
    assert found.explicit is True


def test_at_default_works_even_without_a_configured_default_alias(tmp_path, monkeypatch):
    # 'default' is always a valid target (falls back to [agent] provider),
    # even with no config.toml on disk at all.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    found = prose_request("@default why is the gpu slow", 127)
    assert found is not None
    assert found.agent == "default"


_TARGET_NEGATIVES = [
    "@notaharness/model what is up",  # unregistered backend segment
    "@notanalias what is up",  # unregistered plain name
    "@claude// what is up",  # empty segment
    "@claude/sonnet/medium/extra what is up",  # too many segments
    "@/claude/sonnet what is up",  # first char rule: '/' cannot lead
    "@claude/sonnet/ what is up",  # trailing slash, empty final segment
    "echo user@example.com",  # e-mail-like '@' in argument position
    "mail a@b.c/d",  # ditto, with a slash for good measure
]


@pytest.mark.parametrize("line", _TARGET_NEGATIVES)
def test_at_target_grammar_negatives(aliased_config, line):
    assert prose_request(line, 127) is None
