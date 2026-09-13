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
    assert prose_request(line, 127) == line.strip()


@pytest.mark.parametrize("line", _NEGATIVES)
def test_commands_are_not_prose(line):
    assert prose_request(line, 127) is None


def test_only_command_not_found_counts():
    assert prose_request("what are the memory levels?", 2) is None
    assert prose_request("what are the memory levels?", 1) is None


def test_empty_line_is_not_prose():
    assert prose_request("", 127) is None
    assert prose_request("   ", 127) is None
