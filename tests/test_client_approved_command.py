"""The approved-command trust boundary (SonarCloud S6350, command injection).

``nvsh`` runs the *exact* line the operator approved — the spec requires it, so
the argv stays ``["bash", "-c", command]``. What these tests pin is the only
structural defence that makes that safe: a command string may only reach
:func:`nvsh.client._run_command` through :func:`nvsh.client._approved_command`,
which takes the :class:`Proposal` the panel actually showed plus the decision
:func:`nvsh.client._decide_proposal` (or the approval store) returned. Nothing
from ``argv``, an environment variable or an agent's raw output can get there
on its own.
"""

from __future__ import annotations

import pytest

from nvsh import client as client_mod
from nvsh.agent.base import Proposal, ProposalKind
from nvsh.panel import APPROVE, APPROVE_SESSION, APPROVE_USER, IGNORE


def _proposal(command="ls -la"):
    return Proposal(command=command, rationale="look around", kind=ProposalKind.INSPECT)


@pytest.mark.parametrize("decision", [APPROVE, APPROVE_SESSION, APPROVE_USER])
def test_approved_command_returns_the_exact_line(decision):
    proposal = _proposal("kubectl get pods | grep -c Running")
    assert client_mod._approved_command(proposal, decision) == proposal.command


def test_auto_inspect_is_an_approved_decision():
    proposal = _proposal()
    assert client_mod._approved_command(proposal, client_mod.AUTO_INSPECT) == "ls -la"


@pytest.mark.parametrize("decision", [IGNORE, client_mod.STEERED, "", "yes", None])
def test_unapproved_decisions_are_refused(decision):
    proposal = _proposal()
    with pytest.raises(client_mod.UnapprovedCommandError):
        client_mod._approved_command(proposal, decision)


def test_a_bare_string_is_not_a_proposal():
    with pytest.raises(client_mod.UnapprovedCommandError):
        client_mod._approved_command("rm -rf /", APPROVE)  # type: ignore[arg-type]


@pytest.mark.parametrize("command", ["", "   ", "ls\x00-la", "ls\x1b]133;C\x07", "ls\x07"])
def test_commands_with_no_body_or_control_bytes_are_refused(command):
    proposal = _proposal(command)
    with pytest.raises(client_mod.UnapprovedCommandError):
        client_mod._approved_command(proposal, APPROVE)


def test_multiline_fix_commands_are_still_allowed():
    command = "set -x\nmake -j4\techo done"
    assert client_mod._approved_command(_proposal(command), APPROVE) == command


def test_run_approved_refuses_before_spawning_anything(monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "_run_command", lambda *a, **k: calls.append((a, k)))
    proposal = _proposal()
    with pytest.raises(client_mod.UnapprovedCommandError):
        client_mod._run_approved(proposal, IGNORE)
    assert calls == []


def test_run_approved_passes_the_approved_line_through(monkeypatch):
    seen = {}

    def fake(command, timeout=None, login=False):
        seen.update(command=command, timeout=timeout, login=login)
        return client_mod.RunResult(exit_code=0)

    monkeypatch.setattr(client_mod, "_run_command", fake)
    result = client_mod._run_approved(_proposal("echo hi"), APPROVE, login=True)
    assert result.exit_code == 0
    assert seen == {"command": "echo hi", "timeout": None, "login": True}


def test_every_executor_call_site_goes_through_the_boundary():
    """No caller may reach ``_run_command`` except ``_run_approved``."""
    import inspect
    import re

    source = inspect.getsource(client_mod)
    # Drop the definition line and the one call inside _run_approved.
    call_sites = [
        line.strip()
        for line in source.splitlines()
        if re.search(r"(?<![\w.])_run_command\s*\(", line) and not line.strip().startswith("def ")
    ]
    assert len(call_sites) == 1, call_sites
