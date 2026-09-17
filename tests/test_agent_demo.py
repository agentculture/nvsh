"""The ``demo`` adapter (task t1): a committed fixture replayed through the
real daemon and client path.

Four acceptance criteria, in order:

1. :class:`nvsh.agent.demo.DemoAgent` is a :class:`~nvsh.agent.fake.FakeAgent`
   that loads ``nvsh/agent/demo_fixture.json`` -- text deltas plus one
   ``chmod +x`` proposal whose script path is read off the failing command
   line in the :class:`~nvsh.agent.base.AgentRequest`.
2. ``ADAPTERS['demo']`` needs no binary: ``installed('demo')`` is ``True``
   without ``shutil.which`` ever being called, and ``nvsh agent list --json``
   reports it installed.
3. :func:`nvsh.agent.registry.probe` never offers ``demo`` -- the same
   exclusion ``openai-compat`` has -- even when every binary is on PATH.
4. A failing command with ``demo`` as the default target opens the panel
   through a *real* daemon, and the audit log stamps the turn ``demo``.

Nothing here spawns a harness, and nothing touches the network: the whole
point of the adapter under test is that the answer is a file in the repo.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import time
import tomllib
import types
from fnmatch import fnmatch
from pathlib import Path

import pytest

from nvsh import client as client_mod
from nvsh import daemon as daemon_mod
from nvsh import panel as panel_mod
from nvsh.agent import demo as demo_mod
from nvsh.agent import registry
from nvsh.agent.audit import AuditLog
from nvsh.agent.base import AgentContext, AgentRequest, EventKind, ProposalKind, RequestKind
from nvsh.agent.fake import FakeAgent
from nvsh.cli import main
from nvsh.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "nvsh" / "agent" / "demo_fixture.json"


def _request(command: str) -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, command=command, exit_code=126)


def _events(agent, command: str) -> list:
    agent.start()
    return list(agent.run(_request(command), AgentContext()))


def _proposals(events) -> list:
    return [event.proposal for event in events if event.kind is EventKind.PROPOSAL]


def _text(events) -> str:
    return "".join(event.text for event in events if event.kind is EventKind.TEXT_DELTA)


# --- criterion 1: the fixture and the adapter over it ----------------------


def test_the_fixture_is_committed_json_so_a_scenario_edit_needs_no_python():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(data["events"], list)
    assert data["events"]
    kinds = [event["kind"] for event in data["events"]]
    assert "text_delta" in kinds
    assert kinds.count("proposal") == 1
    assert kinds[-1] == "done"


def test_demo_agent_is_a_fake_agent():
    assert issubclass(demo_mod.DemoAgent, FakeAgent)


def test_the_diagnosis_says_the_script_is_not_executable_and_is_short():
    events = _events(demo_mod.DemoAgent(), "./run-model.sh")
    text = _text(events)
    assert "executable" in text.lower()
    assert 2 <= len([line for line in text.splitlines() if line.strip()]) <= 5
    # It must not pass itself off as a real model's answer.
    assert "scripted demo" in text.lower()


def test_the_single_proposal_is_chmod_on_the_script_from_the_failing_line():
    events = _events(demo_mod.DemoAgent(), "./run-model.sh")
    proposals = _proposals(events)
    assert len(proposals) == 1
    assert proposals[0].command == "chmod +x -- ./run-model.sh"
    assert proposals[0].kind is ProposalKind.FIX
    assert proposals[0].rationale
    assert events[-1].kind is EventKind.DONE


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("./run-model.sh", "./run-model.sh"),
        ("./tools/serve.sh --port 8000", "./tools/serve.sh"),
        ("bash scripts/run-model.sh", "scripts/run-model.sh"),
        ("", demo_mod.DEFAULT_SCRIPT),
        ("ls /nope", demo_mod.DEFAULT_SCRIPT),  # no ./path token -> the default
    ],
)
def test_the_script_path_comes_from_the_failing_command_line(command, expected):
    assert demo_mod.script_from_command(command) == expected
    proposals = _proposals(_events(demo_mod.DemoAgent(), command))
    assert proposals[0].command == f"chmod +x -- {expected}"


def test_a_missing_or_broken_fixture_is_an_error_event_not_a_crash(tmp_path):
    agent = demo_mod.DemoAgent({"fixture": str(tmp_path / "nope.json")})
    events = _events(agent, "./run-model.sh")
    assert [event.kind for event in events] == [EventKind.ERROR]
    assert "fixture" in events[0].error

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    events = _events(demo_mod.DemoAgent({"fixture": str(broken)}), "./run-model.sh")
    assert [event.kind for event in events] == [EventKind.ERROR]


def test_the_fixture_is_packaged_into_the_wheel():
    """A data file hatchling is not told about silently misses the wheel."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    rel = "nvsh/agent/demo_fixture.json"
    for key in ("include", "artifacts"):
        assert any(fnmatch(rel, pattern) for pattern in wheel[key]), key


# --- criterion 2: the registry entry needs no binary -----------------------


def test_the_demo_adapter_spec_needs_no_binary_and_is_local():
    spec = registry.ADAPTERS["demo"]
    assert spec.binary is None
    assert spec.hosted is False
    assert spec.path in registry.PATH_VALUES
    assert spec.needs_node is False


def test_the_demo_adapter_declares_tool_calling():
    assert registry.ADAPTERS["demo"].factory(Config()).capabilities().tool_calling is True
    assert registry._tool_calling("demo", Config()) is True


def test_the_demo_adapter_declares_steer_false():
    assert registry.ADAPTERS["demo"].factory(Config()).capabilities().steer is False
    assert registry.steer_capable("demo", Config()) is False


def test_installed_demo_is_true_without_ever_calling_which():
    def _explode(name):  # pragma: no cover - called only on a regression
        raise AssertionError(f"shutil.which was called with {name!r}")

    assert registry.installed("demo", which=_explode) is True


def test_agent_list_json_shows_demo_installed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert main(["agent", "list", "--json"]) == 0
    rows = {row["name"]: row for row in json.loads(capsys.readouterr().out)["adapters"]}
    assert rows["demo"]["installed"] is True
    assert rows["demo"]["binary"] is None
    assert rows["demo"]["capabilities"]["tool_calling"] is True


def test_agent_use_help_still_works_and_names_demo(capsys):
    with pytest.raises(SystemExit):
        main(["agent", "use", "--help"])
    assert "demo" in capsys.readouterr().out


# --- criterion 3: probe never offers demo ---------------------------------


def test_probe_never_returns_demo_even_with_everything_on_path():
    rows = registry.probe(which=lambda _name: "/bin/true")
    names = [row["name"] for row in rows]
    assert "demo" not in names
    assert "openai-compat" not in names  # the exclusion demo is modelled on
    assert "pi" in names  # ... and the fake PATH really did install the rest


def test_choose_never_auto_picks_demo():
    config = Config(agent_provider="pi")
    name, _reason = registry.choose(config, which=lambda _name: "/bin/true")
    assert name != "demo"


def test_choose_still_honours_a_forced_demo_target():
    config = Config(agent_provider="pi")
    name, reason = registry.choose(config, which=lambda _name: None, forced="demo")
    assert name == "demo"
    assert "forced" in reason


# --- criterion 4: end to end, through the daemon and into the audit log ----


pytestmark_unix = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="the daemon needs unix domain sockets"
)


def _start(daemon: daemon_mod.Daemon) -> None:
    threading.Thread(target=daemon.serve_forever, daemon=True).start()
    for _ in range(500):
        if daemon_mod.is_running(daemon.env):
            return
        time.sleep(0.01)
    raise AssertionError("daemon never created its socket")  # pragma: no cover


@pytestmark_unix
def test_a_failing_command_with_demo_as_default_streams_through_the_daemon(
    tmp_path, monkeypatch, capsys
):
    """The panel fills from the fixture, the *daemon* is what answered, and
    every audit line for the turn carries ``target: demo``."""
    for name in ("run", "state", "config"):
        (tmp_path / name).mkdir()
    env = {
        "HOME": str(tmp_path),
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "NVSH_NO_DAEMON": "1",  # never fork one; the in-process daemon is right here
        "NO_COLOR": "1",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config_dir = tmp_path / "config" / "nvsh"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        '[agent]\nprovider = "demo"\n\n[aliases]\ndefault = "demo"\n', encoding="utf-8"
    )
    monkeypatch.setattr(client_mod, "_platform_block", lambda: "platform: fixture")

    daemon = daemon_mod.Daemon(Config(agent_provider="demo", aliases={"default": "demo"}), env=env)
    _start(daemon)
    panel = panel_mod.Panel(
        out=io.StringIO(), in_=io.StringIO("q\n"), env={"NO_COLOR": "1"}, isatty=False
    )
    args = types.SimpleNamespace(
        exit=126,
        pipestatus="126",
        line="./run-model.sh",
        cwd=str(tmp_path),
        log="",
        json=False,
    )
    try:
        # "q" ignores the demo proposal: declined, exit 3 (t17)
        assert client_mod.handle_failure(args, panel=panel) == 3
    finally:
        daemon.shutdown()

    out = panel.out.getvalue()
    assert "./run-model.sh failed (exit 126)" in out
    assert "executable" in out.lower()
    assert "chmod +x -- ./run-model.sh" in out
    assert "chmod" not in capsys.readouterr().out, "the fix must be proposed, never run"

    # It really went through the daemon, not the client's one-shot fallback.
    assert daemon._backend_name == "demo"

    entries = AuditLog(env=env).read_all()
    assert entries, "the proposal was never recorded"
    assert all(entry["target"]["backend"] == "demo" for entry in entries)
    proposals = [entry for entry in entries if entry["event"] == "proposal"]
    assert len(proposals) == 1
    assert proposals[0]["proposal"]["command"] == "chmod +x -- ./run-model.sh"


# --- the platform placeholder (deviation d5) --------------------------------


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("platform: dgx-spark\n  dgx_name: DGX Spark  [file: /etc/dgx-release]", "dgx-spark"),
        ("platform: jetson\n  l4t_release: R36", "jetson"),
        ("", "this machine"),
        ("platform: unknown (detection failed: boom)", "unknown"),
    ],
)
def test_platform_kind_is_the_first_platform_line(block, expected):
    from nvsh.agent.demo import platform_kind

    assert platform_kind(block) == expected


def test_the_reply_names_the_platform_from_the_context():
    from nvsh.agent.base import AgentContext, AgentRequest, EventKind, RequestKind
    from nvsh.agent.demo import DemoAgent

    agent = DemoAgent()
    agent.start()
    request = AgentRequest(kind=RequestKind.FAILURE, command="./run-model.sh", exit_code=126)
    context = AgentContext(platform="platform: jetson\n  l4t_release: R36")
    text = "".join(e.text for e in agent.run(request, context) if e.kind == EventKind.TEXT_DELTA)
    assert "jetson" in text
    assert "{platform}" not in text


# --- review fixes (PR #14, Qodo 1, 4, 6, 7, 10) --------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("./model.sh;id", "./run-model.sh"),  # metacharacters: refused, default used
        ("./a.sh && rm -rf /", "./a.sh"),  # the operator only ever sees chmod +x ./a.sh
        ("'./evil$(id).sh'", "./run-model.sh"),
        ("python tool.py --output report.sh", "./run-model.sh"),  # argument, not command
        ("bash scripts/run-model.sh", "scripts/run-model.sh"),
        ("sh ./x.sh --flag", "./x.sh"),
        ("./ok-1.sh arg", "./ok-1.sh"),
        ("../up.sh", "./run-model.sh"),
        ("", "./run-model.sh"),
    ],
)
def test_script_token_is_command_position_only_and_shell_safe(command, expected):
    from nvsh.agent.demo import script_from_command

    assert script_from_command(command) == expected


def test_proposal_never_carries_shell_metacharacters():
    from nvsh.agent.base import EventKind
    from nvsh.agent.demo import FIXTURE_PATH, load_events

    events = load_events(FIXTURE_PATH, "./model.sh;id")
    proposal = next(e.proposal for e in events if e.kind == EventKind.PROPOSAL)
    assert proposal.command == "chmod +x -- ./run-model.sh"
    assert ";" not in proposal.command


@pytest.mark.parametrize(
    "body", ["[]", "42", '{"events": {}}', '{"events": []}', '{"events": [1]}']
)
def test_wrongly_shaped_fixture_is_an_error_event_not_a_crash(tmp_path, body):
    from nvsh.agent.base import AgentContext, AgentRequest, EventKind, RequestKind
    from nvsh.agent.demo import DemoAgent

    fixture = tmp_path / "bad.json"
    fixture.write_text(body, encoding="utf-8")
    agent = DemoAgent({"fixture": str(fixture)})
    agent.start()
    events = list(
        agent.run(AgentRequest(kind=RequestKind.FAILURE, command="./x.sh"), AgentContext())
    )
    assert [e.kind for e in events] == [EventKind.ERROR]
    assert "unusable" in events[0].error


def test_config_accepts_the_fixture_key(tmp_path, monkeypatch):
    from nvsh import config as nvsh_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "nvsh").mkdir()
    (tmp_path / "nvsh" / "config.toml").write_text(
        '[agents.demo]\nfixture = "/tmp/x.json"\n', encoding="utf-8"
    )
    cfg = nvsh_config.load()
    assert cfg.agents["demo"]["fixture"] == "/tmp/x.json"


def test_doctor_rejects_a_malformed_fixture(tmp_path):
    from nvsh import config as nvsh_config
    from nvsh.doctor_checks import _check_demo_reachable

    fixture = tmp_path / "bad.json"
    fixture.write_text("[]", encoding="utf-8")
    cfg = nvsh_config.Config(agents={"demo": {"fixture": str(fixture)}})
    check = _check_demo_reachable(cfg)
    assert check["passed"] is False
    assert "unusable" in check["message"]


def test_an_option_looking_token_or_fixture_default_is_refused():
    from nvsh.agent.demo import DEFAULT_SCRIPT, safe_script, script_from_command

    assert safe_script("-x.sh") is None
    assert safe_script("--reference=/etc/passwd.sh") is None
    assert safe_script("./ok.sh") == "./ok.sh"
    # a fixture default that is itself unsafe is not trusted either
    assert script_from_command("", default="-evil.sh") == DEFAULT_SCRIPT
    assert script_from_command("-x.sh", default="./d.sh") == "./d.sh"
