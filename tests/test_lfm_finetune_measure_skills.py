"""Skill-routing measurement script: scripts/lfm-finetune/measure_skills.py.

Exercises the whole CLI against a fake OpenAI-compatible endpoint (a real
``http.server`` on 127.0.0.1, random port) -- no network, no real model.
Part of #39 (t9).
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

from nvsh.tiers.runtime import RuntimeUnavailable as RuntimeUnavailableForTest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/measure_skills.py"


def _module():
    spec = importlib.util.spec_from_file_location("measure_skills", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _module()


# ---------------------------------------------------------------------------
# fixtures: tools.json / test.jsonl + a fake endpoint
# ---------------------------------------------------------------------------


def _tool(skill: str, repo: str) -> dict:
    tool_name = skill.replace("-", "_")
    return {
        "skill": skill,
        "repo": repo,
        "tool": {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"the {skill} skill",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    }


def _tools_fixture() -> list[dict]:
    return [
        _tool("jetson-diagnostic", "device"),
        _tool("jetson-thermal", "device"),
        _tool("bsp-flash", "bsp"),
    ]


def _eval(eval_id: str, repo: str, expected_skill: str, names_skill: bool) -> dict:
    return {
        "id": eval_id,
        "repo": repo,
        "skill": expected_skill,
        "text": f"fixture prompt for {eval_id}",
        "expected_skill": expected_skill,
        "ground_truth": None,
        "names_skill": names_skill,
    }


def _evals_fixture() -> list[dict]:
    return [
        _eval("ev-correct", "device", "jetson-diagnostic", False),
        _eval("ev-wrong", "device", "jetson-thermal", False),
        _eval("ev-no-call", "bsp", "bsp-flash", False),
        _eval("ev-several", "device", "jetson-diagnostic", True),
    ]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    tools_path = tmp_path / "tools.json"
    test_path = tmp_path / "test.jsonl"
    tools_path.write_text(json.dumps(_tools_fixture()), encoding="utf-8")
    test_path.write_text(
        "\n".join(json.dumps(ev) for ev in _evals_fixture()) + "\n", encoding="utf-8"
    )
    return tools_path, test_path


def _tool_call(name: str) -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": "{}"}}


#: One scripted response per eval above, in order: correct, wrong skill, no
#: call, several calls.
_SCRIPTED_TOOL_CALLS: list[list[dict]] = [
    [_tool_call("jetson_diagnostic")],
    [_tool_call("jetson_diagnostic")],  # eval expects jetson-thermal: wrong skill
    [],
    [_tool_call("jetson_diagnostic"), _tool_call("bsp_flash")],
]


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    scripted: list[list[dict]] = []
    calls = 0
    requests: list[dict] = []
    #: Optional message content per call, in order (default: empty).
    contents: list[str] = []
    #: What GET /models answers with (issue 46 preflight); every fixture test uses
    #: "fake-model", so this default satisfies preflight_models without per-test setup.
    model_ids: list[str] = ["fake-model"]
    #: 0-based POST call indices that simulate a dead server (issue 46: a call error).
    fail_indices: set[int] = set()

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming convention
        if self.path.rstrip("/") != "/models":
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(
            {"data": [{"id": model_id} for model_id in _FakeHandler.model_ids]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming convention
        index = _FakeHandler.calls
        _FakeHandler.calls += 1
        if index in _FakeHandler.fail_indices:
            # Simulate a crashed/unreachable server for this one call: drop the
            # connection with no response, rather than answering anything.
            self.close_connection = True
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _FakeHandler.requests.append(json.loads(body.decode("utf-8")))
        tool_calls = _FakeHandler.scripted[index] if index < len(_FakeHandler.scripted) else []
        content = _FakeHandler.contents[index] if index < len(_FakeHandler.contents) else ""
        payload = {
            "choices": [
                {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}
            ]
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args) -> None:  # keep test output quiet
        pass


@pytest.fixture
def fake_server():
    _FakeHandler.calls = 0
    _FakeHandler.requests = []
    _FakeHandler.scripted = [list(calls) for calls in _SCRIPTED_TOOL_CALLS]
    _FakeHandler.model_ids = ["fake-model"]
    _FakeHandler.fail_indices = set()
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _base_url(server: http.server.HTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


# ---------------------------------------------------------------------------
# unit-level: scoring
# ---------------------------------------------------------------------------


def test_classify_correct(mod):
    assert mod.classify(("jetson_diagnostic",), "jetson_diagnostic") == mod.OUTCOME_CORRECT


def test_classify_wrong_skill(mod):
    assert mod.classify(("jetson_thermal",), "jetson_diagnostic") == mod.OUTCOME_WRONG_SKILL


def test_classify_no_call(mod):
    assert mod.classify((), "jetson_diagnostic") == mod.OUTCOME_NO_CALL


def test_classify_several_calls(mod):
    assert (
        mod.classify(("jetson_diagnostic", "bsp_flash"), "jetson_diagnostic")
        == mod.OUTCOME_SEVERAL_CALLS
    )


def test_require_localhost_accepts_loopback(mod):
    mod.require_localhost("http://127.0.0.1:8000")
    mod.require_localhost("http://localhost:8000")


def test_require_localhost_refuses_remote(mod):
    with pytest.raises(ValueError):
        mod.require_localhost("http://example.com:8000")


def test_require_localhost_refuses_https(mod):
    with pytest.raises(ValueError):
        mod.require_localhost("https://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# end-to-end against the fake endpoint
# ---------------------------------------------------------------------------


def test_run_measurement_and_aggregate(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    tools = mod.load_tools(tools_path)
    evals = mod.load_evals(test_path)
    results = mod.run_measurement(_base_url(fake_server), "fake-model", tools, evals)

    outcomes = {r.id: r.outcome for r in results}
    assert outcomes == {
        "ev-correct": mod.OUTCOME_CORRECT,
        "ev-wrong": mod.OUTCOME_WRONG_SKILL,
        "ev-no-call": mod.OUTCOME_NO_CALL,
        "ev-several": mod.OUTCOME_SEVERAL_CALLS,
    }

    aggregated = mod.aggregate(results)
    assert aggregated.overall.correct == 1
    assert aggregated.overall.total == 4
    # only ev-several names its skill in the prompt, and it is not correct
    assert aggregated.named_in_prompt.total == 1
    assert aggregated.named_in_prompt.correct == 0
    assert aggregated.not_named.total == 3
    assert aggregated.not_named.correct == 1
    assert aggregated.per_repo["device"].total == 3
    assert aggregated.per_repo["bsp"].total == 1
    assert aggregated.outcome_counts[mod.OUTCOME_SEVERAL_CALLS] == 1

    # requests were sent with tool_choice "auto" and temperature 0 (operator spec)
    for request in _FakeHandler.requests:
        assert request["tool_choice"] == "auto"
        assert request["temperature"] == 0
        assert request["model"] == "fake-model"


def test_run_measurement_refuses_unknown_expected_skill(mod, fake_server, tmp_path):
    tools_path, _test_path = _write_inputs(tmp_path)
    tools = mod.load_tools(tools_path)
    bad_eval = [_eval("ev-bad", "device", "no-such-skill", False)]
    base_url = _base_url(fake_server)
    with pytest.raises(ValueError):
        mod.run_measurement(base_url, "fake-model", tools, bad_eval)


# ---------------------------------------------------------------------------
# CLI: stock run
# ---------------------------------------------------------------------------


def test_main_stock_run_writes_results_file(mod, fake_server, tmp_path, capsys):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--url",
            _base_url(fake_server),
            "--model",
            "fake-model",
            "--label",
            "stock",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "## Margin claimed before scoring" not in text
    assert "1 of 4" in text
    assert "device" in text
    assert "bsp" in text
    captured = capsys.readouterr()
    assert str(out_path) in captured.out


# ---------------------------------------------------------------------------
# CLI: tuned run requires --margin, written ahead of the numbers
# ---------------------------------------------------------------------------


def test_main_tuned_run_without_margin_is_refused(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    argv = [
        "--tools",
        str(tools_path),
        "--test",
        str(test_path),
        "--url",
        _base_url(fake_server),
        "--model",
        "fake-model",
        "--label",
        "tuned",
        "--out",
        str(tmp_path / "results.md"),
    ]
    with pytest.raises(SystemExit) as excinfo:
        mod.main(argv)
    assert excinfo.value.code != 0
    # no request should have been sent before the refusal
    assert _FakeHandler.calls == 0


def test_main_tuned_flag_without_margin_is_refused_even_with_stock_label(
    mod, fake_server, tmp_path
):
    tools_path, test_path = _write_inputs(tmp_path)
    argv = [
        "--tools",
        str(tools_path),
        "--test",
        str(test_path),
        "--url",
        _base_url(fake_server),
        "--model",
        "fake-model",
        "--tuned",
        "--out",
        str(tmp_path / "results.md"),
    ]
    with pytest.raises(SystemExit):
        mod.main(argv)
    assert _FakeHandler.calls == 0


def test_main_tuned_run_with_margin_writes_it_before_the_numbers(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--url",
            _base_url(fake_server),
            "--model",
            "fake-model",
            "--label",
            "tuned",
            "--margin",
            "+15 points overall",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    margin_index = text.index("## Margin claimed before scoring")
    results_index = text.index("## Results")
    assert margin_index < results_index
    assert "+15 points overall" in text
    assert text.index("+15 points overall") < results_index


# ---------------------------------------------------------------------------
# CLI: --url localhost enforcement and manifest provenance
# ---------------------------------------------------------------------------


def test_main_refuses_non_localhost_url(mod, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    with pytest.raises(SystemExit):
        mod.main(
            [
                "--tools",
                str(tools_path),
                "--test",
                str(test_path),
                "--url",
                "http://example.com:8000",
                "--model",
                "fake-model",
                "--out",
                str(tmp_path / "results.md"),
            ]
        )


def test_main_requires_model(mod, tmp_path, monkeypatch):
    monkeypatch.delenv(mod.MODEL_ENV_VAR, raising=False)
    tools_path, test_path = _write_inputs(tmp_path)
    with pytest.raises(SystemExit):
        mod.main(
            [
                "--tools",
                str(tools_path),
                "--test",
                str(test_path),
                "--out",
                str(tmp_path / "results.md"),
            ]
        )


def test_main_records_manifest_provenance(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "repositories": [
                    {
                        "repo": "device",
                        "url": "https://github.com/NVIDIA-AI-IOT/jetson-device-skills",
                        "commit": "deadbeef" * 5,
                    },
                    {
                        "repo": "bsp",
                        "url": "https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills",
                        "commit": "cafef00d" * 5,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "results.md"
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--url",
            _base_url(fake_server),
            "--model",
            "fake-model",
            "--model-revision",
            "9e6c6ccf47cd",
            "--manifest",
            str(manifest_path),
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "deadbeef" * 5 in text
    assert "cafef00d" * 5 in text
    assert "9e6c6ccf47cd" in text


def test_default_out_path_uses_label_and_date(mod):
    path = mod.default_out_path("stock", "2026-09-22")
    assert path.name == "2026-09-22-skills-stock.md"
    assert path.parent.name == "benchmarks"


# ---------------------------------------------------------------------------
# finding 1: credentials/endpoints must never leak
# ---------------------------------------------------------------------------


def test_require_localhost_refuses_userinfo(mod):
    with pytest.raises(ValueError):
        mod.require_localhost("http://user:example-secret@127.0.0.1:8000/v1")
    with pytest.raises(ValueError):
        mod.require_localhost("http://user@127.0.0.1:8000/v1")


def test_main_refuses_url_with_credentials(mod, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    with pytest.raises(SystemExit):
        mod.main(
            [
                "--tools",
                str(tools_path),
                "--test",
                str(test_path),
                "--url",
                "http://user:example-secret@127.0.0.1:8000/v1",
                "--model",
                "fake-model",
                "--out",
                str(tmp_path / "results.md"),
            ]
        )


def test_main_never_writes_endpoint_url_into_results(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"
    url = _base_url(fake_server)
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--url",
            url,
            "--model",
            "fake-model",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    assert url not in text
    assert mod.LOCAL_ENDPOINT_LABEL in text


def test_main_strips_url_value_from_recorded_command_line(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"
    url = _base_url(fake_server)
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--url",
            url,
            "--model",
            "fake-model",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    command_line = next(line for line in text.splitlines() if line.startswith("- command:"))
    assert url not in command_line
    assert mod.LOCAL_ENDPOINT_LABEL in command_line


def test_redact_command_line_replaces_url_value(mod):
    argv = ["--url", "http://user:secret@127.0.0.1:8000/v1", "--model", "m"]
    rendered = mod.redact_command_line(argv)
    assert "secret" not in rendered
    assert mod.LOCAL_ENDPOINT_LABEL in rendered
    assert "--model m" in rendered


def test_redact_command_line_handles_equals_form(mod):
    argv = ["--url=http://user:secret@127.0.0.1:8000/v1", "--model", "m"]
    rendered = mod.redact_command_line(argv)
    assert "secret" not in rendered
    assert mod.LOCAL_ENDPOINT_LABEL in rendered


def test_main_passes_results_through_redact(mod, fake_server, tmp_path, monkeypatch):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"

    calls: list[bytes] = []
    real_redact = mod.redact

    def spy(data: bytes) -> bytes:
        calls.append(data)
        return real_redact(data)

    monkeypatch.setattr(mod, "redact", spy)
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--url",
            _base_url(fake_server),
            "--model",
            "fake-model",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    assert calls, "redact() must be called on the recorded results text"


# ---------------------------------------------------------------------------
# finding 2: --launch serves the model through nvsh's own Tier 2 launcher
# ---------------------------------------------------------------------------


class _FakeLaunchRuntime:
    def __init__(self, base_url: str, fail: str = "") -> None:
        self.base_url = base_url
        self.fail = fail
        self.ensured = 0
        self.stopped = 0

    def ensure(self) -> str:
        self.ensured += 1
        if self.fail:
            raise RuntimeUnavailableForTest(self.fail)
        return self.base_url

    def stop(self) -> None:
        self.stopped += 1

    def status(self) -> str:
        return "fake tier 2 runtime"


class _LaunchHarness:
    """Fake docker + fake config + fake runtime builder for --launch."""

    def __init__(self, mod, *, base_url: str, running_container: str | None = None, fail: str = ""):
        self.mod = mod
        self.base_url = base_url
        self.running_container = running_container
        self.fail = fail
        self.docker_calls: list[list[str]] = []
        self.build_calls: list[tuple[dict, object]] = []
        self.runtime: _FakeLaunchRuntime | None = None
        self.lfm_config = {"engine": "vllm", "mode": "managed", "tool_call_parser": "lfm2"}

        def run_docker(argv, timeout):
            self.docker_calls.append(argv)
            if argv[0] == "docker" and argv[1] == "ps" and "--filter" in argv:
                names = f"{self.running_container}\n" if self.running_container else ""
                return (0, names)
            return (0, "")

        def load_config(path):
            from types import SimpleNamespace

            return SimpleNamespace(tiers={"memory_floor_mb": 1024, "lfm": dict(self.lfm_config)})

        def build_runtime(lfm_settings, platform, *, floor_check):
            self.build_calls.append((dict(lfm_settings), platform))
            self.runtime = _FakeLaunchRuntime(self.base_url, fail=self.fail)
            return self.runtime

        self.seams = mod.LaunchSeams(
            run_docker=run_docker,
            detect_platform=lambda: object(),
            load_config=load_config,
            build_runtime=build_runtime,
            uid=lambda: 1234,
        )


def test_launch_guard_refuses_when_container_already_running(mod, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    harness = _LaunchHarness(
        mod, base_url="http://127.0.0.1:9/v1", running_container="nvsh-tier2-1234"
    )
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--launch",
            "--out",
            str(tmp_path / "results.md"),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 2
    assert harness.runtime is None
    assert not (tmp_path / "results.md").exists()


def test_launch_serves_and_measures_then_stops_runtime(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"
    harness = _LaunchHarness(mod, base_url=_base_url(fake_server))
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--launch",
            "--out",
            str(out_path),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 0
    assert harness.runtime is not None
    assert harness.runtime.ensured == 1
    assert harness.runtime.stopped == 1
    # the model override reached [tiers.lfm] settings used to build the runtime
    assert harness.build_calls[0][0]["model"] == "fake-model"
    text = out_path.read_text(encoding="utf-8")
    assert _base_url(fake_server) not in text
    assert mod.LOCAL_ENDPOINT_LABEL in text


def test_launch_stops_runtime_even_when_measurement_fails(mod, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    # base_url with no server listening -> the issue-46 preflight refuses first
    harness = _LaunchHarness(mod, base_url="http://127.0.0.1:1/v1")
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--launch",
            "--out",
            str(tmp_path / "results.md"),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 2
    assert harness.runtime is not None
    assert harness.runtime.stopped == 1
    assert not (tmp_path / "results.md").exists()


def test_launch_reports_runtime_unavailable_without_measuring(mod, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    harness = _LaunchHarness(mod, base_url="http://127.0.0.1:9/v1", fail="no gpu")
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--launch",
            "--out",
            str(tmp_path / "results.md"),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 2
    assert harness.runtime is not None
    assert harness.runtime.stopped == 1
    assert not (tmp_path / "results.md").exists()


def test_launch_ignores_url_localhost_check_but_still_uses_local_runtime(
    mod, fake_server, tmp_path
):
    """Without --launch, --url is validated; with --launch, only the runtime's
    own base URL matters (still required to be local)."""
    tools_path, test_path = _write_inputs(tmp_path)
    harness = _LaunchHarness(mod, base_url=_base_url(fake_server))
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--launch",
            "--url",
            "http://example.com:8000",  # would be refused outside --launch
            "--out",
            str(tmp_path / "results.md"),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 0


# ---------------------------------------------------------------------------
# finding 3: --model-revision is verified against the --launch cache
# ---------------------------------------------------------------------------


def _hf_cache(tmp_path: Path, model: str, commit: str, *, hub: bool = True) -> Path:
    """A host HF cache whose ``refs/main`` resolves *model* to *commit*."""
    cache = tmp_path / "hf-cache"
    base = cache / "hub" if hub else cache
    ref = base / f"models--{model.replace('/', '--')}" / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(commit, encoding="utf-8")
    return cache


def test_launch_revision_mismatch_refuses_before_starting_runtime(mod, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    cache = _hf_cache(tmp_path, "fake-model", "cachedrev")
    harness = _LaunchHarness(mod, base_url="http://127.0.0.1:9/v1")
    harness.lfm_config["hf_cache_dir"] = str(cache)
    out_path = tmp_path / "results.md"
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--model-revision",
            "wantedrev",
            "--launch",
            "--out",
            str(out_path),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 2
    assert harness.runtime is None
    assert not out_path.exists()


def test_launch_revision_missing_from_cache_refuses(mod, tmp_path, capsys):
    tools_path, test_path = _write_inputs(tmp_path)
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    harness = _LaunchHarness(mod, base_url="http://127.0.0.1:9/v1")
    harness.lfm_config["hf_cache_dir"] = str(cache)
    out_path = tmp_path / "results.md"
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--model-revision",
            "wantedrev",
            "--launch",
            "--out",
            str(out_path),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "refs/main" in err
    assert harness.runtime is None
    assert not out_path.exists()


def test_launch_revision_verified_is_recorded(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    cache = _hf_cache(tmp_path, "fake-model", "wantedrev")
    harness = _LaunchHarness(mod, base_url=_base_url(fake_server))
    harness.lfm_config["hf_cache_dir"] = str(cache)
    out_path = tmp_path / "results.md"
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--model-revision",
            "wantedrev",
            "--launch",
            "--out",
            str(out_path),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "`wantedrev` (revision verified from the cache)" in text


def test_launch_revision_unverified_for_attached_endpoint(mod, fake_server, tmp_path):
    """A cache mismatch never blocks an attach-mode run: nothing on this host
    is downloaded, so there is nothing to check the revision against."""
    tools_path, test_path = _write_inputs(tmp_path)
    harness = _LaunchHarness(mod, base_url=_base_url(fake_server))
    harness.lfm_config["mode"] = "attach"
    out_path = tmp_path / "results.md"
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--model-revision",
            "wantedrev",
            "--launch",
            "--out",
            str(out_path),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "`wantedrev` (operator-supplied, not verified: attached endpoint)" in text


def test_launch_without_model_revision_skips_verification(mod, fake_server, tmp_path):
    """No --model-revision means nothing to verify (and no cache is read)."""
    tools_path, test_path = _write_inputs(tmp_path)
    harness = _LaunchHarness(mod, base_url=_base_url(fake_server))
    # no hf_cache_dir configured at all; would blow up if verification ran
    out_path = tmp_path / "results.md"
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--model",
            "fake-model",
            "--launch",
            "--out",
            str(out_path),
        ],
        launch_seams=harness.seams,
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "model revision" not in text


def test_the_command_line_writes_the_home_directory_as_home(monkeypatch, tmp_path) -> None:
    module = _module()
    monkeypatch.setattr(module.Path, "home", classmethod(lambda cls: tmp_path))
    line = module.redact_command_line(["--tools", f"{tmp_path}/skills/tools.json"])
    assert line == "measure_skills.py --tools $HOME/skills/tools.json"


# ---------------------------------------------------------------------------
# issue 46: thinking off on request, non-empty think blocks counted
# ---------------------------------------------------------------------------


def _skills_argv(tools_path, test_path, server, out_path, *extra):
    return [
        "--tools",
        str(tools_path),
        "--test",
        str(test_path),
        "--url",
        _base_url(server),
        "--model",
        "fake-model",
        "--out",
        str(out_path),
        *extra,
    ]


def test_enable_thinking_false_is_sent_as_chat_template_kwargs(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"
    argv = _skills_argv(tools_path, test_path, fake_server, out_path, "--enable-thinking", "false")
    assert mod.main(argv) == 0
    assert _FakeHandler.requests
    for request in _FakeHandler.requests:
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
    text = out_path.read_text(encoding="utf-8")
    assert "chat_template_kwargs enable_thinking=false" in text
    assert "| Non-empty think blocks (must be 0) | 0 |" in text


def test_thinking_is_not_sent_unless_configured(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    assert mod.main(_skills_argv(tools_path, test_path, fake_server, tmp_path / "r.md")) == 0
    for request in _FakeHandler.requests:
        assert "chat_template_kwargs" not in request


def test_nonempty_think_blocks_are_counted_and_reported(mod, fake_server, tmp_path, monkeypatch):
    tools_path, test_path = _write_inputs(tmp_path)
    monkeypatch.setattr(
        _FakeHandler, "contents", ["<think>let me see</think>", "<think>\n\n</think>", "", ""]
    )
    out_path = tmp_path / "results.md"
    code = mod.main(_skills_argv(tools_path, test_path, fake_server, out_path))
    assert code == 0
    assert "| Non-empty think blocks (must be 0) | 1 |" in out_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"content": "<think>reasoning</think>answer"}, True),
        ({"content": "<think>\n\n</think>\n\nanswer"}, False),
        ({"content": "reasoning left open</think>answer"}, True),
        ({"content": "</think>answer"}, False),
        ({"content": "no tags at all"}, False),
        ({"content": None, "reasoning_content": "thought"}, True),
        ({"content": "", "reasoning": "  "}, False),
        ("not a message", False),
    ],
)
def test_nonempty_think(mod, message, expected):
    assert mod.nonempty_think(message) is expected


# ---------------------------------------------------------------------------
# Issue 46 finding: a crashed/unreachable endpoint must not produce a
# plausible-looking results page -- preflight before the first eval, and
# refuse the results page when any eval is a call error.
# ---------------------------------------------------------------------------


def test_preflight_refuses_the_wrong_model_name(mod, fake_server):
    with pytest.raises(RuntimeError, match="other-model"):
        mod.preflight_models(_base_url(fake_server), "other-model")


def test_preflight_refuses_a_refused_connection(mod):
    with pytest.raises(RuntimeError):
        mod.preflight_models("http://127.0.0.1:1", "fake-model")


def test_preflight_accepts_a_server_serving_the_model(mod, fake_server):
    mod.preflight_models(_base_url(fake_server), "fake-model")  # does not raise


def test_main_refuses_a_wrong_model_name_before_any_eval(mod, fake_server, tmp_path):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--url",
            _base_url(fake_server),
            "--model",
            "wrong-model",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 2
    assert not out_path.exists()
    assert _FakeHandler.calls == 0  # refused before the first eval


def test_a_call_error_fails_the_run_and_writes_no_results_page(
    mod, fake_server, tmp_path, monkeypatch
):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"
    monkeypatch.setattr(
        _FakeHandler, "fail_indices", {0}
    )  # the first eval's call drops the connection
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--url",
            _base_url(fake_server),
            "--model",
            "fake-model",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 2
    assert not out_path.exists()


def test_allow_tier_errors_writes_the_page_with_the_count(mod, fake_server, tmp_path, monkeypatch):
    tools_path, test_path = _write_inputs(tmp_path)
    out_path = tmp_path / "results.md"
    monkeypatch.setattr(_FakeHandler, "fail_indices", {0})
    exit_code = mod.main(
        [
            "--tools",
            str(tools_path),
            "--test",
            str(test_path),
            "--url",
            _base_url(fake_server),
            "--model",
            "fake-model",
            "--allow-tier-errors",
            "1",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "1 call-error eval(s) permitted" in text
    assert "--allow-tier-errors 1" in text
