"""Augmentation pipeline: scripts/lfm-finetune/augment.py (part of #39).

Drives the real pipeline (generator -> corrector -> two reviewers) against
fake OpenAI-compatible endpoints served in-process on 127.0.0.1 -- no
network, no real model calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/augment.py"


def _module():
    spec = importlib.util.spec_from_file_location("lfm_augment", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses' field-type resolution looks the module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aug = _module()

Responder = Callable[[dict[str, Any], str | None], str]


class _FakeHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.requests.append(body)  # type: ignore[attr-defined]
        model = body.get("model")
        responder = self.server.responders.get(model)  # type: ignore[attr-defined]
        if responder is None:
            self.send_response(404)
            self.end_headers()
            return
        result = responder(body, self.headers.get("Authorization"))
        if isinstance(result, tuple):
            # (status, extra_headers) -- simulates a transient/non-transient
            # HTTP error, optionally carrying a Retry-After header.
            status, extra_headers = result
            self.send_response(status)
            for header_name, header_value in (extra_headers or {}).items():
                self.send_header(header_name, header_value)
            self.end_headers()
            return
        content = result
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # silence test noise
        pass


class _FakeHandlerWithReasoning(_FakeHandler):
    """Like _FakeHandler, but every reply carries an empty ``content`` plus a
    non-empty ``reasoning_content`` -- proving the pipeline never reads the
    reasoning field as an answer."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.requests.append(body)  # type: ignore[attr-defined]
        payload = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "I am thinking very hard about this...",
                        }
                    }
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_server(request):
    responders: dict[str, Responder] = {}
    handler_cls = getattr(request, "param", _FakeHandler)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    server.responders = responders  # type: ignore[attr-defined]
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    yield server, url
    server.shutdown()
    thread.join(timeout=5)


def _set_roles(monkeypatch, url: str, models: dict[str, str], **extra_env: str) -> None:
    for role, model in models.items():
        monkeypatch.setenv(f"NVSH_AUG_{role}_URL", url)
        monkeypatch.setenv(f"NVSH_AUG_{role}_MODEL", model)
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)


DEFAULT_MODELS = {
    "GENERATOR": "gen-model",
    "CORRECTOR": "cor-model",
    "REVIEWER_A": "rev-a-model",
    "REVIEWER_B": "rev-b-model",
}


def _always(text: str) -> Responder:
    return lambda body, auth: text


def _flaky(
    fail_count: int,
    status: int = 503,
    headers: dict[str, str] | None = None,
    success: str = "ok",
) -> Responder:
    """A responder that fails *fail_count* times with an HTTP *status*
    (optionally carrying *headers*, e.g. Retry-After) then returns *success*
    forever after."""
    calls = {"n": 0}

    def _responder(body: dict[str, Any], auth: str | None):
        calls["n"] += 1
        if calls["n"] <= fail_count:
            return (status, headers)
        return success

    _responder.calls = calls  # type: ignore[attr-defined]
    return _responder


def _split_seed_file(
    tmp_path: Path, name: str = "val.json", header: str = "fixture", **entry_overrides: Any
) -> Path:
    entry = {
        "id": "dev-e01~x",
        "kind": "explicit",
        "text": "How hot is this machine?",
        "expect": {"operation": "thermal_stats", "args": {}},
        "source_id": "dev-e01",
    }
    entry.update(entry_overrides)
    path = tmp_path / name
    path.write_text(json.dumps({"header": header, "entries": [entry]}), encoding="utf-8")
    return path


def _tools_seed_file(tmp_path: Path, name: str = "tools.json") -> Path:
    records = [
        {
            "skill": "jetson-diagnostic",
            "repo": "device",
            "tool": {
                "type": "function",
                "function": {
                    "name": "jetson_diagnostic",
                    "description": "Diagnose common Jetson boot and driver failures.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        }
    ]
    path = tmp_path / name
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# side / answer / source inheritance
# ---------------------------------------------------------------------------


def test_accepted_variation_inherits_side_answer_and_source(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    responders = {
        "gen-model": _always("How warm is the box right now?"),
        "cor-model": _always("How warm is the box right now?"),
        "rev-a-model": _always("yes, same request"),
        "rev-b-model": _always("yes: matches"),
    }
    _server.responders.update(responders)

    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=accepted,
        rejected_out=rejected,
        per_source=1,
    )

    assert counts.as_dict() == {
        "generated": 1,
        "corrected": 1,
        "accepted": 1,
        "rejected_by_a": 0,
        "rejected_by_b": 0,
        "errors": 0,
        "retries": 0,
    }
    lines = accepted.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["id"] == "dev-e01~v1"
    assert record["source_id"] == "dev-e01"
    assert record["side"] == "val"  # inferred from the val.json filename
    assert record["expect"] == {"operation": "thermal_stats", "args": {}}
    assert record["text"] == "How warm is the box right now?"
    # bug 4: the record keeps the source entry's own corpus "kind"
    # (explicit/failure), never a "split"/"skill" seed-format label, and
    # records the seed format separately.
    assert record["kind"] == "explicit"
    assert record["seed_format"] == "split"
    assert not rejected.exists()


def test_side_can_be_given_explicitly(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path, name="side_unknown.json")
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
        side="train",
    )
    record = json.loads((tmp_path / "accepted.jsonl").read_text().splitlines()[0])
    assert record["side"] == "train"


def test_side_required_when_not_inferrable(tmp_path):
    seed_file = _split_seed_file(tmp_path, name="side_unknown.json")
    with pytest.raises(aug.ConfigError):
        aug.load_seeds(seed_file)


# ---------------------------------------------------------------------------
# NVIDIA evals refused as seeds
# ---------------------------------------------------------------------------


def test_refuses_eval_file_by_field_names(tmp_path):
    eval_path = tmp_path / "test.jsonl"
    record = {
        "id": "bsp-1",
        "repo": "bsp",
        "skill": "some-skill",
        "text": "do the thing",
        "expected_skill": "some-skill",
        "ground_truth": "some-skill",
        "names_skill": False,
    }
    eval_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(aug.SeedRefused):
        aug.load_seeds(eval_path)


def test_refuses_held_out_file_by_name_regardless_of_content(tmp_path):
    held_out = tmp_path / "held-out.json"
    held_out.write_text(json.dumps({"header": "h", "entries": []}), encoding="utf-8")
    with pytest.raises(aug.SeedRefused):
        aug.load_seeds(held_out)


# ---------------------------------------------------------------------------
# both-reviewers rule + rejection log contents
# ---------------------------------------------------------------------------


def test_rejected_when_either_reviewer_says_no(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("rephrased request"),
            "cor-model": _always("rephrased request, cleaned up"),
            "rev-a-model": _always("yes, matches"),
            "rev-b-model": _always("no, this now asks for something else"),
        }
    )
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=accepted,
        rejected_out=rejected,
        per_source=1,
    )

    assert not accepted.exists()
    assert counts.accepted == 0
    assert counts.rejected_by_b == 1
    assert counts.rejected_by_a == 0

    record = json.loads(rejected.read_text(encoding="utf-8").splitlines()[0])
    assert record["source_id"] == "dev-e01"
    assert record["models"] == {
        "GENERATOR": "gen-model",
        "CORRECTOR": "cor-model",
        "REVIEWER_A": "rev-a-model",
        "REVIEWER_B": "rev-b-model",
    }
    assert record["verdicts"]["reviewer_a"]["accept"] is True
    assert record["verdicts"]["reviewer_b"]["accept"] is False
    assert "something else" in record["verdicts"]["reviewer_b"]["reason"]


def test_empty_reviewer_reply_is_an_error_retried_on_resume(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("rephrased request"),
            "cor-model": _always("rephrased request, cleaned up"),
            "rev-a-model": _always(""),  # e.g. a reasoning model with too small a budget
            "rev-b-model": _always("yes"),
        }
    )
    accepted, rejected = tmp_path / "accepted.jsonl", tmp_path / "rejected.jsonl"
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=accepted,
        rejected_out=rejected,
        per_source=1,
    )
    # Not a judgement: nothing is written, so the next resume tries it again.
    assert counts.errors == 1 and counts.rejected_by_a == 0
    assert not rejected.exists() or rejected.read_text(encoding="utf-8") == ""
    assert not accepted.exists() or accepted.read_text(encoding="utf-8") == ""


def test_reasoning_field_is_never_read_as_the_answer(tmp_path, monkeypatch, request):
    # A server that always returns empty content + a reasoning_content trace.
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandlerWithReasoning)
    server.responders = {}
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request.addfinalizer(lambda: (server.shutdown(), thread.join(timeout=5)))
    url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"

    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    seed_file = _split_seed_file(tmp_path)
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
    )
    # The generator's reply is empty (reasoning is never the answer) -> error,
    # not a fabricated variation built from the reasoning trace.
    assert counts.generated == 0
    assert counts.errors == 1
    assert not (tmp_path / "accepted.jsonl").exists()
    assert not (tmp_path / "rejected.jsonl").exists()


# ---------------------------------------------------------------------------
# env-var errors
# ---------------------------------------------------------------------------


def test_missing_url_raises_named_config_error(monkeypatch):
    monkeypatch.delenv("NVSH_AUG_GENERATOR_URL", raising=False)
    with pytest.raises(aug.ConfigError, match="NVSH_AUG_GENERATOR_URL"):
        aug.load_role_config("GENERATOR", env={})


def test_missing_model_raises_named_config_error():
    with pytest.raises(aug.ConfigError, match="NVSH_AUG_GENERATOR_MODEL"):
        aug.load_role_config("GENERATOR", env={"NVSH_AUG_GENERATOR_URL": "http://127.0.0.1:1/x"})


def test_missing_key_env_target_raises_named_config_error():
    env = {
        "NVSH_AUG_GENERATOR_URL": "http://127.0.0.1:1/x",
        "NVSH_AUG_GENERATOR_MODEL": "m",
        "NVSH_AUG_GENERATOR_KEY_ENV": "SOME_MISSING_KEY_VAR",
    }
    with pytest.raises(aug.ConfigError, match="SOME_MISSING_KEY_VAR"):
        aug.load_role_config("GENERATOR", env=env)


def test_key_env_resolves_bearer_header(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seen_auth: dict[str, str | None] = {}

    def _capture(body, auth):
        seen_auth["value"] = auth
        return "yes"

    _server.responders["rev-a-model"] = _capture
    env = {
        "NVSH_AUG_REVIEWER_A_URL": url,
        "NVSH_AUG_REVIEWER_A_MODEL": "rev-a-model",
        "NVSH_AUG_REVIEWER_A_KEY_ENV": "MY_KEY_VAR",
        "MY_KEY_VAR": "sekrit-test-value",
    }
    role = aug.load_role_config("REVIEWER_A", env=env)
    aug.default_caller(role, "sys", "user")
    assert seen_auth["value"] == "Bearer sekrit-test-value"


# ---------------------------------------------------------------------------
# tools.json seed mode
# ---------------------------------------------------------------------------


def test_tools_json_seed_mode(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _tools_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    seen_prompts: dict[str, str] = {}

    def _generator(body, auth):
        seen_prompts["generator_user"] = body["messages"][-1]["content"]
        return "My board won't finish booting, what's wrong?"

    _server.responders.update(
        {
            "gen-model": _generator,
            "cor-model": _always("My board won't finish booting, what's wrong?"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
    )
    assert counts.accepted == 1
    record = json.loads((tmp_path / "accepted.jsonl").read_text().splitlines()[0])
    assert record["source_id"] == "jetson-diagnostic"
    assert record["seed_format"] == "skills"
    # A skill seed is not a corpus entry: it must not carry a corpus "kind".
    assert "kind" not in record
    assert record["expect"] == {"skill": "jetson-diagnostic"}
    assert record["id"] == "jetson-diagnostic~v1"
    assert "Diagnose common Jetson boot" in seen_prompts["generator_user"]


# ---------------------------------------------------------------------------
# resumable runs
# ---------------------------------------------------------------------------


def test_resume_skips_ids_already_present(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    calls = {"n": 0}

    def _counting(body, auth):
        calls["n"] += 1
        return "should not be called"

    _server.responders.update(
        {
            "gen-model": _counting,
            "cor-model": _counting,
            "rev-a-model": _counting,
            "rev-b-model": _counting,
        }
    )
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text(
        json.dumps({"id": "dev-e01~v1", "source_id": "dev-e01", "text": "already accepted"}) + "\n",
        encoding="utf-8",
    )
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=accepted,
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
    )
    assert calls["n"] == 0
    assert counts.as_dict()["generated"] == 0
    assert accepted.read_text(encoding="utf-8").count("\n") == 1


# ---------------------------------------------------------------------------
# --limit
# ---------------------------------------------------------------------------


def test_limit_caps_new_variations(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=5,
        limit=2,
    )
    assert counts.accepted == 2


# ---------------------------------------------------------------------------
# h30: read-only/escalate answers get the extra "could this be a change" guard
# ---------------------------------------------------------------------------


def test_change_check_added_for_read_only_operation(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)  # thermal_stats, read_only=True
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    seen = {}

    def _reviewer(body, auth):
        seen["system"] = body["messages"][0]["content"]
        return "yes"

    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _reviewer,
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
    )
    assert "carried out by one of the changes listed" in seen["system"]


def test_no_change_check_for_mutating_operation(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(
        tmp_path, expect={"operation": "power_set", "args": {"mode": "balanced"}}
    )
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    seen = {}

    def _reviewer(body, auth):
        seen["system"] = body["messages"][0]["content"]
        return "yes"

    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _reviewer,
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
    )
    assert "carried out by one of the changes listed" not in seen["system"]


def test_change_check_added_for_escalate(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path, expect={"escalate": True})
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    seen = {}

    def _reviewer(body, auth):
        seen["system"] = body["messages"][0]["content"]
        return "yes"

    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _reviewer,
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
    )
    assert "carried out by one of the changes listed" in seen["system"]


# ---------------------------------------------------------------------------
# max_tokens / disable_thinking passthrough
# ---------------------------------------------------------------------------


def test_max_tokens_default_and_override(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seen = {}

    def _generator(body, auth):
        seen["max_tokens"] = body["max_tokens"]
        seen["chat_template_kwargs"] = body.get("chat_template_kwargs")
        return "rephrased"

    _server.responders.update(
        {
            "gen-model": _generator,
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    _set_roles(
        monkeypatch,
        url,
        DEFAULT_MODELS,
        NVSH_AUG_GENERATOR_MAX_TOKENS="2048",
        NVSH_AUG_GENERATOR_DISABLE_THINKING="true",
    )
    seed_file = _split_seed_file(tmp_path)
    roles = aug.load_all_roles()
    aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
    )
    assert seen["max_tokens"] == 2048
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert roles["CORRECTOR"].max_tokens == aug.DEFAULT_MAX_TOKENS


# ---------------------------------------------------------------------------
# no non-localhost URL, no key, in this committed file (scan-secrets' own job,
# but a quick sanity check here keeps a regression from silently reappearing)
# ---------------------------------------------------------------------------


def test_source_has_no_non_localhost_url_literal():
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "https://" not in text
    for line in text.splitlines():
        if "http://" in line:
            assert "127.0.0.1" in line or "localhost" in line


# ---------------------------------------------------------------------------
# bug 1: parse_verdict must only accept a clear, unhedged "yes"
# ---------------------------------------------------------------------------


def test_parse_verdict_rejects_yes_no_no():
    accepted, reason = aug.parse_verdict("yes/no: no")
    assert accepted is False
    assert reason == "yes/no: no"


def test_parse_verdict_rejects_yes_question_then_no():
    accepted, reason = aug.parse_verdict("yes? No, this changes the answer.")
    assert accepted is False
    assert reason == "yes? No, this changes the answer."


def test_parse_verdict_rejects_hedged_yes_but():
    text = "Yes, but it could also be read as a request to restart"
    accepted, reason = aug.parse_verdict(text)
    assert accepted is False
    assert reason == text


def test_parse_verdict_rejects_empty_reply():
    assert aug.parse_verdict("") == (False, "empty reply")


@pytest.mark.parametrize(
    "text",
    [
        "yes",
        "Yes.",
        "yes, matches exactly",
        "**Yes**, this matches",
        "yes: same request",
    ],
)
def test_parse_verdict_accepts_clear_yes(text):
    accepted, _reason = aug.parse_verdict(text)
    assert accepted is True


@pytest.mark.parametrize(
    "text",
    [
        "no",
        "no, this drifts",
        "maybe",
        "yesterday this would work",  # not the word "yes"
        "yes, however it could be read differently",
        "yes, although unclear",
        "yes, this is ambiguous",
        "yes, but only partially",
    ],
)
def test_parse_verdict_rejects_non_clean_yes(text):
    accepted, _reason = aug.parse_verdict(text)
    assert accepted is False


# ---------------------------------------------------------------------------
# bug 2: --side must not override a file's own inferable side
# ---------------------------------------------------------------------------


def test_side_flag_conflicting_with_filename_is_refused(tmp_path):
    seed_file = _split_seed_file(tmp_path, name="test.json")
    with pytest.raises(aug.ConfigError, match="test"):
        aug.load_seeds(seed_file, side="train")


def test_side_flag_agreeing_with_filename_is_allowed(tmp_path):
    seed_file = _split_seed_file(tmp_path, name="test.json")
    seeds = aug.load_seeds(seed_file, side="test")
    assert seeds[0].side == "test"


def test_side_inferred_from_split_py_header_when_filename_is_renamed(tmp_path):
    header = "Split 'train' of dev.json (seed=42)."
    seed_file = _split_seed_file(tmp_path, name="renamed_batch.json", header=header)
    seeds = aug.load_seeds(seed_file)
    assert seeds[0].side == "train"


def test_side_flag_conflicting_with_header_inferred_side_is_refused(tmp_path):
    header = "Split 'train' of dev.json (seed=42)."
    seed_file = _split_seed_file(tmp_path, name="renamed_batch.json", header=header)
    with pytest.raises(aug.ConfigError, match="train"):
        aug.load_seeds(seed_file, side="val")


# ---------------------------------------------------------------------------
# bug 3: one source_id must not carry two sides or two expect blocks, and an
# id can never be written twice in one run
# ---------------------------------------------------------------------------


def test_conflicting_side_for_shared_source_id_is_refused_before_any_model_call(
    tmp_path, monkeypatch, fake_server
):
    _server, url = fake_server
    train_file = _split_seed_file(tmp_path, name="train.json")
    test_file = _split_seed_file(tmp_path, name="test.json")  # same source_id, different side
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    calls = {"n": 0}

    def _counting(body, auth):
        calls["n"] += 1
        return "yes"

    _server.responders.update(
        {m: _counting for m in ("gen-model", "cor-model", "rev-a-model", "rev-b-model")}
    )
    roles = aug.load_all_roles()
    with pytest.raises(ValueError, match="dev-e01"):
        aug.run_pipeline(
            seed_files=[train_file, test_file],
            roles=roles,
            accepted_out=tmp_path / "accepted.jsonl",
            rejected_out=tmp_path / "rejected.jsonl",
            per_source=1,
        )
    assert calls["n"] == 0


def test_conflicting_expect_for_shared_source_id_is_refused(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    file_a = _split_seed_file(tmp_path, name="val.json")
    file_b = _split_seed_file(
        tmp_path,
        name="val2.jsonl",
    )
    # val2.jsonl: same source_id, different expect block, side inferred as "val"
    # via explicit --side since the filename can't be inferred.
    file_b.write_text(
        json.dumps(
            {
                "id": "dev-e01~y",
                "kind": "explicit",
                "text": "How hot is this machine?",
                "expect": {"operation": "power_stats", "args": {}},
                "source_id": "dev-e01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    roles = aug.load_all_roles()
    with pytest.raises(ValueError, match="dev-e01"):
        aug.run_pipeline(
            seed_files=[file_a, file_b],
            roles=roles,
            accepted_out=tmp_path / "accepted.jsonl",
            rejected_out=tmp_path / "rejected.jsonl",
            per_source=1,
            side="val",
        )


def test_duplicate_consistent_seed_across_files_never_writes_the_same_id_twice(
    tmp_path, monkeypatch, fake_server
):
    _server, url = fake_server
    file_a = _split_seed_file(tmp_path, name="val.json")
    file_b = _split_seed_file(tmp_path, name="val_copy.jsonl")
    # val_copy.jsonl: the identical source, same side and expect -- e.g. the
    # same seed accidentally included in two input files.
    file_b.write_text(
        json.dumps(
            {
                "id": "dev-e01~x",
                "kind": "explicit",
                "text": "How hot is this machine?",
                "expect": {"operation": "thermal_stats", "args": {}},
                "source_id": "dev-e01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[file_a, file_b],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
        side="val",
    )
    assert counts.accepted == 1
    lines = (tmp_path / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "dev-e01~v1"


# ---------------------------------------------------------------------------
# bug 4: an accepted split-seed record must load via nvsh.tiers.bench's
# load_corpus (it keeps the source entry's own corpus "kind"), while a skill
# seed record is clearly marked as not being a corpus entry at all
# ---------------------------------------------------------------------------


def test_accepted_split_record_loads_via_bench_load_corpus(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("How warm is the box right now?"),
            "cor-model": _always("How warm is the box right now?"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
    )
    record = json.loads((tmp_path / "accepted.jsonl").read_text().splitlines()[0])

    from nvsh.tiers.bench import load_corpus

    corpus_path = tmp_path / "wrapped_corpus.json"
    corpus_path.write_text(json.dumps({"entries": [record]}), encoding="utf-8")
    result = load_corpus(corpus_path)
    assert result.problems == ()
    assert len(result.entries) == 1
    assert result.entries[0].kind == "explicit"


def _many_entry_seed_file(tmp_path: Path, count: int, name: str = "val.json") -> Path:
    entries = [
        {
            "id": f"dev-e{i:02d}~x",
            "kind": "explicit",
            "text": f"How hot is machine {i}?",
            "expect": {"operation": "thermal_stats", "args": {}},
            "source_id": f"dev-e{i:02d}",
        }
        for i in range(count)
    ]
    path = tmp_path / name
    path.write_text(json.dumps({"header": "fixture", "entries": entries}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# --workers: bounded thread pool, deterministic ids, no double writes
# ---------------------------------------------------------------------------


def test_workers_process_variations_concurrently_and_write_whole_records(
    tmp_path, monkeypatch, fake_server
):
    _server, url = fake_server
    seed_file = _many_entry_seed_file(tmp_path, 6)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)

    def _slow_always(text: str) -> Responder:
        def _responder(body, auth):
            time.sleep(0.02)  # encourage overlap across worker threads
            return text

        return _responder

    _server.responders.update(
        {
            "gen-model": _slow_always("rephrased"),
            "cor-model": _slow_always("rephrased"),
            "rev-a-model": _slow_always("yes"),
            "rev-b-model": _slow_always("yes"),
        }
    )
    roles = aug.load_all_roles()
    accepted = tmp_path / "accepted.jsonl"
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=accepted,
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
        workers=4,
    )
    assert counts.accepted == 6
    assert counts.errors == 0
    lines = accepted.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    ids = set()
    for line in lines:
        record = json.loads(line)  # a corrupted/interleaved write would fail to parse
        ids.add(record["id"])
    assert len(ids) == 6  # every id is unique -- none written twice


def test_workers_default_is_two_and_backward_compatible(tmp_path, monkeypatch, fake_server):
    # No `workers=` passed at all -- existing callers must keep working.
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
    )
    assert counts.accepted == 1


# ---------------------------------------------------------------------------
# retry with backoff: transient failures, Retry-After, non-transient 4xx,
# and exhaustion counting as a (still-retryable-on-resume) error
# ---------------------------------------------------------------------------


def test_is_transient_status_codes():
    def _http_error(code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError("http://127.0.0.1/x", code, "msg", {}, None)

    for code in (429, 500, 502, 503, 504):
        transient, _retry_after = aug._is_transient(_http_error(code))
        assert transient is True, code
    for code in (400, 401, 403, 404):
        transient, _retry_after = aug._is_transient(_http_error(code))
        assert transient is False, code


def test_is_transient_parses_retry_after_header():
    exc = urllib.error.HTTPError("http://127.0.0.1/x", 503, "msg", {"Retry-After": "7"}, None)
    transient, retry_after = aug._is_transient(exc)
    assert transient is True
    assert retry_after == 7.0


def test_is_transient_connection_error_and_timeout():
    conn_exc = urllib.error.URLError(ConnectionRefusedError())
    transient, retry_after = aug._is_transient(conn_exc)
    assert transient is True
    assert retry_after is None

    timeout_transient, _timeout_retry_after = aug._is_transient(TimeoutError("timed out"))
    assert timeout_transient is True


def test_compute_backoff_honours_retry_after_and_caps_exponential_growth():
    assert aug._compute_backoff(1, 2.0, retry_after=10.0, rand_fn=lambda: 0.9) == 10.0
    # Large attempt count: raw would be huge, but the cap always wins.
    assert aug._compute_backoff(20, 2.0, retry_after=None, rand_fn=lambda: 1.0) == (
        aug.MAX_BACKOFF_WAIT
    )
    # Zero jitter draw -> zero wait, never negative or the full cap.
    assert aug._compute_backoff(1, 2.0, retry_after=None, rand_fn=lambda: 0.0) == 0.0


def test_retry_recovers_after_transient_failures_and_counts_retries(
    tmp_path, monkeypatch, fake_server
):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    flaky_generator = _flaky(2, status=503, success="rephrased")
    _server.responders.update(
        {
            "gen-model": flaky_generator,
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    sleeps: list[float] = []
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
        workers=1,
        sleep_fn=sleeps.append,
        rand_fn=lambda: 0.0,
    )
    assert counts.errors == 0
    assert counts.accepted == 1
    assert counts.retries == 2
    assert len(sleeps) == 2
    assert flaky_generator.calls["n"] == 3  # 2 failures + 1 success


def test_retry_exhaustion_counts_as_error_and_stays_retryable(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    always_503 = _flaky(10_000, status=503)  # never succeeds within this run
    _server.responders.update(
        {
            "gen-model": always_503,
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=accepted,
        rejected_out=rejected,
        per_source=1,
        workers=1,
        max_retries=2,
        sleep_fn=lambda _seconds: None,
        rand_fn=lambda: 0.0,
    )
    assert counts.errors == 1
    assert counts.retries == 2
    assert always_503.calls["n"] == 3  # initial attempt + 2 retries
    assert not accepted.exists()
    assert not rejected.exists()  # never written -- stays retryable on resume


def test_non_transient_4xx_is_not_retried(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    calls = {"n": 0}

    def _bad_request(body, auth):
        calls["n"] += 1
        return (400, None)

    _server.responders.update(
        {
            "gen-model": _bad_request,
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
        workers=1,
        sleep_fn=lambda _seconds: None,
    )
    assert counts.errors == 1
    assert counts.retries == 0
    assert calls["n"] == 1  # no retry attempted at all


def test_retry_after_header_is_honoured_over_computed_backoff(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    flaky_generator = _flaky(1, status=503, headers={"Retry-After": "5"}, success="rephrased")
    _server.responders.update(
        {
            "gen-model": flaky_generator,
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    sleeps: list[float] = []
    roles = aug.load_all_roles()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
        workers=1,
        sleep_fn=sleeps.append,
        rand_fn=lambda: 0.99,  # would blow up a naive backoff*rand computation
    )
    assert counts.accepted == 1
    assert sleeps == [5.0]  # Retry-After honoured exactly, not backoff*jitter


# ---------------------------------------------------------------------------
# per-role timeout (NVSH_AUG_<ROLE>_TIMEOUT, default 120s, replaces the old
# fixed 60s)
# ---------------------------------------------------------------------------


def test_role_config_timeout_default_and_override():
    role = aug.load_role_config(
        "GENERATOR",
        env={"NVSH_AUG_GENERATOR_URL": "http://127.0.0.1:1/x", "NVSH_AUG_GENERATOR_MODEL": "m"},
    )
    assert role.timeout == 120.0

    role_override = aug.load_role_config(
        "GENERATOR",
        env={
            "NVSH_AUG_GENERATOR_URL": "http://127.0.0.1:1/x",
            "NVSH_AUG_GENERATOR_MODEL": "m",
            "NVSH_AUG_GENERATOR_TIMEOUT": "45",
        },
    )
    assert role_override.timeout == 45.0


def test_role_config_timeout_bad_value_raises_named_config_error():
    with pytest.raises(aug.ConfigError, match="NVSH_AUG_GENERATOR_TIMEOUT"):
        aug.load_role_config(
            "GENERATOR",
            env={
                "NVSH_AUG_GENERATOR_URL": "http://127.0.0.1:1/x",
                "NVSH_AUG_GENERATOR_MODEL": "m",
                "NVSH_AUG_GENERATOR_TIMEOUT": "not-a-number",
            },
        )


def test_post_chat_completion_passes_role_timeout(monkeypatch):
    seen: dict[str, Any] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")

    def _fake_urlopen(request, timeout=None):
        seen["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(aug.urllib.request, "urlopen", _fake_urlopen)
    role = aug.RoleConfig(role="GENERATOR", url="http://127.0.0.1:1/x", model="m", timeout=45.0)
    result = aug.default_caller(role, "sys", "user")
    assert result == "ok"
    assert seen["timeout"] == 45.0


# ---------------------------------------------------------------------------
# progress reporting
# ---------------------------------------------------------------------------


def test_progress_prints_periodic_lines_with_expected_fields(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _many_entry_seed_file(tmp_path, 3)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    import io

    progress_out = io.StringIO()
    counts = aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
        workers=1,
        progress_every=1e-9,  # effectively force a line after every completed task
        progress_out=progress_out,
    )
    assert counts.accepted == 3
    lines = [line for line in progress_out.getvalue().splitlines() if line.startswith("progress:")]
    assert len(lines) >= 1
    line = lines[-1]
    assert "attempted" in line
    assert "accepted=" in line
    assert "rejected=" in line
    assert "errors=" in line
    assert "retries=" in line
    assert "rate=" in line
    assert "eta=" in line


def test_progress_every_zero_disabled_via_negative_is_never_forced(
    tmp_path, monkeypatch, fake_server
):
    # progress_every > the whole run's duration -> no progress line at all,
    # just the (tested elsewhere) final counts.
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    roles = aug.load_all_roles()
    import io

    progress_out = io.StringIO()
    aug.run_pipeline(
        seed_files=[seed_file],
        roles=roles,
        accepted_out=tmp_path / "accepted.jsonl",
        rejected_out=tmp_path / "rejected.jsonl",
        per_source=1,
        workers=1,
        progress_every=3600,
        progress_out=progress_out,
    )
    assert progress_out.getvalue() == ""


def test_final_counts_include_retries_field(tmp_path, monkeypatch, fake_server, capsys):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    _server.responders.update(
        {
            "gen-model": _always("rephrased"),
            "cor-model": _always("rephrased"),
            "rev-a-model": _always("yes"),
            "rev-b-model": _always("yes"),
        }
    )
    rc = aug.main(
        [
            str(seed_file),
            "--per-source",
            "1",
            "--accepted-out",
            str(tmp_path / "accepted.jsonl"),
            "--rejected-out",
            str(tmp_path / "rejected.jsonl"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "retries=0" in out.splitlines()


# ---------------------------------------------------------------------------
# --dry-run: reports the count, never calls an endpoint, needs no role config
# ---------------------------------------------------------------------------


def test_dry_run_reports_count_without_calling_endpoints(tmp_path, monkeypatch, fake_server):
    _server, url = fake_server
    seed_file = _split_seed_file(tmp_path)
    _set_roles(monkeypatch, url, DEFAULT_MODELS)
    calls = {"n": 0}

    def _counting(body, auth):
        calls["n"] += 1
        return "yes"

    _server.responders.update(
        {m: _counting for m in ("gen-model", "cor-model", "rev-a-model", "rev-b-model")}
    )
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = aug.main(
            [
                str(seed_file),
                "--per-source",
                "3",
                "--dry-run",
                "--accepted-out",
                str(tmp_path / "accepted.jsonl"),
                "--rejected-out",
                str(tmp_path / "rejected.jsonl"),
            ]
        )
    assert rc == 0
    assert calls["n"] == 0
    assert "3" in buf.getvalue()


def test_dry_run_does_not_require_role_config(tmp_path, monkeypatch):
    seed_file = _split_seed_file(tmp_path)
    for role in aug.ROLES:
        monkeypatch.delenv(f"NVSH_AUG_{role}_URL", raising=False)
        monkeypatch.delenv(f"NVSH_AUG_{role}_MODEL", raising=False)
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = aug.main(
            [
                str(seed_file),
                "--per-source",
                "2",
                "--dry-run",
                "--accepted-out",
                str(tmp_path / "accepted.jsonl"),
                "--rejected-out",
                str(tmp_path / "rejected.jsonl"),
            ]
        )
    assert rc == 0
    assert "2" in buf.getvalue()


def test_dry_run_subtracts_already_written_ids(tmp_path):
    seed_file = _split_seed_file(tmp_path)
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text(
        json.dumps({"id": "dev-e01~v1", "source_id": "dev-e01"}) + "\n", encoding="utf-8"
    )
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = aug.main(
            [
                str(seed_file),
                "--per-source",
                "3",
                "--dry-run",
                "--accepted-out",
                str(accepted),
                "--rejected-out",
                str(seed_file.parent / "rejected.jsonl"),
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    assert "dry-run: 2 " in out  # 3 requested minus the 1 already written


def test_a_skill_seed_reviewer_sees_the_capability_description() -> None:
    module = _module()
    seed = module.Seed(
        source_id="jetson-diagnostic",
        side="train",
        seed_text="Read-only Jetson health snapshot.",
        expect={"skill": "jetson-diagnostic"},
        seed_format="skills",
        needs_change_check=False,
    )
    system, user = module.reviewer_prompt(seed, "Give me a health check of this Jetson")
    assert system == module.REVIEWER_SYSTEM_SKILL
    assert "Read-only Jetson health snapshot." in user
    assert "Give me a health check of this Jetson" in user


def test_the_expected_answer_is_described_in_words_not_json() -> None:
    module = _module()
    words = module._answer_in_words({"operation": "power_set", "args": {"mode": "max_performance"}})
    assert "power mode" in words and "mode max performance" in words and "{" not in words
    assert "power_set" not in words
    assert "more capable assistant" in module._answer_in_words({"escalate": True})
    assert "It pins clocks." in module._answer_in_words(
        {"explain": True, "answer": "It pins clocks."}
    )


def test_each_variation_number_asks_for_a_different_phrasing_style() -> None:
    module = _module()
    seed = module.Seed(
        source_id="g1",
        seed_format="split",
        side="train",
        seed_text="Show GPU load",
        expect={"operation": "gpu_stats", "args": {}},
        needs_change_check=True,
    )
    styles = {module.generator_prompt(seed, n)[1] for n in range(len(module.PHRASING_STYLES))}
    assert len(styles) == len(module.PHRASING_STYLES)
    assert module._variation_number("g1~v12") == 12 and module._variation_number("g1") == 0


@pytest.mark.parametrize(
    "text,leak",
    [
        (
            "What is the operation 'memory_stats' (Show the current memory usage statistics.)",
            "memory_stats",
        ),
        ("run POWER_SET to max", "power_set"),
        ("How much memory is this Jetson using?", ""),
        ("show the memory stats please", ""),
    ],
)
def test_a_variation_that_names_an_internal_operation_is_caught(text, leak) -> None:
    assert _module().names_internal_operation(text) == leak


def test_tasks_are_planned_round_robin_across_seeds() -> None:
    module = _module()

    def seed(source_id: str):
        return module.Seed(
            source_id=source_id,
            seed_format="split",
            side="train",
            seed_text="x",
            expect={"escalate": True},
            needs_change_check=True,
        )

    tasks = module._plan_tasks([seed("a"), seed("b")], per_source=2, limit=3, done=set())
    assert [variation_id for _, variation_id in tasks] == ["a~v1", "b~v1", "a~v2"]


@pytest.mark.parametrize(
    "text,copied",
    [
        ("Propose this action: Restart a named container, with container = inference", True),
        ("Hand the request to the full agent: a linker issue", True),
        ("Restart the inference container", False),
        ("Why does the trainer keep dying?", False),
    ],
)
def test_a_variation_copying_the_answer_template_is_caught(text, copied) -> None:
    assert bool(_module().copies_answer_template(text)) is copied


def test_the_generator_never_sees_the_expected_answer() -> None:
    module = _module()
    seed = module.Seed(
        source_id="g1",
        seed_format="split",
        side="train",
        seed_text="Restart the trainer container",
        expect={"operation": "container_restart", "args": {"container": "trainer"}},
        needs_change_check=False,
    )
    _system, user = module.generator_prompt(seed, 1)
    assert "Restart the trainer container" in user
    assert "take this action" not in user and "container =" not in user


@pytest.mark.parametrize(
    "reply,accepted",
    [
        ("yes\nThe response runs a read-only check with no machine changes involved.", True),
        ("yes/no: no", False),
        ("yes? No, this changes the answer.", False),
        ("Yes. No change is needed.", False),
    ],
)
def test_only_a_verdict_like_no_rejects_a_yes(reply, accepted) -> None:
    assert _module().parse_verdict(reply)[0] is accepted
