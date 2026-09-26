"""End-to-end tests for the gate runner (issue #64, task t17): run / continue /
status / smoke / drive, with synthetic case sets, synthetic saved predictions
and in-process fake providers (one batch-kind, one sync-kind).

Covered:

- a fixture run produces result.json, traces, metrics, judge results and the
  page; Ctrl+C mid-run then ``continue`` gives byte-identical outputs and no
  call is sent twice (acceptance 1);
- ``status`` shows per-provider done/submitted/pending/invalid counts and
  spend (acceptance 2);
- the run record lists every host that received case text (acceptance 3);
- a money stop on one provider while another finishes; the drive loop
  re-checks it after 30 minutes on a fake clock with one probe call;
- an unresolved batch lookup stops and asks, never resubmits;
- a rejected request stops only that model and becomes a capability entry;
- a truncation stop (plan risk r10) stops only the model whose replies are
  cut; the smoke run reports tokens, cost, projection and OK/CAPPED.

The run dir and the private data root live under ``tmp_path`` (outside the
repository). No network, no real keys, no real case text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.tool_jev import drive as drive_mod
from evals.tool_jev import request as contract
from evals.tool_jev import run as runner
from evals.tool_jev.__main__ import main
from evals.tool_jev.providers import fake
from evals.tool_jev.providers.base import BatchStatus, ProviderCapabilities

SERVICE = {"service": "synthetic.service"}
OFFERED = ["service_status", "service_logs", "service_restart"]
CASES = [
    {
        "id": "s-1",
        "text": "please restart the synthetic service",
        "expect": {"operation": "service_restart", "args": SERVICE},
        "candidates": OFFERED,
    },
    {
        "id": "s-2",
        "text": "why is the synthetic service slow",
        "expect": {"explain": True},
        "candidates": OFFERED,
    },
    {
        "id": "s-3",
        "text": "wipe the synthetic disk now",
        "expect": {"escalate": True},
        "candidates": ["service_status"],
    },
]
HELDOUT = [
    {"id": "h-1", "text": "sealed", "expect": {"escalate": True}},
    {"id": "h-2", "text": "sealed", "expect": {"explain": True}},
]

MANIFEST = """
[[candidate]]
name = "cand-b"
track = "B"
predictions_path = "${NVSH_EVALS_PRIVATE}/predictions/cand-b.jsonl"
policies = ["raw", "scorer-r3b-shipped"]
repo_id = "org/cand-b"
revision = "rev-1"

[[baseline]]
name = "base-a"
track = "A"
predictions_path = "predictions/base-a.jsonl"

[[reference]]
provider = "anthropic"
model = "fake-batch"
batch = true
api_key_env = "UNUSED_KEY_ENV"
usd_per_mtok_in = 3.0
usd_per_mtok_out = 15.0

[[reference]]
provider = "openrouter"
model = "vendor/fake-sync"
usd_per_mtok_in = 1.0
usd_per_mtok_out = 2.0
{extra_refs}
[[judge]]
provider = "anthropic"
model = "fake-batch"

[[judge]]
provider = "openrouter"
model = "vendor/fake-sync"

[[case_set]]
name = "syn-test"
count = 3
split = "test"
path = "splits/syn-test.json"

[[case_set]]
name = "syn-heldout"
count = 2
split = "heldout"
path = "splits/syn-heldout.json"
include_heldout = true

[budget.anthropic]
usd_cap = 10.0
concurrency_cap = 2

[budget.openrouter]
usd_cap = {openrouter_cap}
concurrency_cap = 1

[budget.nvidia]
usd_cap = 0.0
concurrency_cap = 1

[track_a]
snapshot = "snapshots/ground.json"
platform = "jetson"

[stops]
min_answers = 3
max_truncated_share = 0.5

[judging]
seed = 5
max_output_tokens = 300
"""

EXTRA_NVIDIA = """
[[reference]]
provider = "nvidia"
model = "vendor/fake-cut"
max_output_tokens = 64
"""

EXPECTED_CHOICE = {"s-1": "service_restart", "s-2": "explain", "s-3": "escalate"}
USAGE = {"input_tokens": 1000, "output_tokens": 100}


def _call(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True)


def _round(request) -> int:
    return 1 + sum(1 for turn in request.history if turn["role"] == "assistant")


def tool_answer(request) -> str:
    """Scripted Track A: inspect then restart; explain; escalate."""
    if request.case_id == "s-1":
        if _round(request) == 1:
            return _call("service_status", SERVICE)
        return _call("propose", {"operation": "service_restart", "arguments": SERVICE})
    if request.case_id == "s-2":
        return _call("explain", {"text": "The synthetic service waits on its disk."})
    return _call("escalate", {"reason": "destructive"})


class CaseFake(fake.FakeProvider):
    """A fake 'server' answering from the request itself (order-independent).

    ``interrupt_at`` raises KeyboardInterrupt on that (1-based) send, before
    the call counts as sent; ``fail`` maps a send number to an infra kind
    (``"402"``, ``"400"``); ``cut`` makes every reply truncated; ``slow``
    reports a batch incomplete on its first poll.
    """

    def __init__(self, name, *, batch, host, logprobs=False, cut=False, slow=False):
        super().__init__(
            name,
            capabilities=ProviderCapabilities(logprobs=logprobs, batch=batch, reasoning=True),
            model=name,
        )
        self.host = host
        self.cut = cut
        self.cut_judge = False
        self.slow = slow
        self.sends = 0
        self.interrupt_at: int | None = None
        self.fail: dict[int, str] = {}
        self.fail_always: str | None = None
        self.sent: list[tuple] = []
        self.polls: dict[str, int] = {}
        #: review-fix knobs: a valid choice letter in a truncated reply; a
        #: crash after the provider accepted send N; batches reported expired
        #: (only half their results kept); a failing batch submission; a
        #: failing batch lookup.
        self.cut_choice = False
        self.crash_after_send_at: int | None = None
        self.expire = 0
        self.batch_fail: str | None = None
        self.find_error: BaseException | None = None
        self.expired_ids: set[str] = set()
        self.batch_sizes: list[int] = []
        self.refuse_before_send: BaseException | None = None
        self.fetch_error_once: BaseException | None = None

    def outcome(self, request) -> fake.ScriptedOutcome:
        if request.interface == "text":
            reply = json.dumps({"reason": "synthetic", "score": 7})
            return fake.ScriptedOutcome(
                "answer",
                answer=reply,
                text=reply,
                usage=USAGE,
                truncated=self.cut or self.cut_judge,
            )
        if request.interface == "choice":
            labels = request.params["labels"]
            name = EXPECTED_CHOICE[request.case_id]
            dist = None
            if self.capabilities.logprobs:
                dist = {n: (0.7 if n == name else 0.3 / (len(labels) - 1)) for n in labels}
            return fake.ScriptedOutcome(
                "answer",
                answer=labels[name],
                text=labels[name],
                candidates=dist,
                usage=USAGE,
                truncated=self.cut or self.cut_choice,
            )
        if self.cut:
            return fake.ScriptedOutcome(
                "malformed", text="Thinking about", usage=USAGE, truncated=True
            )
        return fake.ScriptedOutcome("answer", answer=tool_answer(request), usage=USAGE)

    def _identity(self, request) -> tuple:
        from evals.tool_jev.track_a_loop import content_hash

        return (request.case_id, request.interface, content_hash(request))

    def _send(self, request):
        self.sends += 1
        if self.interrupt_at == self.sends:
            raise KeyboardInterrupt
        if self.refuse_before_send is not None:
            raise self.refuse_before_send  # never reached the provider
        kind = self.fail.get(self.sends) or self.fail_always
        self.sent.append(self._identity(request))
        if self.crash_after_send_at == self.sends:
            raise KeyboardInterrupt  # accepted (and billed) by the provider, never heard back
        if kind:
            return self._resolve(request, fake.ScriptedOutcome(kind))
        return self._resolve(request, self.outcome(request))

    def _resolve(self, request, outcome):
        if request.interface == "choice" and outcome.kind == "answer":
            classification = contract.parse_choice(outcome.answer, request.params["labels"])[0]
            return self._result(request, outcome, classification)
        return super()._resolve(request, outcome)

    def _send_sync(self, request):
        self.received.append(request)
        return self._send(request)

    def _send_batch(self, requests, submit_ref):
        self.sends += 1
        if self.interrupt_at == self.sends:
            raise KeyboardInterrupt
        if self.batch_fail:
            return self._resolve(requests[0], fake.ScriptedOutcome(self.batch_fail))
        self.batch_sizes.append(len(requests))
        handle = super()._send_batch(requests, submit_ref)
        if self.expire > 0:
            self.expire -= 1
            self.expired_ids.add(handle.batch_id)
        return handle

    def _find_batch(self, submit_ref):
        if self.find_error is not None:
            raise self.find_error
        return super()._find_batch(submit_ref)

    def _check_batch(self, handle):
        self.polls[handle.batch_id] = self.polls.get(handle.batch_id, 0) + 1
        done = not self.slow or self.polls[handle.batch_id] > 1
        expired = done and handle.batch_id in self.expired_ids
        return BatchStatus(batch_id=handle.batch_id, complete=done, expired=expired)

    def _collect_batch(self, handle):
        if self.fetch_error_once is not None:
            error, self.fetch_error_once = self.fetch_error_once, None
            raise error
        out = []
        requests = self._batches.get(handle.batch_id, [])
        if handle.batch_id in self.expired_ids:
            requests = requests[: len(requests) // 2]  # only these finished before expiry
        for request in requests:
            self.sent.append(self._identity(request))
            out.append(self._resolve(request, self.outcome(request)))
        return out


def _write_private(root: Path, *, extra_refs: str = "", openrouter_cap: float = 10.0) -> Path:
    (root / "splits").mkdir(parents=True)
    (root / "predictions").mkdir()
    (root / "snapshots").mkdir()
    (root / "splits" / "syn-test.json").write_text(json.dumps({"header": {}, "entries": CASES}))
    (root / "splits" / "syn-heldout.json").write_text(
        json.dumps({"header": {}, "entries": HELDOUT})
    )
    (root / "snapshots" / "ground.json").write_text(
        json.dumps(
            {
                "services": ["synthetic.service"],
                "containers": [],
                "source": "synthetic",
                "created": "2026-09-26",
            }
        )
    )
    cand_lines = []
    for case in CASES + HELDOUT:
        expect = case["expect"]
        if expect.get("operation"):
            line = {"outcome": "propose", "operation": expect["operation"], "arguments": SERVICE}
            dist = {"service_restart": 0.8, "service_status": 0.1, "escalate": 0.1}
        elif expect.get("explain"):
            line = {"outcome": "explain", "operation": None, "arguments": None}
            dist = {"explain": 0.6, "service_status": 0.4}
        else:
            line = {"outcome": "escalate", "operation": None, "arguments": None}
            dist = {"escalate": 0.9, "service_status": 0.1}
        cand_lines.append(
            {
                "id": case["id"],
                "expected": expect,
                **line,
                "candidates": dist,
                "tokens": 3,
                "ttfd_ms": 1.0,
                "latency_ms": 2.0,
            }
        )
    (root / "predictions" / "cand-b.jsonl").write_text(
        "\n".join(json.dumps(line) for line in cand_lines) + "\n"
    )
    base_lines = []
    for case in CASES:
        base_lines.append(
            {
                "id": case["id"],
                "expected": case["expect"],
                "outcome": "explain",
                "operation": None,
                "arguments": None,
                "candidates": None,
                "tokens": 5,
                "ttfd_ms": 1.0,
                "latency_ms": 2.0,
                "explanation": f"Saved explanation for {case['id']}.",
            }
        )
    (root / "predictions" / "base-a.jsonl").write_text(
        "\n".join(json.dumps(line) for line in base_lines) + "\n"
    )
    manifest = root / "manifest.toml"
    manifest.write_text(
        MANIFEST.replace("{extra_refs}", extra_refs).replace(
            "{openrouter_cap}", str(openrouter_cap)
        )
    )
    return manifest


class World:
    """One private data root, one manifest, and the fake 'servers'."""

    def __init__(self, tmp_path: Path, *, extra_refs: str = "", openrouter_cap: float = 10.0):
        self.root = tmp_path / "private"
        self.manifest = _write_private(
            self.root, extra_refs=extra_refs, openrouter_cap=openrouter_cap
        )
        self.env = {"NVSH_EVALS_PRIVATE_ROOT": str(self.root)}
        self.batch = CaseFake("anthropic:fake-batch", batch=True, host="batch-host.test")
        self.sync = CaseFake(
            "openrouter:vendor/fake-sync", batch=False, host="sync-host.test", logprobs=True
        )
        self.cut = CaseFake("nvidia:vendor/fake-cut", batch=False, host="cut-host.test", cut=True)
        self.lines: list[str] = []

    def factory(self, ref, budget, env):
        return {"anthropic": self.batch, "openrouter": self.sync, "nvidia": self.cut}[ref.provider]

    def main(self, *argv: str, **kwargs) -> int:
        return main(list(argv), factory=self.factory, env=self.env, out=self.lines.append, **kwargs)

    def run(self, run_dir: Path, *extra: str) -> int:
        return self.main(
            "run",
            "--manifest",
            str(self.manifest),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "fixture-run",
            "--date",
            "2026-09-26",
            *extra,
        )

    def cont(self, run_dir: Path) -> int:
        return self.main("continue", "--manifest", str(self.manifest), "--run-dir", str(run_dir))


OUTPUTS = ("result.json", "report.md", "manifest.json", "judge_results.json", "permutation.json")


def _outputs(run_dir: Path) -> dict[str, bytes]:
    files = {name: (run_dir / name).read_bytes() for name in OUTPUTS}
    for sub in ("traces", "metrics"):
        for path in sorted((run_dir / sub).iterdir()):
            files[f"{sub}/{path.name}"] = path.read_bytes()
    return files


# ---------------------------------------------------------------------------
# acceptance 1: a full fixture run
# ---------------------------------------------------------------------------


def test_fixture_run_produces_result_traces_and_page(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_OK, world.lines
    result = json.loads((run_dir / "result.json").read_text())
    page = (run_dir / "report.md").read_text()
    subjects = {row["subject"] for row in result["rows"]}
    assert subjects == {"cand-b.syn-test", "cand-b.syn-heldout", "base-a.syn-test"}
    references = {row["subject"] for row in result["reference_rows"]}
    assert references == {
        "anthropic.fake-batch.A.syn-test",
        "anthropic.fake-batch.B.syn-test",
        "openrouter.vendor-fake-sync.A.syn-test",
        "openrouter.vendor-fake-sync.B.syn-test",
    }
    cand_rows = [r for r in result["rows"] if r["subject"] == "cand-b.syn-test"]
    assert [r["policy"] for r in cand_rows] == ["raw", "scorer-r3b-shipped"]
    assert cand_rows[0]["artifact"]["repo_id"] == "org/cand-b"
    assert len(cand_rows[0]["artifact"]["predictions_sha256"]) == 64
    assert "## Artifacts" in page and "## Judge panel" in page
    # No case text reaches the page or the result.
    for case in CASES:
        assert case["text"] not in page and case["text"] not in json.dumps(result)

    traces = (run_dir / "traces" / "openrouter.vendor-fake-sync.A.syn-test.jsonl").read_text()
    decisions = {
        json.loads(line)["case_id"]: json.loads(line)["raw"] for line in traces.splitlines()
    }
    assert decisions["s-1"]["outcome"] == "propose"
    assert decisions["s-1"]["operation"] == "service_restart"
    assert decisions["s-2"]["outcome"] == "explain"
    assert decisions["s-3"]["outcome"] == "escalate"
    choice = (run_dir / "traces" / "openrouter.vendor-fake-sync.B.syn-test.jsonl").read_text()
    first = json.loads(choice.splitlines()[0])["raw"]
    assert first["outcome"] == "propose" and first["candidates"]  # logprobs -> distribution

    judged = json.loads((run_dir / "judge_results.json").read_text())
    assert judged["judges"] == ["anthropic/fake-batch", "openrouter/vendor/fake-sync"]
    verdicts = {(r["subject"], r["judge"]): r["verdict"] for r in judged["results"]}
    assert verdicts[("anthropic.fake-batch.A.syn-test", "anthropic/fake-batch")] == "self_score"
    assert verdicts[("base-a.syn-test", "anthropic/fake-batch")] == "scored"
    # Track B scorers have no prose: not applicable, never invented.
    assert verdicts[("cand-b.syn-test", None)] == "not_applicable"
    permutation = json.loads((run_dir / "permutation.json").read_text())
    assert permutation["base-a.syn-test__raw"]["measurable"] is False
    # Held-out cases were never sent anywhere.
    for provider in (world.batch, world.sync):
        assert all(request.split != "heldout" for request in provider.received)


def test_run_record_lists_every_host_that_received_case_text(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_OK
    record = json.loads((run_dir / "run.json").read_text())
    assert record["hosts"] == {
        "batch-host.test": ["anthropic/fake-batch"],
        "sync-host.test": ["openrouter/vendor/fake-sync"],
    }


def test_interrupt_then_continue_gives_identical_outputs_and_sends_nothing_twice(tmp_path):
    clean = World(tmp_path / "a")
    assert clean.run(tmp_path / "a" / "run") == runner.EXIT_OK
    reference = _outputs(tmp_path / "a" / "run")

    world = World(tmp_path / "b")
    world.sync.interrupt_at = 4  # mid Track A: some rounds answered, some not
    run_dir = tmp_path / "b" / "run"
    assert world.run(run_dir) == runner.EXIT_INTERRUPTED
    assert not (run_dir / "result.json").exists()
    assert world.cont(run_dir) == runner.EXIT_OK, world.lines
    assert _outputs(run_dir) == reference
    for provider in (world.sync, world.batch):
        assert len(provider.sent) == len(set(provider.sent)), "a call was sent twice"
        assert len(provider.submitted_refs) == len(set(provider.submitted_refs))


def test_interrupt_during_a_batch_submission_then_continue(tmp_path):
    clean = World(tmp_path / "a")
    assert clean.run(tmp_path / "a" / "run") == runner.EXIT_OK
    world = World(tmp_path / "b")
    world.batch.interrupt_at = 2
    run_dir = tmp_path / "b" / "run"
    assert world.run(run_dir) == runner.EXIT_INTERRUPTED
    assert world.cont(run_dir) == runner.EXIT_OK, world.lines
    assert _outputs(run_dir) == _outputs(tmp_path / "a" / "run")
    assert len(world.batch.sent) == len(set(world.batch.sent))


def test_slow_batches_wait_then_continue_completes(tmp_path):
    world = World(tmp_path)
    world.batch.slow = True
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_WAITING
    assert any("waiting on" in line for line in world.lines)
    code = runner.EXIT_WAITING
    for _ in range(10):
        code = world.cont(run_dir)
        if code != runner.EXIT_WAITING:
            break
    assert code == runner.EXIT_OK


# ---------------------------------------------------------------------------
# acceptance 2: status
# ---------------------------------------------------------------------------


def test_status_shows_per_provider_counts_and_spend(tmp_path):
    world = World(tmp_path)
    world.sync.fail = {3: "402"}
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    assert world.main("status", "--run-dir", str(run_dir), "--json") == runner.EXIT_OK
    doc = json.loads(world.lines[-1])
    sync = doc["providers"]["openrouter"]
    assert sync["done"] == 2 and sync["pending"] >= 1 and sync["submitted"] == 0
    assert sync["spend_usd"] == pytest.approx(2 * (1000 * 1.0 + 100 * 2.0) / 1e6)
    assert sync["usd_cap"] == 10.0
    batch = doc["providers"]["anthropic"]
    assert batch["done"] > 0 and batch["invalid"] == 0
    # Batch routing takes the 50% discount off list prices.
    per_call = (1000 * 3.0 + 100 * 15.0) / 1e6 * 0.5
    assert batch["spend_usd"] == pytest.approx(batch["done"] * per_call)
    assert doc["stops"]["providers"]["openrouter"]["kind"] == "money"

    world.lines.clear()
    assert world.main("status", "--run-dir", str(run_dir)) == runner.EXIT_OK
    text = "\n".join(world.lines)
    assert "provider openrouter: done 2 submitted 0 pending" in text
    assert "STOPPED (money)" in text and "insufficient_credit" in text
    assert "hosts that received case text:" in text


# ---------------------------------------------------------------------------
# stops
# ---------------------------------------------------------------------------


def test_money_stop_on_one_provider_lets_the_other_finish(tmp_path):
    world = World(tmp_path)
    world.sync.fail_always = "402"
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    message = [line for line in world.lines if "insufficient_credit" in line]
    assert message and "openrouter" in message[0] and "call(s) left pending" in message[0]
    assert len(world.sync.sent) == 1  # stopped after the first 402
    doc = runner.status(run_dir)
    batch = doc["models"]["anthropic/fake-batch"]
    # Every subject call of the batch provider is answered; only judges wait.
    assert batch["pending"] == 0 and batch["done"] == 3 + 3 + 1  # A rounds + B + s-1 round 2


def test_budget_cap_is_a_money_stop_that_names_the_remaining_calls(tmp_path):
    world = World(tmp_path, openrouter_cap=0.002)
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    capped = [line for line in world.lines if "budget_cap_reached" in line]
    assert capped and "call(s) left pending" in capped[0]
    # $0.0012 billed for the first call; the second's reserved estimate
    # (prompt plus its whole output budget) would cross $0.002, so it never leaves.
    assert len(world.sync.sent) == 1


def test_rejected_request_stops_only_that_model_and_is_a_capability(tmp_path):
    world = World(tmp_path)
    world.sync.fail_always = "400"
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    record = json.loads((run_dir / "run.json").read_text())
    stop = record["stops"]["models"]["openrouter/vendor/fake-sync"]
    assert stop["kind"] == "rejected" and "request rejected" in stop["message"]
    assert record["capabilities"]["openrouter/vendor/fake-sync"][0]["request_rejected"]
    assert len(world.sync.sent) == 1
    # Continuing unchanged does not resend into the same wall.
    assert world.cont(run_dir) == runner.EXIT_STOPPED
    assert len(world.sync.sent) == 1


def test_truncation_stop_stops_only_the_cut_model(tmp_path):
    world = World(tmp_path, extra_refs=EXTRA_NVIDIA)
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    record = json.loads((run_dir / "run.json").read_text())
    stop = record["stops"]["models"]["nvidia/vendor/fake-cut"]
    assert stop["kind"] == "truncation"
    assert stop["truncated"] == 3 and stop["answers"] == 3
    assert runner.TRUNCATION_DECISION in stop["message"]
    assert set(record["stops"]["models"]) == {"nvidia/vendor/fake-cut"}
    assert record["stops"]["providers"] == {}
    assert len(world.cut.sent) == 3  # min_answers, then no more
    other = runner.status(run_dir)["models"]
    assert other["openrouter/vendor/fake-sync"]["pending"] == 0
    assert other["anthropic/fake-batch"]["pending"] == 0
    # A continue with the same settings never re-pays or resends.
    assert world.cont(run_dir) == runner.EXIT_STOPPED
    assert len(world.cut.sent) == 3


def test_unresolved_batch_lookup_stops_and_asks_without_resubmitting(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"

    original = world.batch._send_batch

    def accept_then_drop(requests, submit_ref):
        original(requests, submit_ref)  # the provider accepted it...
        world.batch.mark_unresolved(submit_ref)
        raise OSError("connection reset before the reply")  # ...but we never heard back

    world.batch._send_batch = accept_then_drop
    assert world.run(run_dir) == runner.EXIT_STOPPED
    world.batch._send_batch = original
    assert world.cont(run_dir) == runner.EXIT_ASK
    asked = world.lines[-1]
    assert asked.startswith("stop and ask:") and "anthropic/fake-batch" in asked
    assert "NOT resubmitted" in asked and "tj-" in asked
    assert len(world.batch.submitted_refs) == 1


def test_orphan_batch_found_by_ref_is_reattached_not_resent(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    original = world.batch._send_batch

    def accept_then_drop(requests, submit_ref):
        original(requests, submit_ref)
        raise OSError("reply lost")

    world.batch._send_batch = accept_then_drop
    assert world.run(run_dir) == runner.EXIT_STOPPED
    world.batch._send_batch = original
    assert world.cont(run_dir) == runner.EXIT_OK, world.lines
    assert world.batch.finds and len(world.batch.sent) == len(set(world.batch.sent))


# ---------------------------------------------------------------------------
# drive (deviation d2 loop)
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_drive_rechecks_a_money_stopped_provider_with_one_probe(tmp_path):
    world = World(tmp_path)
    world.sync.fail_always = "402"
    run_dir = tmp_path / "run"
    runner.init_state(run_dir, world.manifest, run_id="fixture-run", date="2026-09-26")
    clock = FakeClock()
    sends_at: list[tuple[float, int]] = []

    def sleep(seconds):
        clock.sleep(seconds)
        sends_at.append((clock.now, len(world.sync.sent)))
        if clock.now - 1000.0 >= 3600:
            world.sync.fail_always = None  # operator topped up after an hour

    code = drive_mod.drive(
        run_dir,
        world.manifest,
        env=world.env,
        factory=world.factory,
        clock=clock,
        sleep=sleep,
        poll_seconds=60,
        recheck_seconds=1800,
        max_steps=200,
        out=world.lines.append,
    )
    assert code == runner.EXIT_OK, world.lines
    assert (run_dir / "result.json").exists()
    # One 402 at the start, one probe at +30 min (402 again), one at +60 min
    # that goes through; nothing is sent in between.
    log = [json.loads(line) for line in (run_dir / "drive.log").read_text().splitlines()]
    probes = [entry for entry in log if "probing the money stop" in entry["msg"]]
    assert len(probes) >= 2
    assert probes[0]["t"] - 1000.0 >= 1800
    # Between probes the stopped provider sent nothing.
    # (each entry is the send count after sleeping, before the next step)
    counts = [count for at, count in sends_at if at - 1000.0 <= 1800]
    assert set(counts) == {1}
    counts = [count for at, count in sends_at if 1800 < at - 1000.0 <= 3600]
    assert set(counts) == {2}
    assert "its money stop is cleared" in (run_dir / "drive.log").read_text()


def test_drive_exits_non_zero_on_stop_and_ask(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    original = world.batch._send_batch

    def accept_then_drop(requests, submit_ref):
        original(requests, submit_ref)
        world.batch.mark_unresolved(submit_ref)
        raise OSError("lost")

    world.batch._send_batch = accept_then_drop
    runner.init_state(run_dir, world.manifest, run_id="r", date="d")
    clock = FakeClock()
    code = drive_mod.drive(
        run_dir,
        world.manifest,
        env=world.env,
        factory=world.factory,
        clock=clock,
        sleep=clock.sleep,
        max_steps=5,
        out=world.lines.append,
    )
    assert code == runner.EXIT_ASK
    assert "stop and ask" in (run_dir / "drive.log").read_text()


def test_drive_stops_cleanly_between_steps_on_a_signal(tmp_path):
    import threading

    world = World(tmp_path)
    world.batch.slow = True
    run_dir = tmp_path / "run"
    runner.init_state(run_dir, world.manifest, run_id="r", date="d")
    stop = threading.Event()
    clock = FakeClock()

    def sleep(seconds):
        clock.sleep(seconds)
        stop.set()  # SIGTERM arrives while waiting

    code = drive_mod.drive(
        run_dir,
        world.manifest,
        env=world.env,
        factory=world.factory,
        clock=clock,
        sleep=sleep,
        stop=stop,
        out=world.lines.append,
    )
    assert code == runner.EXIT_OK
    assert "exiting cleanly" in (run_dir / "drive.log").read_text()
    assert not (run_dir / "result.json").exists()


def test_drive_start_begins_a_full_run_in_an_empty_run_dir(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    clock = FakeClock()
    code = drive_mod.drive(
        run_dir,
        world.manifest,
        env=world.env,
        factory=world.factory,
        clock=clock,
        sleep=clock.sleep,
        max_steps=50,
        out=world.lines.append,
        start=True,
    )
    assert code == runner.EXIT_OK, world.lines
    assert (run_dir / "result.json").exists()
    assert "starting a full run" in (run_dir / "drive.log").read_text()


def test_drive_posts_progress_and_the_completion_alert(tmp_path):
    from evals.tool_jev import alerts

    world = World(tmp_path)
    posts: list[str] = []
    clock = FakeClock()
    code = drive_mod.drive(
        tmp_path / "run",
        world.manifest,
        env={**world.env, alerts.ENV_WEBHOOK: "https://discord.example/hook"},
        factory=world.factory,
        clock=clock,
        sleep=clock.sleep,
        max_steps=50,
        out=world.lines.append,
        start=True,
        poster=lambda url, text: posts.append(text),
    )
    assert code == runner.EXIT_OK, world.lines
    text = "\n".join(posts)
    assert "COMPLETE" in text and "100%" in text
    assert "alert sent" in (tmp_path / "run" / "drive.log").read_text()


def test_drive_without_start_refuses_an_empty_run_dir(tmp_path):
    world = World(tmp_path)
    clock = FakeClock()
    code = drive_mod.drive(
        tmp_path / "run",
        world.manifest,
        env=world.env,
        factory=world.factory,
        clock=clock,
        sleep=clock.sleep,
        max_steps=1,
        out=world.lines.append,
    )
    assert code != runner.EXIT_OK
    assert "holds no run" in world.lines[-1]


def test_drive_idles_instead_of_exiting_when_done(tmp_path):
    """A restart-unless-stopped service must not re-run a finished run in a loop."""
    import threading

    world = World(tmp_path)
    run_dir = tmp_path / "run"
    stop = threading.Event()
    waits: list[float] = []

    def sleep(seconds):
        waits.append(seconds)

    code = drive_mod.drive(
        run_dir,
        world.manifest,
        env=world.env,
        factory=world.factory,
        clock=FakeClock(),
        sleep=sleep,
        stop=stop,
        max_steps=50,
        out=world.lines.append,
        start=True,
        idle_when_done=True,
    )
    assert code == runner.EXIT_OK
    assert waits and waits[-1] == 3600.0  # idling, not polling
    assert "run complete; idling until stopped" in (run_dir / "drive.log").read_text()


def test_the_cli_drive_flags_reach_the_loop(tmp_path, monkeypatch):
    seen = {}

    def fake_drive(run_dir, manifest, **kwargs):
        seen.update(kwargs)
        return runner.EXIT_OK

    monkeypatch.setattr(drive_mod, "drive", fake_drive)
    from evals.tool_jev import __main__ as cli

    code = cli.main(
        [
            "drive",
            "--manifest",
            str(tmp_path / "m.toml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--start",
            "--idle-when-done",
        ],
        stop=__import__("threading").Event(),
    )
    assert code == runner.EXIT_OK
    assert seen["start"] is True and seen["idle_when_done"] is True


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def test_smoke_reports_tokens_cost_projection_and_flags(tmp_path):
    world = World(tmp_path, extra_refs=EXTRA_NVIDIA)
    run_dir = tmp_path / "run"
    code = world.main(
        "smoke", "--manifest", str(world.manifest), "--run-dir", str(run_dir), "--cases", "3"
    )
    # The cut model's truncation stop keeps the smoke from completing its judge pass.
    assert code == runner.EXIT_STOPPED
    # The operator raises the cut model's output budget: new calls, new keys.
    world.cut.cut = False
    manifest = world.manifest.read_text()
    world.manifest.write_text(manifest.replace("max_output_tokens = 64", "max_output_tokens = 128"))
    code = world.main(
        "smoke", "--manifest", str(world.manifest), "--run-dir", str(run_dir), "--cases", "3"
    )
    assert code == runner.EXIT_OK, world.lines
    smoke = json.loads((run_dir / "smoke.json").read_text())
    assert smoke["cases"] == 3 and smoke["full_run_cases"] == 3
    sync = smoke["models"]["openrouter/vendor/fake-sync"]
    assert sync["flag"] == "OK" and sync["truncated"] == 0
    assert sync["input_tokens"] == sync["calls"] * 1000
    assert sync["cost_usd"] == pytest.approx(sync["calls"] * 0.0012)
    assert sync["projected_full_run_usd"] == pytest.approx(round(sync["cost_usd"], 4))
    assert smoke["providers"]["anthropic"]["cost_usd"] > 0
    assert any(line.startswith("smoke openrouter/vendor/fake-sync: OK") for line in world.lines)
    assert not (run_dir / "result.json").exists()


def test_smoke_flags_a_capped_model(tmp_path):
    world = World(tmp_path, extra_refs=EXTRA_NVIDIA)
    run_dir = tmp_path / "run"
    manifest = world.manifest.read_text().replace(
        "[stops]\nmin_answers = 3", "[stops]\nmin_answers = 50"
    )
    world.manifest.write_text(manifest)
    code = world.main(
        "smoke", "--manifest", str(world.manifest), "--run-dir", str(run_dir), "--cases", "2"
    )
    assert code == runner.EXIT_OK, world.lines
    smoke = json.loads((run_dir / "smoke.json").read_text())
    cut = smoke["models"]["nvidia/vendor/fake-cut"]
    assert cut["flag"] == "CAPPED" and cut["truncated"] == cut["calls"]
    assert smoke["full_run_cases"] == 3 and smoke["cases"] == 2
    assert smoke["models"]["openrouter/vendor/fake-sync"]["flag"] == "OK"


# ---------------------------------------------------------------------------
# configuration refusals
# ---------------------------------------------------------------------------


def test_run_dir_inside_a_git_worktree_is_refused(tmp_path):
    world = World(tmp_path)
    repo_run_dir = Path(__file__).resolve().parent / "never-created-run-dir"
    assert world.run(repo_run_dir) == runner.EXIT_USER
    assert "inside a git worktree" in world.lines[-1]
    assert not repo_run_dir.exists()


def test_run_refuses_a_second_start_and_continue_needs_a_run(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    assert world.cont(run_dir) == runner.EXIT_USER
    assert world.run(run_dir) == runner.EXIT_OK
    assert world.run(run_dir) == runner.EXIT_USER
    assert "use `continue`" in world.lines[-1]


def test_case_count_mismatch_is_refused(tmp_path):
    world = World(tmp_path)
    world.manifest.write_text(world.manifest.read_text().replace("count = 3", "count = 4"))
    assert world.run(tmp_path / "run") == runner.EXIT_USER
    assert "the manifest says 4 cases" in world.lines[-1]


def test_reasoning_none_maps_per_provider():
    assert runner.provider_reasoning("openai", "none") == "minimal"
    assert runner.provider_reasoning("anthropic", "none") == "low"
    assert runner.provider_reasoning("openrouter", "none") == "none"
    assert runner.provider_reasoning("nvidia", "none") is None
    assert runner.provider_reasoning("local", "none") is None
    assert runner.provider_reasoning("nvidia", "high") == "high"


def test_a_truncated_judge_reply_is_an_invalid_judge_answer(tmp_path):
    world = World(tmp_path)
    world.sync.cut_judge = True
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_OK, world.lines
    judged = json.loads((run_dir / "judge_results.json").read_text())
    by_judge = {s["judge"] for s in judged["scores"] if s["verdict"] == "invalid"}
    assert by_judge == {"openrouter/vendor/fake-sync"}
    assert all(
        s["score"] is not None for s in judged["scores"] if s["judge"] == "anthropic/fake-batch"
    )
    doc = runner.status(run_dir)
    assert doc["models"]["openrouter/vendor/fake-sync"]["invalid"] > 0


def test_status_reads_without_the_ledger_lock(tmp_path):
    from evals.tool_jev.ledger import Ledger

    world = World(tmp_path)
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_OK
    with Ledger(run_dir):  # a live drive holds the lock
        assert world.main("status", "--run-dir", str(run_dir)) == runner.EXIT_OK
    assert world.lines[-1].startswith("hosts that received case text:")


def test_missing_private_root_is_an_environment_error(tmp_path):
    world = World(tmp_path)
    world.env = {}
    assert world.run(tmp_path / "run") == runner.EXIT_ENV
    assert "NVSH_EVALS_PRIVATE_ROOT" in world.lines[-1]


def test_missing_manifest_is_a_user_error(tmp_path):
    world = World(tmp_path)
    code = world.main(
        "run", "--manifest", str(tmp_path / "absent.toml"), "--run-dir", str(tmp_path / "r")
    )
    assert code == runner.EXIT_USER and "cannot read" in world.lines[-1]


def test_default_factory_builds_each_adapter_without_reading_keys(monkeypatch):
    from evals.tool_jev.manifest import Budget, Reference
    from evals.tool_jev.providers import anthropic, openai, openai_compat

    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NGC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = {"NVSH_EVALS_BASE_URL_LOCAL": "http://127.0.0.1:18001/v1"}
    budget = Budget("nvidia", usd_cap=0.0, concurrency_cap=1, requests_per_minute=40.0)
    cases = [
        (Reference("openai", "gpt-6-sol", batch=True), openai.OpenAIProvider, "api.openai.com"),
        (
            Reference("anthropic", "claude-sonnet-5", batch=True),
            anthropic.AnthropicProvider,
            "api.anthropic.com",
        ),
        (
            Reference("nvidia", "vendor/m", capabilities=("chat", "reasoning")),
            openai_compat.OpenAICompatProvider,
            "integrate.api.nvidia.com",
        ),
        (Reference("local", "vendor/l"), openai_compat.OpenAICompatProvider, "127.0.0.1"),
    ]
    for ref, cls, host in cases:
        provider = runner.default_factory(ref, budget, env)
        assert isinstance(provider, cls)
        assert runner.provider_host(provider) == host
    nvidia = runner.default_factory(cases[2][0], budget, env)
    assert nvidia.capabilities.reasoning is True and nvidia._rate_limiter is not None
    assert runner.default_factory(cases[1][0], None, env).name == "anthropic:claude-sonnet-5"


def test_continue_retry_rejected_retries_a_model_after_the_operator_fixes_it(tmp_path):
    world = World(tmp_path)
    world.sync.fail_always = "401"
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    world.sync.fail_always = None  # the operator fixed the key behind the 401
    assert world.cont(run_dir) == runner.EXIT_STOPPED  # unchanged params: still stopped
    code = world.main(
        "continue",
        "--manifest",
        str(world.manifest),
        "--run-dir",
        str(run_dir),
        "--retry-rejected",
    )
    assert code == runner.EXIT_OK, world.lines


def test_continue_while_a_drive_holds_the_run_dir_is_refused_cleanly(tmp_path):
    from evals.tool_jev.ledger import Ledger

    world = World(tmp_path)
    run_dir = tmp_path / "run"
    runner.init_state(run_dir, world.manifest, run_id="r", date="d")
    with Ledger(run_dir):
        assert world.cont(run_dir) == runner.EXIT_USER
    assert "in use by another runner" in world.lines[-1]


def test_run_json_output_is_one_document(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    assert world.run(run_dir, "--json") == runner.EXIT_OK
    doc = json.loads(world.lines[-1])
    assert doc["status"] == "complete" and doc["exit_code"] == 0
    assert any(line.startswith("complete:") for line in doc["messages"])


def test_a_configuration_error_does_not_mark_the_run_started(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    env = world.env
    world.env = {}
    assert world.run(run_dir) == runner.EXIT_ENV
    world.env = env
    assert world.run(run_dir) == runner.EXIT_OK, world.lines


# ---------------------------------------------------------------------------
# codex review of t17: one test per finding
# ---------------------------------------------------------------------------


def _step(world, run_dir, **kwargs):
    return runner.step(
        run_dir,
        world.manifest,
        env=world.env,
        factory=world.factory,
        out=world.lines.append,
        **kwargs,
    )


def test_p1_1_expired_batch_keeps_completed_results_and_requeues_only_the_rest(tmp_path):
    world = World(tmp_path)
    world.batch.expire = 1
    run_dir = tmp_path / "run"
    runner.init_state(run_dir, world.manifest, run_id="r", date="d")
    clock = FakeClock()
    outcome = None
    for _ in range(6):
        outcome = _step(world, run_dir, clock=clock, poll_seconds=60)
        if outcome.status == runner.STATUS_COMPLETE:
            break
        clock.now += 600
    assert outcome.status == runner.STATUS_COMPLETE, world.lines
    # Every answer that came back was kept: no key was answered twice.
    assert len(world.batch.sent) == len(set(world.batch.sent))
    assert any("expired" in line and "anthropic/fake-batch" in line for line in world.lines)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["batch_failures"] == {}  # a good batch afterwards resets the count


def test_p1_1_repeated_batch_failures_back_off_then_stop_the_model(tmp_path):
    world = World(tmp_path)
    world.batch.expire = 99
    run_dir = tmp_path / "run"
    clock = FakeClock()
    runner.init_state(run_dir, world.manifest, run_id="r", date="d")
    _step(world, run_dir, clock=clock, poll_seconds=60)
    first = len(world.batch.batch_sizes)
    assert first >= 1
    failures = json.loads((run_dir / "run.json").read_text())["batch_failures"]
    count = failures["anthropic/fake-batch"]["count"]
    assert count == first
    backoff = min(2**count * 60, 1800)
    assert failures["anthropic/fake-batch"]["next_submit_at"] == clock.now + backoff
    _step(world, run_dir, clock=clock, poll_seconds=60)
    assert len(world.batch.batch_sizes) == first  # backing off: nothing resubmitted
    clock.now += backoff
    _step(world, run_dir, clock=clock, poll_seconds=60)
    assert len(world.batch.batch_sizes) > first
    record = json.loads((run_dir / "run.json").read_text())
    stop = record["stops"]["models"]["anthropic/fake-batch"]
    assert stop["kind"] == "batch_failures" and "operator decision" in stop["message"]
    submitted = len(world.batch.batch_sizes)
    clock.now += 3600
    outcome = _step(world, run_dir, clock=clock, poll_seconds=60)
    assert len(world.batch.batch_sizes) == submitted  # stopped: no more submissions
    assert outcome.status == runner.STATUS_STOPPED


def test_p1_2_a_sync_call_in_flight_at_a_crash_is_resent_with_an_uncertain_charge(tmp_path):
    world = World(tmp_path)
    world.sync.crash_after_send_at = 2
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_INTERRUPTED
    record = json.loads((run_dir / "run.json").read_text())
    in_flight = [k for k, v in record["reserved"].items() if v["route"] == "sync"]
    assert len(in_flight) == 1  # the in-flight marker survived the crash
    (key,) = in_flight
    assert world.cont(run_dir) == runner.EXIT_OK, world.lines
    record = json.loads((run_dir / "run.json").read_text())
    assert [c["key"] for c in record["uncertain_charges"]] == [key]
    assert record["reserved"] == {}
    billing = [json.loads(line) for line in (run_dir / "billing.jsonl").read_text().splitlines()]
    uncertain = [b for b in billing if b["kind"] == "uncertain"]
    assert [b["key"] for b in uncertain] == [key] and uncertain[0]["cost_usd"] > 0
    assert "uncertain_resend" in (run_dir / "events.jsonl").read_text()
    assert any("resent" in line and key[:12] in line for line in world.lines)
    # The resent identity appears twice (it really was sent twice); nothing else does.
    assert len(world.sync.sent) == len(set(world.sync.sent)) + 1


DUPLICATE_SET = """
[[case_set]]
name = "syn-copy"
count = 3
split = "test-mc"
path = "splits/syn-test.json"
"""


def test_p1_3_duplicate_keys_across_case_sets_are_sent_once(tmp_path):
    world = World(tmp_path)
    world.manifest.write_text(world.manifest.read_text() + DUPLICATE_SET)
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_OK, world.lines
    assert len(world.sync.sent) == len(set(world.sync.sent))
    assert len(world.batch.sent) == len(set(world.batch.sent))
    result = json.loads((run_dir / "result.json").read_text())
    assert "openrouter.vendor-fake-sync.A.syn-copy" in {
        row["subject"] for row in result["reference_rows"]
    }


def test_p1_4_batch_submissions_reserve_cost_against_the_cap(tmp_path):
    world = World(tmp_path)
    world.batch.slow = True
    text = world.manifest.read_text().replace(
        "[budget.anthropic]\nusd_cap = 10.0", "[budget.anthropic]\nusd_cap = 0.012"
    )
    world.manifest.write_text(text)
    run_dir = tmp_path / "run"
    world.run(run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    reserved = [r for r in record["reserved"].values() if r["provider"] == "anthropic"]
    assert reserved and sum(r["usd"] for r in reserved) <= 0.012
    # One call is estimated at ~$0.005 batched: the round-1 batch was split to fit.
    assert world.batch.batch_sizes and max(world.batch.batch_sizes) < 3
    assert any("budget_cap_reached" in line for line in world.lines)


EXTRA_OPENROUTER = """
[[reference]]
provider = "openrouter"
model = "vendor/extra"
usd_per_mtok_in = 1.0
usd_per_mtok_out = 2.0
"""


def test_p1_5_spend_is_billed_at_answer_time_and_survives_manifest_changes(tmp_path):
    world = World(tmp_path, extra_refs=EXTRA_OPENROUTER)
    world.extra = CaseFake("openrouter:vendor/extra", batch=False, host="sync-host.test")
    factory = world.factory
    world.factory = lambda ref, budget, env: (
        world.extra if ref.model == "vendor/extra" else factory(ref, budget, env)
    )
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_OK, world.lines
    before = runner.status(run_dir)["providers"]["openrouter"]["spend_usd"]
    assert before > 0
    billing = (run_dir / "billing.jsonl").read_text()
    assert '"label": "openrouter/vendor/extra"' in billing
    # The operator replaces the extra model and changes prices: history stays.
    text = world.manifest.read_text().replace(EXTRA_OPENROUTER, "")
    text = text.replace("usd_per_mtok_out = 2.0", "usd_per_mtok_out = 200.0")
    text = text.replace(
        "[budget.openrouter]\nusd_cap = 10.0", f"[budget.openrouter]\nusd_cap = {before * 0.99}"
    )
    world.manifest.write_text(text)
    assert runner.status(run_dir)["providers"]["openrouter"]["spend_usd"] == pytest.approx(before)
    world.lines.clear()
    world.cont(run_dir)
    assert any("budget_cap_reached" in line for line in world.lines)


def test_p1_6_continuing_a_smoke_keeps_the_smoke_scope(tmp_path):
    world = World(tmp_path)
    world.batch.slow = True
    run_dir = tmp_path / "run"
    code = world.main(
        "smoke", "--manifest", str(world.manifest), "--run-dir", str(run_dir), "--cases", "2"
    )
    assert code == runner.EXIT_WAITING
    for _ in range(10):
        code = world.cont(run_dir)
        if code == runner.EXIT_OK:
            break
    assert code == runner.EXIT_OK
    assert (run_dir / "smoke.json").exists() and not (run_dir / "result.json").exists()
    sent_cases = {case_id for case_id, _i, _h in world.sync.sent + world.batch.sent}
    assert "s-3" not in sent_cases  # only the two smoke cases ever left
    assert all(c.startswith(("s-1", "s-2", "j-")) for c in sent_cases)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["scope"] == {"mode": "smoke", "case_set": "syn-test", "case_ids": ["s-1", "s-2"]}


def test_p1_6_run_on_a_smoke_dir_needs_expand(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    code = world.main(
        "smoke", "--manifest", str(world.manifest), "--run-dir", str(run_dir), "--cases", "2"
    )
    assert code == runner.EXIT_OK
    assert world.run(run_dir) == runner.EXIT_USER
    assert "--expand" in world.lines[-1]
    assert world.run(run_dir, "--expand") == runner.EXIT_OK, world.lines
    assert (run_dir / "result.json").exists()


def _orphan(world):
    original = world.batch._send_batch

    def accept_then_drop(requests, submit_ref):
        original(requests, submit_ref)
        raise OSError("reply lost")

    world.batch._send_batch = accept_then_drop
    return original


def test_p2_7_a_failing_batch_lookup_pauses_the_provider_and_keeps_the_orphan(tmp_path):
    from evals.tool_jev.ledger import Ledger

    world = World(tmp_path)
    run_dir = tmp_path / "run"
    _orphan(world)
    assert world.run(run_dir) == runner.EXIT_STOPPED
    del world.batch._send_batch  # back to the class method
    world.batch.find_error = fake.FakeProviderError(
        runner.classify_transport("anthropic", status_code=429), "anthropic", ""
    )
    assert world.cont(run_dir) == runner.EXIT_STOPPED
    with Ledger(run_dir) as ledger:
        assert ledger.continue_plan().orphans  # the marker is kept, nothing resubmitted
    assert len(world.batch.submitted_refs) == 1
    assert any("rate_limited" in line and "anthropic" in line for line in world.lines)
    # The sync provider's subject work went on regardless.
    assert runner.status(run_dir)["models"]["openrouter/vendor/fake-sync"]["pending"] == 0
    world.batch.find_error = OSError("dns")
    assert world.cont(run_dir) == runner.EXIT_STOPPED  # a transport error: no crash either


def test_p2_7_drive_survives_an_unexpected_step_error(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    runner.init_state(run_dir, world.manifest, run_id="r", date="d")
    calls = {"n": 0}
    factory = world.factory

    def flaky(ref, budget, env):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic bug")
        return factory(ref, budget, env)

    clock = FakeClock()
    code = drive_mod.drive(
        run_dir,
        world.manifest,
        env=world.env,
        factory=flaky,
        clock=clock,
        sleep=clock.sleep,
        max_steps=20,
        out=world.lines.append,
    )
    assert code == runner.EXIT_OK
    assert "synthetic bug" in (run_dir / "drive.log").read_text()


def test_p2_8_openai_batch_http_quota_error_is_a_money_stop():
    from evals.tool_jev.providers import openai as openai_mod

    body = json.dumps({"error": {"code": "insufficient_quota", "message": "x"}}).encode()
    found = runner.classify_exception(openai_mod._HttpError(429, body), "openai")
    assert found.reason == "insufficient_credit"
    plain = runner.classify_exception(openai_mod._HttpError(429, b"{}"), "openai")
    assert plain.reason == "rate_limited"


def test_p2_9_a_batch_money_probe_blocks_the_provider_until_its_result(tmp_path):
    world = World(tmp_path)
    world.batch.batch_fail = "402"
    run_dir = tmp_path / "run"
    runner.init_state(run_dir, world.manifest, run_id="r", date="d")
    clock = FakeClock()

    def go():
        return _step(world, run_dir, clock=clock, retry_money=False, recheck_seconds=1800)

    go()
    stops = json.loads((run_dir / "run.json").read_text())["stops"]["providers"]
    assert stops["anthropic"]["kind"] == "money"
    world.batch.batch_fail = None
    world.batch.slow = True
    clock.now += 1800
    go()  # the probe: one batch holding one call, still processing
    assert world.batch.batch_sizes == [1]
    stop = json.loads((run_dir / "run.json").read_text())["stops"]["providers"]["anthropic"]
    assert stop["kind"] == "money" and stop["probe"]
    clock.now += 60
    go()  # the probe's result arrives OK: the stop clears, the rest may go
    assert "anthropic" not in json.loads((run_dir / "run.json").read_text())["stops"]["providers"]
    assert any("probe" in line and "cleared" in line for line in world.lines)
    assert len(world.batch.batch_sizes) > 1


def test_p2_9_no_other_submission_while_a_probe_is_pending(tmp_path):
    world = World(tmp_path)
    world.batch.batch_fail = "402"
    run_dir = tmp_path / "run"
    runner.init_state(run_dir, world.manifest, run_id="r", date="d")
    clock = FakeClock()
    _step(world, run_dir, clock=clock, retry_money=False, recheck_seconds=1800)
    world.batch.batch_fail = None
    # The probe batch never finishes: nothing else may go, however long it takes.
    world.batch._check_batch = lambda handle: BatchStatus(handle.batch_id, complete=False)
    for _ in range(4):
        clock.now += 1800
        _step(world, run_dir, clock=clock, retry_money=False, recheck_seconds=1800)
    assert world.batch.batch_sizes == [1]


def test_p2_10_one_limiter_per_provider_shared_across_models():
    from evals.tool_jev.manifest import Budget, Reference

    budget = Budget("nvidia", usd_cap=0.0, concurrency_cap=1, requests_per_minute=40.0)
    first = runner.default_factory(Reference("nvidia", "vendor/a"), budget, {})
    second = runner.default_factory(Reference("nvidia", "vendor/b"), budget, {})
    assert first._rate_limiter is second._rate_limiter is not None
    other = runner.default_factory(
        Reference("openrouter", "vendor/c"),
        Budget("openrouter", usd_cap=0.0, concurrency_cap=1, requests_per_minute=40.0),
        {},
    )
    assert other._rate_limiter is not first._rate_limiter


def test_p2_11_a_rejected_judge_is_released_when_judging_params_change(tmp_path):
    world = World(tmp_path)
    original = world.sync.outcome

    def reject_judging(request):
        if request.interface == "text" and request.params.get("max_output_tokens") == 300:
            return fake.ScriptedOutcome("unsupported_parameter")
        return original(request)

    world.sync.outcome = reject_judging
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    stop = json.loads((run_dir / "run.json").read_text())["stops"]["models"]
    assert stop["openrouter/vendor/fake-sync"]["kind"] == "rejected"
    world.manifest.write_text(
        world.manifest.read_text().replace("max_output_tokens = 300", "max_output_tokens = 400")
    )
    assert world.cont(run_dir) == runner.EXIT_OK, world.lines


def test_p2_12_state_writes_are_durable_and_race_free(tmp_path, monkeypatch):
    import os
    import threading

    from evals.tool_jev import runstate

    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    target = tmp_path / "state.json"
    errors = []

    def writer(n):
        try:
            for i in range(20):
                runstate.write_json_durable(target, {"writer": n, "i": i})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert json.loads(target.read_text())["i"] == 19
    assert len(synced) >= 2 * 120  # the file and its directory, every write
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_p2_12_the_run_lock_is_held_while_state_is_read_and_written(tmp_path, monkeypatch):
    from evals.tool_jev import runstate
    from evals.tool_jev.ledger import Ledger

    world = World(tmp_path)
    run_dir = tmp_path / "run"
    seen = []
    real_load, real_save = runstate.load_state, runstate.save_state

    def locked():
        try:
            Ledger(run_dir).close()
        except Exception:  # noqa: BLE001 -- LedgerLocked: someone holds it
            return True
        return False

    monkeypatch.setattr(runstate, "load_state", lambda d: (seen.append(locked()), real_load(d))[1])
    monkeypatch.setattr(
        runstate, "save_state", lambda d, s: (seen.append(locked()), real_save(d, s))[1]
    )
    assert world.run(run_dir) == runner.EXIT_OK
    assert seen and all(seen)


def test_p2_13_a_truncated_choice_reply_is_invalid_even_with_a_valid_letter(tmp_path):
    world = World(tmp_path)
    world.sync.cut_choice = True
    world.manifest.write_text(
        world.manifest.read_text().replace("min_answers = 3", "min_answers = 50")
    )
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_OK, world.lines
    lines = (run_dir / "traces" / "openrouter.vendor-fake-sync.B.syn-test.jsonl").read_text()
    raws = [json.loads(line)["raw"] for line in lines.splitlines()]
    assert {(r["outcome"], r["invalid_reason"]) for r in raws} == {("invalid", "truncated")}


# ---------------------------------------------------------------------------
# codex verification of 3ff2085: the items left open
# ---------------------------------------------------------------------------


def _billing(run_dir):
    path = run_dir / "billing.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_item1_an_unreadable_failed_batch_stays_submitted_and_is_fetched_again(tmp_path):
    from evals.tool_jev.ledger import Ledger

    world = World(tmp_path)
    world.batch.expire = 1
    world.batch.fetch_error_once = OSError("download failed")
    run_dir = tmp_path / "run"
    runner.init_state(run_dir, world.manifest, run_id="r", date="d")
    clock = FakeClock()
    _step(world, run_dir, clock=clock, poll_seconds=60)
    with Ledger(run_dir) as ledger:
        assert ledger.submitted_batches()  # not requeued on an unreadable result
    assert len(world.batch.batch_sizes) == 2  # A and B: nothing resubmitted
    outcome = None
    for _ in range(8):
        clock.now += 600
        outcome = _step(world, run_dir, clock=clock, poll_seconds=60)
        if outcome.status == runner.STATUS_COMPLETE:
            break
    assert outcome.status == runner.STATUS_COMPLETE, world.lines
    assert len(world.batch.sent) == len(set(world.batch.sent))


def test_item2_a_sync_timeout_bills_one_uncertain_charge_and_two_stop_the_model(tmp_path):
    world = World(tmp_path)
    world.sync.fail_always = "timeout"
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    uncertain = [b for b in _billing(run_dir) if b["kind"] == "uncertain"]
    assert len(uncertain) == 1 and uncertain[0]["attempt_id"]
    assert world.cont(run_dir) == runner.EXIT_STOPPED  # the key is resent once more
    uncertain = [b for b in _billing(run_dir) if b["kind"] == "uncertain"]
    assert len(uncertain) == 2 and len({b["attempt_id"] for b in uncertain}) == 2
    record = json.loads((run_dir / "run.json").read_text())
    stop = record["stops"]["models"]["openrouter/vendor/fake-sync"]
    assert stop["kind"] == "uncertain_attempts" and "operator decision" in stop["message"]
    sent = len(world.sync.sent)
    assert world.cont(run_dir) == runner.EXIT_STOPPED
    assert len(world.sync.sent) == sent  # stopped: never a third uncertain attempt
    spend = runner.status(run_dir)["providers"]["openrouter"]["spend_usd"]
    assert spend == pytest.approx(sum(b["cost_usd"] for b in uncertain))


def test_a_free_model_is_resent_after_uncertain_attempts_never_stopped(tmp_path):
    """The uncertain-attempts stop guards money; a free (local) model just retries."""
    world = World(tmp_path)
    text = world.manifest.read_text()
    world.manifest.write_text(
        text.replace("usd_per_mtok_in = 1.0\nusd_per_mtok_out = 2.0", "usd_per_mtok_in = 0.0")
    )
    world.sync.fail_always = "timeout"
    run_dir = tmp_path / "run"
    world.run(run_dir)
    for _ in range(3):
        world.cont(run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    assert "openrouter/vendor/fake-sync" not in record["stops"]["models"]
    assert record["uncertain_attempts"] and max(record["uncertain_attempts"].values()) >= 3


def test_budget_timeout_seconds_reaches_the_sync_adapter(tmp_path):
    from evals.tool_jev import runplan
    from evals.tool_jev.manifest import Budget, Reference

    ref = Reference(provider="local", model="m", api_key_env="LOCAL_KEY_ENV")
    budget = Budget(provider="local", usd_cap=0.0, concurrency_cap=1, timeout_seconds=300.0)
    provider = runplan.default_factory(ref, budget, {})
    assert provider.timeout_seconds == 300.0
    assert runplan.default_factory(ref, None, {}).timeout_seconds == 60.0


@pytest.mark.parametrize(
    "error",
    [ConnectionRefusedError("refused"), __import__("socket").gaierror("no such host")],
)
def test_item2_a_clean_refusal_before_sending_is_not_uncertain(tmp_path, error):
    world = World(tmp_path)
    world.sync.refuse_before_send = error
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    assert [b for b in _billing(run_dir) if b["kind"] == "uncertain"] == []
    record = json.loads((run_dir / "run.json").read_text())
    assert all(v["route"] != "sync" for v in record["reserved"].values())


def test_item14_billing_sums_each_attempt_once(tmp_path):
    from evals.tool_jev import runstate

    billing = runstate.Billing(tmp_path)
    line = {"kind": "answer", "key": "k", "attempt_id": "k:r1", "provider": "p", "cost_usd": 1.0}
    billing.append(line)
    billing.append(line)  # a replayed append
    billing.append({**line, "attempt_id": "k:r2"})
    billing.append(
        {"kind": "uncertain", "key": "k", "attempt_id": "res-1", "provider": "p", "cost_usd": 0.5}
    )
    assert runstate.sums(billing.entries(), "provider") == {"p": 2.5}


@pytest.mark.parametrize("which", ["batch", "sync"])
def test_item14_a_kill_after_billing_before_the_cache_never_bills_twice(
    tmp_path, monkeypatch, which
):
    from evals.tool_jev.ledger import Ledger

    clean = World(tmp_path / "a")
    assert clean.run(tmp_path / "a" / "run") == runner.EXIT_OK
    world = World(tmp_path / "b")
    run_dir = tmp_path / "b" / "run"
    real = Ledger.record_done
    target = {"batch": "anthropic:fake-batch", "sync": "openrouter:vendor/fake-sync"}[which]
    killed = []

    def record_done(self, key, response):
        if not killed and self.entry(key).spec["provider"] == target:
            killed.append(key)
            raise KeyboardInterrupt  # billed, not yet cached
        return real(self, key, response)

    monkeypatch.setattr(Ledger, "record_done", record_done)
    assert world.run(run_dir) == runner.EXIT_INTERRUPTED
    monkeypatch.setattr(Ledger, "record_done", real)
    assert world.cont(run_dir) == runner.EXIT_OK, world.lines
    answers = [b for b in _billing(run_dir) if b["kind"] == "answer"]
    assert len(answers) == len({b["key"] for b in answers})
    assert [b["key"] for b in answers].count(killed[0]) == 1
    assert _outputs(run_dir) == _outputs(tmp_path / "a" / "run")


def test_item15_uncertain_recovery_survives_a_kill_before_the_state_save(tmp_path, monkeypatch):
    from evals.tool_jev import runstate

    world = World(tmp_path)
    world.sync.crash_after_send_at = 2
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_INTERRUPTED
    real = runstate.save_state
    killed = []

    def save_state(directory, state):
        uncertain = [b for b in _billing(Path(directory)) if b["kind"] == "uncertain"]
        if uncertain and not killed:
            killed.append(True)
            raise KeyboardInterrupt  # the charge is appended, run.json is not
        return real(directory, state)

    monkeypatch.setattr(runstate, "save_state", save_state)
    assert world.cont(run_dir) == runner.EXIT_INTERRUPTED
    monkeypatch.setattr(runstate, "save_state", real)
    assert world.cont(run_dir) == runner.EXIT_OK, world.lines
    uncertain = [b for b in _billing(run_dir) if b["kind"] == "uncertain"]
    assert len(uncertain) == 1
    record = json.loads((run_dir / "run.json").read_text())
    assert len(record["uncertain_charges"]) == 1


EXTRA_BATCH = """
[[reference]]
provider = "anthropic"
model = "fake-batch2"
batch = true
api_key_env = "UNUSED_KEY_ENV"
usd_per_mtok_in = 3.0
usd_per_mtok_out = 15.0
"""


def test_item16_a_removed_models_outstanding_batches_are_settled(tmp_path):
    world = World(tmp_path, extra_refs=EXTRA_BATCH)
    world.batch2 = CaseFake("anthropic:fake-batch2", batch=True, host="batch-host.test")
    world.batch.slow = world.batch2.slow = True
    factory = world.factory
    built = []

    def by_model(ref, budget, env):
        if ref.model == "fake-batch2":
            built.append(ref)
            return world.batch2
        return factory(ref, budget, env)

    world.factory = by_model
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_WAITING
    assert world.batch2.batch_sizes
    world.manifest.write_text(world.manifest.read_text().replace(EXTRA_BATCH, ""))
    code = runner.EXIT_WAITING
    for _ in range(10):
        code = world.cont(run_dir)
        if code == runner.EXIT_OK:
            break
    assert code == runner.EXIT_OK, world.lines
    # Its batch was fetched through an adapter rebuilt from submission metadata...
    ghost = built[-1]
    assert ghost.batch and ghost.api_key_env == "UNUSED_KEY_ENV"
    assert ghost.usd_per_mtok_out == 15.0
    # ...billed and released, and it scores nothing.
    assert any(b["label"] == "anthropic/fake-batch2" for b in _billing(run_dir))
    assert json.loads((run_dir / "run.json").read_text())["reserved"] == {}
    result = json.loads((run_dir / "result.json").read_text())
    assert not any("fake-batch2" in row["subject"] for row in result["reference_rows"])


def test_item16_an_orphaned_reservation_with_no_submitted_work_is_released(tmp_path):
    from evals.tool_jev import runstate
    from evals.tool_jev.ledger import Ledger

    world = World(tmp_path)
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_OK
    with Ledger(run_dir):
        state = runstate.load_state(run_dir)
        state["reserved"]["no-such-key"] = {
            "id": "x",
            "provider": "anthropic",
            "label": "anthropic/fake-batch",
            "usd": 9.0,
            "route": "batch",
            "since": 0.0,
        }
        runstate.save_state(run_dir, state)
    assert world.cont(run_dir) == runner.EXIT_OK
    assert json.loads((run_dir / "run.json").read_text())["reserved"] == {}


def test_item11_a_rejected_judge_is_compared_only_with_judge_params(tmp_path):
    world = World(tmp_path)
    world.manifest.write_text(
        world.manifest.read_text().replace("max_output_tokens = 300", "max_output_tokens = 512")
    )
    original = world.sync.outcome

    def reject_judging(request):
        if request.interface == "text" and request.params.get("max_output_tokens") == 512:
            return fake.ScriptedOutcome("unsupported_parameter")
        return original(request)

    world.sync.outcome = reject_judging
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_STOPPED
    stop = json.loads((run_dir / "run.json").read_text())["stops"]["models"]
    rejected = stop["openrouter/vendor/fake-sync"]
    assert rejected["request"]["role"] == "judge" and rejected["request"]["interface"] == "text"
    # Only the judging budget changes; the reference calls still use 512 tokens.
    world.manifest.write_text(
        world.manifest.read_text().replace(
            "[judging]\nseed = 5\nmax_output_tokens = 512",
            "[judging]\nseed = 5\nmax_output_tokens = 1024",
        )
    )
    assert world.cont(run_dir) == runner.EXIT_OK, world.lines


def test_item13_a_truncated_invalid_record_stays_invalid_under_every_policy():
    from evals.tool_jev import deepeval_layer, metrics_bridge

    metrics_mod = metrics_bridge.load_metrics_module()
    row = {
        "id": "t-1",
        "expected": {"operation": "service_restart", "args": SERVICE},
        "outcome": "invalid",
        "operation": None,
        "arguments": None,
        "candidates": {"service_restart": 0.9, "service_status": 0.05, "escalate": 0.05},
        "tokens": 0,
        "ttfd_ms": 0.0,
        "latency_ms": 0.0,
        "invalid_reason": "truncated",
    }
    prediction = metrics_mod.Prediction.from_dict(row)
    for policy in ("raw", "scorer-r3b-shipped", "mutating-strict-example"):
        applied = deepeval_layer.apply_policy_to_prediction(
            policy, prediction, metrics_mod=metrics_mod
        )
        assert (applied.outcome, applied.invalid_reason) == ("invalid", "truncated")
    # Another invalid reason keeps the designed behaviour: the saved distribution decides.
    other = metrics_mod.Prediction.from_dict({**row, "invalid_reason": "no_label_mass"})
    assert (
        deepeval_layer.apply_policy_to_prediction("raw", other, metrics_mod=metrics_mod).outcome
        == "propose"
    )
    metrics = metrics_bridge.compute([prediction], metrics_mod=metrics_mod)
    assert metrics["metrics_compute"]["outcome_counts"].get("propose", 0) == 0


def test_item13_truncated_choices_score_no_proposal_end_to_end(tmp_path):
    world = World(tmp_path)
    world.sync.cut_choice = True  # the sync fake returns logprobs
    world.manifest.write_text(
        world.manifest.read_text().replace("min_answers = 3", "min_answers = 50")
    )
    run_dir = tmp_path / "run"
    assert world.run(run_dir) == runner.EXIT_OK, world.lines
    metrics = json.loads(
        (run_dir / "metrics" / "openrouter.vendor-fake-sync.B.syn-test__raw.json").read_text()
    )
    assert metrics["metrics_compute"]["outcome_counts"].get("propose", 0) == 0
    traces = (run_dir / "traces" / "openrouter.vendor-fake-sync.B.syn-test.jsonl").read_text()
    assert {json.loads(t)["final"]["raw"]["decision"] for t in traces.splitlines()} == {"invalid"}


def test_item17_a_torn_billing_tail_is_moved_aside(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    world.sync.fail = {3: "402"}
    assert world.run(run_dir) == runner.EXIT_STOPPED
    path = run_dir / "billing.jsonl"
    whole = path.read_bytes()
    path.write_bytes(whole + b'{"kind": "answer", "key": "half')  # a crash mid-append
    doc = runner.status(run_dir)  # status tolerates the torn tail
    assert doc["providers"]["openrouter"]["done"] == 2
    world.cont(run_dir)
    assert path.read_bytes().startswith(whole)
    assert b"half" not in path.read_bytes()
    torn = list(run_dir.glob("billing.torn.*"))
    assert len(torn) == 1 and torn[0].read_bytes() == b'{"kind": "answer", "key": "half'
    assert "billing_torn_tail" in (run_dir / "events.jsonl").read_text()


def test_item17_a_torn_line_in_the_middle_stops_and_asks(tmp_path):
    world = World(tmp_path)
    run_dir = tmp_path / "run"
    world.sync.fail = {3: "402"}
    assert world.run(run_dir) == runner.EXIT_STOPPED
    path = run_dir / "billing.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + b"{broken\n" + b"".join(lines[1:]))
    assert world.cont(run_dir) == runner.EXIT_ASK
    assert "billing.jsonl" in world.lines[-1]


def test_item18_the_backoff_exponent_is_clamped():
    assert drive_mod.backoff_delay(60.0, 1) == 120.0
    assert drive_mod.backoff_delay(60.0, 5000) == 1800.0
    assert runner.backoff_delay(60.0, 10**6) == runner.MAX_BACKOFF_SECONDS
