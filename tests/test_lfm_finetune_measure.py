"""The stock-versus-tuned Tier 2 measurement script: scripts/lfm-finetune/measure.py.

Everything that would touch the machine -- docker, nvidia-smi, git, the Tier 2
container and model -- goes through the script's ``Seams``; these tests pass a
fake docker runner and a scripted :class:`~nvsh.tiers.fake.FakeTier`, so no
container is launched, no model is loaded and nothing reaches the network.
Results files are written under ``tmp_path`` via ``--out``, never into the repo.
"""

from __future__ import annotations

import http.server
import importlib.util
import itertools
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvsh.platform._model import Platform
from nvsh.tiers import bench as tier_bench
from nvsh.tiers.base import Decline, DeclineReason, Explanation
from nvsh.tiers.fake import FakeTier
from nvsh.tiers.runtime import RuntimeUnavailable

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/measure.py"
UID = 4242
OWN = f"nvsh-tier2-{UID}"
STOCK = "LiquidAI/LFM2.5-350M"
TUNED = "jetson-ai-lab/lfm2.5-350m-nvsh-triage"


@pytest.fixture(scope="module")
def measure():
    spec = importlib.util.spec_from_file_location("lfm_measure", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lfm_measure"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixtures: a split file, a fake docker, a fake runtime, scripted tiers
# ---------------------------------------------------------------------------


def _entry(entry_id: str, source_id: str, expect: dict) -> dict:
    return {
        "id": entry_id,
        "source_id": source_id,
        "kind": "explicit",
        "text": f"fixture request {entry_id}",
        "expect": expect,
        "source": "fixture",
    }


def _entries() -> list[dict]:
    """Two variations of s1, one of s2 (mutating), two of s3 (escalate), one explain."""
    return [
        _entry("op1", "s1", {"operation": "gpu_stats", "args": {}}),
        _entry("op1v", "s1", {"operation": "gpu_stats", "args": {}}),
        _entry("op2", "s2", {"operation": "container_restart", "args": {"container": "trainer"}}),
        _entry("esc1", "s3", {"escalate": True}),
        _entry("esc1v", "s3", {"escalate": True}),
        _entry("exp1", "s4", {"explain": True}),
    ]


def _split(tmp_path: Path, name: str = "val.json", header: str | None = None) -> Path:
    header = header or "Fixture corpus. Split 'val' of fixture.json (seed=39)."
    path = tmp_path / "splits" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"header": header, "entries": _entries()}), encoding="utf-8")
    return path


def _call(name: str, **arguments) -> list[dict]:
    return [{"name": name, "arguments": arguments}]


def _decline() -> Decline:
    return Decline(DeclineReason.TIER_UNAVAILABLE, "scripted decline")


def _stock_script() -> list[object]:
    """op1 right, op1v escalated (s1 ties: wrong per source), op2 a wrong mutating pick,
    esc1 escalated, esc1v proposed (s3 ties: wrong per source), exp1 explained."""
    return [
        _call("gpu_stats"),
        _decline(),
        _call("service_restart", service="nginx.service"),
        _decline(),
        _call("gpu_stats"),
        Explanation(text="the GPU is idle"),
    ]


def _tuned_script() -> list[object]:
    """Everything right except exp1, which proposes a mutating operation."""
    return [
        _call("gpu_stats"),
        _call("gpu_stats"),
        _call("container_restart", container="trainer"),
        _decline(),
        _decline(),
        _call("container_restart", container="inference"),
    ]


class FakeDocker:
    """Answers the argv lists the script runs; records every call."""

    def __init__(self, running=(), backgrounds=(("vllm-server",),), ps_code: int = 0) -> None:
        self.running = list(running)
        self.backgrounds = list(backgrounds)
        self.ps_code = ps_code
        self.calls: list[list[str]] = []
        self._names_calls = 0

    def __call__(self, argv: list[str], timeout: float) -> tuple[int, str]:
        del timeout
        self.calls.append(list(argv))
        if argv[:3] == ["docker", "ps", "--filter"]:
            wanted = argv[3].removeprefix("name=^").removesuffix("$")
            names = [name for name in self.running if name == wanted]
            return (self.ps_code, "".join(f"{name}\n" for name in names))
        if argv == ["docker", "ps", "--format", "{{.Names}}"]:
            index = min(self._names_calls, len(self.backgrounds) - 1)
            self._names_calls += 1
            names = list(self.backgrounds[index]) + self.running
            return (0, "".join(f"{name}\n" for name in names))
        if argv == ["docker", "ps"]:
            return (0, "CONTAINER ID   IMAGE   NAMES\nabc   vllm/vllm-openai   vllm-server\n")
        if argv == ["nvidia-smi"]:
            return (0, "NVIDIA-SMI 580.126.09   Driver Version: 580.126.09   GB10\n")
        if argv[:2] == ["docker", "stats"]:
            return (0, "4.5GiB / 121GiB\n")
        if argv[:1] == ["git"]:
            return (0, "abc1234def\n")
        return (127, "not found")


class FakeRuntime:
    def __init__(self, fail: str = "") -> None:
        self.fail = fail
        self.ensured = 0
        self.stopped = 0

    def ensure(self) -> str:
        self.ensured += 1
        if self.fail:
            raise RuntimeUnavailable(self.fail)
        return "http://127.0.0.1:1/v1"

    def stop(self) -> None:
        self.stopped += 1

    def status(self) -> str:
        return "fake runtime"


class ClosingFakeTier(FakeTier):
    """A FakeTier whose close() releases its runtime, as LfmTier.close() does."""

    def __init__(self, script, runtime: FakeRuntime) -> None:
        super().__init__(script, name="lfm")
        self.runtime = runtime

    def close(self) -> None:
        super().close()
        self.runtime.stop()


def _hf_cache(tmp_path: Path, refs: dict[str, str], *, hub: bool = True) -> Path:
    """A host HF cache whose ``refs/main`` resolves each repo id to a commit."""
    cache = tmp_path / "hf-cache"
    base = cache / "hub" if hub else cache
    for repo_id, commit in refs.items():
        ref = base / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(commit, encoding="utf-8")
    return cache


class Harness:
    def __init__(
        self, measure, tmp_path: Path, docker: FakeDocker, scripts=None, lfm=None, fail=None
    ) -> None:
        self.docker = docker
        self.scripts = dict(scripts or {STOCK: _stock_script(), TUNED: _tuned_script()})
        self.fail = dict(fail or {})
        self.specs: list = []
        self.tiers: list[ClosingFakeTier] = []
        self.lfm = lfm or {
            "engine": "vllm",
            "mode": "managed",
            "tool_call_parser": "lfm2",
            "gpu_memory_fraction": 0.08,
            "hf_cache_dir": str(_hf_cache(tmp_path, {STOCK: "rev0", TUNED: "rev1"})),
        }
        counter = itertools.count()
        self.seams = measure.Seams(
            run=docker,
            build_tier=self.build_tier,
            detect_platform=lambda: Platform(kind="dgx_spark"),
            today=lambda: "2026-09-22",
            clock=lambda: next(counter) * 0.25,
            uid=lambda: UID,
            load_config=lambda _path: SimpleNamespace(
                tiers={"memory_floor_mb": 1024, "lfm": dict(self.lfm)}
            ),
            # FakeRuntime.ensure() answers a URL nothing actually listens on; the
            # issue-46 preflight is exercised by its own dedicated tests instead.
            preflight=lambda base_url, model, ctx=None: None,
        )

    def build_tier(self, spec):
        self.specs.append(spec)
        model = spec.lfm_settings["model"]
        runtime = FakeRuntime(self.fail.get(model, ""))
        tier = ClosingFakeTier(self.scripts[model], runtime)
        self.tiers.append(tier)
        return tier, runtime


def _argv(split: Path, out: Path, *extra: str, models=(STOCK, TUNED)) -> list[str]:
    argv = ["--split", str(split), "--out", str(out), "--label", "val-350m"]
    for index, model in enumerate(models):
        argv += ["--model", model, "--revision", f"rev{index}"]
    return argv + list(extra)


# ---------------------------------------------------------------------------
# Split refusals
# ---------------------------------------------------------------------------


def test_refuses_held_out_without_acceptance(measure, tmp_path, capsys):
    split = _split(tmp_path, "held-out.json")
    harness = Harness(measure, tmp_path, FakeDocker())
    code = measure.main(_argv(split, tmp_path / "r.md"), seams=harness.seams)
    assert code == 1
    assert "--acceptance" in capsys.readouterr().err
    assert harness.specs == []
    assert not (tmp_path / "r.md").exists()


def test_held_out_runs_with_acceptance(measure, tmp_path):
    split = _split(tmp_path, "held-out.json")
    harness = Harness(measure, tmp_path, FakeDocker())
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out, "--acceptance"), seams=harness.seams) == 0
    assert "- Acceptance run: yes" in out.read_text(encoding="utf-8")


def test_refuses_test_side_without_final(measure, tmp_path, capsys):
    split = _split(tmp_path, "test.json")
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, tmp_path / "r.md"), seams=harness.seams) == 1
    assert "--final" in capsys.readouterr().err
    assert harness.specs == []


@pytest.mark.parametrize("flag", ["--final", "--acceptance"])
def test_final_and_acceptance_refused_on_other_files(measure, tmp_path, flag):
    split = _split(tmp_path, "val.json")
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, tmp_path / "r.md", flag), seams=harness.seams) == 1
    assert harness.specs == []


def test_final_run_counts_previous_final_runs(measure, tmp_path):
    split = _split(tmp_path, "test.json")
    results = tmp_path / "benchmarks"
    results.mkdir()
    (results / "2026-09-01-lfm-old.md").write_text("# old\n\n- Final run: yes\n", encoding="utf-8")
    (results / "2026-09-02-lfm-val.md").write_text("# val\n\n- Final run: no\n", encoding="utf-8")
    out = results / "2026-09-22-lfm-final.md"
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, out, "--final"), seams=harness.seams) == 0
    text = out.read_text(encoding="utf-8")
    assert "- Final run: yes" in text
    assert "- Final runs on the test side, including this one: 2" in text


def test_revision_required_per_model(measure, tmp_path, capsys):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    argv = ["--split", str(split), "--out", str(tmp_path / "r.md"), "--model", STOCK]
    assert measure.main(argv, seams=harness.seams) == 1
    assert "--revision" in capsys.readouterr().err


def test_refuses_to_overwrite_results(measure, tmp_path):
    split = _split(tmp_path)
    out = tmp_path / "r.md"
    out.write_text("keep me\n", encoding="utf-8")
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, out), seams=harness.seams) == 1
    assert out.read_text(encoding="utf-8") == "keep me\n"
    assert measure.main(_argv(split, out, "--force"), seams=harness.seams) == 0


# ---------------------------------------------------------------------------
# The container guard
# ---------------------------------------------------------------------------


def test_refuses_when_own_container_running_and_never_stops_it(measure, tmp_path, capsys):
    split = _split(tmp_path)
    docker = FakeDocker(running=[OWN])
    harness = Harness(measure, tmp_path, docker)
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out), seams=harness.seams) == 2
    err = capsys.readouterr().err
    assert OWN in err
    assert "already running" in err
    assert harness.specs == []
    assert not out.exists()
    assert ["docker", "ps", "--filter", f"name=^{OWN}$", "--format", "{{.Names}}"] in docker.calls
    for argv in docker.calls:
        assert not ({"stop", "rm", "kill"} & set(argv)), argv


def test_refuses_when_docker_ps_fails(measure, tmp_path):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker(ps_code=1))
    assert measure.main(_argv(split, tmp_path / "r.md"), seams=harness.seams) == 2
    assert harness.specs == []


def test_other_containers_do_not_trip_the_guard(measure, tmp_path):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker(running=["nvsh-tier2-1000", "lobes-stt"]))
    assert measure.main(_argv(split, tmp_path / "r.md"), seams=harness.seams) == 0


def test_attach_mode_skips_the_guard(measure, tmp_path):
    split = _split(tmp_path)
    docker = FakeDocker(running=[OWN])
    harness = Harness(measure, tmp_path, docker, lfm={"engine": "vllm", "mode": "attach"})
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out), seams=harness.seams) == 0
    assert not any(argv[:3] == ["docker", "ps", "--filter"] for argv in docker.calls)
    assert "base_url" not in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A full stock-versus-tuned run
# ---------------------------------------------------------------------------


def _row(text: str, metric: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith(f"| {metric} |"):
            return [cell.strip() for cell in line.strip().strip("|").split("|")][1:]
    raise AssertionError(f"no row {metric!r} in:\n{text}")


def test_full_run_scores_and_records(measure, tmp_path, capsys):
    split = _split(tmp_path)
    out = tmp_path / "2026-09-22-lfm-val-350m.md"
    docker = FakeDocker()
    harness = Harness(measure, tmp_path, docker)
    assert measure.main(_argv(split, out), seams=harness.seams) == 0
    text = out.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert f"| Metric | `{STOCK}` | `{TUNED}` |" in printed

    # Provenance: command line, seed, model repo ids and revisions, nvsh commit.
    assert "--model LiquidAI/LFM2.5-350M --revision rev0" in text
    assert "- Seed: 39 (from the split header)" in text
    assert (
        f"`{STOCK}` @ `rev0` (revision verified from the cache); "
        f"`{TUNED}` @ `rev1` (revision verified from the cache)"
    ) in text
    assert "commit `abc1234def`" in text
    assert "- Final run: no" in text
    assert "(6 entries, 4 sources)" in text

    # Per-source and per-variation columns.
    assert _row(text, "Right operation and arguments proposed, per source") == [
        "0 of 2",
        "2 of 2",
    ]
    assert _row(text, "Right operation and arguments proposed, per variation") == [
        "1 of 3",
        "3 of 3",
    ]
    assert _row(text, "Should-escalate asks escalated, per source") == ["0 of 1", "1 of 1"]
    assert _row(text, "Should-escalate asks escalated, per variation") == ["1 of 2", "2 of 2"]
    assert _row(text, "Wrong mutating proposals, per source / per variation") == ["1 / 1", "1 / 1"]
    assert _row(text, "Explain asks explained, per source") == ["1 of 1", "0 of 1"]
    assert _row(text, "Explain asks: explained / proposed / escalated, per variation") == [
        "1 / 0 / 0 of 1",
        "0 / 1 / 0 of 1",
    ]
    assert _row(text, "Mutating proposals on explain asks") == ["0", "1"]
    assert _row(text, "Container memory (`docker stats`)") == ["4.5 GiB", "4.5 GiB"]
    assert _row(text, "Start-up, including first download") == ["0.2 s", "0.2 s"]
    assert _row(text, "First request after start")[0].endswith(" s")
    assert "ms /" in _row(text, "Warm latency, median / p95")[0]
    assert "not comparable" not in text

    # docker ps and nvidia-smi captured before each run.
    assert text.count("CONTAINER ID   IMAGE   NAMES") == 2
    assert text.count("NVIDIA-SMI 580.126.09") == 2
    first_bench_call = next(
        i for i, argv in enumerate(docker.calls) if argv[:2] == ["docker", "stats"]
    )
    assert ["nvidia-smi"] in docker.calls[:first_bench_call]

    # Each tier was started, then closed (its runtime stopped) after its run.
    assert [tier.closed for tier in harness.tiers] == [True, True]
    assert [tier.runtime.ensured for tier in harness.tiers] == [1, 1]
    assert [tier.runtime.stopped for tier in harness.tiers] == [1, 1]


def test_runs_differ_only_in_the_model(measure, tmp_path):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, tmp_path / "r.md"), seams=harness.seams) == 0
    stock, tuned = harness.specs
    assert stock.lfm_settings["model"] == STOCK
    assert tuned.lfm_settings["model"] == TUNED
    assert {k: v for k, v in stock.lfm_settings.items() if k != "model"} == {
        k: v for k, v in tuned.lfm_settings.items() if k != "model"
    }
    assert stock.memory_floor_mb == tuned.memory_floor_mb == 1024
    assert stock.tier_platform == tuned.tier_platform
    assert stock.runtime_platform == Platform(kind="dgx_spark")


def test_seed_flag_overrides_header(measure, tmp_path):
    split = _split(tmp_path, header="Fixture with no seed recorded.")
    harness = Harness(measure, tmp_path, FakeDocker())
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out, "--seed", "7"), seams=harness.seams) == 0
    assert "- Seed: 7 (--seed)" in out.read_text(encoding="utf-8")


def test_latency_marked_not_comparable_when_background_differs(measure, tmp_path):
    split = _split(tmp_path)
    docker = FakeDocker(backgrounds=[("vllm-server",), ("vllm-server", "lobes-stt")])
    harness = Harness(measure, tmp_path, docker)
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out), seams=harness.seams) == 0
    text = out.read_text(encoding="utf-8")
    assert "Latency figures are not comparable" in text
    assert _row(text, "Warm latency, median / p95 (not comparable)")
    assert "Other running containers: lobes-stt, vllm-server" in text


def test_startup_failure_is_recorded_not_hidden(measure, tmp_path):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker(), fail={TUNED: "image not pinned"})
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out), seams=harness.seams) == 2
    text = out.read_text(encoding="utf-8")
    assert _row(text, "Start-up, including first download")[1] == (
        "start-up failed: image not pinned"
    )
    assert _row(text, "Right operation and arguments proposed, per source")[1] == "not measured"
    assert harness.tiers[1].closed


# ---------------------------------------------------------------------------
# It drives nvsh.tiers.bench, not its own loop or scoring
# ---------------------------------------------------------------------------


def test_uses_bench_with_tier2_and_its_scoring_helpers(measure, tmp_path, monkeypatch):
    calls = {"bench": [], "_is_correct": 0, "compute_escalation": 0, "compute_false_mutating": 0}
    real_bench = tier_bench.bench

    def spy_bench(entries, *, split, tier1, platform, options):
        calls["bench"].append((tier1, options.tier2, split))
        return real_bench(entries, split=split, tier1=tier1, platform=platform, options=options)

    monkeypatch.setattr(tier_bench, "bench", spy_bench)
    for name in ("_is_correct", "compute_escalation", "compute_false_mutating"):
        real = getattr(tier_bench, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            calls[_name] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(tier_bench, name, spy)

    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, tmp_path / "r.md"), seams=harness.seams) == 0
    assert [(t1, t2, s) for t1, t2, s in calls["bench"]] == [
        (None, harness.tiers[0], "val"),
        (None, harness.tiers[1], "val"),
    ]
    assert calls["_is_correct"]
    assert calls["compute_escalation"]
    assert calls["compute_false_mutating"]


def test_per_source_tie_counts_against_the_model(measure):
    assert measure._majority([True, False]) is False
    assert measure._majority([True, True, False]) is True
    assert measure._majority([]) is False


def test_seed_from_header(measure):
    assert measure.seed_from_header("Split 'val' of dev.json (seed=39).") == 39
    assert measure.seed_from_header({"seed": 5}) == 5
    assert measure.seed_from_header("no seed") is None
    assert measure.seed_from_header(None) is None


def test_default_factory_builds_lfm_tier_like_the_daemon(measure, tmp_path):
    """The real factory: build_runtime over [tiers.lfm] + LfmTier, model overridden.

    Constructing either object starts nothing (no docker, no model)."""
    from nvsh.tiers.lfm import LfmTier
    from nvsh.tiers.runtime_docker import DockerRuntime

    platform = Platform(kind="dgx_spark")
    spec = measure.TierSpec(
        lfm_settings={
            "engine": "vllm",
            "mode": "managed",
            "model": TUNED,
            "hf_cache_dir": str(tmp_path / "hf"),
        },
        runtime_platform=platform,
        tier_platform=platform,
        runner=lambda argv, timeout: (127, ""),
        memory_floor_mb=1024,
    )
    tier, runtime = measure.build_lfm_tier(spec)
    assert isinstance(tier, LfmTier)
    assert isinstance(runtime, DockerRuntime)
    assert tier._model == TUNED
    assert "nvsh-tier2-" in runtime.status()
    assert "not started" in runtime.status()


def test_script_names_no_endpoint_or_key():
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "http://" not in text
    assert "https://" not in text
    assert "api_key" not in text.lower()


def test_scoring_helpers_on_explain_rows(measure):
    """items_from_result turns a handled, operation-less row into an explanation."""
    loaded = tier_bench.CorpusEntry(
        id="e", kind="explicit", text="t", expect={"explain": True}, source="f"
    )
    result = {"items": [{"id": "e", "handled_by": "lfm", "operation": None, "args": {}}]}
    (item,) = measure.items_from_result(result, [loaded])
    assert tier_bench._is_correct(item)
    assert item.outcome.escalated_to is None
    assert item.outcome.explanation == ""


# ---------------------------------------------------------------------------
# Split sides come from the file name AND its header (renaming is no bypass)
# ---------------------------------------------------------------------------

_TEST_HEADER = "Fixture corpus. Split 'test' of dev.json (seed=39)."
_HELD_OUT_HEADER = json.loads(
    (Path(__file__).resolve().parents[1] / "nvsh/tiers/corpus/held-out.json").read_text(
        encoding="utf-8"
    )
)["header"]


def test_renamed_test_side_still_needs_final(measure, tmp_path):
    with pytest.raises(measure.MeasureError, match="--final"):
        measure.check_split_allowed(Path("test-copy.json"), acceptance=False, final=False)
    renamed = _split(tmp_path, "renamed.json", header=_TEST_HEADER)
    with pytest.raises(measure.MeasureError, match="--final"):
        measure.check_split_allowed(renamed, acceptance=False, final=False)
    measure.check_split_allowed(renamed, acceptance=False, final=True)


def test_val_named_file_with_test_header_needs_final(measure, tmp_path):
    split = _split(tmp_path, "val.json", header=_TEST_HEADER)
    with pytest.raises(measure.MeasureError, match="--final"):
        measure.check_split_allowed(split, acceptance=False, final=False)


def test_renamed_held_out_still_needs_acceptance(measure, tmp_path, capsys):
    split = _split(tmp_path, "adoption.json", header=_HELD_OUT_HEADER)
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, tmp_path / "r.md"), seams=harness.seams) == 1
    assert "--acceptance" in capsys.readouterr().err
    assert harness.specs == []
    measure.check_split_allowed(split, acceptance=True, final=False)


def test_split_of_held_out_is_held_out(measure, tmp_path):
    header = "Held-out corpus. Split 'val' of held-out.json (seed=3)."
    split = _split(tmp_path, "val.json", header=header)
    with pytest.raises(measure.MeasureError, match="--acceptance"):
        measure.check_split_allowed(split, acceptance=False, final=False)


def test_undetermined_side_is_refused(measure, tmp_path):
    split = _split(tmp_path, "notes.json", header="Some corpus with no split note.")
    with pytest.raises(measure.MeasureError, match="cannot tell"):
        measure.check_split_allowed(split, acceptance=False, final=False)


def test_plain_dev_corpus_is_allowed(measure):
    dev_path = tier_bench.dev_corpus_path()
    measure.check_split_allowed(dev_path, acceptance=False, final=False)
    with pytest.raises(measure.MeasureError):
        measure.check_split_allowed(dev_path, acceptance=False, final=True)


def test_train_split_of_dev_is_allowed(measure, tmp_path):
    dev_header = json.loads(tier_bench.dev_corpus_path().read_text(encoding="utf-8"))["header"]
    # dev.json's own header mentions held-out.json; that must not mark its splits held-out.
    split = _split(
        tmp_path, "train.json", header=f"{dev_header} Split 'train' of dev.json (seed=39)."
    )
    measure.check_split_allowed(split, acceptance=False, final=False)


# ---------------------------------------------------------------------------
# --revision is verified against what the engine will actually serve
# ---------------------------------------------------------------------------


def test_revision_mismatch_refuses_before_any_run(measure, tmp_path, capsys):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    _hf_cache(tmp_path, {TUNED: "someothercommit"})
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out), seams=harness.seams) == 2
    err = capsys.readouterr().err
    assert TUNED in err
    assert "someothercommit" in err
    assert "rev1" in err
    assert harness.specs == []
    assert not out.exists()


def test_revision_missing_from_cache_refuses(measure, tmp_path, capsys):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    ref = tmp_path / "hf-cache/hub" / f"models--{STOCK.replace('/', '--')}" / "refs/main"
    ref.unlink()
    assert measure.main(_argv(split, tmp_path / "r.md"), seams=harness.seams) == 2
    assert "refs/main" in capsys.readouterr().err
    assert harness.specs == []


def test_revision_verified_is_recorded(measure, tmp_path):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out), seams=harness.seams) == 0
    text = out.read_text(encoding="utf-8")
    assert f"`{STOCK}` @ `rev0` (revision verified from the cache)" in text
    assert f"`{TUNED}` @ `rev1` (revision verified from the cache)" in text


def test_revision_found_without_hub_level(measure, tmp_path):
    split = _split(tmp_path)
    cache = tmp_path / "flat"
    for repo_id, commit in ((STOCK, "rev0"), (TUNED, "rev1")):
        ref = cache / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_text(commit + "\n", encoding="utf-8")
    lfm = {"engine": "vllm", "mode": "managed", "hf_cache_dir": str(cache)}
    harness = Harness(measure, tmp_path, FakeDocker(), lfm=lfm)
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out), seams=harness.seams) == 0
    assert "(revision verified from the cache)" in out.read_text(encoding="utf-8")


def test_mounted_model_revision_is_operator_supplied(measure, tmp_path):
    split = _split(tmp_path)
    lfm = {"engine": "llama-server", "mode": "managed", "model_dir": str(tmp_path / "gguf")}
    scripts = {"stock.gguf": _stock_script(), "tuned.gguf": _tuned_script()}
    harness = Harness(measure, tmp_path, FakeDocker(), lfm=lfm, scripts=scripts)
    out = tmp_path / "r.md"
    argv = _argv(split, out, models=("stock.gguf", "tuned.gguf"))
    assert measure.main(argv, seams=harness.seams) == 0
    text = out.read_text(encoding="utf-8")
    assert "`stock.gguf` @ `rev0` (operator-supplied, not verified)" in text


# ---------------------------------------------------------------------------
# A mutating proposal with the right operation but wrong arguments
# ---------------------------------------------------------------------------


def test_wrong_arguments_to_mutating_operation_are_reported(measure, tmp_path):
    split = _split(tmp_path)
    wrong_args = _tuned_script()
    wrong_args[2] = _call("container_restart", container="inference")  # expected "trainer"
    wrong_args[5] = Explanation(text="nothing to restart")
    harness = Harness(measure, tmp_path, FakeDocker(), scripts={STOCK: wrong_args})
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out, models=(STOCK,)), seams=harness.seams) == 0
    text = out.read_text(encoding="utf-8")
    # bench's own row still compares operation names only (unchanged scoring) ...
    assert _row(text, "Wrong mutating proposals, per source / per variation") == ["0 / 0"]
    # ... and the separate row catches the wrong target.
    assert _row(text, "Mutating proposals with wrong arguments, per source / per variation") == [
        "1 / 1"
    ]
    assert "judged on the sum of both rows" in text


def test_wrong_arguments_row_is_zero_when_arguments_match(measure, tmp_path):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    out = tmp_path / "r.md"
    assert measure.main(_argv(split, out), seams=harness.seams) == 0
    text = out.read_text(encoding="utf-8")
    assert _row(text, "Mutating proposals with wrong arguments, per source / per variation") == [
        "0 / 0",
        "0 / 0",
    ]


def test_details_write_one_row_per_entry_per_model(measure, tmp_path):
    split = _split(tmp_path, "val.json")
    harness = Harness(measure, tmp_path, FakeDocker())
    details = tmp_path / "details.jsonl"
    argv = _argv(split, tmp_path / "r.md", "--details", str(details))
    assert measure.main(argv, seams=harness.seams) == 0
    rows = [json.loads(line) for line in details.read_text(encoding="utf-8").splitlines()]
    assert {row["model"] for row in rows} == {STOCK, TUNED}
    assert all({"id", "expect", "did", "correct"} <= set(row) for row in rows)
    assert {row["did"] for row in rows} <= {"propose", "escalate", "explain"}


@pytest.mark.parametrize("flag", ["--final", "--acceptance"])
def test_details_are_refused_on_test_and_held_out_runs(measure, tmp_path, flag):
    name = "test.json" if flag == "--final" else "held-out.json"
    split = _split(tmp_path, name)
    harness = Harness(measure, tmp_path, FakeDocker())
    argv = _argv(split, tmp_path / "r.md", flag, "--details", str(tmp_path / "d.jsonl"))
    assert measure.main(argv, seams=harness.seams) == 1
    assert harness.specs == []
    assert not (tmp_path / "d.jsonl").exists()


def test_reports_write_the_home_directory_as_home(measure, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(measure.Path, "home", classmethod(lambda cls: tmp_path))
    line = measure.home_relative(f"measure.py --split {tmp_path}/work/val.json")
    assert line == "measure.py --split $HOME/work/val.json"


# ---------------------------------------------------------------------------
# Issue 46: the shared predictions file, scored by metrics.py
# ---------------------------------------------------------------------------

_METRICS = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/metrics.py"


@pytest.fixture(scope="module")
def metrics():
    spec = importlib.util.spec_from_file_location("lfm_metrics_for_measure", _METRICS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prediction_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _predictions_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.predictions.jsonl"))


def test_predictions_follow_the_metrics_schema(measure, metrics, tmp_path):
    split = _split(tmp_path)
    out = tmp_path / "r.md"
    predictions = tmp_path / "predictions"
    harness = Harness(measure, tmp_path, FakeDocker())
    argv = _argv(split, out, "--predictions", str(predictions))
    assert measure.main(argv, seams=harness.seams) == 0
    files = _predictions_files(predictions)
    assert len(files) == 2
    stock = _prediction_lines(files[0])
    for row in stock:
        metrics.Prediction.from_dict(row)  # raises on any schema problem
    assert [row["id"] for row in stock] == ["op1", "op1v", "op2", "esc1", "esc1v", "exp1"]
    by_id = {row["id"]: row for row in stock}
    assert by_id["op1"]["outcome"] == "propose"
    assert by_id["op1"]["operation"] == "gpu_stats"
    assert by_id["op1"]["arguments"] == {}
    assert by_id["op2"]["operation"] == "service_restart"
    assert by_id["op2"]["arguments"] == {"service": "nginx.service"}
    # A scripted tier that is unavailable did not abstain: it made no decision.
    assert by_id["op1v"]["outcome"] == "invalid"
    assert by_id["op1v"]["invalid_reason"] == "tier_unavailable"
    assert by_id["exp1"]["outcome"] == "explain"
    assert by_id["exp1"]["expected"] == {"explain": True}
    # A scripted tier generates nothing and returns no log-probabilities.
    assert all(row["tokens"] == 0 and row["candidates"] is None for row in stock)
    assert all(row["ttfd_ms"] <= row["latency_ms"] for row in stock)
    # Scored with metrics.py, next to the predictions and in the report.
    scored = json.loads(
        files[0]
        .with_name(files[0].name.replace(".predictions.jsonl", ".metrics.json"))
        .read_text(encoding="utf-8")
    )
    assert scored == metrics.compute(metrics.read_predictions(files[0]))
    text = out.read_text(encoding="utf-8")
    assert "## Issue 46 metrics" in text
    assert _row(text, "Right proposals (metrics.py)") == ["1 of 3", "3 of 3"]
    assert _row(text, "Non-empty think blocks (must be 0)") == ["0", "0"]


def test_predictions_are_scored_with_metrics_py(measure, tmp_path, monkeypatch):
    seen = []
    real = measure.metrics.compute

    def spy(predictions):
        seen.append(len(predictions))
        return real(predictions)

    monkeypatch.setattr(measure.metrics, "compute", spy)
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, tmp_path / "r.md"), seams=harness.seams) == 0
    assert seen == [6, 6]


def test_predictions_are_accepted_on_final_runs_without_request_text(measure, metrics, tmp_path):
    """Deviation d6: track_a_calibration.py reads a final run's predictions file."""
    split = _split(tmp_path, "test.json")
    out = tmp_path / "r.md"
    predictions = tmp_path / "p"
    harness = Harness(measure, tmp_path, FakeDocker())
    argv = _argv(split, out, "--final", "--predictions", str(predictions))
    assert measure.main(argv, seams=harness.seams) == 0
    files = _predictions_files(predictions)
    assert len(files) == 2
    rows = _prediction_lines(files[0])
    assert rows  # the run actually produced lines
    for row in rows:
        metrics.Prediction.from_dict(row)  # raises on any schema problem
        assert "text" not in row
    assert "## Issue 46 metrics" in out.read_text(encoding="utf-8")


def test_final_runs_still_report_metrics(measure, tmp_path):
    split = _split(tmp_path, "test.json")
    out = tmp_path / "r.md"
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, out, "--final"), seams=harness.seams) == 0
    assert "## Issue 46 metrics" in out.read_text(encoding="utf-8")


# -- a real LfmTier against a fake OpenAI-compatible server ------------------


def _lp(p: float) -> float:
    import math

    return math.log(p)


def _tok(token: str, p: float, top: dict[str, float] | None = None) -> dict:
    alternatives = {token: p, **(top or {})}
    return {
        "token": token,
        "logprob": _lp(p),
        "top_logprobs": [{"token": t, "logprob": _lp(q)} for t, q in alternatives.items()],
    }


def _propose_reply(operation: str, *, think: str = "") -> dict:
    arguments = json.dumps({"operation": operation, "arguments": {}})
    tokens = [
        _tok("<tool_call>", 0.99),
        _tok("\n<function=", 0.99),
        _tok("propose", 0.8, {"escalate>": 0.15, "explain>": 0.05}),
        _tok(">\n<parameter=operation>\n", 1.0),
        _tok(operation, 0.9, {"thermal_stats\n": 0.1}),
        _tok("\n</parameter>", 1.0),
    ]
    return {
        "message": {
            "role": "assistant",
            "content": think,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "propose", "arguments": arguments},
                }
            ],
        },
        "logprobs": {"content": tokens},
        "usage": {"completion_tokens": 17},
    }


def _escalate_reply(*, think: str = "") -> dict:
    tokens = [
        _tok("<tool_call>", 0.99),
        _tok("\n<function=", 0.99),
        _tok("escalate", 0.7, {"explain>": 0.2, "gpu_stats>": 0.1}),
        _tok(">\n", 1.0),
    ]
    return {
        "message": {
            "role": "assistant",
            "content": think,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "escalate", "arguments": '{"reason": "big"}'},
                }
            ],
        },
        "logprobs": {"content": tokens},
        "usage": {"completion_tokens": 9},
    }


class _ChatHandler(http.server.BaseHTTPRequestHandler):
    replies: list[dict] = []
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming convention
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _ChatHandler.requests.append(json.loads(body.decode("utf-8")))
        index = len(_ChatHandler.requests) - 1
        choice = _ChatHandler.replies[index % len(_ChatHandler.replies)]
        data = json.dumps({"choices": [choice], "usage": choice.get("usage")}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def chat_server():
    _ChatHandler.replies = []
    _ChatHandler.requests = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


class _UrlRuntime(FakeRuntime):
    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url

    def ensure(self) -> str:
        self.ensured += 1
        return self.url


def _small_split(tmp_path: Path, name: str = "val.json") -> Path:
    entries = [
        _entry("op1", "s1", {"operation": "gpu_stats", "args": {}}),
        _entry("esc1", "s2", {"escalate": True}),
    ]
    path = tmp_path / "splits" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "Fixture corpus. Split 'val' of fixture.json (seed=39)."
    path.write_text(json.dumps({"header": header, "entries": entries}), encoding="utf-8")
    return path


def _lfm_harness(measure, tmp_path, server, lfm=None) -> Harness:
    """Real LfmTiers built from each spec, talking to *server* through spec.chat_factory."""
    from nvsh.tiers.lfm import LfmTier

    harness = Harness(measure, tmp_path, FakeDocker(), lfm=lfm)
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"

    def build(spec):
        harness.specs.append(spec)
        runtime = _UrlRuntime(url)
        tier = LfmTier(
            runtime,
            spec.tier_platform,
            model=str(spec.lfm_settings["model"]),
            runner=spec.runner,
            chat_factory=spec.chat_factory,
        )
        harness.tiers.append(tier)
        return tier, runtime

    harness.seams.build_tier = build
    return harness


def test_served_generative_run_records_tokens_logprobs_and_thinking(
    measure, metrics, tmp_path, chat_server
):
    _ChatHandler.replies = [_propose_reply("gpu_stats"), _escalate_reply()]
    split = _small_split(tmp_path)
    out = tmp_path / "r.md"
    predictions = tmp_path / "p"
    harness = _lfm_harness(measure, tmp_path, chat_server)
    argv = _argv(split, out, "--predictions", str(predictions), "--enable-thinking", "false")
    argv += ["--top-logprobs", "5"]
    assert measure.main(argv, seams=harness.seams) == 0
    for request in _ChatHandler.requests:
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
        assert request["logprobs"] is True
        assert request["top_logprobs"] == 5
        assert request["stream"] is False
    rows = _prediction_lines(_predictions_files(predictions)[0])
    for row in rows:
        metrics.Prediction.from_dict(row)
    op1, esc1 = rows
    assert (op1["outcome"], op1["operation"], op1["tokens"]) == ("propose", "gpu_stats", 17)
    assert (esc1["outcome"], esc1["tokens"]) == ("escalate", 9)
    # propose 0.8 x (gpu_stats 0.9, thermal_stats 0.1); escalate 0.15; explain 0.05
    assert op1["candidates"] == pytest.approx(
        {"gpu_stats": 0.72, "thermal_stats": 0.08, "(escalate)": 0.15, "(explain)": 0.05}
    )
    # an inspection tool's mass is not a decision; escalate 0.7 and explain 0.2, normalised
    assert esc1["candidates"] == pytest.approx({"(escalate)": 0.7 / 0.9, "(explain)": 0.2 / 0.9})
    text = out.read_text(encoding="utf-8")
    assert "chat_template_kwargs enable_thinking=false" in text
    assert _row(text, "Non-empty think blocks (must be 0)") == ["0", "0"]
    assert _row(text, "Lines with a candidate distribution") == ["2 of 2", "2 of 2"]


def test_thinking_not_sent_unless_configured(measure, tmp_path, chat_server):
    _ChatHandler.replies = [_propose_reply("gpu_stats"), _escalate_reply()]
    split = _small_split(tmp_path)
    harness = _lfm_harness(measure, tmp_path, chat_server)
    assert measure.main(_argv(split, tmp_path / "r.md", models=(STOCK,)), seams=harness.seams) == 0
    assert _ChatHandler.requests
    for request in _ChatHandler.requests:
        assert "chat_template_kwargs" not in request


def test_nonempty_think_blocks_are_counted_and_fail_the_run(measure, tmp_path, chat_server, capsys):
    _ChatHandler.replies = [
        _propose_reply("gpu_stats", think="<think>hmm</think>"),
        _escalate_reply(think="<think>\n\n</think>"),
    ]
    split = _small_split(tmp_path)
    out = tmp_path / "r.md"
    harness = _lfm_harness(measure, tmp_path, chat_server)
    argv = _argv(split, out, "--enable-thinking", "false", models=(STOCK,))
    assert measure.main(argv, seams=harness.seams) == 2
    assert _row(out.read_text(encoding="utf-8"), "Non-empty think blocks (must be 0)") == ["1"]
    assert "think block" in capsys.readouterr().err


def test_missing_candidate_slice_restricts_the_tools_offered(measure, tmp_path, chat_server):
    _ChatHandler.replies = [_escalate_reply()]
    split = _small_split(tmp_path)
    predictions = tmp_path / "p"
    out = tmp_path / "r.md"
    harness = _lfm_harness(measure, tmp_path, chat_server)
    argv = _argv(split, out, "--slice", "missing-candidate", "--predictions", str(predictions))
    assert measure.main(argv, seams=harness.seams) == 0
    rows = _prediction_lines(_predictions_files(predictions)[0])
    assert [row["id"] for row in rows] == ["op1-nocand"]
    assert rows[0]["expected"] == {"escalate": True}
    assert rows[0]["outcome"] == "escalate"
    offered = [tool["function"]["name"] for tool in _ChatHandler.requests[0]["tools"]]
    assert "gpu_stats" not in offered
    assert {"propose", "explain", "escalate", "thermal_stats"} <= set(offered)
    propose = next(
        t for t in _ChatHandler.requests[0]["tools"] if t["function"]["name"] == "propose"
    )
    assert "gpu_stats" not in propose["function"]["parameters"]["properties"]["operation"]["enum"]
    assert "- Slice: missing-candidate" in out.read_text(encoding="utf-8")


# -- the generative candidate distribution, by hand ---------------------------


def _tokens(measure, rows):
    return measure.parse_logprobs({"content": rows})


def test_candidates_need_log_probabilities(measure):
    from nvsh.tiers.toolchat import ToolCall

    call = ToolCall(name="escalate", arguments={})
    distribution, reason = measure.generative_candidates(None, call, ("gpu_stats",))
    assert distribution is None
    assert "log-probabilities" in reason


def test_candidates_refuse_an_ambiguous_alternative(measure):
    from nvsh.tiers.toolchat import ToolCall

    rows = [
        _tok("<function=", 0.99),
        _tok("propose", 1.0),
        _tok(">", 1.0),
        _tok("service", 0.9, {"container": 0.1}),
        _tok("_status", 1.0),
        _tok("\n", 1.0),
    ]
    call = ToolCall(name="propose", arguments={"operation": "service_status", "arguments": {}})
    ops = ("service_status", "container_list", "container_restart")
    distribution, reason = measure.generative_candidates(_tokens(measure, rows), call, ops)
    assert distribution is None
    assert "not observed" in reason


def test_candidates_refuse_a_unique_prefix_whose_continuation_was_not_observed(measure):
    """Codex #3: ``gpu`` at the first token is not ``gpu_stats``: ``_stats`` was never scored."""
    from nvsh.tiers.toolchat import ToolCall

    rows = [
        _tok("<function=", 0.99),
        _tok("propose", 1.0),
        _tok(">", 1.0),
        _tok("service", 0.9, {"gpu": 0.1}),
        _tok("_restart", 1.0),
        _tok("\n", 1.0),
    ]
    call = ToolCall(name="propose", arguments={"operation": "service_restart", "arguments": {}})
    ops = ("gpu_stats", "service_restart")
    distribution, reason = measure.generative_candidates(_tokens(measure, rows), call, ops)
    assert distribution is None
    assert reason == "an alternative token's continuation was not observed"


def test_candidates_refuse_a_whole_name_without_its_end(measure):
    """An alternative spelling all of ``escalate`` still has an unscored continuation."""
    from nvsh.tiers.toolchat import ToolCall

    rows = [
        _tok("<function=", 0.99),
        _tok("propose", 0.9, {"escalate": 0.1}),
        _tok(">", 1.0),
        _tok("gpu_stats", 1.0),
        _tok("\n", 1.0),
    ]
    call = ToolCall(name="propose", arguments={"operation": "gpu_stats", "arguments": {}})
    distribution, reason = measure.generative_candidates(
        _tokens(measure, rows), call, ("gpu_stats",)
    )
    assert distribution is None
    assert "not observed" in reason


def test_candidates_read_the_token_that_ends_the_generated_name(measure):
    """A longer label offered at the token after the generated name is not the name's mass."""
    from nvsh.tiers.toolchat import ToolCall

    rows = [
        _tok("<function=", 0.99),
        _tok("propose", 1.0),
        _tok(">", 1.0),
        _tok("gpu", 1.0),
        _tok("\n", 0.8, {"_stats": 0.2}),
    ]
    call = ToolCall(name="propose", arguments={"operation": "gpu", "arguments": {}})
    distribution, reason = measure.generative_candidates(
        _tokens(measure, rows), call, ("gpu", "gpu_stats")
    )
    assert distribution is None
    assert "not observed" in reason


def test_candidates_need_the_name_to_end_inside_the_recorded_tokens(measure):
    from nvsh.tiers.toolchat import ToolCall

    rows = [_tok("<function=", 0.99), _tok("escalate", 1.0)]
    call = ToolCall(name="escalate", arguments={})
    distribution, reason = measure.generative_candidates(
        _tokens(measure, rows), call, ("gpu_stats",)
    )
    assert distribution is None
    assert "runs past" in reason


def test_candidates_walk_every_token_of_the_name(measure):
    from nvsh.tiers.toolchat import ToolCall

    rows = [
        _tok("<function=", 0.99),
        _tok("propose", 0.8, {"escalate>": 0.15, "explain>": 0.05}),
        _tok(">", 1.0),
        _tok("service", 0.9, {"gpu_stats\n": 0.1}),
        _tok("_restart", 0.6, {"_status\n": 0.3, "_logs\n": 0.1}),
        _tok("\n", 1.0),
    ]
    call = ToolCall(name="propose", arguments={"operation": "service_restart", "arguments": {}})
    ops = ("gpu_stats", "service_status", "service_logs", "service_restart")
    distribution, reason = measure.generative_candidates(_tokens(measure, rows), call, ops)
    assert reason == ""
    assert distribution == pytest.approx(
        {
            "(escalate)": 0.15,
            "(explain)": 0.05,
            "gpu_stats": 0.08,
            "service_status": 0.8 * 0.9 * 0.3,
            "service_logs": 0.8 * 0.9 * 0.1,
            "service_restart": 0.8 * 0.9 * 0.6,
        }
    )


def test_candidates_multiply_in_the_token_that_ends_the_name(measure):
    """The generated name's mass includes the token closing it; a no-label alternative drops."""
    from nvsh.tiers.toolchat import ToolCall

    rows = [
        _tok("<function=", 0.99),
        _tok("escalate", 0.9, {"explain>": 0.1}),
        _tok(">", 0.5, {"_now>": 0.5}),
    ]
    call = ToolCall(name="escalate", arguments={})
    distribution, reason = measure.generative_candidates(
        _tokens(measure, rows), call, ("gpu_stats",)
    )
    assert reason == ""
    assert distribution == pytest.approx({"(escalate)": 0.45 / 0.55, "(explain)": 0.1 / 0.55})


def test_candidates_do_not_split_proposal_mass_they_never_saw(measure):
    from nvsh.tiers.toolchat import ToolCall

    rows = [_tok("<function=", 0.99), _tok("escalate", 0.9, {"propose>": 0.1}), _tok(">", 1.0)]
    call = ToolCall(name="escalate", arguments={})
    distribution, reason = measure.generative_candidates(
        _tokens(measure, rows), call, ("gpu_stats",)
    )
    assert distribution is None
    assert "split" in reason


def test_unobserved_continuations_leave_the_line_without_a_distribution(
    measure, tmp_path, chat_server
):
    """End to end: the refused line keeps its outcome, gets null candidates and a counted note."""
    reply = _propose_reply("gpu_stats")
    reply["logprobs"]["content"][4] = _tok("gpu_stats", 0.9, {"thermal": 0.1})
    _ChatHandler.replies = [reply, _escalate_reply()]
    split = _small_split(tmp_path)
    predictions = tmp_path / "p"
    out = tmp_path / "r.md"
    harness = _lfm_harness(measure, tmp_path, chat_server)
    argv = _argv(split, out, "--predictions", str(predictions), models=(STOCK,))
    assert measure.main(argv, seams=harness.seams) == 0
    op1, esc1 = _prediction_lines(_predictions_files(predictions)[0])
    assert (op1["outcome"], op1["operation"], op1["candidates"]) == ("propose", "gpu_stats", None)
    assert esc1["candidates"] is not None
    text = out.read_text(encoding="utf-8")
    assert _row(text, "Lines with a candidate distribution") == ["1 of 2"]
    assert "an alternative token's continuation was not observed" in text


# -- an explanation that is really an unparsed tool call (Codex #6) ----------

_QWEN_XML = (
    "<tool_call>\n<function=propose>\n<parameter=operation>\ngpu_stats\n"
    "</parameter>\n</function>\n</tool_call>"
)


def _select(measure, result, text: str = "", calls=()):
    from nvsh.tiers.toolchat import ChatReply

    reply = measure.ReplyRecord(
        reply=ChatReply(text=text, tool_calls=tuple(calls)),
        tokens=3,
        think=False,
        logprobs=None,
        at=0.0,
    )
    return measure.SelectRecord(
        request=None, started=0.0, ended=0.0, result=result, replies=[reply]
    )


@pytest.mark.parametrize(
    "text",
    [
        _QWEN_XML,
        "<function=gpu_stats>\n</function>",
        "I will check.\n<parameter=operation>\ngpu_stats\n</parameter>",
        '<tool_call>\n{"name": "gpu_stats"}',
    ],
)
def test_an_explanation_holding_tool_call_markup_is_invalid(measure, text):
    row = {"id": "x", "handled_by": "lfm", "operation": None}
    select = _select(measure, Explanation(text=text), text=text)
    assert measure._decided(row, select) == ("invalid", None, None, "unparsed_tool_call")
    # the router's row is not needed: the tier's own result says the same
    assert measure._decided(None, select)[3] == "unparsed_tool_call"


def test_markup_in_the_raw_reply_is_found_even_if_the_explanation_lost_it(measure):
    row = {"id": "x", "handled_by": "lfm", "operation": None}
    select = _select(measure, Explanation(text="gpu_stats"), text=_QWEN_XML)
    assert measure._decided(row, select)[3] == "unparsed_tool_call"


def test_a_plain_explanation_is_still_an_explanation(measure):
    row = {"id": "x", "handled_by": "lfm", "operation": None}
    text = "The GPU is idle; a <function> in C returns a value."
    select = _select(measure, Explanation(text=text), text=text)
    assert measure._decided(row, select) == ("explain", None, None, None)


def test_served_unparsed_qwen_xml_is_an_invalid_output(measure, metrics, tmp_path, chat_server):
    """End to end: Qwen XML in message.content with no structured tool_calls."""
    unparsed = {
        "message": {"role": "assistant", "content": _QWEN_XML},
        "usage": {"completion_tokens": 20},
    }
    _ChatHandler.replies = [unparsed]
    split = _small_split(tmp_path)
    predictions = tmp_path / "p"
    harness = _lfm_harness(measure, tmp_path, chat_server)
    argv = _argv(split, tmp_path / "r.md", "--predictions", str(predictions), models=(STOCK,))
    assert measure.main(argv, seams=harness.seams) == 0
    rows = _prediction_lines(_predictions_files(predictions)[0])
    for row in rows:
        metrics.Prediction.from_dict(row)
        assert row["outcome"] == "invalid"
        assert row["invalid_reason"] == "unparsed_tool_call"
        assert row["candidates"] is None


# -- serving record: ctx, engine, image digest, tool_call_parser -------------

_DIGEST = "vllm/vllm-openai@sha256:" + "8bd082c2" * 8


def test_ctx_is_selectable_and_the_run_record_names_the_serving(measure, tmp_path):
    split = _split(tmp_path)
    out = tmp_path / "r.md"
    lfm = {
        "engine": "vllm",
        "mode": "managed",
        "image": _DIGEST,
        "tool_call_parser": "qwen3_coder",
        "hf_cache_dir": str(_hf_cache(tmp_path, {STOCK: "rev0", TUNED: "rev1"})),
    }
    harness = Harness(measure, tmp_path, FakeDocker(), lfm=lfm)
    assert measure.main(_argv(split, out, "--ctx", "2048"), seams=harness.seams) == 0
    assert [spec.lfm_settings["ctx"] for spec in harness.specs] == [2048, 2048]
    text = out.read_text(encoding="utf-8")
    serving = next(line for line in text.splitlines() if line.startswith("- Serving:"))
    assert "engine=vllm" in serving
    assert "ctx=2048" in serving
    assert f"image=`{_DIGEST}`" in serving
    assert "tool_call_parser=qwen3_coder" in serving


def test_ctx_defaults_to_the_launcher_default(measure, tmp_path):
    split = _split(tmp_path)
    out = tmp_path / "r.md"
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, out), seams=harness.seams) == 0
    serving = next(
        line
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.startswith("- Serving:")
    )
    assert "ctx=4096" in serving
    assert "tool_call_parser=lfm2" in serving


def test_bad_ctx_is_refused_before_any_run(measure, tmp_path):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    assert measure.main(_argv(split, tmp_path / "r.md", "--ctx", "10"), seams=harness.seams) == 1
    assert harness.specs == []


# -- the fixed grounding snapshot (deviation d1) ------------------------------


def _snapshot(tmp_path: Path, services=("nginx.service",), containers=("trainer",)) -> Path:
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "services": list(services),
                "containers": list(containers),
                "source": "fixture",
                "created": "2026-09-23",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ground_snapshot_is_the_runner_and_its_hash_is_recorded(measure, tmp_path):
    import hashlib

    from nvsh.ops import ground as ops_ground

    split = _split(tmp_path)
    snapshot = _snapshot(tmp_path)
    out = tmp_path / "r.md"
    harness = Harness(measure, tmp_path, FakeDocker())
    argv = _argv(split, out, "--ground-snapshot", str(snapshot))
    assert measure.main(argv, seams=harness.seams) == 0
    runner = harness.specs[0].runner
    assert runner(list(ops_ground.SERVICE_LOOKUP_ARGV), 3.0)[1].split()[0] == "nginx.service"
    assert runner(list(ops_ground.CONTAINER_LOOKUP_ARGV), 3.0) == (0, "trainer\n")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert digest in out.read_text(encoding="utf-8")


def test_ground_snapshot_and_live_are_exclusive(measure, tmp_path):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    argv = _argv(split, tmp_path / "r.md", "--live", "--ground-snapshot", str(_snapshot(tmp_path)))
    assert measure.main(argv, seams=harness.seams) == 1


def test_a_malformed_snapshot_is_refused(measure, tmp_path):
    split = _split(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"services": "nginx.service"}), encoding="utf-8")
    harness = Harness(measure, tmp_path, FakeDocker())
    argv = _argv(split, tmp_path / "r.md", "--ground-snapshot", str(bad))
    assert measure.main(argv, seams=harness.seams) == 1
    assert harness.specs == []


def test_snapshot_builder_merges_live_and_split_values_and_prints_counts_only(
    measure, tmp_path, capsys
):
    from nvsh.ops import ground as ops_ground

    split = _split(tmp_path)  # expects container "trainer"
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps(
            {
                "header": "x",
                "entries": [
                    _entry(
                        "o1", "o1", {"operation": "service_logs", "args": {"service": "secretd"}}
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    def run(argv, timeout):
        del timeout
        if argv == ops_ground.SERVICE_LOOKUP_ARGV:
            return (0, "livesvc.service loaded active running x\n")
        if argv == ops_ground.CONTAINER_LOOKUP_ARGV:
            return (0, "livebox\n")
        return (127, "")

    harness = Harness(measure, tmp_path, FakeDocker())
    harness.seams.run = run
    out = tmp_path / "snap.json"
    argv = ["snapshot", "--out", str(out), "--from-split", str(split), "--from-split", str(other)]
    assert measure.main(argv, seams=harness.seams) == 0
    snapshot = json.loads(out.read_text(encoding="utf-8"))
    assert snapshot["services"] == ["livesvc.service", "secretd.service"]
    assert snapshot["containers"] == ["livebox", "trainer"]
    assert snapshot["created"] == "2026-09-22"
    assert snapshot["source"]
    printed = capsys.readouterr()
    for value in ("livesvc", "secretd", "livebox", "trainer"):
        assert value not in printed.out + printed.err
    assert "2 services" in printed.out
    assert "2 containers" in printed.out


def test_snapshot_keys_cover_every_grounded_argument(measure):
    from nvsh.ops import ground as ops_ground

    assert set(measure.SNAPSHOT_KEYS) == set(ops_ground._KINDS)


# -- Track B: --scorer writes the same predictions file -----------------------


class _FakeScorer:
    """score_next_token with scripted label log-probabilities, per prompt."""

    def __init__(self, pick) -> None:
        self.pick = pick
        self.prompts: list[str] = []
        self.tops: list[int] = []

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        self.prompts.append(prompt)
        self.tops.append(top)
        return self.pick(prompt)


def _scorer_seams(measure, harness, fake: _FakeScorer, built: list):
    def build_scorer(spec):
        built.append(spec)
        return measure.ScorerHandle(
            scorer=fake, render=lambda messages: json.dumps(messages), close=lambda: None
        )

    harness.seams.build_scorer = build_scorer


def _label_logprobs(measure, winner: str, offered=None) -> dict[str, float]:
    labels = measure.scorer.labels_for(offered or measure.scorer.candidates())
    rest = (1.0 - 0.9) / (len(labels) - 1)
    return {label: _lp(0.9 if name == winner else rest) for name, label in labels.items()}


def test_scorer_mode_writes_the_same_predictions_file(measure, metrics, tmp_path):
    entries = [
        _entry("op2", "s2", {"operation": "container_restart", "args": {"container": "trainer"}}),
        _entry("esc1", "s3", {"escalate": True}),
        _entry("op3", "s4", {"operation": "service_logs", "args": {"service": "nginx.service"}}),
    ]
    entries[0]["text"] = "restart the trainer container"
    entries[2]["text"] = "show me the logs"  # names no service: cannot be grounded
    split = tmp_path / "splits" / "val.json"
    split.parent.mkdir(parents=True)
    header = "Fixture corpus. Split 'val' of fixture.json (seed=39)."
    split.write_text(json.dumps({"header": header, "entries": entries}), encoding="utf-8")

    def pick(prompt: str) -> dict[str, float]:
        request = json.loads(prompt)[-1]["content"]  # the fake render is json.dumps(messages)
        if "trainer" in request:
            return _label_logprobs(measure, "container_restart")
        if "logs" in request:
            return _label_logprobs(measure, "service_logs")
        return _label_logprobs(measure, "escalate")

    fake = _FakeScorer(pick)
    harness = Harness(measure, tmp_path, FakeDocker(), lfm={"engine": "vllm", "mode": "attach"})
    built: list = []
    _scorer_seams(measure, harness, fake, built)
    predictions = tmp_path / "p"
    out = tmp_path / "r.md"
    argv = _argv(split, out, "--scorer", "served", "--max-logprobs", "24", models=(STOCK,))
    argv += ["--predictions", str(predictions), "--ground-snapshot", str(_snapshot(tmp_path))]
    assert measure.main(argv, seams=harness.seams) == 0
    assert harness.specs == []  # no LfmTier in scorer mode
    rows = _prediction_lines(_predictions_files(predictions)[0])
    for row in rows:
        metrics.Prediction.from_dict(row)
    op2, esc1, op3 = rows
    assert (op2["outcome"], op2["operation"], op2["arguments"]) == (
        "propose",
        "container_restart",
        {"container": "trainer"},
    )
    assert op2["candidates"]["container_restart"] == pytest.approx(0.9)
    assert set(op2["candidates"]) >= {"(escalate)", "(explain)", "gpu_stats"}
    assert esc1["outcome"] == "escalate"
    assert op3["outcome"] == "invalid"
    assert op3["invalid_reason"] == "not_grounded"
    assert all(row["tokens"] == 0 for row in rows)
    assert fake.tops == [len(measure.scorer.candidates()) + measure.scorer.TOP_MARGIN] * 3
    text = out.read_text(encoding="utf-8")
    assert "candidate scorer (served)" in text
    assert "max-logprobs 24" in text
    assert "## Issue 46 metrics" in text


def test_served_scorer_needs_enough_max_logprobs(measure, tmp_path, capsys):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker(), lfm={"engine": "vllm", "mode": "attach"})
    built: list = []
    _scorer_seams(measure, harness, _FakeScorer(lambda _p: {}), built)
    for extra in ([], ["--max-logprobs", "20"]):
        argv = _argv(split, tmp_path / "r.md", "--scorer", "served", *extra, models=(STOCK,))
        assert measure.main(argv, seams=harness.seams) == 1
        assert "max-logprobs" in capsys.readouterr().err
    assert built == []


def test_served_scorer_refused_on_a_managed_launch(measure, tmp_path, capsys):
    split = _split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    built: list = []
    _scorer_seams(measure, harness, _FakeScorer(lambda _p: {}), built)
    argv = _argv(split, tmp_path / "r.md", "--scorer", "served", "--max-logprobs", "24")
    assert measure.main(argv, seams=harness.seams) == 2
    assert "max-logprobs" in capsys.readouterr().err
    assert built == []


def test_scorer_offers_only_the_slice_candidates(measure, tmp_path):
    split = _small_split(tmp_path)
    fake = _FakeScorer(lambda _p: {})
    harness = Harness(measure, tmp_path, FakeDocker())
    built: list = []
    _scorer_seams(measure, harness, fake, built)
    predictions = tmp_path / "p"
    argv = _argv(split, tmp_path / "r.md", "--scorer", "in-process", "--slice", "missing-candidate")
    argv += ["--predictions", str(predictions)]
    assert measure.main(argv, seams=harness.seams) == 0
    (prompt,) = fake.prompts[:1]
    assert "gpu_stats" not in prompt
    assert "thermal_stats" in prompt
    rows = _prediction_lines(_predictions_files(predictions)[0])
    assert rows[0]["outcome"] == "invalid"  # no label had mass
    assert rows[0]["candidates"] is None


# ---------------------------------------------------------------------------
# Issue 46 finding: a crashed/unreachable server must not produce a
# plausible-looking results page -- preflight before the first entry, and
# refuse the results page when any prediction is a tier_error.
# ---------------------------------------------------------------------------


class _ModelsHandler(http.server.BaseHTTPRequestHandler):
    status = 200
    ids: list[str] = ["good-model"]
    #: ``max_model_len`` reported for every id; ``None`` omits the field.
    max_model_len: int | None = None
    #: llama-server's shape (issue 46, t25): ``owned_by`` "llamacpp" and the
    #: context in ``GET /props``; ``None`` serves no /props at all.
    owned_by: str | None = None
    props_n_ctx: int | None = None
    props_redirect = False

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming convention
        if self.path.rstrip("/") == "/props" and _ModelsHandler.props_redirect:
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.2:9/props")
            self.end_headers()
            return
        if self.path.rstrip("/") == "/props" and _ModelsHandler.props_n_ctx is not None:
            body = {"default_generation_settings": {"n_ctx": _ModelsHandler.props_n_ctx}}
            data = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.rstrip("/") not in ("/models", "/v1/models"):
            self.send_response(404)
            self.end_headers()
            return
        entries = []
        for i in _ModelsHandler.ids:
            entry: dict = {"id": i}
            if _ModelsHandler.max_model_len is not None:
                entry["max_model_len"] = _ModelsHandler.max_model_len
            if _ModelsHandler.owned_by is not None:
                entry["owned_by"] = _ModelsHandler.owned_by
            entries.append(entry)
        data = json.dumps({"data": entries}).encode("utf-8")
        self.send_response(_ModelsHandler.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def models_server():
    _ModelsHandler.status = 200
    _ModelsHandler.ids = ["good-model"]
    _ModelsHandler.max_model_len = None
    _ModelsHandler.owned_by = None
    _ModelsHandler.props_n_ctx = None
    _ModelsHandler.props_redirect = False
    server = http.server.HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _models_url(server: http.server.HTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_preflight_accepts_a_server_serving_the_model(measure, models_server):
    measure.preflight_models(_models_url(models_server), "good-model")  # does not raise


def test_preflight_refuses_the_wrong_model_name(measure, models_server):
    with pytest.raises(measure.MeasureError) as excinfo:
        measure.preflight_models(_models_url(models_server), "other-model")
    assert excinfo.value.code == measure.EXIT_ENV
    assert "other-model" in excinfo.value.message


def test_preflight_refuses_a_wrong_http_status(measure, models_server):
    _ModelsHandler.status = 500
    with pytest.raises(measure.MeasureError) as excinfo:
        measure.preflight_models(_models_url(models_server), "good-model")
    assert excinfo.value.code == measure.EXIT_ENV
    assert "500" in excinfo.value.message


def test_preflight_refuses_a_refused_connection(measure):
    with pytest.raises(measure.MeasureError) as excinfo:
        measure.preflight_models("http://127.0.0.1:1", "good-model")
    assert excinfo.value.code == measure.EXIT_ENV


def test_a_tier_error_prediction_fails_the_run_and_writes_no_results_page(measure, tmp_path):
    split = _small_split(tmp_path)
    out = tmp_path / "r.md"
    predictions = tmp_path / "p"
    harness = Harness(
        measure,
        tmp_path,
        FakeDocker(),
        scripts={STOCK: [Decline(DeclineReason.TIER_ERROR, "server dropped"), _call("escalate")]},
    )
    argv = _argv(split, out, "--predictions", str(predictions), models=(STOCK,))
    assert measure.main(argv, seams=harness.seams) == 2
    assert not out.exists()
    # kept for debugging even though the results page is refused
    files = _predictions_files(predictions)
    assert len(files) == 1
    rows = _prediction_lines(files[0])
    assert any(row["invalid_reason"] == "tier_error" for row in rows)


def test_allow_tier_errors_writes_the_page_with_the_count(measure, tmp_path):
    split = _small_split(tmp_path)
    out = tmp_path / "r.md"
    harness = Harness(
        measure,
        tmp_path,
        FakeDocker(),
        scripts={STOCK: [Decline(DeclineReason.TIER_ERROR, "server dropped"), _call("escalate")]},
    )
    argv = _argv(split, out, "--allow-tier-errors", "1", models=(STOCK,))
    assert measure.main(argv, seams=harness.seams) == 0
    text = out.read_text(encoding="utf-8")
    assert "1 tier-error prediction(s) permitted" in text
    assert "--allow-tier-errors 1" in text


def test_allow_tier_errors_rejects_a_negative_value(measure, tmp_path, capsys):
    split = _small_split(tmp_path)
    harness = Harness(measure, tmp_path, FakeDocker())
    argv = _argv(split, tmp_path / "r.md", "--allow-tier-errors", "-1", models=(STOCK,))
    assert measure.main(argv, seams=harness.seams) == 1
    assert "--allow-tier-errors" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Issue 46 (lead follow-up): Track A's preflight/tier-error rules apply to
# Track B's served scorer (score_one) too -- a dead or wrong server there
# must fail the same way, never a plausible-looking "no label mass" run.
# ---------------------------------------------------------------------------


class _ScorerHandler(http.server.BaseHTTPRequestHandler):
    model_ids: list[str] = [STOCK]
    completions_status = 200
    #: One {token: logprob} dict per POST /completions call, in order.
    logprobs_by_call: list[dict[str, float]] = []
    calls = 0

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming convention
        if self.path.rstrip("/") != "/models":
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(
            {
                "data": [
                    # vLLM reports each served model's max_model_len (lapse l3 check).
                    {"id": model_id, "max_model_len": 4096}
                    for model_id in _ScorerHandler.model_ids
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming convention
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        index = _ScorerHandler.calls
        _ScorerHandler.calls += 1
        if _ScorerHandler.completions_status != 200:
            self.send_response(_ScorerHandler.completions_status)
            self.end_headers()
            return
        logprobs = (
            _ScorerHandler.logprobs_by_call[index]
            if index < len(_ScorerHandler.logprobs_by_call)
            else {}
        )
        payload = {"choices": [{"logprobs": {"top_logprobs": [logprobs]}}]}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def scorer_server():
    _ScorerHandler.model_ids = [STOCK]
    _ScorerHandler.completions_status = 200
    _ScorerHandler.logprobs_by_call = []
    _ScorerHandler.calls = 0
    server = http.server.HTTPServer(("127.0.0.1", 0), _ScorerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _served_scorer_harness(measure, tmp_path, server) -> tuple[Harness, list]:
    """A real ToolChat-backed ScorerHandle talking to *server* (issue 46 preflight/tier-error)."""
    from nvsh.tiers import toolchat

    harness = Harness(measure, tmp_path, FakeDocker(), lfm={"engine": "vllm", "mode": "attach"})
    url = f"http://127.0.0.1:{server.server_address[1]}"
    built: list = []

    def build_scorer(spec):
        built.append(spec)
        chat = toolchat.ToolChat(url, spec.model, stream=False)
        return measure.ScorerHandle(
            scorer=chat,
            render=lambda messages: json.dumps(messages),
            close=lambda: None,
            base_url=url,
        )

    harness.seams.build_scorer = build_scorer
    # Restore the real preflight for these tests -- the default Harness fake
    # always passes, but this is exactly what is under test here.
    harness.seams.preflight = measure.preflight_models
    return harness, built


def test_served_scorer_preflight_refuses_the_wrong_model_name(measure, tmp_path, scorer_server):
    _ScorerHandler.model_ids = ["a-different-model"]
    split = _small_split(tmp_path)
    harness, built = _served_scorer_harness(measure, tmp_path, scorer_server)
    argv = _argv(
        split, tmp_path / "r.md", "--scorer", "served", "--max-logprobs", "24", models=(STOCK,)
    )
    assert measure.main(argv, seams=harness.seams) == 2
    assert not (tmp_path / "r.md").exists()
    assert _ScorerHandler.calls == 0  # refused before the first scored entry


def test_served_scorer_call_error_fails_the_run_and_writes_no_results_page(
    measure, tmp_path, scorer_server
):
    _ScorerHandler.completions_status = 500
    split = _small_split(tmp_path)
    out = tmp_path / "r.md"
    predictions = tmp_path / "p"
    harness, built = _served_scorer_harness(measure, tmp_path, scorer_server)
    argv = _argv(split, out, "--scorer", "served", "--max-logprobs", "24", models=(STOCK,))
    argv += ["--predictions", str(predictions)]
    assert measure.main(argv, seams=harness.seams) == 2
    assert not out.exists()
    files = _predictions_files(predictions)
    assert len(files) == 1
    rows = _prediction_lines(files[0])
    assert all(row["invalid_reason"] == "tier_error" for row in rows)


def test_served_scorer_healthy_run_is_not_a_tier_error(measure, tmp_path, scorer_server):
    """A real round trip that answers every label cleanly writes the results page."""
    split = _small_split(tmp_path)
    out = tmp_path / "r.md"
    harness, built = _served_scorer_harness(measure, tmp_path, scorer_server)
    full = _label_logprobs(measure, "escalate")
    _ScorerHandler.logprobs_by_call = [full, full]
    argv = _argv(split, out, "--scorer", "served", "--max-logprobs", "24", models=(STOCK,))
    assert measure.main(argv, seams=harness.seams) == 0
    assert out.exists()


# ---------------------------------------------------------------------------
# served context check (issue 46, lapse l3: a "4K" run was served at 2048
# while its report said ctx=4096)
# ---------------------------------------------------------------------------


def test_preflight_accepts_a_matching_served_context(measure, models_server):
    _ModelsHandler.max_model_len = 4096
    measure.preflight_models(_models_url(models_server), "good-model", ctx=4096)


def test_preflight_refuses_a_different_served_context(measure, models_server):
    _ModelsHandler.max_model_len = 2048
    with pytest.raises(measure.MeasureError) as excinfo:
        measure.preflight_models(_models_url(models_server), "good-model", ctx=4096)
    assert excinfo.value.code == measure.EXIT_ENV
    assert "2048" in excinfo.value.message and "4096" in excinfo.value.message


def test_preflight_refuses_when_the_served_context_cannot_be_read(measure, models_server):
    _ModelsHandler.max_model_len = None
    with pytest.raises(measure.MeasureError) as excinfo:
        measure.preflight_models(_models_url(models_server), "good-model", ctx=4096)
    assert "max_model_len" in excinfo.value.message


@pytest.mark.parametrize("suffix", ["", "/v1"])
def test_preflight_reads_a_llama_server_context_from_props(measure, models_server, suffix):
    """Issue 46, t25: llama-server's /v1/models has no max_model_len; its
    context is /props default_generation_settings.n_ctx at the server root."""
    _ModelsHandler.owned_by = "llamacpp"
    _ModelsHandler.props_n_ctx = 2048
    measure.preflight_models(_models_url(models_server) + suffix, "good-model", ctx=2048)


def test_preflight_refuses_a_llama_server_with_another_context(measure, models_server):
    _ModelsHandler.owned_by = "llamacpp"
    _ModelsHandler.props_n_ctx = 4096
    with pytest.raises(measure.MeasureError) as excinfo:
        measure.preflight_models(_models_url(models_server) + "/v1", "good-model", ctx=2048)
    assert "4096" in excinfo.value.message and "2048" in excinfo.value.message


def test_preflight_refuses_a_llama_server_without_props(measure, models_server):
    _ModelsHandler.owned_by = "llamacpp"
    with pytest.raises(measure.MeasureError) as excinfo:
        measure.preflight_models(_models_url(models_server) + "/v1", "good-model", ctx=2048)
    assert "n_ctx" in excinfo.value.message or "max_model_len" in excinfo.value.message


def test_a_redirected_props_answer_is_refused(measure, models_server):
    """Codex: a redirect could carry the /props answer off localhost."""
    _ModelsHandler.owned_by = "llamacpp"
    _ModelsHandler.props_redirect = True
    with pytest.raises(measure.MeasureError):
        measure.preflight_models(_models_url(models_server) + "/v1", "good-model", ctx=2048)


def test_props_are_not_consulted_for_a_server_that_is_not_llama_cpp(measure, models_server):
    _ModelsHandler.props_n_ctx = 2048  # present, but the server does not say it is llama.cpp
    with pytest.raises(measure.MeasureError):
        measure.preflight_models(_models_url(models_server), "good-model", ctx=2048)


def test_preflight_without_ctx_is_unchanged(measure, models_server):
    _ModelsHandler.max_model_len = None
    measure.preflight_models(_models_url(models_server), "good-model")  # does not raise


def test_served_ctx_defaults_to_what_the_report_shows(measure) -> None:
    # Codex review of lapse l3: no ctx in the config must still be checked.
    assert measure._served_ctx({}) == measure.DEFAULT_CTX
    assert measure._served_ctx({"ctx": 2048}) == 2048


# ---------------------------------------------------------------------------
# Track B failures surface; the tokenizer comes from a path (issue 46, t23)
# ---------------------------------------------------------------------------


def test_scorer_failure_reason_reaches_stderr(measure, tmp_path, capsys):
    split = _small_split(tmp_path)
    fake = _FakeScorer(lambda prompt: {})
    built: list = []
    harness = Harness(measure, tmp_path, FakeDocker(), lfm={"engine": "vllm", "mode": "attach"})
    _scorer_seams(measure, harness, fake, built)

    def failing(spec):
        raise ImportError("No module named 'transformers'")

    harness.seams.build_scorer = failing
    argv = _argv(split, tmp_path / "r.md", "--scorer", "in-process", models=(STOCK,))
    assert measure.main(argv, seams=harness.seams) == 2
    err = capsys.readouterr().err
    assert "scorer start-up failed" in err and "transformers" in err


def test_tokenizer_option_reaches_the_scorer_spec(measure, tmp_path):
    split = _small_split(tmp_path)
    full = _label_logprobs(measure, "escalate")
    fake = _FakeScorer(lambda prompt: full)
    built: list = []
    harness = Harness(measure, tmp_path, FakeDocker(), lfm={"engine": "vllm", "mode": "attach"})
    _scorer_seams(measure, harness, fake, built)
    argv = _argv(split, tmp_path / "r.md", "--scorer", "in-process", models=(STOCK,))
    argv += ["--tokenizer", "/models/merged"]
    measure.main(argv, seams=harness.seams)
    assert built and built[0].tokenizer == "/models/merged"


@pytest.mark.parametrize(
    ("label", "ok"),
    [
        ("final-a3.q4_k_m", True),
        ("val-350m", True),
        ("Final-A3", False),
        ("_x", False),
        ("a b", False),
    ],
)
def test_label_allows_a_quantized_build_name(measure, tmp_path, label, ok):
    """Issue 46, t25: a GGUF build is named <run>.q4_k_m, so its labels carry '_'."""
    split = _split(tmp_path, "val.json")
    harness = Harness(measure, tmp_path, FakeDocker())
    argv = _argv(split, tmp_path / "r.md")
    argv[argv.index("--label") + 1] = label
    assert (measure.main(argv, seams=harness.seams) == 0) is ok
