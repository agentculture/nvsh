"""The stock-versus-tuned Tier 2 measurement script: scripts/lfm-finetune/measure.py.

Everything that would touch the machine -- docker, nvidia-smi, git, the Tier 2
container and model -- goes through the script's ``Seams``; these tests pass a
fake docker runner and a scripted :class:`~nvsh.tiers.fake.FakeTier`, so no
container is launched, no model is loaded and nothing reaches the network.
Results files are written under ``tmp_path`` via ``--out``, never into the repo.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
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
