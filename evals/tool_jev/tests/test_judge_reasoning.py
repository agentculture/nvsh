"""A judge may set its own reasoning effort (deviation d8, 2026-09-29).

qwen3.8-max as a judge truncated 19% of its replies at medium reasoning;
the output cap and effort are shared, and changing them for every judge
would re-key already-paid judge batches.
"""

from __future__ import annotations

import tomllib
from types import SimpleNamespace

import pytest

from evals.tool_jev import run as runner
from evals.tool_jev.manifest import ManifestError, parse_manifest
from evals.tool_jev.tests.test_manifest import _BASE_MANIFEST

_JUDGE = '[[judge]]\nprovider = "anthropic"\nmodel = "claude-opus-5-5"\n'


def _with_judge_reasoning(value: str) -> str:
    return _BASE_MANIFEST.replace(_JUDGE, _JUDGE + f'reasoning = "{value}"\n')


def test_a_judge_without_reasoning_follows_its_reference():
    manifest = parse_manifest(tomllib.loads(_BASE_MANIFEST))
    assert manifest.judges[0].reasoning is None


def test_a_judge_can_set_its_own_reasoning():
    manifest = parse_manifest(tomllib.loads(_with_judge_reasoning("low")))
    assert manifest.judges[0].reasoning == "low"


def test_a_judge_reasoning_must_be_a_known_level():
    with pytest.raises(ManifestError, match="reasoning"):
        parse_manifest(tomllib.loads(_with_judge_reasoning("extreme")))


def _judge_params(judge_reasoning, ref_reasoning="medium"):
    ref = SimpleNamespace(
        provider="openrouter", model="qwen/qwen3.8-max-0902", reasoning=ref_reasoning
    )
    judge = SimpleNamespace(
        provider="openrouter", model="qwen/qwen3.8-max-0902", reasoning=judge_reasoning
    )
    manifest = SimpleNamespace(judging=SimpleNamespace(max_output_tokens=1024), judges=(judge,))
    r = object.__new__(runner.Runner)
    r.plan = SimpleNamespace(manifest=manifest)
    return r._judge_params(SimpleNamespace(ref=ref))


def test_judge_params_use_the_judges_own_reasoning():
    assert _judge_params("low") == {"max_output_tokens": 1024, "reasoning": "low"}


def test_judge_params_default_to_the_references_reasoning():
    assert _judge_params(None) == {"max_output_tokens": 1024, "reasoning": "medium"}
