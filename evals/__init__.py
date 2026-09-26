"""evals — DeepEval-based release-gate evaluations for nvsh.

This package is NOT part of the ``nvsh`` runtime distribution: it lives
outside the ``nvsh/`` package, is never imported by it, and is not listed
in ``[project.dependencies]`` or packaged into the wheel (see
``pyproject.toml``'s ``[tool.hatch.build.targets.wheel]``). Its only
dependency, ``deepeval``, lives in the ``evals`` dependency group, which
is not a default group — a plain ``uv sync`` never installs it.
"""

from __future__ import annotations
