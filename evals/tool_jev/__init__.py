"""tool_jev — DeepEval release-gate evaluation suite for the Tool-Jev harness.

Importing this package is the single choke point every ``evals.tool_jev.*``
module and test must go through before ``deepeval`` itself is imported,
because ``deepeval`` registers a ``pytest11`` entry point
(``deepeval.plugins.plugin``) that starts telemetry and reads its config at
*import* time, not at pytest-collection time. Two environment variables have
to be set before that first import happens anywhere in the process:

- ``DEEPEVAL_TELEMETRY_OPT_OUT=1`` — disable deepeval's telemetry pings.
- ``DEEPEVAL_DISABLE_DOTENV=1`` — stop deepeval from reading a stray
  ``.env`` file for its own config (nvsh's env/config discipline lives in
  ``nvsh/config.py`` and ``$XDG_CONFIG_HOME/nvsh/``; deepeval's ``.env``
  auto-loading is unrelated and unwanted here).

Both are set as soon as this module is imported, and only if the caller has
not already set them (an operator's own explicit value always wins).

This package also refuses to run at all if ``CONFIDENT_API_KEY`` is set in
the environment: that variable opts deepeval's Confident AI integration into
uploading eval results to a hosted, non-local service, which has no place in
an offline-first CI gate that runs with no secrets configured.
"""

from __future__ import annotations

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")

if os.environ.get("CONFIDENT_API_KEY"):
    raise RuntimeError(
        "evals.tool_jev refuses to run with CONFIDENT_API_KEY set: this "
        "suite is offline-first and must never upload results to Confident "
        "AI. Unset CONFIDENT_API_KEY before running these evals."
    )
