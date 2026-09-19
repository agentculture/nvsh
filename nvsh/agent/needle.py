"""``needle``: the explicit Tier 1 (Needle3) adapter (task t14).

Every other adapter in this package drives a real harness/CLI. ``needle`` is
different: it wraps :class:`nvsh.tiers.router.TierRouter`, restricted to
Tier 1 only (``tier2=None``, no verifier), so an operator who explicitly asks
for ``@needle`` (or ``nvsh --agent needle``) gets exactly the same
tool-selecting local model the daemon's tier ladder would try first -- and
nothing else. It never falls through to the full agent on its own: an
explicit ``@needle`` request that Tier 1 cannot answer is not silently
upgraded to a real harness call (that would defeat the point of asking for
Needle specifically); instead this adapter explains, in one line, why it had
no answer, and ends the turn there. Escalating past a *failed* command is a
different door entirely (``nvsh``'s normal failure flow, or a plain
``@<harness>``), not this one.

The turn loop, decline-text formatting and lifecycle this adapter shares
with :class:`~nvsh.agent.lfm.LfmAgent` (task t19) live in
:mod:`nvsh.agent._tier_adapter`; this module only builds the Tier-1-only
router and fills in the four hooks that module calls into.

``nvsh.tiers`` is a large, native-adjacent subtree (child processes,
zipfile/urllib prefetch, ctypes-loaded engines) that must never load on the
hot success path (CLAUDE.md's stdlib-only import-time contract). So, like
every lazy adapter factory in :mod:`nvsh.agent.registry`, this module keeps
``__init__`` to plain attribute assignment -- no tier, no router, no
subprocess -- and imports ``nvsh.tiers`` only from inside :meth:`start` and
the handful of helpers :meth:`run` calls into.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

from ._tier_adapter import TierOnlyAdapter
from .base import Capabilities

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from ..platform._model import Platform

#: What this adapter reports (see :class:`~nvsh.agent.base.Capabilities`'s
#: docstring for each field): it streams, its PROPOSAL goes through nvsh's
#: own approve loop (the same reading ``pi``/``claude``/``agy`` give for
#: ``tool_calling=True, approval='nvsh'``), it needs nothing off this
#: machine (``local_model=True``), it reads no file on its own
#: (``unmediated_file_access=False``), and it has no mid-turn steering
#: channel -- a Tier 1 selection is one request/response round trip, not an
#: open conversation (``steer=False``, matching the design constraint that
#: :class:`~nvsh.agent._tier_adapter.TierOnlyAdapter`'s ``steer`` always
#: returns ``False``).
NEEDLE_CAPABILITIES = Capabilities(
    streaming=True,
    tool_calling=True,
    cancellation=True,
    persistent_session=True,
    local_model=True,
    thinking=False,
    effort=False,
    path="inproc",
    approval="nvsh",
    unmediated_file_access=False,
    steer=False,
)

#: Explanation given when Tier 1 was never even consulted: a FAILURE-kind
#: request (a command that exited non-zero) is not the instruction-shaped ask
#: Needle selects from (see ``nvsh/tiers/router.py``'s ``Route._order``).
_FAILURE_TEXT = "needle only answers instruction-shaped requests, not command failures"


#: The tier name the router puts on its hand-off-to-the-full-agent status.
_FULL_AGENT = "agent"


class NeedleAgent(TierOnlyAdapter):
    """The explicit ``@needle`` adapter: Tier 1 only, propose-or-explain.

    ``config`` is the ``[tiers]`` table (:attr:`nvsh.config.Config.tiers`),
    not ``[agents.needle]`` -- Tier 1 has no harness-style settings
    (``model``, ``effort``, ...) of its own, only the knobs already defined
    for the tier ladder (``needle_min_confidence``, ``memory_floor_mb``,
    ``records_cap_mb``, ``store_request_text``). The remaining keyword-only
    arguments exist so tests can drive this adapter against
    ``tests/fakes/needle_worker`` without ``cactus-needle`` installed and
    without writing to a real records path.
    """

    def __init__(
        self,
        config: Mapping[str, object] | None = None,
        *,
        worker_argv: list[str] | None = None,
        weights_path: str | None = None,
        tier_env: Mapping[str, str] | None = None,
        records_path: Path | None = None,
        availability: Callable[[], str | None] | None = None,
        floor_reader: Callable[[], str] | None = None,
        runner: Callable[[list[str], float], tuple[int, str]] | None = None,
        platform: "Platform | None" = None,
    ) -> None:
        """Cheap: only plain attributes. Spawns nothing, imports nothing heavy."""
        super().__init__()
        settings = dict(config or {})
        self._min_confidence = _as_float(settings.get("needle_min_confidence"), 0.0)
        self._memory_floor_mb = _as_int(settings.get("memory_floor_mb"), 1024)
        self._records_cap_mb = _as_int(settings.get("records_cap_mb"), 8)
        self._store_request_text = bool(settings.get("store_request_text", False))
        self._worker_argv = list(worker_argv) if worker_argv is not None else None
        self._weights_path = weights_path
        self._tier_env = dict(tier_env) if tier_env is not None else None
        self._records_path = records_path
        self._availability = availability
        self._floor_reader = floor_reader
        self._runner = runner
        self._platform = platform
        self._tier = None  # the same object as self._router.tier1, kept for close()/status()

    # -- lifecycle --

    def start(self) -> None:
        """Build the Tier-1-only router. Idempotent; spawns no child yet --
        the worker only starts on the first :meth:`run`."""
        if self._router is not None:
            return
        from ..tiers.memfloor import check_floor, default_reader
        from ..tiers.needle import NeedleTier
        from ..tiers.records import TierRecords
        from ..tiers.router import TierRouter

        reader = self._floor_reader or default_reader
        floor_mb = self._memory_floor_mb

        def floor_check():
            return check_floor(floor_mb, reader)

        tier_kwargs: dict[str, object] = {
            "min_confidence": self._min_confidence,
            "floor_check": floor_check,
        }
        if self._worker_argv is not None:
            tier_kwargs["worker_argv"] = self._worker_argv
        if self._weights_path is not None:
            tier_kwargs["weights_path"] = self._weights_path
        if self._tier_env is not None:
            tier_kwargs["env"] = self._tier_env
        if self._availability is not None:
            tier_kwargs["availability"] = self._availability

        self._tier = NeedleTier(**tier_kwargs)
        records = TierRecords(
            path=self._records_path,
            cap_bytes=self._records_cap_mb * 1024 * 1024,
            store_request_text=self._store_request_text,
        )
        platform = self._platform
        if platform is None:
            from ..platform import detect as detect_platform

            platform = detect_platform()
        router_kwargs: dict[str, object] = {"min_confidence": self._min_confidence}
        if self._runner is not None:
            router_kwargs["runner"] = self._runner
        self._router = TierRouter(
            self._tier,
            None,
            records,
            platform,
            **router_kwargs,
        )

    # -- TierOnlyAdapter hooks --

    def _tier_label(self) -> str:
        return "needle"

    def _failure_text(self) -> str:
        return _FAILURE_TEXT

    def _unavailable_text(self) -> str:
        return self._tier.status() if self._tier is not None else "needle tier unavailable"

    def _active_tier(self) -> object | None:
        return self._tier

    def capabilities(self) -> Capabilities:
        return NEEDLE_CAPABILITIES


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)
