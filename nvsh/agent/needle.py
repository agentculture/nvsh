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
from typing import TYPE_CHECKING, Callable, Iterator, Mapping

from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind, NvshAgent

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
#: :meth:`NeedleAgent.steer` always returns ``False``).
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

#: One-line, human-readable text for a :class:`~nvsh.tiers.base.DeclineReason`
#: value the router reported (``TierOutcome.declines`` only carries the enum,
#: not the tier's own detail string -- see :meth:`NeedleAgent._no_answer_text`).
#: ``tier_unavailable`` is handled separately, from :meth:`NeedleTier.status`,
#: because that is the one reason whose detail text nvsh already has to hand.
_DECLINE_TEXT: dict[str, str] = {
    "no_call": "needle had no operation to propose for this request",
    "multiple_calls": "needle proposed more than one operation for this request",
    "unknown_operation": "needle proposed an operation nvsh does not know",
    "bad_argument": "needle's proposed arguments did not validate",
    "raw_shell": "needle proposed a raw shell command, which nvsh refuses to run",
    "malformed_output": "needle's output could not be understood",
    "tier_error": "needle failed while answering",
    "low_confidence": "needle was not confident enough in its answer",
    "memory_floor": "needle was skipped: not enough free memory on this machine",
    "not_grounded": "needle's proposed operation could not be grounded on this machine",
    "loop_limit": "needle hit its loop limit",
    "not_renderable": "needle's proposed operation has no single command on this platform",
}

#: Explanation given when Tier 1 was never even consulted: a FAILURE-kind
#: request (a command that exited non-zero) is not the instruction-shaped ask
#: Needle selects from (see ``nvsh/tiers/router.py``'s ``Route._order``).
_FAILURE_TEXT = "needle only answers instruction-shaped requests, not command failures"


class NeedleAgent(NvshAgent):
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
        self._router = None  # a nvsh.tiers.router.TierRouter, built in start()
        self._tier = None  # the same object as self._router.tier1, kept for close()/status()
        self._cancelled = False

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

    def close(self) -> None:
        """Kill the Tier 1 child, if one is running. Idempotent."""
        self._close_tier()

    # -- one turn --

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        """Route *request* through Tier 1 only. Never reaches the full agent:
        an escalation is turned into one explanatory event, then DONE."""
        self._cancelled = False
        return self._turn(request, context)

    def _turn(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        self.start()
        route = self._router.route(request, context)
        for event in route:
            if self._cancelled:
                return
            yield event
        # Route.outcome is only assigned *after* its generator's last yield
        # (see nvsh/tiers/router.py's Route._run/_proposed/_explained), so it
        # must not be read until the for-loop above has fully drained the
        # generator -- reading it inside the loop, right after a yield,
        # would still see the outcome from the *previous* turn.
        outcome = route.outcome
        if outcome is not None and outcome.escalated_to is not None:
            # Tier 1 declined (or a FAILURE-kind request never reached it):
            # the router's own generator ends here with no DONE, because a
            # normal caller would go on to the full agent. This adapter
            # never does that -- it is what the operator explicitly asked
            # for -- so it closes the turn itself instead.
            yield AgentEvent(
                kind=EventKind.STATUS,
                text=self._no_answer_text(outcome),
                args={"tier": "needle"},
            )
            yield AgentEvent(kind=EventKind.DONE, args={"tier": "needle"})

    def _no_answer_text(self, outcome) -> str:
        if not outcome.declines:
            return _FAILURE_TEXT
        _tier_name, reason = outcome.declines[-1]
        value = reason.value
        if value == "tier_unavailable":
            return self._tier.status() if self._tier is not None else "needle tier unavailable"
        return _DECLINE_TEXT.get(value, f"needle declined: {value}")

    def steer(self, text: str) -> bool:
        """Never -- a Tier 1 selection is one request/response, not an open
        turn a correction could land in."""
        del text
        return False

    def cancel(self) -> None:
        """Stop the in-flight selection by killing the Tier 1 child.

        There is no cooperative interrupt point inside a blocking
        ``select()`` on the worker pipe (see ``nvsh/tiers/needle.py``), so
        "ask it to stop" and "kill it" are the same operation here -- the
        next request gets a fresh child (:class:`~nvsh.tiers.needle.NeedleTier`
        respawns lazily).
        """
        self._cancelled = True
        self._close_tier()

    def force_stop(self) -> None:
        """Same as :meth:`cancel`: there is nothing gentler to escalate from."""
        self.cancel()

    def _close_tier(self) -> None:
        tier = self._tier
        if tier is not None:
            tier.close()

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
