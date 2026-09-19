"""Shared skeleton for the Tier-only explicit adapters (``needle``, ``lfm``).

Both :class:`~nvsh.agent.needle.NeedleAgent` and :class:`~nvsh.agent.lfm.LfmAgent`
wrap a single-tier :class:`~nvsh.tiers.router.TierRouter` and answer
propose-or-explain, never falling through to the full agent on their own: an
explicit ``@needle``/``@lfm`` request that its tier cannot answer is not
silently upgraded to a real harness call -- this adapter explains, in one
line, why it had no answer, and ends the turn there.

The two adapters' turn loop, decline-text formatting and lifecycle
(``steer``/``cancel``/``force_stop``/``close``) are otherwise identical, so
they live here once rather than being copied (SonarCloud flags duplication
past a few dozen lines): a subclass only builds its own single-tier router
in :meth:`~nvsh.agent.base.NvshAgent.start` and fills in the four hooks
below.
"""

from __future__ import annotations

from typing import Iterator

from .base import AgentContext, AgentEvent, AgentRequest, EventKind, NvshAgent

#: The tier name the router puts on its hand-off-to-the-full-agent status.
#: Dropped from the stream by :meth:`TierOnlyAdapter._turn`: an explicit
#: request to one of these adapters never actually reaches the full agent,
#: so echoing the router's "asking the full agent" line would be untrue.
FULL_AGENT = "agent"

#: One-line decline text, shared by every Tier-only adapter and filled in
#: with the asking tier's own name. Keys are
#: :class:`~nvsh.tiers.base.DeclineReason` values; ``tier_unavailable`` is
#: handled separately (each adapter's own ``_unavailable_text``), because
#: that is the one reason whose detail text the tier itself already has to
#: hand.
DECLINE_TEXT: dict[str, str] = {
    "no_call": "{tier} had no operation to propose for this request",
    "multiple_calls": "{tier} proposed more than one operation for this request",
    "unknown_operation": "{tier} proposed an operation nvsh does not know",
    "bad_argument": "{tier}'s proposed arguments did not validate",
    "raw_shell": "{tier} proposed a raw shell command, which nvsh refuses to run",
    "malformed_output": "{tier}'s output could not be understood",
    "tier_error": "{tier} failed while answering",
    "low_confidence": "{tier} was not confident enough in its answer",
    "memory_floor": "{tier} was skipped: not enough free memory on this machine",
    "not_grounded": "{tier}'s proposed operation could not be grounded on this machine",
    "loop_limit": "{tier} hit its loop limit",
    "not_renderable": "{tier}'s proposed operation has no single command on this platform",
    #: Tier 2's own third outcome (its ``escalate`` tool, nothing usable, or
    #: its round budget ran out) -- worded to avoid the router's own "asking
    #: the full agent" phrase, which this adapter never actually does.
    "escalated": "{tier} could not settle this within its inspection budget",
}


class TierOnlyAdapter(NvshAgent):
    """One tier, propose-or-explain, never falling through to the full agent.

    A subclass builds a single-tier :class:`~nvsh.tiers.router.TierRouter`
    in its own ``start()`` (assigning ``self._router``, or leaving it
    ``None`` when its tier is not configured at all -- see
    :meth:`_unrouted`), and implements four small hooks: :meth:`_tier_label`
    (the name used in every event and decline message), :meth:`_failure_text`
    and :meth:`_unavailable_text` (the two decline texts each tier's own
    router-ordering/availability rules produce), and :meth:`_active_tier`
    (the object :meth:`close`/:meth:`cancel` stop).
    """

    def __init__(self) -> None:
        self._router = None  # a nvsh.tiers.router.TierRouter, built in start()
        self._cancelled = False

    # -- one turn --

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        """Route *request* through this adapter's one tier. Never reaches the
        full agent: an escalation is turned into one explanatory event, then
        DONE."""
        self._cancelled = False
        return self._turn(request, context)

    def _turn(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        self.start()
        router = self._router
        if router is None:
            yield from self._unrouted()
            return
        route = router.route(request, context)
        for event in route:
            if self._cancelled:
                return
            if event.args.get("tier") == FULL_AGENT:
                # The router's "asking the full agent" line: untrue here,
                # because an explicit request never goes on to one.
                continue
            yield event
        # Route.outcome is only assigned *after* its generator's last yield
        # (see nvsh/tiers/router.py's Route._run/_proposed/_explained), so it
        # must not be read until the for-loop above has fully drained the
        # generator -- reading it inside the loop, right after a yield,
        # would still see the outcome from the *previous* turn.
        outcome = route.outcome
        if outcome is not None and outcome.escalated_to is not None:
            label = self._tier_label()
            yield AgentEvent(
                kind=EventKind.STATUS,
                text=self._no_answer_text(outcome),
                args={"tier": label},
            )
            yield AgentEvent(kind=EventKind.DONE, args={"tier": label})

    def _unrouted(self) -> Iterator[AgentEvent]:
        """Yielded when :meth:`start` built no router at all.

        Only reachable for an adapter whose tier can be entirely
        unconfigured (``lfm``, with no ``[tiers.lfm] model`` -- see
        :meth:`~nvsh.agent.lfm.LfmAgent._unrouted`); ``needle`` always builds
        a router in ``start()``, so its ``_turn`` never takes this branch.
        The default here is one bare DONE -- a subclass that can leave
        ``self._router`` unset overrides this to explain why first.
        """
        yield AgentEvent(kind=EventKind.DONE, args={"tier": self._tier_label()})

    def _no_answer_text(self, outcome) -> str:
        if not outcome.declines:
            return self._failure_text()
        _tier_name, reason = outcome.declines[-1]
        value = reason.value
        if value == "tier_unavailable":
            return self._unavailable_text()
        label = self._tier_label()
        template = DECLINE_TEXT.get(value)
        if template is not None:
            return template.format(tier=label)
        return f"{label} declined: {value}"

    # -- hooks each adapter fills in --

    def _tier_label(self) -> str:
        """This adapter's name, as it appears in every event and message."""
        raise NotImplementedError

    def _failure_text(self) -> str:
        """Text for a request whose router order held no tier at all."""
        raise NotImplementedError

    def _unavailable_text(self) -> str:
        """Text for a ``tier_unavailable`` decline (the tier's own detail)."""
        raise NotImplementedError

    def _active_tier(self) -> object | None:
        """The tier object :meth:`close`/:meth:`cancel` should stop, if any."""
        raise NotImplementedError

    # -- lifecycle/steering shared by both --

    def steer(self, text: str) -> bool:
        """Never -- one selection is one request/response, not an open turn
        a correction could land in."""
        del text
        return False

    def cancel(self) -> None:
        """Stop the in-flight selection by killing the tier's child/runtime.

        There is no cooperative interrupt point inside a tier's blocking
        round trip to its child/container, so "ask it to stop" and "kill it"
        are the same operation here -- the next request gets a fresh one
        (each tier respawns lazily).
        """
        self._cancelled = True
        self._close_tier()

    def force_stop(self) -> None:
        """Same as :meth:`cancel`: there is nothing gentler to escalate from."""
        self.cancel()

    def close(self) -> None:
        """Release the active tier, if one was built. Idempotent."""
        self._close_tier()

    def _close_tier(self) -> None:
        tier = self._active_tier()
        if tier is not None:
            tier.close()
