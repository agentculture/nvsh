"""Tier residency: what the daemon holds, when it builds it, when it drops it.

The tiers live in the per-user daemon so a model is loaded once for every
hooked shell instead of once per request -- but a daemon that loaded them at
*start* would pay for them on every machine, including the ones where no
operator ever enables the tiers. So this manager builds **nothing** when it
is constructed: the first tier request builds the router, its records file
and Tier 1; :meth:`TierManager.sweep` drops them again after
``[tiers] idle_unload_seconds`` of quiet, and the next request rebuilds.

``daemon.py`` only wires this in. Everything about *which* tier answers,
what it may propose and what is recorded stays in
:mod:`nvsh.tiers.router` -- nothing here inspects an operation name, and
nothing here executes anything.

Tier 2 and the confidence verifier have no implementation yet: their
factories default to ``None``, which is exactly what
:class:`~nvsh.tiers.router.TierRouter` already understands as "that tier is
not installed". The seam is here so the LFM tier plugs in without
``daemon.py`` changing.

This module is **not** importable on the hot success path: the daemon
imports it lazily, inside the first tier request.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping

from ..agent.base import AgentContext, AgentEvent, AgentRequest, EventKind
from ..platform._model import Platform
from .base import Tier
from .records import TierRecords, default_records_path
from .router import Route, TierRouter, Verifier

#: The two outcomes of one tier request. ``handled`` means a tier answered
#: (the client shows the proposal or the explanation and stops); ``escalate``
#: means the request still needs the full agent. The daemon puts one of
#: these on the wire -- see ``docs/daemon.md`` -- and ``nvsh.daemon`` keeps
#: its own copy of both strings, pinned equal by a test.
HANDLED = "handled"
ESCALATE = "escalate"

#: How many handled routes the manager keeps, so the operator's later
#: approve/decline can be recorded against the tier that proposed it. Bounded
#: on purpose: a long-lived daemon must not grow a route per request.
RECENT_ROUTES = 32

#: Defaults for the ``[tiers]`` keys this module reads. The config module
#: owns the real defaults; these only keep the manager working when it is
#: handed a config object whose table is partial (a test's, or one written
#: by an older nvsh).
_DEFAULTS: dict[str, object] = {
    "enabled": False,
    "needle_min_confidence": 0.0,
    "memory_floor_mb": 1024,
    "idle_unload_seconds": 900,
    "records_cap_mb": 8,
    "store_request_text": False,
}

TierFactory = Callable[[], Tier]
VerifierFactory = Callable[[], Verifier]


@dataclass(frozen=True)
class TierAnswer:
    """How one tier request ended, in the shape the wire carries.

    A flattened, JSON-ready view of
    :class:`~nvsh.tiers.router.TierOutcome`: ``outcome`` is
    :data:`HANDLED` or :data:`ESCALATE`, ``tier`` names who answered,
    ``route_id`` is the handle a later :meth:`TierManager.record_decision`
    quotes, and ``declines``/``escalation_context`` are what the escalating
    client hands the full agent so it does not repeat the tiers' work.
    """

    outcome: str
    tier: str = ""
    route_id: str = ""
    declines: tuple[tuple[str, str], ...] = ()
    escalation_context: tuple[tuple[str, str], ...] = ()
    text: str = ""


#: What a request that never reached a tier answers with.
ESCALATED = TierAnswer(outcome=ESCALATE, text="no local tier answered; asking the full agent")


class TierSession:
    """One tier request: iterate it for events, then read :attr:`answer`.

    The same shape :class:`~nvsh.tiers.router.Route` has, because it is
    mostly a thin wrapper around one: iterating yields the very
    :class:`~nvsh.agent.base.AgentEvent` objects an adapter would yield, so
    a client renders a tier answer through the code path it already has.

    A session with no route -- the tiers are off, or the flavor could not be
    built -- yields at most one ``status`` line and answers
    :data:`ESCALATE`. It loads nothing.
    """

    def __init__(
        self,
        manager: "TierManager",
        route: Route | None,
        *,
        route_id: str = "",
        problem: str = "",
    ) -> None:
        self._manager = manager
        self._route = route
        self._route_id = route_id
        self._problem = problem
        self._answer: TierAnswer | None = None

    def __iter__(self) -> Iterator[AgentEvent]:
        route = self._route
        if route is None:
            if self._problem:
                yield AgentEvent(kind=EventKind.STATUS, text=self._problem, args={"tier": "tiers"})
            self._answer = ESCALATED
            return
        yield from route
        self._answer = self._manager.finish(self._route_id, route)

    @property
    def answer(self) -> TierAnswer:
        """How the request ended; :data:`ESCALATED` until it has been run."""
        return self._answer if self._answer is not None else ESCALATED


def _declines(outcome: object) -> tuple[tuple[str, str], ...]:
    raw = getattr(outcome, "declines", ()) or ()
    return tuple((str(tier), getattr(reason, "value", str(reason))) for tier, reason in raw)


def _context(outcome: object) -> tuple[tuple[str, str], ...]:
    raw = getattr(outcome, "escalation_context", ()) or ()
    return tuple((str(operation), str(excerpt)) for operation, excerpt in raw)


@dataclass
class _Loaded:
    """What one build of the tiers put in memory, so it can be dropped whole."""

    router: TierRouter
    tiers: tuple[Tier, ...] = field(default_factory=tuple)


class TierManager:
    """Holds the tiers for one daemon. Thread-safe; builds on first use.

    Handler threads call :meth:`open` concurrently -- the build is
    serialised here, the *selection* is not (a tier that needs one turn at a
    time enforces that itself, as :class:`~nvsh.tiers.needle.NeedleTier`
    does), so one shell's cold model load does not become every shell's.
    """

    def __init__(
        self,
        config: object,
        platform: Platform,
        *,
        clock: Callable[[], float] = time.monotonic,
        env: Mapping[str, str] | None = None,
        tier1_factory: TierFactory | None = None,
        tier2_factory: TierFactory | None = None,
        verifier_factory: VerifierFactory | None = None,
    ) -> None:
        self._config = config
        self._platform = platform
        self._clock = clock
        self._env = env
        self._tier1_factory = tier1_factory
        self._tier2_factory = tier2_factory
        self._verifier_factory = verifier_factory

        self._lock = threading.RLock()
        self._loaded: _Loaded | None = None
        self._problem = ""
        self._last_used = clock()
        self._next_id = 0
        self._routes: "OrderedDict[str, Route]" = OrderedDict()

    # -- configuration -----------------------------------------------------

    def _setting(self, key: str) -> object:
        table = getattr(self._config, "tiers", None)
        if isinstance(table, Mapping) and key in table:
            return table[key]
        return _DEFAULTS[key]

    @property
    def enabled(self) -> bool:
        """Is automatic tier routing switched on in the operator's config?"""
        return bool(self._setting("enabled"))

    def status(self) -> dict:
        """What ``nvsh daemon status`` reports about the tiers. Loads nothing."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "loaded": self._loaded is not None,
                "problem": self._problem,
            }

    # -- one request -------------------------------------------------------

    def open(self, request: AgentRequest, context: AgentContext) -> TierSession:
        """A session for *request*. Nothing runs until the session is iterated."""
        with self._lock:
            if not self.enabled:
                return TierSession(self, None)
            self._last_used = self._clock()
            loaded = self._ensure_loaded()
            if loaded is None:
                return TierSession(self, None, problem=self._problem)
            self._next_id += 1
            route_id = str(self._next_id)
            route = loaded.router.route(request, context)
        return TierSession(self, route, route_id=route_id)

    def finish(self, route_id: str, route: Route) -> TierAnswer:
        """Read *route*'s outcome, keeping it only when a tier answered."""
        outcome = route.outcome
        declines, escalated = _declines(outcome), _context(outcome)
        handled = getattr(outcome, "handled_by", None)
        if not handled:
            return TierAnswer(
                outcome=ESCALATE,
                declines=declines,
                escalation_context=escalated,
                text=ESCALATED.text,
            )
        self._remember(route_id, route)
        return TierAnswer(
            outcome=HANDLED,
            tier=str(handled),
            route_id=route_id,
            declines=declines,
            escalation_context=escalated,
            text=f"{handled} answered",
        )

    def _remember(self, route_id: str, route: Route) -> None:
        with self._lock:
            self._routes[route_id] = route
            while len(self._routes) > RECENT_ROUTES:
                self._routes.popitem(last=False)

    def record_decision(self, route_id: str, decision: str) -> bool:
        """Record what the operator did with *route_id*'s proposal.

        The tiers never learn the answer themselves -- the client does,
        after approval -- so this is how ``approved``/``declined`` lands in
        the measurement log against the tier that proposed it. ``False``
        when the route is unknown or already reported: a decision is
        recorded once, and a daemon that has since unloaded (or rotated the
        route out) says so rather than writing a record it cannot attribute.
        """
        with self._lock:
            route = self._routes.pop(route_id, None)
        if route is None or not decision:
            return False
        route.record_decision(decision)
        return True

    # -- loading and unloading ---------------------------------------------

    def _ensure_loaded(self) -> _Loaded | None:
        """The built tiers, building them on this call if they are not there."""
        if self._loaded is not None:
            return self._loaded
        try:
            self._loaded = self._build()
        except Exception as exc:  # noqa: BLE001 - a broken flavor costs one line
            # A missing engine, an unreadable records directory, a flavor
            # whose import blows up: the request escalates with one status
            # line and the full agent answers it (spec claim c17).
            self._problem = f"local tiers unavailable: {type(exc).__name__}: {exc}"
            return None
        self._problem = ""
        return self._loaded

    def _build(self) -> _Loaded:
        records = TierRecords(
            default_records_path(self._env),
            cap_bytes=int(self._setting("records_cap_mb")) * 1024 * 1024,
            store_request_text=bool(self._setting("store_request_text")),
        )
        tier1 = self._tier1_factory() if self._tier1_factory is not None else self._needle()
        tier2 = self._tier2_factory() if self._tier2_factory is not None else None
        verifier = self._verifier_factory() if self._verifier_factory is not None else None
        router = TierRouter(
            tier1,
            tier2,
            records,
            self._platform,
            verifier=verifier,
            min_confidence=float(self._setting("needle_min_confidence")),
        )
        return _Loaded(router=router, tiers=tuple(t for t in (tier1, tier2) if t is not None))

    def _needle(self) -> Tier:
        """The stock Tier 1: Needle3 in its own child process, floor-checked."""
        from .memfloor import check_floor  # lazy: nothing here on the hot path
        from .needle import NeedleTier

        floor_mb = int(self._setting("memory_floor_mb"))
        return NeedleTier(
            min_confidence=float(self._setting("needle_min_confidence")),
            floor_check=lambda: check_floor(floor_mb),
            env=self._env,
        )

    def sweep(self) -> bool:
        """Unload the tiers when they have been idle long enough.

        Called from the daemon's existing watchdog, so there is no thread of
        this module's own: until a tier request arrives this is two attribute
        reads. ``idle_unload_seconds`` of ``0`` means "never unload".
        Returns ``True`` when something was dropped.
        """
        window = float(self._setting("idle_unload_seconds") or 0)
        if window <= 0:
            return False
        with self._lock:
            if self._loaded is None or self._clock() - self._last_used < window:
                return False
        self.close()
        return True

    def close(self) -> None:
        """Drop the tiers and everything they hold (a Needle child). Never raises."""
        with self._lock:
            loaded, self._loaded = self._loaded, None
            self._routes.clear()
        if loaded is None:
            return
        for tier in loaded.tiers:
            # Teardown must never raise: a tier that cannot close is a tier
            # that is already gone, and the daemon still has to stop.
            with contextlib.suppress(Exception):
                tier.close()
