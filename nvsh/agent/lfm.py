"""``lfm``: the explicit Tier 2 (LFM) adapter (task t19).

The exact analogue of :class:`~nvsh.agent.needle.NeedleAgent`: it wraps
:class:`nvsh.tiers.router.TierRouter`, restricted to Tier 2 only
(``tier1=None``, no verifier), so an operator who explicitly asks for
``@lfm`` (or ``nvsh --agent lfm``) gets exactly the bounded
inspect-interpret-propose loop the daemon's tier ladder would try second --
and nothing else. It never falls through to the full agent on its own: an
explicit ``@lfm`` request its tier cannot answer is not silently upgraded to
a real harness call; instead this adapter explains, in one line, why it had
no answer, and ends the turn there.

Unlike Tier 1, Tier 2 answers a FAILURE-kind request as readily as an
EXPLICIT one (``nvsh/tiers/router.py``'s ``Route._order`` puts Tier 2 in
every order), so there is no FAILURE-never-reaches-this-tier case here the
way there is for ``needle``.

"The flavor is configured" for ``lfm`` means ``[tiers.lfm] model`` is a
non-empty string -- the ``lfm`` extra has no dependencies to import-check,
unlike ``needle``'s ``cactus-needle`` package (see
:func:`nvsh.agent.registry._lfm_flavor_installed`). With no model
configured, :meth:`LfmAgent.start` builds no router at all, and
:meth:`LfmAgent.run` answers with one status line naming the missing key and
how to set it (see :meth:`LfmAgent._unrouted`) -- it never builds a broken
``LfmTier`` just to have it decline.

The turn loop, decline-text formatting and lifecycle this adapter shares
with :class:`~nvsh.agent.needle.NeedleAgent` live in
:mod:`nvsh.agent._tier_adapter`; this module only builds the Tier-2-only
router and fills in the four hooks that module calls into.

``nvsh.tiers`` (and everything it pulls in: the Docker-launching runtime,
the chat client, ``nvsh.ops``) must never load on the hot success path
(CLAUDE.md's stdlib-only import-time contract). So, like every lazy adapter
factory in :mod:`nvsh.agent.registry`, this module keeps ``__init__`` to
plain attribute assignment -- no tier, no router, no runtime -- and imports
``nvsh.tiers`` only from inside :meth:`start`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Mapping

from ._tier_adapter import TierOnlyAdapter
from .base import AgentEvent, Capabilities, EventKind

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from ..platform._model import Platform
    from ..tiers.lfm import ChatLike
    from ..tiers.runtime import Runtime

#: What this adapter reports -- the same shape as ``needle``'s (see
#: :class:`~nvsh.agent.needle.NEEDLE_CAPABILITIES`'s docstring for each
#: field): it streams, its PROPOSAL goes through nvsh's own approve loop,
#: it needs nothing off this machine (``local_model=True``: the container
#: Tier 2 talks to is one nvsh itself starts on localhost), it reads no
#: file on its own, and it has no mid-turn steering channel -- one Tier 2
#: loop is one request/response round trip, not an open conversation.
LFM_CAPABILITIES = Capabilities(
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

#: Explanation given when Tier 2 was never even consulted. Kept for parity
#: with ``needle``'s ``TierOnlyAdapter._failure_text`` hook; unlike Tier 1,
#: Tier 2 is in every router order this adapter ever builds (FAILURE and
#: EXPLICIT alike), so this text is not expected to surface in practice.
_FAILURE_TEXT = "lfm was not asked to answer this request"

#: What :meth:`LfmAgent.run` says when ``[tiers.lfm] model`` is unset --
#: the one case where :meth:`LfmAgent.start` builds no router at all.
_NO_MODEL_TEXT = "lfm has no model configured; set [tiers.lfm] model in config.toml"


class LfmAgent(TierOnlyAdapter):
    """The explicit ``@lfm`` adapter: Tier 2 only, propose-or-explain.

    ``config`` is the ``[tiers]`` table (:attr:`nvsh.config.Config.tiers`),
    including its nested ``[tiers.lfm]`` table -- Tier 2 has no
    ``[agents.lfm]`` settings of its own. The remaining keyword-only
    arguments exist so tests can drive this adapter against a fake
    :class:`~nvsh.tiers.runtime.Runtime`/chat client without Docker or a
    real model server, and without writing to a real records path.
    """

    def __init__(
        self,
        config: Mapping[str, object] | None = None,
        *,
        runtime: "Runtime | None" = None,
        runner: Callable[[list[str], float], tuple[int, str]] | None = None,
        tier_runner: Callable[[list[str], float], tuple[int, str]] | None = None,
        chat_factory: Callable[[str], "ChatLike"] | None = None,
        records_path: Path | None = None,
        floor_reader: Callable[[], str] | None = None,
        platform: "Platform | None" = None,
    ) -> None:
        """Cheap: only plain attributes. Spawns nothing, imports nothing heavy."""
        super().__init__()
        settings = dict(config or {})
        lfm_settings = settings.get("lfm")
        self._lfm_settings = dict(lfm_settings) if isinstance(lfm_settings, Mapping) else {}
        model = self._lfm_settings.get("model")
        self._model = model if isinstance(model, str) else ""
        self._min_confidence = _as_float(settings.get("needle_min_confidence"), 0.0)
        self._memory_floor_mb = _as_int(settings.get("memory_floor_mb"), 1024)
        self._records_cap_mb = _as_int(settings.get("records_cap_mb"), 8)
        self._store_request_text = bool(settings.get("store_request_text", False))
        self._runtime = runtime
        #: Grounds a proposed operation's arguments (the router's own
        #: runner) -- distinct from *tier_runner*, which runs Tier 2's own
        #: read-only inspections. Both default to the real system runner in
        #: production; tests inject fakes for either or both.
        self._runner = runner
        self._tier_runner = tier_runner
        self._chat_factory = chat_factory
        self._records_path = records_path
        self._floor_reader = floor_reader
        self._platform = platform
        self._tier2 = None  # the same object as self._router.tier2, kept for close()/status()

    # -- lifecycle --

    def start(self) -> None:
        """Build the Tier-2-only router when ``[tiers.lfm] model`` is set.

        Idempotent. Leaves :attr:`_router` unset (``None``) when no model is
        configured -- see :meth:`_unrouted` -- rather than building a
        ``LfmTier`` doomed to decline every request; spawns no container yet
        either way, only :class:`~nvsh.tiers.lfm.LfmTier.select` does that.
        """
        if self._router is not None or not self._model:
            return
        from ..tiers.lfm import LfmTier
        from ..tiers.memfloor import check_floor, default_reader
        from ..tiers.records import TierRecords
        from ..tiers.router import TierRouter
        from ..tiers.runtime_docker import build_runtime

        reader = self._floor_reader or default_reader
        floor_mb = self._memory_floor_mb

        def floor_check():
            return check_floor(floor_mb, reader)

        platform = self._platform
        if platform is None:
            from ..platform import detect as detect_platform

            platform = detect_platform()

        runtime = self._runtime
        if runtime is None:
            runtime = build_runtime(self._lfm_settings, platform, floor_check=floor_check)

        tier_kwargs: dict[str, object] = {"model": self._model, "floor_check": floor_check}
        if self._tier_runner is not None:
            tier_kwargs["runner"] = self._tier_runner
        if self._chat_factory is not None:
            tier_kwargs["chat_factory"] = self._chat_factory
        self._tier2 = LfmTier(runtime, platform, **tier_kwargs)

        records = TierRecords(
            path=self._records_path,
            cap_bytes=self._records_cap_mb * 1024 * 1024,
            store_request_text=self._store_request_text,
        )
        router_kwargs: dict[str, object] = {"min_confidence": self._min_confidence}
        if self._runner is not None:
            router_kwargs["runner"] = self._runner
        self._router = TierRouter(None, self._tier2, records, platform, **router_kwargs)

    # -- TierOnlyAdapter hooks --

    def _unrouted(self) -> Iterator[AgentEvent]:
        """No ``[tiers.lfm] model`` configured: explain, then DONE -- the one
        case where :meth:`start` left :attr:`_router` unset."""
        yield AgentEvent(kind=EventKind.STATUS, text=_NO_MODEL_TEXT, args={"tier": "lfm"})
        yield AgentEvent(kind=EventKind.DONE, args={"tier": "lfm"})

    def _tier_label(self) -> str:
        return "lfm"

    def _failure_text(self) -> str:
        return _FAILURE_TEXT

    def _unavailable_text(self) -> str:
        return self._tier2.status() if self._tier2 is not None else "lfm tier unavailable"

    def _active_tier(self) -> object | None:
        return self._tier2

    def capabilities(self) -> Capabilities:
        return LFM_CAPABILITIES


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)
