"""The seam between Tier 2 and whatever serves its model.

Tier 2 talks to one OpenAI-compatible endpoint on localhost. *What* stands
behind that endpoint -- a container nvsh started, or a server the operator
already runs -- is a :class:`Runtime`. The tier asks it for a base URL and
never learns more; the launcher (``runtime_docker``) never learns what the
tier does with it. Switching engine is therefore a config change only.

A runtime never raises anything but :class:`RuntimeUnavailable` from
:meth:`Runtime.ensure`, and its ``detail`` is one line an operator can read:
the tier turns it into a single-status-line decline and the request moves up
to the full agent.
"""

from __future__ import annotations

from typing import Protocol

from .toolchat import ToolChatError, require_localhost


class RuntimeUnavailable(Exception):
    """The model runtime cannot serve right now. ``str(exc)`` is one line."""


class Runtime(Protocol):
    """What Tier 2 needs from the thing serving its model."""

    def ensure(self) -> str:
        """Return the base URL of a ready endpoint, starting it if needed.

        Raises :class:`RuntimeUnavailable` when it cannot.
        """

    def stop(self) -> None:
        """Release whatever :meth:`ensure` started. Never raises."""

    def status(self) -> str:
        """One line for ``nvsh overview`` / ``doctor``. Starts nothing."""


class AttachedRuntime:
    """A server the operator already runs: nvsh starts and stops nothing."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def ensure(self) -> str:
        try:
            require_localhost(self._base_url)
        except ToolChatError as exc:
            raise RuntimeUnavailable(str(exc)) from exc
        return self._base_url

    def stop(self) -> None:
        """Nothing to stop: the server is the operator's, not nvsh's."""

    def status(self) -> str:
        return f"attached to {self._base_url}"
