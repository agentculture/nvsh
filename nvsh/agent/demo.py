"""DemoAgent: the README recording's backend -- a committed fixture, replayed.

The demo has to run the *real* path (hook -> client -> daemon -> adapter ->
panel), or it would be a recording of something nvsh does not do. What it
must not run is a model: a recording has to come out the same on every
machine, with no harness installed, no key and no network. So ``demo`` is a
registered adapter like any other (``ADAPTERS['demo']``, no binary, always
"installed") whose answer comes from ``demo_fixture.json`` next to this file.

Two things keep the fixture honest:

* **It is JSON.** Re-scripting the demo -- different wording, a different
  proposal -- is an edit to that file, never to this module. The event
  dicts are exactly what :func:`nvsh.agent.base.event_from_dict` decodes,
  the same shape that travels the daemon socket.
* **The script path is the operator's, not the fixture's.** Whatever
  ``./path`` the failing command line named is substituted for
  ``{script}``, so the proposal is ``chmod +x`` on the file that actually
  failed rather than a name baked into a fixture. A line with no such token
  falls back to the fixture's ``default_script``.

The reply says it is scripted (see the fixture's closing line): the demo
shows nvsh's mechanism, and must not be mistaken for a model's diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Mapping

from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind, event_from_dict
from .fake import FakeAgent

#: What the fixture writes where the failing script path goes.
SCRIPT_PLACEHOLDER = "{script}"
#: What the fixture writes where the detected platform kind goes.
PLATFORM_PLACEHOLDER = "{platform}"
#: The platform word when the context carries no ``platform:`` line.
DEFAULT_PLATFORM = "this machine"

#: Used when the failing command line carries no path to blame, and the
#: fixture does not name a ``default_script`` of its own.
DEFAULT_SCRIPT = "./run-model.sh"

#: The committed scenario. ``[agents.demo] fixture = "..."`` points the
#: adapter at another one (a second recording, a test's own file).
FIXTURE_PATH = Path(__file__).with_name("demo_fixture.json")

#: What this adapter reports: it streams, it proposes commands nvsh's own
#: approve loop gates, and it needs nothing off this machine.
DEMO_CAPABILITIES = Capabilities(
    streaming=True,
    tool_calling=True,
    cancellation=True,
    persistent_session=False,
    local_model=True,
    path="fixture",
    approval="nvsh",
)


def script_from_command(command: str, default: str = DEFAULT_SCRIPT) -> str:
    """The script the failing command line was trying to run.

    A ``./path`` token wins outright (that is how the demo's planted script
    is invoked). Failing that, any token ending in ``.sh`` is taken, so
    ``bash scripts/run-model.sh`` still blames the script rather than
    ``bash``. A line naming neither -- an ordinary failure that happened to
    reach this adapter -- gets *default*, never a guess at some other
    argument on the line.
    """
    tokens = (command or "").split()
    for token in tokens:
        if token.startswith("./"):
            return token
    for token in tokens:
        if token.endswith(".sh"):
            return token
    return default


def platform_kind(platform_block: str, default: str = DEFAULT_PLATFORM) -> str:
    """The ``kind`` named on the context's first ``platform: <kind>`` line.

    The block :func:`nvsh.client._platform_block` builds starts with
    ``platform: dgx-spark`` (or ``jetson``, ``rtx``, ``generic``); anything
    else -- an empty context, a detection failure -- yields *default* so the
    reply still reads as a sentence.
    """
    for line in (platform_block or "").splitlines():
        head, sep, rest = line.strip().partition(":")
        if sep and head == "platform" and rest.strip():
            return rest.strip().split()[0]
    return default


def _substituted(value: object, replacements: Mapping[str, str]) -> object:
    """Apply every placeholder -> text pair to every string inside *value*."""
    if isinstance(value, str):
        for placeholder, text in replacements.items():
            value = value.replace(placeholder, text)
        return value
    if isinstance(value, Mapping):
        return {key: _substituted(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_substituted(item, replacements) for item in value]
    return value


def load_events(path: Path, command: str, platform_block: str = "") -> list[AgentEvent]:
    """Decode the fixture at *path* into events for one failing *command*.

    *platform_block* is the context's platform text; its kind fills the
    fixture's ``{platform}`` placeholder so a recording names the device
    it was made on.

    Raises nothing the caller has to catch beyond the usual file/JSON
    errors; :meth:`DemoAgent.run` turns those into an ERROR event so a
    mistyped ``fixture`` path degrades the panel instead of the shell.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    placeholder = str(data.get("script_placeholder") or SCRIPT_PLACEHOLDER)
    default = str(data.get("default_script") or DEFAULT_SCRIPT)
    platform_placeholder = str(data.get("platform_placeholder") or PLATFORM_PLACEHOLDER)
    replacements = {
        placeholder: script_from_command(command, default),
        platform_placeholder: platform_kind(platform_block),
    }
    events = []
    for raw in data.get("events") or []:
        if isinstance(raw, Mapping):
            events.append(event_from_dict(_substituted(raw, replacements)))
    return events


class DemoAgent(FakeAgent):
    """A :class:`~nvsh.agent.fake.FakeAgent` whose script is the fixture.

    Everything else -- cancellation between yields, ending on an ERROR
    event, ``start``/``close`` -- is FakeAgent's, deliberately: the demo
    adapter should behave like the scripted adapter the conformance suite
    already knows, not like a second implementation of it.
    """

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        super().__init__([], capabilities=DEMO_CAPABILITIES)
        settings = dict(config or {})
        fixture = settings.get("fixture")
        self._fixture_path = Path(str(fixture)) if fixture else FIXTURE_PATH

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        try:
            self._script = list(load_events(self._fixture_path, request.command, context.platform))
        except (OSError, ValueError) as exc:
            yield AgentEvent(
                kind=EventKind.ERROR,
                error=f"demo fixture unreadable ({self._fixture_path}): {exc}",
            )
            return
        yield from super().run(request, context)
