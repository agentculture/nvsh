"""The inline failure panel: the one piece of nvsh the operator actually sees.

Hard rules this module exists to enforce (spec c33 / the terminal-portability
entry, and CLAUDE.md's headless-over-SSH constraint):

* **Hard-coded SGR only.** thor has no ``xterm-ghostty`` capability entry,
  so any capability-database helper would fail on an ssh in from Ghostty.
  This module consults no such database and shells out to nothing; colour is
  a handful of literal escape constants, switched off entirely when
  ``NO_COLOR`` is set, when ``TERM`` is ``dumb``/empty, or when the output is
  not a tty.
* **First text on screen immediately.** :meth:`Panel.stream` writes and
  flushes every ``text_delta`` as it arrives; nothing is buffered until the
  answer is complete, and there is no spinner to repaint.
* **Propose, don't run.** :meth:`Panel.show_proposal` renders the proposed
  command *exactly* as the agent proposed it and returns the operator's
  choice. It never pre-types anything and never touches ``READLINE_LINE``:
  the panel reports a decision, the caller (``nvsh.client``) acts on it.
* **Ctrl+C always returns the prompt.** While streaming, SIGINT is handled
  here: the caller's ``cancel`` callback runs (aborting the daemon/one-shot
  run), the terminal's termios settings are restored in a ``finally``, one
  line is printed, and the stream ends. The caller turns that into exit 130.

Stdlib only; nothing here is imported at shell start (the bash hook only
runs ``nvsh`` on a qualifying failure).
"""

from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, TextIO

from .agent.base import AgentEvent, EventKind, Proposal

#: The panel's fixed width for its ASCII boxes.
_BOX_WIDTH = 72

#: What :meth:`Panel.show_proposal` can return.
APPROVE = "approve"
EXPLAIN = "explain"
DETAILS = "details"
IGNORE = "ignore"


@dataclass(frozen=True)
class Style:
    """Literal SGR sequences, or empty strings when styling is off.

    Every attribute is a plain string constant -- no capability database is
    consulted, so a terminal with no capability entry behaves exactly like
    one with a complete entry (it just receives the same bytes).
    """

    enabled: bool = False

    def _seq(self, code: str) -> str:
        return code if self.enabled else ""

    @property
    def bold(self) -> str:
        return self._seq("\x1b[1m")

    @property
    def dim(self) -> str:
        return self._seq("\x1b[2m")

    @property
    def red(self) -> str:
        return self._seq("\x1b[31m")

    @property
    def yellow(self) -> str:
        return self._seq("\x1b[33m")

    @property
    def cyan(self) -> str:
        return self._seq("\x1b[36m")

    @property
    def reset(self) -> str:
        return self._seq("\x1b[0m")


def style(env: Mapping[str, str] | None, isatty: bool) -> Style:
    """Decide whether the panel may emit colour at all.

    Off when ``NO_COLOR`` is set (any value), when ``TERM`` is unset, empty
    or ``dumb``, or when the output stream is not a terminal.
    """
    resolved = os.environ if env is None else env
    if not isatty:
        return Style(enabled=False)
    if resolved.get("NO_COLOR") is not None:
        return Style(enabled=False)
    term = resolved.get("TERM", "")
    if term in ("", "dumb"):
        return Style(enabled=False)
    return Style(enabled=True)


@dataclass
class StreamResult:
    """What one :meth:`Panel.stream` produced."""

    text: str = ""
    error: str = ""
    done: bool = False
    interrupted: bool = False
    proposals: list[Proposal] = field(default_factory=list)


class _Interrupted(BaseException):
    """Raised inside the SIGINT handler to break out of a blocking read."""


class Panel:
    """Renders one failure panel onto ``out``, reading keys from ``in_``."""

    def __init__(
        self,
        out: TextIO | None = None,
        in_: TextIO | None = None,
        env: Mapping[str, str] | None = None,
        isatty: bool | None = None,
    ) -> None:
        self.out = out if out is not None else sys.stdout
        self.in_ = in_ if in_ is not None else sys.stdin
        self.env = dict(os.environ if env is None else env)
        if isatty is None:
            isatty = _safe_isatty(self.out)
        self.isatty = bool(isatty)
        self.style = style(self.env, self.isatty)

    # -- low-level writing -------------------------------------------------

    def write(self, text: str) -> None:
        """Write and flush immediately -- nothing waits for the answer to end."""
        try:
            self.out.write(text)
            self.out.flush()
        except (OSError, ValueError):  # closed pipe: the operator moved on
            pass

    def line(self, text: str = "") -> None:
        self.write(text + "\n")

    def note(self, text: str) -> None:
        """One dim informational line (never mixed into the agent's text)."""
        self.line(f"{self.style.dim}{text}{self.style.reset}")

    def header(self, command: str, exit_code: int) -> None:
        """The panel's first line: what failed and with which status."""
        s = self.style
        self.line(f"{s.bold}{s.red}nvsh:{s.reset} {command} failed (exit {exit_code})")

    # -- streaming ---------------------------------------------------------

    def stream(
        self,
        events: Iterable[AgentEvent],
        *,
        on_proposal: Callable[[Proposal, AgentEvent], object] | None = None,
        cancel: Callable[[], object] | None = None,
    ) -> StreamResult:
        """Render ``events`` as they arrive; return what happened.

        SIGINT during the stream calls ``cancel`` and ends the stream with
        ``interrupted=True``. Terminal attributes and the previous SIGINT
        handler are always restored.
        """
        result = StreamResult()
        saved_attrs = _save_termios(self.in_, self.isatty)
        previous = _install_sigint(self._on_sigint(cancel))
        started_text = False
        try:
            for event in events:
                if event.kind is EventKind.TEXT_DELTA:
                    started_text = True
                    result.text += event.text
                    self.write(event.text)
                elif event.kind is EventKind.STATUS:
                    if event.text:
                        self.note(f"... {event.text}")
                elif event.kind is EventKind.TOOL_CALL:
                    self.note(f"... tool {event.tool}")
                elif event.kind is EventKind.TOOL_RESULT:
                    self.note(f"... tool {event.tool} finished")
                elif event.kind is EventKind.PROPOSAL and event.proposal is not None:
                    if started_text:
                        self.line()
                        started_text = False
                    result.proposals.append(event.proposal)
                    if on_proposal is not None:
                        on_proposal(event.proposal, event)
                elif event.kind is EventKind.ERROR:
                    result.error = event.error
                    if started_text:
                        self.line()
                        started_text = False
                    self.line(f"{self.style.red}nvsh: {event.error}{self.style.reset}")
                    break
                elif event.kind is EventKind.DONE:
                    result.done = True
                    break
        except (KeyboardInterrupt, _Interrupted):
            result.interrupted = True
        finally:
            _restore_sigint(previous)
            _restore_termios(self.in_, saved_attrs)
            if started_text:
                self.line()
            if result.interrupted:
                self.line(f"{self.style.dim}nvsh: interrupted{self.style.reset}")
        return result

    def _on_sigint(self, cancel: Callable[[], object] | None) -> Callable[[int, object], None]:
        def handler(_signum: int, _frame: object) -> None:
            if cancel is not None:
                _quiet(cancel)
            raise _Interrupted()

        return handler

    # -- proposals ---------------------------------------------------------

    def render_proposal(self, proposal: Proposal) -> None:
        """Print the proposed command verbatim in a plain-ASCII box."""
        s = self.style
        kind = getattr(proposal.kind, "value", str(proposal.kind))
        if proposal.rationale:
            self.line(f"{s.dim}{proposal.rationale}{s.reset}")
        rule = "+" + "-" * (_BOX_WIDTH - 2) + "+"
        self.line(rule)
        # Never wrapped and never re-quoted: the operator must be able to read
        # (and search for) the command exactly as the agent proposed it. Long
        # lines are left to the terminal to wrap.
        for chunk in proposal.command.splitlines() or [""]:
            self.line(f"  {s.cyan}{chunk}{s.reset}")
        self.line(rule)
        self.line(f"{s.dim}({kind}){s.reset}")

    def explain_proposal(self, proposal: Proposal) -> None:
        """Print why the agent proposed this (the ``e`` key's answer)."""
        self.line(f"why: {proposal.rationale or '(no rationale given)'}")

    def detail_proposal(self, proposal: Proposal) -> None:
        """Print the proposal's full fields (the ``d`` key's answer)."""
        kind = getattr(proposal.kind, "value", str(proposal.kind))
        self.line(f"kind: {kind}")
        self.line(f"command: {proposal.command}")
        self.line(f"rationale: {proposal.rationale}")

    def show_proposal(self, proposal: Proposal) -> str:
        """Show ``proposal`` and return ``approve``/``explain``/``details``/``ignore``.

        The command is never pre-typed anywhere: the operator sees it and
        presses a key. Enter approves, ``e`` explains, ``d`` shows details,
        Esc (or Ctrl+C, or anything else) ignores. Re-asking after ``e``/``d``
        is the caller's job -- this method reports one keypress and returns.
        """
        self.render_proposal(proposal)
        self.line("[Enter] run   [e] explain   [d] details   [Esc] ignore")
        return self._read_choice()

    def _read_choice(self) -> str:
        if self.isatty:
            key = _read_key(self.in_)
        else:
            key = _read_line_key(self.in_)
        if key in ("\r", "\n"):
            return APPROVE
        if key in ("e", "E"):
            return EXPLAIN
        if key in ("d", "D"):
            return DETAILS
        return IGNORE


# ---------------------------------------------------------------------------
# terminal helpers -- termios only, no capability lookup of any kind
# ---------------------------------------------------------------------------


def _safe_isatty(stream: object) -> bool:
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - a StringIO or closed file is simply "not a tty"
        return False


def _fileno(stream: object) -> int | None:
    try:
        return int(stream.fileno())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None


def _save_termios(stream: object, isatty: bool):
    if not isatty:
        return None
    fd = _fileno(stream)
    if fd is None:
        return None
    try:
        import termios

        return fd, termios.tcgetattr(fd)
    except Exception:  # noqa: BLE001 - not a real terminal after all
        return None


def _quiet(call: Callable[..., object], *args: object) -> None:
    """Run ``call``, swallowing any error.

    Used only for teardown (restoring the terminal, cancelling a run): the
    operator must get their prompt back even if the cleanup itself fails.
    """
    try:
        call(*args)
    except Exception:  # noqa: BLE001 - teardown must never raise at the prompt
        return


def _restore_termios(stream: object, saved) -> None:
    if not saved:
        return
    fd, attrs = saved
    import termios

    _quiet(termios.tcsetattr, fd, termios.TCSADRAIN, attrs)


def _install_sigint(handler):
    try:
        return signal.signal(signal.SIGINT, handler)
    except ValueError:  # not the main thread
        return None


def _restore_sigint(previous) -> None:
    if previous is None:
        return
    _quiet(signal.signal, signal.SIGINT, previous)


def _read_key(stream) -> str:
    """Read one raw keypress. Falls back to a line read if raw mode fails."""
    fd = _fileno(stream)
    if fd is None:
        return _read_line_key(stream)
    try:
        import termios
        import tty
    except ImportError:  # pragma: no cover - POSIX only
        return _read_line_key(stream)
    try:
        saved = termios.tcgetattr(fd)
    except Exception:  # noqa: BLE001
        return _read_line_key(stream)
    try:
        tty.setraw(fd)
        data = os.read(fd, 1)
    except Exception:  # noqa: BLE001
        return IGNORE
    finally:
        _quiet(termios.tcsetattr, fd, termios.TCSADRAIN, saved)
    return data.decode("utf-8", errors="replace") if data else ""


def _read_line_key(stream) -> str:
    """Non-tty fallback: read one line; an empty line means Enter."""
    try:
        raw = stream.readline()
    except Exception:  # noqa: BLE001
        return ""
    if raw == "":
        return ""  # EOF -> ignore
    stripped = raw.strip()
    return stripped[:1] if stripped else "\n"
