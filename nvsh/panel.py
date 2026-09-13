"""The inline failure panel: the one piece of nvsh the operator actually sees.

Hard rules this module exists to enforce (spec c33 / the terminal-portability
entry, and CLAUDE.md's headless-over-SSH constraint):

* **Hard-coded SGR only.** thor has no ``xterm-ghostty`` capability entry,
  so any capability-database helper would fail on an ssh in from Ghostty.
  This module consults no such database and shells out to nothing; colour is
  a handful of literal escape constants, switched off entirely when
  ``NO_COLOR`` is set, when ``TERM`` is ``dumb``/empty, or when the output is
  not a tty.
* **First text on screen immediately, and progress while there is none.**
  :meth:`Panel.stream` writes and flushes every ``text_delta`` as it
  arrives; nothing is buffered until the answer is complete. When the
  backend goes quiet for more than a second (the real model routinely takes
  tens of seconds before its first token) a background ticker paints one
  dim ``... waiting for the agent (12s)`` line, repainted in place with a
  literal ``\r`` plus ``ESC [ 2 K`` -- only when styling is on, i.e. on a
  tty without ``NO_COLOR``/``TERM=dumb``. Off a tty the panel prints that
  line once, in plain text, and never repaints. The line is always erased
  before the next real event reaches the screen, so nothing interleaves.
  This is deliberately not an animated spinner: it is one line whose only
  moving part is an honest elapsed count (deviation d13, which revised this
  module's earlier "no spinner to repaint" rule).
* **Propose, don't run.** :meth:`Panel.show_proposal` renders the proposed
  command *exactly* as the agent proposed it and returns the operator's
  choice. It never pre-types anything and never touches ``READLINE_LINE``:
  the panel reports a decision, the caller (``nvsh.client``) acts on it.
  It does print one acknowledgement line the instant a key is pressed
  (``nvsh: running ...`` / ``running (approved for this session) ...`` /
  ``running (approved for this user) ...`` / ``explaining`` / ``details`` /
  ``ignored``), so a keypress is never followed by silence while the
  backend reacts. That ack is the *first* line; the caller's own outcome
  lines (``nvsh: not run``, ``nvsh: <cmd> -> exit N``) still follow it and
  are worded differently.
* **Approving a class is a decision, so it is guarded.** ``[s]`` and ``[u]``
  run the command *and* approve it for the login session / for this user
  (deviation d15, after an operator re-approved the same command class on
  every failure). The panel owns neither the policy nor the store: it hands
  the scope to the caller's ``guard``, prints the one-line reason a refused
  scope comes back with (``sudo``, ``rm``, a bare ``*``), and returns
  ``refused`` so the caller asks again with run-once still available.
* **A fallback is news, not noise.** A ``status`` event announcing that the
  daemon could not be used (``one-shot ...``, ``daemon did not start`` /
  ``refused`` / ``connection lost``) is rendered as a visible
  ``nvsh: falling back - <original text>`` line rather than another dim
  ``...`` line. Every other status stays dim, and an empty one prints
  nothing at all.
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
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, TextIO

from .agent.base import AgentEvent, EventKind, Proposal

#: The panel's fixed width for its ASCII boxes.
_BOX_WIDTH = 72

#: Seconds of silence before the panel admits it is waiting.
_WAIT_AFTER = 1.0

#: How often the waiting ticker wakes up to consider a repaint.
_WAIT_TICK = 0.2

#: The waiting line's text (the elapsed count is appended on a tty).
_WAIT_TEXT = "waiting for the agent"

#: Carriage return plus erase-to-end-of-line. Hard-coded CSI, never tput.
_ERASE_LINE = "\r\x1b[2K"

#: Status texts that mean "the daemon was not usable" -- see
#: ``nvsh.client_transport``, which is the only producer of these.
_FALLBACK_MARKERS = (
    "daemon did not start",
    "daemon refused",
    "daemon connection lost",
)

#: What :meth:`Panel.show_proposal` can return.
APPROVE = "approve"
#: Run *and* approve this command class for the rest of the login session /
#: for this user. These two are deliberately spelled with the same tokens
#: the pi approval extension's ``ctx.ui.select`` offers
#: ("once"/"session"/"user"/"deny"), so the client can forward the
#: operator's answer verbatim as ``{"value": choice}``.
APPROVE_SESSION = "session"
APPROVE_USER = "user"
EXPLAIN = "explain"
DETAILS = "details"
#: Tell the agent something. The panel reads one line at an ``nvsh> `` prompt
#: (cooked mode) and the caller injects it into the *same* conversation
#: (deviation d16): a proposal the operator does not want is worth more as a
#: correction than as a silent "ignore".
TELL = "tell"
IGNORE = "ignore"
#: A scope key the caller's ``guard`` refused. The caller re-asks; run-once
#: stays available.
REFUSED = "refused"

#: The one-line key legend. Kept under 80 columns so it never wraps on a
#: bare ssh into a Jetson, where a wrapped legend costs the panel a line and
#: reads as two half-legends. ``+session``/``+user`` are the abbreviation
#: that buys the room, and still say which scope each key approves for.
LEGEND = "[Enter] run [s] +session [u] +user [e] explain [d] details [t] tell [Esc] ignore"

#: The widest a panel line may get before it is truncated (the running-command
#: line, the scope line). 80 is the narrowest terminal nvsh promises to read
#: on; 100 is what d19 asks of the running-command line specifically.
_SCOPE_WIDTH = 80
_COMMAND_WIDTH = 100

#: Keys a tool result may spell its exit status with. pi's bash tool says
#: ``exitCode``; other backends (and nvsh's own runner) spell it differently,
#: and a result with none of them simply has no exit code to report.
_EXIT_KEYS = ("exitCode", "exit_code", "exit", "returncode")

#: What the panel prints the instant a proposal key is pressed. Worded so it
#: never collides with the caller's own outcome lines (``nvsh: not run``).
_ACK = {
    APPROVE: "nvsh: running ...",
    APPROVE_SESSION: "nvsh: running (approved for this session) ...",
    APPROVE_USER: "nvsh: running (approved for this user) ...",
    EXPLAIN: "nvsh: explaining ...",
    DETAILS: "nvsh: details ...",
    TELL: "",
    IGNORE: "nvsh: ignored",
}

#: How each scope key describes, in the ack, what it just stored. Filled in
#: from the caller's ``scopes`` mapping; the bare wording above is the
#: fallback for a caller that did not say (operator feedback on d15: the
#: old ack never named the pattern, so ``[u]`` read as "approve this exact
#: argument" when it approves every argument).
_SCOPE_ACK = {
    APPROVE_SESSION: "nvsh: running; {what} approved for this session",
    APPROVE_USER: "nvsh: running; {what} approved for this user",
}

#: How the scope line offers each key.
_SCOPE_OFFER = {
    APPROVE_SESSION: "[s] allows {what} for this session",
    APPROVE_USER: "[u] allows {what} for you, persisted",
}


def one_line(text: str, width: int) -> str:
    """``text`` collapsed onto one line and truncated to ``width`` columns."""
    flat = " ".join(str(text).split())
    if len(flat) <= width:
        return flat
    return flat[: width - 1] + "\u2026"


def tool_exit_code(result: object) -> int | None:
    """The exit status a tool result reports, or ``None`` when it reports none."""
    if not isinstance(result, Mapping):
        return None
    for key in _EXIT_KEYS:
        value = result.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.lstrip("-").isdigit():
                return int(text)
    return None


def is_fallback_status(text: str) -> bool:
    """True when a ``status`` text announces a fall back off the daemon."""
    stripped = text.strip()
    if stripped.startswith("one-shot "):
        return True
    return any(marker in stripped for marker in _FALLBACK_MARKERS)


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
        # One lock guards every write, so the waiting ticker (a background
        # thread) can never interleave its repaint with the agent's text.
        self._write_lock = threading.RLock()
        self._waiting_shown = False
        self._waiting_plain = False
        self._waiting_since = 0.0
        self._waiting_paused = True

    # -- low-level writing -------------------------------------------------

    def write(self, text: str) -> None:
        """Write and flush immediately -- nothing waits for the answer to end."""
        with self._write_lock:
            try:
                self.out.write(text)
                self.out.flush()
            except UnicodeEncodeError:
                # An ASCII-only locale (a bare ssh, a cron-ish TERM) must not
                # swallow a line: degrade the characters, keep the message.
                self._write_raw(text.encode("ascii", "replace").decode("ascii"))
            except (OSError, ValueError):  # closed pipe: the operator moved on
                pass

    def _write_raw(self, text: str) -> None:
        try:
            self.out.write(text)
            self.out.flush()
        except (OSError, ValueError):
            pass

    def line(self, text: str = "") -> None:
        self.write(text + "\n")

    def note(self, text: str) -> None:
        """One dim informational line (never mixed into the agent's text)."""
        self.line(f"{self.style.dim}{text}{self.style.reset}")

    def status(self, text: str) -> None:
        """Render one ``status`` event.

        A fallback off the daemon is news the operator must see (it explains
        the wait that follows), so it is bold, not dim, and keeps the
        original text verbatim after the prefix. Everything else is a dim
        ``...`` line, and an empty status prints nothing.
        """
        if not text:
            return
        if is_fallback_status(text):
            s = self.style
            self.line(f"{s.bold}{s.yellow}nvsh: falling back - {s.reset}{text}")
            return
        self.note(f"... {text}")

    def running(self, command: str) -> None:
        """Say which command is running -- the command, not the tool (d19).

        ``... running tool: bash`` told the operator nothing they could act
        on: every proposal nvsh forwards runs through the bash tool, so the
        line was the same whatever the agent was doing. One bare command
        line, collapsed onto one line and truncated at 100 columns, says it.
        """
        self.note(f"... running: {one_line(command, _COMMAND_WIDTH)}")

    def finished(self, exit_code: int) -> None:
        """Say how the command that was running ended (d19)."""
        self.note(f"... finished (exit {exit_code})")

    def tool_call(self, tool: str, command: object = None) -> None:
        """One ``tool_call`` event: the command when there is one, else the tool."""
        if isinstance(command, str) and command.strip():
            self.running(command)
            return
        self.note(f"... running tool: {tool}")

    def tool_result(self, tool: str, exit_code: int | None) -> None:
        """One ``tool_result`` event: the exit code when the backend gave one."""
        if exit_code is None:
            self.note(f"... tool {tool} finished")
            return
        self.finished(exit_code)

    def header(
        self,
        command: str,
        exit_code: int,
        *,
        backend_label: str = "",
        ask: str | None = None,
    ) -> None:
        """The panel's first line: what nvsh is doing, and who it is asking.

        Two forms (deviation d22, after an operator could not tell which
        backend the panel was waiting on):

        * a failure -- ``nvsh: <command> failed (exit N), forwarding to
          <backend>``;
        * a question the operator typed at the prompt (d20) -- ``nvsh:
          asking <backend>: <text>``.

        ``backend_label`` is the caller's ``<harness>/<model>`` string; an
        empty one simply drops the clause, so a panel that does not know
        which backend will serve it still prints a complete first line.
        """
        s = self.style
        lead = f"{s.bold}{s.red}nvsh:{s.reset}"
        if ask is not None:
            who = f" {backend_label}" if backend_label else ""
            self.line(f"{lead} asking{who}: {ask}")
            return
        tail = f", forwarding to {backend_label}" if backend_label else ""
        self.line(f"{lead} {command} failed (exit {exit_code}){tail}")

    # -- the waiting indicator ---------------------------------------------

    def _paint_waiting(self, seconds: int) -> None:
        """Repaint (tty) or print once (everything else) the waiting line."""
        s = self.style
        with self._write_lock:
            if s.enabled:
                self._waiting_shown = True
                self.write(f"{_ERASE_LINE}{s.dim}... {_WAIT_TEXT} ({seconds}s){s.reset}")
            elif not self._waiting_plain:
                self._waiting_plain = True
                self.line(f"... {_WAIT_TEXT}")

    def _clear_waiting(self) -> None:
        """Erase the in-place waiting line, if one is on screen."""
        with self._write_lock:
            if self._waiting_shown:
                self._waiting_shown = False
                self.write(_ERASE_LINE)

    def _wait_ticker(self, stop: threading.Event) -> None:
        """Background thread: the only thing that ever repaints in place.

        It owns no state of its own beyond the last count it painted; the
        stream loop pauses it (``_waiting_paused``) for as long as it is
        handling an event, so a proposal prompt is never painted over.
        """
        painted = -1
        while not stop.wait(_WAIT_TICK):
            with self._write_lock:
                if stop.is_set() or self._waiting_paused:
                    painted = -1
                    continue
                elapsed = time.monotonic() - self._waiting_since
                if elapsed < _WAIT_AFTER:
                    continue
                seconds = int(elapsed)
                if seconds == painted:
                    continue
                painted = seconds
                self._paint_waiting(seconds)

    def _arm_waiting(self) -> None:
        with self._write_lock:
            self._waiting_since = time.monotonic()
            self._waiting_paused = False

    def _pause_waiting(self) -> None:
        """Stop the ticker and erase its line before anything else prints."""
        with self._write_lock:
            self._waiting_paused = True
            self._clear_waiting()

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
        stop_waiting = threading.Event()
        self._waiting_shown = False
        self._waiting_plain = False
        ticker = threading.Thread(
            target=self._wait_ticker, args=(stop_waiting,), daemon=True, name="nvsh-wait"
        )
        try:
            # Started inside the try: a SIGINT that lands during start() must
            # still end as an interrupted stream, not a traceback.
            ticker.start()
            self._arm_waiting()
            for event in events:
                # Everything below prints; the ticker must be off and its
                # line erased first, and stay off until the event is handled
                # (``on_proposal`` blocks on a keypress).
                self._pause_waiting()
                if event.kind is EventKind.TEXT_DELTA:
                    started_text = True
                    result.text += event.text
                    self.write(event.text)
                elif event.kind is EventKind.STATUS:
                    if event.text:
                        if started_text:
                            self.line()
                            started_text = False
                        self.status(event.text)
                elif event.kind is EventKind.TOOL_CALL:
                    self.tool_call(event.tool, (event.args or {}).get("command"))
                elif event.kind is EventKind.TOOL_RESULT:
                    self.tool_result(event.tool, tool_exit_code(event.result))
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
                self._arm_waiting()
        except (KeyboardInterrupt, _Interrupted):
            result.interrupted = True
        finally:
            stop_waiting.set()
            if ticker.ident is not None:
                _quiet(ticker.join, 2.0)
            self._pause_waiting()
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

    def detail_proposal(self, proposal: Proposal, details: Mapping[str, str] | None = None) -> None:
        """Print everything nvsh knows about this proposal (the ``d`` key).

        The proposal's own three fields, then whatever the caller knows and
        the panel cannot: the approval state and the exact patterns
        ``[s]``/``[u]`` would store, which backend and conversation the
        proposal came from, and how many redacted bytes of output went to
        the model (deviation d18 -- the old three lines repeated what the
        box above already showed).
        """
        kind = getattr(proposal.kind, "value", str(proposal.kind))
        self.line(f"kind: {kind}")
        self.line(f"command: {proposal.command}")
        self.line(f"rationale: {proposal.rationale}")
        for label, value in (details or {}).items():
            self.line(f"{label}: {value}")

    def scope_lines(
        self,
        scopes: Mapping[str, str],
        guard: Callable[[str], str | None] | None = None,
    ) -> list[str]:
        """What ``[s]``/``[u]`` would actually store, in the operator's words.

        Operator feedback on d15: the keys looked like they approved the
        exact argument, when ``[u]`` approves *every* argument of that
        command. So the panel says which, per key, before the legend -- and
        for a scope the caller's ``guard`` refuses (a privileged or
        destructive command) it says the key is not available and why,
        rather than promising something that will be declined.
        """
        parts: list[str] = []
        for scope in (APPROVE_SESSION, APPROVE_USER):
            what = scopes.get(scope)
            if not what:
                continue
            key = "[s]" if scope == APPROVE_SESSION else "[u]"
            reason = guard(scope) if guard is not None else None
            if reason:
                parts.append(f"{key} not available: {reason}")
            else:
                parts.append(_SCOPE_OFFER[scope].format(what=what))
        if not parts:
            return []
        joined = " \u00b7 ".join(parts)
        if len(joined) <= _SCOPE_WIDTH:
            return [joined]
        return [one_line(part, _SCOPE_WIDTH) for part in parts]

    def show_proposal(
        self,
        proposal: Proposal,
        guard: Callable[[str], str | None] | None = None,
        scopes: Mapping[str, str] | None = None,
    ) -> str:
        """Show ``proposal`` and return the operator's one keypress.

        Returns ``approve`` / ``session`` / ``user`` / ``explain`` /
        ``details`` / ``ignore`` / ``refused``. The command is never
        pre-typed anywhere: the operator sees it and presses a key. Enter
        runs it once, ``s`` runs it and approves it for this login session,
        ``u`` runs it and approves it for this user, ``e`` explains, ``d``
        shows details, Esc (or Ctrl+C, or anything else) ignores.

        ``guard`` is consulted only for ``s``/``u`` -- the panel does not
        know the approval policy, the caller does. It is handed the scope
        (``"session"``/``"user"``) and returns a one-line reason why that
        scope may not be approved, or ``None`` to allow it. A refusal is
        printed as one line, the keypress is *not* acknowledged as a run,
        and ``refused`` comes back so the caller can ask again with
        run-once still on the table. Re-asking after ``e``/``d``/``refused``
        is the caller's job -- this method reports one keypress and returns.
        """
        self.render_proposal(proposal)
        s = self.style
        for line in self.scope_lines(scopes or {}, guard):
            self.line(f"{s.dim}{line}{s.reset}")
        self.line(LEGEND)
        choice = self._read_choice()
        if choice == TELL:
            # No ack: the ``nvsh> `` prompt read_tell() puts on screen *is*
            # the acknowledgement, and nothing has been decided yet.
            return TELL
        if choice in (APPROVE_SESSION, APPROVE_USER) and guard is not None:
            reason = guard(choice)
            if reason:
                self.line(f"nvsh: cannot approve for this {choice}: {reason}")
                return REFUSED
        self.acknowledge(choice, scopes)
        return choice

    def acknowledge(self, choice: str, scopes: Mapping[str, str] | None = None) -> None:
        """Echo the decision the instant the key is pressed.

        Whatever happens next (a command, a backend round-trip, nothing)
        can take seconds; the operator must not be left wondering whether
        the keypress registered at all. A scope key's ack names the pattern
        it stored, so ``[u]`` can never be mistaken for "approve this one
        argument" (operator feedback on d15).
        """
        what = (scopes or {}).get(choice)
        if what and choice in _SCOPE_ACK:
            self.line(_SCOPE_ACK[choice].format(what=what))
            return
        text = _ACK.get(choice, _ACK[IGNORE])
        if text:
            self.line(text)

    def read_tell(self) -> str:
        """Read one line at an ``nvsh> `` prompt; ``""`` means "never mind".

        Deliberately a *cooked* line read, not the raw single keypress the
        proposal keys use: the operator is typing a sentence, so the tty
        driver must do the echoing, the backspace and the kill-line. The
        terminal is already back in cooked mode by the time this runs
        (``_read_key`` restores it with ``TCSADRAIN``), so nothing here
        changes termios at all.

        Ctrl+C cancels back to the proposal rather than ending the panel:
        for the length of the read, SIGINT is *this* method's, not the
        stream's -- the stream's handler would cancel the agent's turn,
        which is the opposite of what an operator who mistyped wants.
        """
        self.write("nvsh> ")
        previous = _install_sigint(_interrupt_handler)
        try:
            raw = self._read_line()
        except (KeyboardInterrupt, _Interrupted):
            self.line()
            return ""
        finally:
            _restore_sigint(previous)
        return raw.strip()

    def _read_line(self) -> str:
        try:
            raw = self.in_.readline()
        except (KeyboardInterrupt, _Interrupted):
            raise
        except Exception:  # noqa: BLE001 - a closed stdin is simply "nothing typed"
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw or "")

    def _read_choice(self) -> str:
        if self.isatty:
            key = _read_key(self.in_)
        else:
            key = _read_line_key(self.in_)
        if key in ("\r", "\n"):
            return APPROVE
        if key in ("s", "S"):
            return APPROVE_SESSION
        if key in ("u", "U"):
            return APPROVE_USER
        if key in ("e", "E"):
            return EXPLAIN
        if key in ("d", "D"):
            return DETAILS
        if key in ("t", "T"):
            return TELL
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


def _interrupt_handler(_signum: int, _frame: object) -> None:
    """SIGINT while the panel is reading a line: cancel the read, nothing else."""
    raise KeyboardInterrupt()


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
