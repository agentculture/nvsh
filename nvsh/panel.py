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
from typing import Callable, Iterable, Mapping, Sequence, TextIO

from .agent.base import AgentEvent, EventKind, Proposal, Target
from .approvals import parse_stages

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
#: Deviation d24: the uppercase keys. ``[S]``/``[U]`` approve the same kind
#: of command *for this first argument only* -- ``ssh orin *`` rather than
#: ``ssh *``. Spelled as the pi extension's own choice tokens too, so the
#: client can still forward the operator's answer verbatim.
APPROVE_SESSION_SPECIFIC = "session-specific"
APPROVE_USER_SPECIFIC = "user-specific"
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

#: The one-line key legend. Kept within 80 columns so it never wraps on a
#: bare ssh into a Jetson, where a wrapped legend costs the panel a line and
#: reads as two half-legends. ``+session``/``+user`` are the abbreviation
#: that buys the room, and still say which scope each key approves for.
#: ``[s/S]``/``[u/U]`` are d24's two forms of each scope (the uppercase key
#: keeps the first argument); ``[e] why`` is the four columns that paid for
#: them, and asks the same question ``explain`` did.
LEGEND = "[Enter] run [s/S] +session [u/U] +user [e] why [d] details [t] tell [Esc] ignore"

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
    APPROVE_SESSION_SPECIFIC: "nvsh: running (approved for this session) ...",
    APPROVE_USER: "nvsh: running (approved for this user) ...",
    APPROVE_USER_SPECIFIC: "nvsh: running (approved for this user) ...",
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
    APPROVE_SESSION_SPECIFIC: "nvsh: running; {what} approved for this session",
    APPROVE_USER: "nvsh: running; {what} approved for this user",
    APPROVE_USER_SPECIFIC: "nvsh: running; {what} approved for this user",
}

#: The two scope families, each offered on one line as its pair of keys:
#: ``(plain scope, specific scope, plain key, specific key, lifetime)``.
_SCOPE_FAMILIES = (
    (APPROVE_SESSION, APPROVE_SESSION_SPECIFIC, "[s]", "[S]", "this session"),
    (APPROVE_USER, APPROVE_USER_SPECIFIC, "[u]", "[U]", "persisted for you"),
)


def _clip(text: str, width: int = _SCOPE_WIDTH) -> str:
    """``text`` truncated to ``width`` columns, with its own spacing kept.

    Unlike :func:`one_line` this does *not* collapse runs of spaces: the
    scope line uses double spaces to separate its two key offers, and
    collapsing them would run ``[s] ...`` straight into ``[S] ...``.
    """
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"


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


class Panel:
    """Renders one failure panel onto ``out``, reading keys from ``in_``."""

    def __init__(
        self,
        out: TextIO | None = None,
        in_: TextIO | None = None,
        env: Mapping[str, str] | None = None,
        isatty: bool | None = None,
        target: Target | None = None,
        path: str = "",
        warm: bool = False,
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
        # d26: which pipeline stages the last scope keypress should cover.
        # ``None`` means "not asked" (a single-stage command, or a key that
        # stores nothing); a list is what the ``stages [all,1,2]: `` prompt
        # read back, and the caller stores exactly those stages.
        self.stage_choice: list[int] | None = None
        # t18: which backend/model/effort this panel is talking to, and
        # whether the run is served warm (a live daemon session) or as a
        # one-shot process -- rendered as one header line at the start of
        # :meth:`stream`. ``target`` stays ``None`` for a caller that does
        # not know (or does not care) which target is serving the request,
        # in which case :meth:`stream` renders nothing new (backward
        # compatible with every pre-t18 caller).
        self._target: Target | None = None
        self._target_path = ""
        self._target_warm = False
        # Whether a THINKING run is currently open (styled mode only; the
        # plain-mode fallback prints one line per delta and never needs to
        # track an open run).
        self._thinking_open = False
        self.set_target(target, path, warm)

    def set_target(self, target: Target | None, path: str = "", warm: bool = False) -> None:
        """Set (or clear) which target this panel is talking to (t18).

        ``target=None`` (the default) opts a panel out of the header line
        entirely -- :meth:`stream` renders nothing new, exactly as before
        this method existed. Callers that know the target may pass it here
        instead of through the constructor, e.g. once the daemon resolves an
        alias to a concrete backend/model after the panel is already built.
        """
        self._target = target
        self._target_path = path
        self._target_warm = warm

    def _target_header_line(self) -> str:
        """``harness/model/effort · path · warm|one-shot`` (t18)."""
        target = self._target
        assert target is not None
        parts = [target.backend]
        if target.model:
            parts.append(target.model)
        if target.effort:
            parts.append(target.effort)
        harness = "/".join(parts)
        warmth = "warm" if self._target_warm else "one-shot"
        return f"{harness} · {self._target_path} · {warmth}"

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
        if self._target is not None:
            self.line(self._target_header_line())
        saved_attrs = _save_termios(self.in_, self.isatty)
        previous = _install_sigint(self._on_sigint(cancel))
        started_text = False
        self._thinking_open = False
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
                started_text, last = self._render_event(
                    event, result, started_text, on_proposal=on_proposal
                )
                if last:
                    break
                self._arm_waiting()
        except KeyboardInterrupt:
            result.interrupted = True
        finally:
            stop_waiting.set()
            if ticker.ident is not None:
                _quiet(ticker.join, 2.0)
            self._pause_waiting()
            _restore_sigint(previous)
            _restore_termios(saved_attrs)
            self._close_thinking()
            if started_text:
                self.line()
            if result.interrupted:
                self.line(f"{self.style.dim}nvsh: interrupted{self.style.reset}")
        return result

    def _end_text_run(self, started_text: bool) -> bool:
        """Close an open run of ``text_delta`` output with a newline.

        Text arrives in fragments and is written without one, so anything
        that is *not* more text (a status, a proposal, an error) has to
        break the line first or it lands mid-sentence. Returns the new
        ``started_text`` (always ``False``: the run is over).
        """
        if started_text:
            self.line()
        return False

    def _render_thinking(self, text: str) -> None:
        """Render one THINKING delta: a dimmed run on a tty, plain lines off it.

        Styled mode opens the dim SGR once (``\\x1b[2m``) and writes each
        delta into that same run as it arrives; the run is closed (reset +
        newline) by :meth:`_close_thinking` before the first TEXT_DELTA,
        TOOL_CALL, PROPOSAL or DONE reaches the screen (or at the end of
        :meth:`stream`, if none of those follow). Off a tty, or under
        NO_COLOR/TERM=dumb, no SGR is ever emitted: each delta becomes its
        own ``thinking: `` prefixed plain line.
        """
        if not text:
            return
        if self.style.enabled:
            if not self._thinking_open:
                self.write(self.style.dim)
                self._thinking_open = True
            self.write(text)
        else:
            self.line(f"thinking: {text}")

    def _close_thinking(self) -> None:
        """Close an open dim THINKING run, if one is open. No-op otherwise."""
        if self._thinking_open:
            self.write(f"{self.style.reset}\n")
            self._thinking_open = False

    def _render_event(
        self,
        event: AgentEvent,
        result: StreamResult,
        started_text: bool,
        *,
        on_proposal: Callable[[Proposal, AgentEvent], object] | None,
    ) -> tuple[bool, bool]:
        """Render one streamed event onto the panel and record it.

        Returns ``(started_text, last)`` -- whether a run of text is still
        open on the current line, and whether this event ends the stream
        (``error`` and ``done`` do; every other kind does not, including an
        event whose kind carries nothing this panel renders).
        """
        kind = event.kind
        if kind is EventKind.THINKING:
            self._render_thinking(event.text)
            return started_text, False
        # Any non-THINKING event closes an open THINKING run first, so it is
        # always closed before the first TEXT_DELTA/TOOL_CALL/PROPOSAL/DONE
        # (and before STATUS/ERROR too, for the same reason: nothing else may
        # ever land mid-dim-run).
        self._close_thinking()
        if kind is EventKind.TEXT_DELTA:
            result.text += event.text
            self.write(event.text)
            return True, False
        if kind is EventKind.STATUS:
            if event.text:
                started_text = self._end_text_run(started_text)
                self.status(event.text)
            return started_text, False
        if kind is EventKind.TOOL_CALL:
            self.tool_call(event.tool, (event.args or {}).get("command"))
            return started_text, False
        if kind is EventKind.TOOL_RESULT:
            self.tool_result(event.tool, tool_exit_code(event.result))
            return started_text, False
        if kind is EventKind.PROPOSAL and event.proposal is not None:
            started_text = self._end_text_run(started_text)
            result.proposals.append(event.proposal)
            if on_proposal is not None:
                # Blocks on a keypress; the ticker stays paused throughout.
                on_proposal(event.proposal, event)
            return started_text, False
        if kind is EventKind.ERROR:
            result.error = event.error
            started_text = self._end_text_run(started_text)
            self.line(f"{self.style.red}nvsh: {event.error}{self.style.reset}")
            return started_text, True
        if kind is EventKind.DONE:
            result.done = True
            return started_text, True
        return started_text, False

    def _on_sigint(self, cancel: Callable[[], object] | None) -> Callable[[int, object], None]:
        def handler(_signum: int, _frame: object) -> None:
            if cancel is not None:
                _quiet(cancel)
            # KeyboardInterrupt, like the one _interrupt_handler raises: a
            # BaseException is what breaks a blocking read without any of
            # the ``except Exception`` guards on the way out swallowing it.
            raise KeyboardInterrupt()

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

    def stages_line(self, stages: Sequence[str]) -> str:
        """``stages: 1 'ls /tmp/git'  2 'grep -i orin'`` -- what the numbers mean.

        Deviation d26. The d24 scope line told an operator on the Spark that
        ``[s]`` approved "each stage exactly" without ever showing what the
        stages *were*, and offered no way to approve only one of them. This
        line numbers them, and those numbers are what the
        ``stages [all,1,2]: `` prompt then reads back.

        The panel formats what it is given and judges nothing: the caller
        renders each token (the client quotes an approvable stage and passes
        ``(not approvable)`` for one no pattern may ever cover). Clipped to
        80 columns like every other scope line.
        """
        numbered = "  ".join(f"{n} {token}" for n, token in enumerate(stages, 1))
        return _clip(f"stages: {numbered}")

    def read_stages(self, count: int) -> list[int]:
        """Ask which stages the approval just pressed should cover (d26).

        One *cooked* line, read exactly the way :meth:`read_tell` reads a
        sentence -- the operator is typing, so the tty driver must do the
        echo, the backspace and the kill-line, and the terminal is already
        back in cooked mode by the time this runs.

        Everything that is not an explicit, valid pick means **all** stages,
        which is what the pre-d26 keypress always did: an empty line, the
        word ``all``, EOF, Ctrl+C, and a second unreadable answer after one
        re-ask. A stage prompt must never cost the operator their approval.
        """
        every = list(range(1, count + 1))
        choices = ",".join(["all"] + [str(n) for n in every])
        for attempt in range(2):
            self.write(f"stages [{choices}]: ")
            previous = _install_sigint(_interrupt_handler)
            try:
                raw = self._read_line()
            except KeyboardInterrupt:
                self.line()
                return every
            finally:
                _restore_sigint(previous)
            picked = parse_stages(raw.strip(), count)
            if picked:
                return picked
            if attempt == 0:
                self.line(f"nvsh: type 'all' or stage numbers 1-{count}")
        return every

    def scope_lines(
        self,
        scopes: Mapping[str, str],
        guard: Callable[[str], str | None] | None = None,
    ) -> list[str]:
        """What each scope key would actually store, in the operator's words.

        One line per scope family, naming both of its keys (deviation d24):

        .. code-block:: text

            [s] this exact line  [S] 'ssh orin *'  (this session)
            [u] 'ssh *'  [U] 'ssh orin *'  (persisted for you)

        Operator feedback on d15 was that the keys looked like they approved
        the exact argument, when ``[u]`` approves *every* argument; d24's
        complaint from the Spark was the same one sharpened -- ``ssh *`` for
        an ``ssh orin "..."`` proposal is far too broad. So the panel names
        both forms before the legend. A family the caller's ``guard``
        refuses (a privileged, destructive or opaque command) says so once,
        for both of its keys, rather than promising something that will be
        declined. Each line is clipped to 80 columns, which is what keeps a
        long multi-stage pattern list from wrapping.
        """
        lines: list[str] = []
        for plain, specific, plain_key, specific_key, lifetime in _SCOPE_FAMILIES:
            what = scopes.get(plain)
            if not what:
                continue
            reason = guard(plain) if guard is not None else None
            if reason:
                lines.append(_clip(f"{plain_key}/{specific_key} not available: {reason}"))
                continue
            specific_what = scopes.get(specific) or what
            lines.append(_clip(f"{plain_key} {what}  {specific_key} {specific_what}  ({lifetime})"))
        return lines

    def show_proposal(
        self,
        proposal: Proposal,
        guard: Callable[[str], str | None] | None = None,
        scopes: Mapping[str, str] | None = None,
        patterns: Mapping[str, str] | None = None,
        stages: Sequence[str] | None = None,
        stage_patterns: Mapping[str, Sequence[str]] | None = None,
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

        ``stages`` (deviation d26) is one rendered token per pipeline stage.
        With more than one, the panel prints the numbered ``stages:`` line
        above the scope lines and, after a scope key, asks which of them the
        approval should cover. The answer lands on :attr:`stage_choice` --
        the keypress itself stays this method's return value, because the
        caller's five other branches (``e``/``d``/``t``/refused/ignore) do
        not have stages -- and the ack then names only the patterns that
        were actually stored. Whichever stages were picked, the *proposal*
        still runs once: the keypress approved this execution, and storing
        is only about future turns.
        """
        self.stage_choice = None
        self.render_proposal(proposal)
        s = self.style
        if stages and len(stages) > 1:
            self.line(f"{s.dim}{self.stages_line(stages)}{s.reset}")
        for line in self.scope_lines(scopes or {}, guard):
            self.line(f"{s.dim}{line}{s.reset}")
        self.line(LEGEND)
        choice = self._read_choice()
        if choice == TELL:
            # No ack: the ``nvsh> `` prompt read_tell() puts on screen *is*
            # the acknowledgement, and nothing has been decided yet.
            return TELL
        if choice in _SCOPE_ACK and guard is not None:
            reason = guard(choice)
            if reason:
                self.line(f"nvsh: cannot approve for this {choice}: {reason}")
                return REFUSED
        if choice in _SCOPE_ACK and stages and len(stages) > 1:
            self.stage_choice = self.read_stages(len(stages))
        self.acknowledge(
            choice, scopes, patterns, stage_patterns, self.stage_choice, len(stages or ())
        )
        return choice

    def acknowledge(
        self,
        choice: str,
        scopes: Mapping[str, str] | None = None,
        patterns: Mapping[str, str] | None = None,
        stage_patterns: Mapping[str, Sequence[str]] | None = None,
        chosen: Sequence[int] | None = None,
        total: int = 0,
    ) -> None:
        """Echo the decision the instant the key is pressed.

        Whatever happens next (a command, a backend round-trip, nothing)
        can take seconds; the operator must not be left wondering whether
        the keypress registered at all. A scope key's ack names the pattern
        it stored, so ``[u]`` can never be mistaken for "approve this one
        argument" (operator feedback on d15).

        When the operator picked a *subset* of a pipeline's stages (d26) the
        ack names exactly those patterns and says which stages they came
        from -- ``'grep -i *' approved for this session (stage 2 of 2)`` --
        so the line can never over-report what was stored. Picking every
        stage is the pre-d26 case and keeps the pre-d26 wording.
        """
        subset = self._stage_ack(choice, stage_patterns, chosen, total)
        if subset:
            self.line(subset)
            return
        what = (patterns or {}).get(choice) or (scopes or {}).get(choice)
        if what and choice in _SCOPE_ACK:
            self.line(_SCOPE_ACK[choice].format(what=what))
            return
        text = _ACK.get(choice, _ACK[IGNORE])
        if text:
            self.line(text)

    @staticmethod
    def _stage_ack(
        choice: str,
        stage_patterns: Mapping[str, Sequence[str]] | None,
        chosen: Sequence[int] | None,
        total: int,
    ) -> str | None:
        """The d26 ack line for a partial stage pick, or ``None``."""
        if choice not in _SCOPE_ACK or not chosen or total < 2 or len(chosen) >= total:
            return None
        available = (stage_patterns or {}).get(choice) or ()
        picked = [available[n - 1] for n in chosen if 1 <= n <= len(available)]
        if not picked:
            return None
        numbers = ",".join(str(n) for n in chosen)
        noun = "stage" if len(chosen) == 1 else "stages"
        line = _SCOPE_ACK[choice].format(what=" ".join(picked))
        return f"{line} ({noun} {numbers} of {total})"

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
        except KeyboardInterrupt:
            self.line()
            return ""
        finally:
            _restore_sigint(previous)
        return raw.strip()

    def _read_line(self) -> str:
        try:
            raw = self.in_.readline()
        except KeyboardInterrupt:
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
        if key == "s":
            return APPROVE_SESSION
        if key == "S":
            return APPROVE_SESSION_SPECIFIC
        if key == "u":
            return APPROVE_USER
        if key == "U":
            return APPROVE_USER_SPECIFIC
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


def _restore_termios(saved) -> None:
    """Put back what :func:`_save_termios` returned (``(fd, attrs)``, or none)."""
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
