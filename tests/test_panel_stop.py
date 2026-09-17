"""Tests for the panel's stop state (tasks t14 and t5).

Since t5 the *first* Ctrl+C or lone Esc no longer stops anything: it opens
the stop-choice prompt (``nvsh: paused -- [t] steer  [s] stop  [Esc] keep
going``) and nothing reaches the harness until the operator picks. ``[s]``
(or a Ctrl+C typed at the prompt) is the old first press -- the stopping
line, one polite ``cancel``, ``interrupted`` -- and the press after that
still calls ``force_stop`` exactly once. ``[Esc]`` and the
:data:`nvsh.panel.STOP_PROMPT_TIMEOUT` timeout resume rendering with the
turn untouched.

Also covered, unchanged since t14: Esc is seen even while the event source
yields nothing; an arrow key -- during streaming or at the proposal --
neither interrupts nor counts as Esc and leaves no stray bytes behind; and
a terminal that cannot show the prompt (not a tty, ``TERM=dumb``) stops
immediately as before.
"""

from __future__ import annotations

import io
import os
import pty
import select
import signal
import threading
import time

import pytest

from nvsh import panel as panel_mod
from nvsh.agent.base import AgentEvent, EventKind, Proposal, ProposalKind

TTY_ENV = {"TERM": "xterm-256color", "NO_COLOR": "1"}
STYLED_ENV = {"TERM": "xterm-256color"}
STOPPING = "stopping… press again to kill"
PAUSED_LEGEND = "[t] steer  [s] stop  [Esc] keep going"
PAUSED = f"nvsh: paused -- {PAUSED_LEGEND}"


@pytest.fixture
def pty_pair():
    master, slave = pty.openpty()
    tty_in = os.fdopen(slave, "rb", buffering=0)
    try:
        yield master, tty_in
    finally:
        tty_in.close()
        try:
            os.close(master)
        except OSError:
            pass


def _panel(tty_in, out=None, env=None):
    return panel_mod.Panel(
        out=out if out is not None else io.StringIO(),
        in_=tty_in,
        env=TTY_ENV if env is None else env,
        isatty=True,
    )


def _leftover(fd: int) -> bool:
    return bool(select.select([fd], [], [], 0.1)[0])


def _press(kind: str, master: int) -> None:
    if kind == "esc":
        os.write(master, b"\x1b")
    else:
        os.kill(os.getpid(), signal.SIGINT)


def _wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _answer(master: int, key: bytes, reacted, tty_in=None, timeout: float = 15.0) -> bool:
    """Type ``key`` at the choice prompt until ``reacted()`` is true.

    The prompt discards typeahead as it opens (c33), so a key written in the
    microseconds between the legend appearing and that flush would be
    dropped. The settle below makes that window practically unreachable, and
    the retry -- not the settle -- is what makes the helper correct: a lost
    key is typed again rather than hanging the test. One write per attempt
    (never a stream of them), so a *late* copy cannot be read as a fresh
    press once the prompt has closed; anything still queued when the panel
    reacts is flushed.
    """
    time.sleep(0.15)
    deadline = time.monotonic() + timeout
    answered = False
    try:
        while not answered and time.monotonic() < deadline:
            try:
                os.write(master, key)
            except OSError:  # pragma: no cover - the pty went away
                break
            answered = _wait_for(reacted, 2.0)
        return answered or reacted()
    finally:
        _drain(tty_in)


def _drain(tty_in) -> None:
    """Drop anything still queued on the slave side of the pty."""
    if tty_in is None:
        return
    try:
        import termios

        termios.tcflush(tty_in.fileno(), termios.TCIFLUSH)
    except Exception:  # noqa: BLE001 - nothing queued is just as good
        return


# --- t5 criterion 1: the first press only opens the prompt ------------------


def _prompt_run(kind: str, master: int, tty_in, *, env=None, lead=None):
    """Press once, wait for the prompt, and report what was called by then."""
    out = io.StringIO()
    p = _panel(tty_in, out, env=env)
    calls: dict[str, list] = {"cancel": [], "kill": [], "choice": []}
    at_prompt: dict[str, object] = {}

    def events():
        if lead is None:
            yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        else:
            yield lead
        pressed = time.monotonic()
        _press(kind, master)
        assert _wait_for(lambda: PAUSED_LEGEND in out.getvalue()), out.getvalue()
        at_prompt["delay"] = time.monotonic() - pressed
        # Nothing may have been sent to the harness while the prompt is open.
        at_prompt["calls"] = {name: list(seen) for name, seen in calls.items()}
        assert _answer(master, b"\x1b", lambda: calls["choice"], tty_in), out.getvalue()
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" winding down")
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(
        events(),
        cancel=lambda: calls["cancel"].append(1),
        force_stop=lambda: calls["kill"].append(1),
        on_choice=lambda outcome, reason: calls["choice"].append((outcome, reason)),
    )
    return result, out.getvalue(), calls, at_prompt


@pytest.mark.parametrize("kind", ["esc", "sigint"])
def test_first_press_opens_the_prompt_within_1s_and_calls_nothing(pty_pair, kind):
    master, tty_in = pty_pair
    result, text, calls, at_prompt = _prompt_run(kind, master, tty_in)
    assert at_prompt["delay"] < 1.0, at_prompt
    assert at_prompt["calls"] == {"cancel": [], "kill": [], "choice": []}
    assert calls["cancel"] == [] and calls["kill"] == []
    assert PAUSED in text
    assert STOPPING not in text
    # The open run of text was closed before the prompt line.
    assert "working\n" in text
    assert text.index("working\n") < text.index(PAUSED)
    assert result.done is True
    assert result.interrupted is False


def test_first_press_closes_an_open_thinking_run_before_the_prompt(pty_pair):
    master, tty_in = pty_pair
    _result, text, _calls, _at = _prompt_run(
        "esc",
        master,
        tty_in,
        env=STYLED_ENV,
        lead=AgentEvent(kind=EventKind.THINKING, text="pondering"),
    )
    # The dim run is reset and broken before the legend reaches the screen.
    assert "\x1b[0m\n" in text
    assert text.index("\x1b[0m\n") < text.index(PAUSED_LEGEND)


# --- t5 criterion 2: [Esc] and the timeout keep going -----------------------


def _keep_going_run(pty_pair, answer: bytes | None):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    cancelled: list[int] = []
    killed: list[int] = []
    choices: list[tuple[str, str]] = []

    def events():
        for word in ("one ", "two "):
            yield AgentEvent(kind=EventKind.TEXT_DELTA, text=word)
        _press("esc", master)
        assert _wait_for(lambda: PAUSED_LEGEND in out.getvalue()), out.getvalue()
        if answer is None:
            assert _wait_for(lambda: choices, timeout=30), "the prompt never timed out"
        else:
            assert _answer(master, answer, lambda: choices, tty_in), out.getvalue()
        for word in ("three ", "four ", "five"):
            yield AgentEvent(kind=EventKind.TEXT_DELTA, text=word)
        yield AgentEvent(kind=EventKind.DONE)

    started = time.monotonic()
    result = p.stream(
        events(),
        cancel=lambda: cancelled.append(1),
        force_stop=lambda: killed.append(1),
        on_choice=lambda outcome, reason: choices.append((outcome, reason)),
    )
    return result, out.getvalue(), cancelled, killed, choices, time.monotonic() - started


def test_esc_at_the_prompt_keeps_going_with_no_events_lost(pty_pair):
    result, out, cancelled, killed, choices, _elapsed = _keep_going_run(pty_pair, b"\x1b")
    assert choices == [("keep_going", "key")]
    assert result.interrupted is False
    assert result.done is True
    assert result.text == "one two three four five"
    assert cancelled == [] and killed == []
    assert STOPPING not in out
    assert "nvsh: interrupted" not in out


def test_timeout_at_the_prompt_keeps_going(pty_pair, monkeypatch):
    monkeypatch.setattr(panel_mod, "STOP_PROMPT_TIMEOUT", 0.5)
    result, out, cancelled, killed, choices, elapsed = _keep_going_run(pty_pair, None)
    assert choices == [("keep_going", "timeout")]
    # The timeout was honoured (not answered instantly); the upper bound is
    # only a "the stream ended" guard, generous for a loaded machine.
    assert 0.4 < elapsed < 30.0, elapsed
    assert result.interrupted is False
    assert result.done is True
    assert result.text == "one two three four five"
    assert cancelled == [] and killed == []
    assert STOPPING not in out


# --- t5 criterion 5: the [t] label, and what [t] reports ---------------------


@pytest.mark.parametrize("label", ["steer", "stop & correct"])
def test_steer_label_is_used_verbatim_and_t_reports_steer(pty_pair, label):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    cancelled: list[int] = []
    killed: list[int] = []
    choices: list[tuple[str, str]] = []

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("esc", master)
        assert _wait_for(lambda: f"[t] {label}" in out.getvalue()), out.getvalue()
        assert _answer(master, b"t", lambda: choices, tty_in), out.getvalue()
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(
        events(),
        cancel=lambda: cancelled.append(1),
        force_stop=lambda: killed.append(1),
        on_choice=lambda outcome, reason: choices.append((outcome, reason)),
        steer_label=label,
    )
    text = out.getvalue()
    assert f"nvsh: paused -- [t] {label}  [s] stop  [Esc] keep going" in text
    assert choices == [("steer", "key")]
    assert cancelled == [] and killed == []
    assert result.interrupted is False
    assert result.done is True
    # t5 reads no correction line yet: no ``nvsh> `` prompt was shown.
    assert "nvsh> " not in text


# --- t5 criterion 4: no prompt where it cannot be shown ---------------------


def test_non_tty_first_sigint_cancels_at_once_without_a_prompt(capsys):
    out = io.StringIO()
    p = panel_mod.Panel(out=out, in_=io.StringIO(""), env={}, isatty=False)
    cancelled: list[int] = []
    stopped = threading.Event()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        os.kill(os.getpid(), signal.SIGINT)
        assert stopped.wait(5)
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(
        events(),
        cancel=lambda: (cancelled.append(1), stopped.set()),
        force_stop=lambda: None,
    )
    text = out.getvalue()
    assert cancelled == [1]
    assert result.interrupted is True
    assert STOPPING in text
    assert "paused" not in text
    assert PAUSED_LEGEND not in text
    captured = capsys.readouterr()
    assert "paused" not in captured.out
    assert "paused" not in captured.err


def test_term_dumb_first_press_cancels_at_once_without_a_prompt(pty_pair):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out, env={"TERM": "dumb"})
    cancelled: list[int] = []
    stopped = threading.Event()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("sigint", master)
        assert stopped.wait(5)
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(
        events(),
        cancel=lambda: (cancelled.append(1), stopped.set()),
        force_stop=lambda: None,
    )
    text = out.getvalue()
    assert cancelled == [1]
    assert result.interrupted is True
    assert STOPPING in text
    assert "paused" not in text


# --- t5 criterion 3: [s] is today's first press ----------------------------


def _one_press_run(
    kind: str, master: int, tty_in, answer: bytes = b"s"
) -> tuple[panel_mod.StreamResult, str, list, list]:
    out = io.StringIO()
    p = _panel(tty_in, out)
    cancelled: list[int] = []
    killed: list[int] = []
    stopped = threading.Event()

    def cancel():
        cancelled.append(1)
        stopped.set()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press(kind, master)
        assert _wait_for(lambda: PAUSED_LEGEND in out.getvalue()), out.getvalue()
        assert _answer(master, answer, stopped.is_set, tty_in), out.getvalue()
        # The panel stays up after the stop: this still renders.
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" winding down")

    result = p.stream(events(), cancel=cancel, force_stop=lambda: killed.append(1))
    return result, out.getvalue(), cancelled, killed


def test_lone_esc_during_streaming_is_identical_to_sigint(pty_pair):
    master, tty_in = pty_pair
    by_esc, esc_out, esc_cancel, esc_kill = _one_press_run("esc", master, tty_in)
    by_int, int_out, int_cancel, int_kill = _one_press_run("sigint", master, tty_in)
    assert by_esc.interrupted is True
    assert by_esc == by_int
    assert by_esc.text == "working winding down"
    assert esc_cancel == int_cancel == [1]
    assert esc_kill == int_kill == []
    for text in (esc_out, int_out):
        assert PAUSED in text
        assert STOPPING in text
        assert "nvsh: interrupted" in text
        assert text.index(PAUSED) < text.index(STOPPING) < text.index("winding down")
    assert esc_out == int_out


@pytest.mark.parametrize("kind", ["esc", "sigint"])
def test_a_single_press_never_calls_force_stop(pty_pair, kind):
    master, tty_in = pty_pair
    result, out, cancelled, killed = _one_press_run(kind, master, tty_in)
    assert result.interrupted is True
    assert cancelled == [1]
    assert killed == []
    assert out.count(STOPPING) == 1


def test_ctrl_c_typed_at_the_choice_prompt_stops_like_s(pty_pair):
    result, out, cancelled, killed = _one_press_run("esc", pty_pair[0], pty_pair[1], b"\x03")
    assert result.interrupted is True
    assert cancelled == [1]
    assert killed == []
    assert out.count(STOPPING) == 1


# --- second press: kill ----------------------------------------------------


@pytest.mark.parametrize("first,second", [("esc", "esc"), ("sigint", "esc"), ("esc", "sigint")])
def test_second_press_calls_force_stop_exactly_once(pty_pair, first, second):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    cancelled: list[int] = []
    killed: list[int] = []
    gone = threading.Event()
    timing: dict[str, float] = {}

    def force_stop():
        # Timed here, on the thread that does the killing: the stream ends
        # the moment this returns, so the source thread may never be
        # scheduled again to record anything.
        timing["kill"] = time.monotonic() - timing["pressed"]
        killed.append(1)
        gone.set()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press(first, master)
        assert _wait_for(lambda: PAUSED_LEGEND in out.getvalue()), out.getvalue()
        assert _answer(master, b"s", lambda: cancelled, tty_in), out.getvalue()
        time.sleep(0.15)  # a separate keypress, not one ESC ESC burst
        timing["pressed"] = time.monotonic()
        _press(second, master)
        # A harness that ignores the polite cancel: silent until killed.
        gone.wait(10)

    started = time.monotonic()
    result = p.stream(events(), cancel=lambda: cancelled.append(1), force_stop=force_stop)
    assert time.monotonic() - started < 30
    assert timing["kill"] < 3.0, timing
    assert result.interrupted is True
    assert cancelled == [1]
    assert killed == [1]
    assert out.getvalue().count(STOPPING) == 1
    assert out.getvalue().count(PAUSED) == 1


def test_second_press_without_force_stop_still_ends_the_stream(pty_pair):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    cancelled: list[int] = []
    release = threading.Event()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("esc", master)
        assert _wait_for(lambda: PAUSED_LEGEND in out.getvalue()), out.getvalue()
        assert _answer(master, b"s", lambda: cancelled, tty_in), out.getvalue()
        time.sleep(0.15)
        _press("esc", master)
        release.wait(10)

    started = time.monotonic()
    try:
        result = p.stream(events(), cancel=lambda: cancelled.append(1))
    finally:
        release.set()
    assert time.monotonic() - started < 30
    assert result.interrupted is True
    assert cancelled == [1]


# --- silent source ---------------------------------------------------------


def test_esc_prompts_then_stops_within_1s_each_while_the_source_is_silent(pty_pair):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    killed = threading.Event()
    timing: dict[str, float] = {}

    def events():
        # Nothing for 30s unless the harness is killed.
        killed.wait(30)
        return
        yield  # pragma: no cover - makes this a generator

    def operator():
        time.sleep(0.3)
        os.write(master, b"\x1b")
        pressed = time.monotonic()
        if _wait_for(lambda: PAUSED_LEGEND in out.getvalue(), timeout=5):
            timing["paused"] = time.monotonic() - pressed
        chose = time.monotonic()
        if _answer(master, b"s", lambda: STOPPING in out.getvalue(), tty_in):
            timing["stopping"] = time.monotonic() - chose
        time.sleep(0.15)
        os.write(master, b"\x1b")

    thread = threading.Thread(target=operator, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        result = p.stream(events(), cancel=lambda: None, force_stop=killed.set)
    finally:
        killed.set()
        thread.join(5)
    assert "paused" in timing, out.getvalue()
    assert timing["paused"] < 1.0, timing
    assert "stopping" in timing, out.getvalue()
    assert timing["stopping"] < 1.0, timing
    assert result.interrupted is True
    assert time.monotonic() - started < 30


# --- arrow keys ------------------------------------------------------------


def test_arrow_key_during_streaming_is_not_esc_and_leaves_no_bytes(pty_pair):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    cancelled: list[int] = []

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        os.write(master, b"\x1b[A")
        time.sleep(0.4)
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" done")
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(events(), cancel=lambda: cancelled.append(1))
    assert result.interrupted is False
    assert result.done is True
    assert result.text == "working done"
    assert cancelled == []
    assert "stopping" not in out.getvalue()
    assert not _leftover(tty_in.fileno())


def test_arrow_key_at_the_proposal_is_not_esc_and_leaves_no_bytes(pty_pair):
    master, tty_in = pty_pair
    p = _panel(tty_in)

    def typist():
        # After raw mode is entered (tty.setraw flushes earlier input).
        time.sleep(0.3)
        os.write(master, b"\x1b[B")
        time.sleep(0.3)
        os.write(master, b"\r")

    thread = threading.Thread(target=typist, daemon=True)
    thread.start()
    try:
        choice = p.show_proposal(Proposal("df -h", "disk", ProposalKind.INSPECT))
    finally:
        thread.join(5)
    assert choice == panel_mod.APPROVE
    assert not _leftover(tty_in.fileno())


def test_lone_esc_at_the_proposal_still_ignores(pty_pair):
    master, tty_in = pty_pair
    p = _panel(tty_in)
    typist = threading.Timer(0.3, lambda: os.write(master, b"\x1b"))
    typist.start()
    try:
        assert p.show_proposal(Proposal("df -h", "disk", ProposalKind.INSPECT)) == "ignore"
    finally:
        typist.cancel()
    assert not _leftover(tty_in.fileno())


def test_proposal_key_read_does_not_spin_on_a_hung_up_tty():
    master, slave = pty.openpty()
    with os.fdopen(slave, "rb", buffering=0) as tty_in:
        p = _panel(tty_in)
        os.close(master)
        started = time.monotonic()
        assert p.show_proposal(Proposal("df -h", "disk", ProposalKind.INSPECT)) == "ignore"
        assert time.monotonic() - started < 2


def test_raw_key_read_returns_empty_at_eof_instead_of_spinning():
    read_end, write_end = os.pipe()
    os.close(write_end)
    try:
        started = time.monotonic()
        assert panel_mod._read_raw_key(read_end) == ""
        assert time.monotonic() - started < 1
    finally:
        os.close(read_end)


# --- the proposal suspends the watcher -------------------------------------


def test_keys_at_a_proposal_mid_stream_reach_the_proposal_not_the_watcher(pty_pair):
    master, tty_in = pty_pair
    p = _panel(tty_in)
    choices: list[str] = []
    cancelled: list[int] = []

    def on_proposal(proposal, _event):
        threading.Timer(0.3, lambda: os.write(master, b"\x1b")).start()
        choices.append(p.show_proposal(proposal))

    def events():
        yield AgentEvent(kind=EventKind.PROPOSAL, proposal=Proposal("df -h", "d", ProposalKind.FIX))
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(events(), on_proposal=on_proposal, cancel=lambda: cancelled.append(1))
    assert choices == ["ignore"]
    assert cancelled == []
    assert result.interrupted is False
    assert result.done is True


# --- t15: the busy prompt (steer / replace / exit) -------------------------
#
# nvsh/daemon.py (t10) yields a BUSY event to the shell that owns a still
# running turn (or whose owning shell pid is gone), carrying {owner, elapsed,
# steerable}. panel.show_busy renders that prompt and returns one of
# STEER/REPLACE/BUSY_EXIT. steer is offered only when the adapter serving the
# turn has a mid-turn channel (steerable=True); a harness with none must
# never let 't' be read as steer. When steer is chosen but the harness stays
# silent (no event acknowledging the steer arrives within the timeout), the
# prompt is shown again without steer -- there's nothing left to steer with,
# only replace or exit.


def _busy_panel(typed: str, out=None):
    return panel_mod.Panel(
        out=out if out is not None else io.StringIO(),
        in_=io.StringIO(typed),
        env={},
        isatty=False,
    )


def test_show_busy_steerable_offers_all_three_and_reads_steer():
    out = io.StringIO()
    p = _busy_panel("t\n", out)
    choice = p.show_busy("shell-a", 12.0, True)
    assert choice == panel_mod.STEER
    text = out.getvalue()
    assert "[t]" in text
    assert "steer" in text.lower()


def test_show_busy_steerable_reads_replace_and_exit():
    assert _busy_panel("r\n").show_busy("shell-a", 1.0, True) == panel_mod.REPLACE
    assert _busy_panel("\x1b\n").show_busy("shell-a", 1.0, True) == panel_mod.BUSY_EXIT


def test_show_busy_non_steerable_has_no_steer_option_in_the_legend():
    out = io.StringIO()
    p = _busy_panel("r\n", out)
    p.show_busy("shell-a", 5.0, False)
    text = out.getvalue()
    assert "[t]" not in text
    assert "steer" not in text.lower()


def test_show_busy_non_steerable_t_is_not_accepted_as_steer():
    # 't' means nothing when steer isn't offered: it falls through to exit,
    # the same as any other key the legend doesn't list (Esc, 'q', ...).
    choice = _busy_panel("t\n").show_busy("shell-a", 5.0, False)
    assert choice != panel_mod.STEER
    assert choice == panel_mod.BUSY_EXIT


def test_show_busy_steer_then_event_arrives_returns_steer_without_reoffering():
    out = io.StringIO()
    p = _busy_panel("t\n", out)
    seen_timeouts: list[float] = []

    def await_event(timeout: float) -> bool:
        seen_timeouts.append(timeout)
        return True  # the harness acknowledged the steer in time

    choice = p.show_busy("shell-a", 3.0, True, await_event=await_event)
    assert choice == panel_mod.STEER
    assert seen_timeouts == [10.0]
    # Only ever prompted once: no second legend was printed.
    assert out.getvalue().count("[t]") == 1


def test_show_busy_steer_then_silence_reoffers_replace_and_exit():
    out = io.StringIO()
    # First read: 't' for steer. After the silent steer times out, the
    # prompt is shown again (steer no longer offered) and reads 'r'.
    p = _busy_panel("t\nr\n", out)

    def await_event(timeout: float) -> bool:
        assert timeout == 10.0
        return False  # the harness never acknowledged the steer

    choice = p.show_busy("shell-a", 3.0, True, await_event=await_event)
    assert choice == panel_mod.REPLACE
    text = out.getvalue()
    # The steer option was offered on the first prompt only.
    assert text.count("[t]") == 1
    # The reoffer prompt still names replace and exit.
    assert text.lower().count("replace") >= 2
    assert "exit" in text.lower()


def test_show_busy_steer_then_silence_reoffer_can_exit():
    p = _busy_panel("t\n\x1b\n")

    def await_event(timeout: float) -> bool:
        return False

    choice = p.show_busy("shell-a", 3.0, True, await_event=await_event)
    assert choice == panel_mod.BUSY_EXIT


def test_show_busy_without_await_event_returns_steer_immediately():
    # A caller that does not care about the reoffer (or hasn't wired the
    # daemon event source yet) gets the pre-t17 behaviour: steer returns at
    # once, no waiting.
    choice = _busy_panel("t\n").show_busy("shell-a", 3.0, True)
    assert choice == panel_mod.STEER


# --- Ctrl+C typed at a raw-mode prompt (PR #16 review, Qodo 4) --------------
#
# The proposal and busy prompts read their key in raw mode, where the tty
# driver does not turn Ctrl+C into SIGINT: it arrives as the byte 0x03. That
# byte must reach the stream's stop path -- cancel once, stream interrupted --
# and must never be read as a prompt choice (ignore / busy exit).


def _ctrl_c_at_prompt(pty_pair, prompt: str):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    choices: list[str] = []
    cancelled: list[int] = []
    killed: list[int] = []
    stopped = threading.Event()

    def cancel():
        cancelled.append(1)
        stopped.set()

    def on_prompt(*_args):
        # After raw mode is entered (tty.setraw flushes earlier input).
        threading.Timer(0.3, lambda: os.write(master, b"\x03")).start()
        if prompt == "proposal":
            choices.append(p.show_proposal(_args[0]))
        else:
            choices.append(p.show_busy("4242", 3.0, True))

    def events():
        if prompt == "proposal":
            yield AgentEvent(
                kind=EventKind.PROPOSAL, proposal=Proposal("df -h", "d", ProposalKind.FIX)
            )
        else:
            yield AgentEvent(kind=EventKind.BUSY, args={"owner": "4242", "steerable": True})
        stopped.wait(5)
        yield AgentEvent(kind=EventKind.DONE)

    handler = {"on_proposal": on_prompt} if prompt == "proposal" else {"on_busy": on_prompt}
    result = p.stream(events(), cancel=cancel, force_stop=lambda: killed.append(1), **handler)
    return result, out.getvalue(), choices, cancelled, killed


@pytest.mark.parametrize("prompt", ["proposal", "busy"])
def test_ctrl_c_byte_at_a_raw_prompt_stops_the_agent_not_the_prompt(pty_pair, prompt):
    result, out, choices, cancelled, killed = _ctrl_c_at_prompt(pty_pair, prompt)
    assert choices == []  # no ignore, no busy exit
    assert cancelled == [1]
    assert killed == []
    assert result.interrupted is True
    assert STOPPING in out
    assert "nvsh: interrupted" in out


def test_ctrl_c_byte_raw_key_read_raises_keyboard_interrupt():
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, b"\x03")
        with pytest.raises(KeyboardInterrupt):
            panel_mod._read_raw_key(read_end)
    finally:
        os.close(read_end)
        os.close(write_end)
