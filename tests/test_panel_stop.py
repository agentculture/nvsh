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
    """Type ``key`` once at the choice prompt and wait for ``reacted()``.

    One write is enough and always was meant to be: the prompt discards
    typeahead *before* it draws its legend, so every caller here -- which
    only writes once the legend is in the panel's output -- is writing into
    a prompt that is already reading. (It used to be the other way round,
    and this helper used to re-type a key the flush could still eat.)
    Anything still queued when the panel reacts is flushed, so a key that
    arrives late cannot be read as a fresh press at the next prompt.
    """
    try:
        os.write(master, key)
    except OSError:  # pragma: no cover - the pty went away
        return reacted()
    try:
        return _wait_for(reacted, timeout)
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
    outcomes: list[tuple[str, str]] = []
    steers: list[str] = []
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
    result = p.stream(
        events(),
        cancel=cancel,
        force_stop=lambda: killed.append(1),
        on_choice=lambda outcome, reason: outcomes.append((outcome, reason)),
        on_steer=lambda text: steers.append(text) or True,
        **handler,
    )
    return result, out.getvalue(), choices, cancelled, killed, outcomes, steers


@pytest.mark.parametrize("prompt", ["proposal", "busy"])
def test_ctrl_c_byte_at_a_raw_prompt_stops_the_agent_not_the_prompt(pty_pair, prompt):
    result, out, choices, cancelled, killed, _outcomes, _steers = _ctrl_c_at_prompt(
        pty_pair, prompt
    )
    assert choices == []  # no ignore, no busy exit
    assert cancelled == [1]
    assert killed == []
    assert result.interrupted is True
    assert STOPPING in out
    assert "nvsh: interrupted" in out


# --- t6 criterion 3: a press at another prompt never opens the choice prompt --


@pytest.mark.parametrize("prompt", ["proposal", "busy"])
def test_ctrl_c_at_another_prompt_never_opens_the_choice_prompt(pty_pair, prompt):
    # A Ctrl+C typed while a proposal or busy prompt is open goes straight to
    # today's stop path: no paused legend, no correction line, and neither
    # callback the choice prompt reports through is ever called.
    result, out, _choices, cancelled, _killed, outcomes, steers = _ctrl_c_at_prompt(
        pty_pair, prompt
    )
    assert "paused" not in out
    assert PAUSED_LEGEND not in out
    assert "nvsh> " not in out
    assert outcomes == []
    assert steers == []
    assert cancelled == [1]
    assert result.interrupted is True


def test_ctrl_c_byte_raw_key_read_raises_keyboard_interrupt():
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, b"\x03")
        with pytest.raises(KeyboardInterrupt):
            panel_mod._read_raw_key(read_end)
    finally:
        os.close(read_end)
        os.close(write_end)


# --- t6: the correction line after [t] -------------------------------------
#
# With ``on_steer`` wired, [t] reads one free-text line at the existing
# ``nvsh> `` prompt (Panel.read_tell) and hands it to the caller. An empty
# line, a Ctrl+C or end of input there means "never mind": nothing is handed
# over, nothing is cancelled and the turn keeps going (spec c34). Without
# ``on_steer`` the panel reads no line at all (t5's behaviour, asserted by
# test_steer_label_is_used_verbatim_and_t_reports_steer above).

TELL_PROMPT = "nvsh> "


def _correction_run(pty_pair, type_line, *, taken: bool = True):
    """Press once, choose ``[t]``, then let ``type_line`` answer ``nvsh> ``.

    ``type_line(master, tty_in, out)`` is called once the tell prompt is on
    screen; the stream then waits for ``on_choice`` before producing its
    remaining events, so what the panel did with the line is already decided
    by the time the assertions run.
    """
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    calls: dict[str, list] = {"cancel": [], "kill": [], "choice": [], "steer": []}

    def on_steer(text: str) -> bool:
        calls["steer"].append(text)
        return taken

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("esc", master)
        assert _wait_for(lambda: PAUSED_LEGEND in out.getvalue()), out.getvalue()
        assert _answer(master, b"t", lambda: TELL_PROMPT in out.getvalue(), tty_in), out.getvalue()
        type_line(master, tty_in, out)
        assert _wait_for(lambda: calls["choice"]), out.getvalue()
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" winding down")
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(
        events(),
        cancel=lambda: calls["cancel"].append(1),
        force_stop=lambda: calls["kill"].append(1),
        on_choice=lambda outcome, reason: calls["choice"].append((outcome, reason)),
        on_steer=on_steer,
    )
    return result, out.getvalue(), calls


def _type(line: bytes):
    def typist(master, _tty_in, _out):
        os.write(master, line)

    return typist


def _settle_then(action):
    """Run ``action`` a beat after the tell prompt is on screen.

    The settle is not what makes the test correct -- the poll for
    ``on_choice`` that follows is -- it only keeps the signal out of the
    microseconds between ``nvsh> `` reaching the screen and read_tell taking
    ownership of SIGINT.
    """

    def typist(master, tty_in, _out):
        time.sleep(0.15)
        action(master, tty_in)

    return typist


@pytest.mark.parametrize("line", [b"look at nvpmodel instead\n", b"  look at nvpmodel instead  \n"])
def test_t_reads_a_correction_line_and_hands_it_to_on_steer(pty_pair, line):
    result, out, calls = _correction_run(pty_pair, _type(line))
    assert TELL_PROMPT in out
    assert calls["steer"] == ["look at nvpmodel instead"]
    assert calls["choice"] == [("steer", "key")]
    assert calls["cancel"] == [] and calls["kill"] == []
    assert result.interrupted is False
    assert result.done is True
    assert result.text == "working winding down"
    assert STOPPING not in out


def test_a_correction_the_harness_did_not_take_is_still_only_handed_over(pty_pair):
    # on_steer returning False means "the caller will send it as the next
    # request": the panel neither cancels nor prints a verdict of its own.
    result, out, calls = _correction_run(pty_pair, _type(b"try jtop\n"), taken=False)
    assert calls["steer"] == ["try jtop"]
    assert calls["choice"] == [("steer", "key")]
    assert calls["cancel"] == [] and calls["kill"] == []
    assert result.interrupted is False
    assert result.done is True
    assert STOPPING not in out


@pytest.mark.parametrize(
    "typist",
    [
        _type(b"\n"),
        _type(b"   \n"),
        _settle_then(lambda _master, _tty: os.kill(os.getpid(), signal.SIGINT)),
        _settle_then(lambda master, _tty: os.close(master)),
    ],
    ids=["empty", "blank", "ctrl-c", "eof"],
)
def test_never_mind_at_the_correction_line_sends_nothing(pty_pair, typist):
    result, out, calls = _correction_run(pty_pair, typist)
    assert calls["steer"] == []
    assert calls["choice"] == [("keep_going", "key")]
    assert calls["cancel"] == [] and calls["kill"] == []
    assert result.interrupted is False
    assert result.done is True
    assert result.text == "working winding down"
    assert STOPPING not in out
    assert "nvsh: interrupted" not in out


# --- t6 criterion 2: end of input at the choice prompt itself ---------------


def test_eof_at_the_choice_prompt_keeps_going_and_sends_nothing(pty_pair):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    calls: dict[str, list] = {"cancel": [], "kill": [], "choice": [], "steer": []}

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("esc", master)
        assert _wait_for(lambda: PAUSED_LEGEND in out.getvalue()), out.getvalue()
        os.close(master)  # the terminal hung up while the prompt was open
        assert _wait_for(lambda: calls["choice"]), out.getvalue()
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" winding down")
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(
        events(),
        cancel=lambda: calls["cancel"].append(1),
        force_stop=lambda: calls["kill"].append(1),
        on_choice=lambda outcome, reason: calls["choice"].append((outcome, reason)),
        on_steer=lambda text: calls["steer"].append(text) or True,
    )
    text = out.getvalue()
    assert calls["choice"] == [("keep_going", "")]
    assert calls["cancel"] == [] and calls["kill"] == [] and calls["steer"] == []
    assert result.interrupted is False
    assert result.done is True
    assert result.text == "working winding down"
    assert STOPPING not in text
    assert TELL_PROMPT not in text


# --- t6 criterion 4: the turn finishes while the prompt is open (c36) -------


def _finished_run(pty_pair, answers: list[bytes], *, on_steer_taken: bool = False):
    """Answer the prompt with DONE already queued behind it.

    The source yields DONE the moment the legend is on screen: the feeder
    puts it on the queue and waits for an ack that cannot come until the
    prompt closes, so the panel *can* see that the turn is over. If that race
    were ever lost the panel would report an interrupted turn and the
    assertions below would say so.
    """
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    calls: dict[str, list] = {"cancel": [], "kill": [], "choice": [], "steer": []}
    at_prompt = threading.Event()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="the whole answer")
        _press("esc", master)
        assert _wait_for(lambda: PAUSED_LEGEND in out.getvalue()), out.getvalue()
        at_prompt.set()
        yield AgentEvent(kind=EventKind.DONE)

    def typist():
        assert at_prompt.wait(15)
        for key in answers:
            if key.endswith(b"\n"):
                time.sleep(0.15)
                os.write(master, key)
            else:
                _answer(
                    master, key, lambda: calls["choice"] or TELL_PROMPT in out.getvalue(), tty_in
                )

    thread = threading.Thread(target=typist, daemon=True)
    thread.start()
    try:
        result = p.stream(
            events(),
            cancel=lambda: calls["cancel"].append(1),
            force_stop=lambda: calls["kill"].append(1),
            on_choice=lambda outcome, reason: calls["choice"].append((outcome, reason)),
            on_steer=lambda text: (calls["steer"].append(text), on_steer_taken)[1],
        )
    finally:
        thread.join(15)
    return result, out.getvalue(), calls


def test_s_with_done_already_queued_renders_it_all_and_reports_not_running(pty_pair):
    result, out, calls = _finished_run(pty_pair, [b"s"])
    assert calls["choice"] == [("stop", "key")]
    assert result.not_running is True
    # A turn that already finished is never reported as interrupted, and the
    # stopping line is never printed for it.
    assert result.interrupted is False
    assert result.done is True
    assert result.text == "the whole answer"
    assert STOPPING not in out
    assert "nvsh: interrupted" not in out
    assert calls["kill"] == []


def test_t_with_done_already_queued_hands_the_correction_to_the_caller(pty_pair):
    result, out, calls = _finished_run(pty_pair, [b"t", b"ask about the fan instead\n"])
    assert calls["steer"] == ["ask about the fan instead"]
    assert calls["choice"] == [("steer", "key")]
    assert result.not_running is True
    assert result.interrupted is False
    assert result.done is True
    assert result.text == "the whole answer"
    assert calls["cancel"] == [] and calls["kill"] == []
    assert STOPPING not in out


# --- t6 / deviation d2: stop_prompt=False keeps the pre-t5 press ------------


def test_stop_prompt_false_stops_at_once_on_a_tty(pty_pair):
    # --json has a real terminal but no panel to answer a prompt on (t7 passes
    # stop_prompt=False there): the first press must stop at once anyway.
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    calls: dict[str, list] = {"cancel": [], "kill": [], "choice": []}
    stopped = threading.Event()

    def cancel():
        calls["cancel"].append(1)
        stopped.set()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("esc", master)
        assert stopped.wait(10), out.getvalue()
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" winding down")

    result = p.stream(
        events(),
        cancel=cancel,
        force_stop=lambda: calls["kill"].append(1),
        on_choice=lambda outcome, reason: calls["choice"].append((outcome, reason)),
        stop_prompt=False,
    )
    text = out.getvalue()
    assert calls["cancel"] == [1]
    assert calls["kill"] == []
    assert calls["choice"] == []
    assert result.interrupted is True
    assert STOPPING in text
    assert "paused" not in text
    assert PAUSED_LEGEND not in text


# --- t8 / deviation d3, seam A: the caller says a stop has begun ------------
#
# ``on_steer`` may answer :data:`nvsh.panel.STOP_BEGUN` instead of a bool:
# the caller has decided to stop this turn so it can resend the correction.
# The panel then does the visible half of a ``[s]`` press -- the stopping
# line, one polite cancel, the next press as the kill press -- but never
# reports the turn as interrupted, because the operator asked for a
# correction, not for the agent to stop.


#: The label a harness with no mid-turn channel is offered under -- the one
#: the stop-and-correct path is actually reached through.
STOP_AND_CORRECT_LABEL = "stop & correct"
CORRECT_LEGEND = f"[t] {STOP_AND_CORRECT_LABEL}  [s] stop  [Esc] keep going"


CORRECTION = b"look at nvpmodel instead\n"


def _stop_to_correct_run(pty_pair, *, second_press: str | None = None):
    """Press once, choose [t], type a line, and answer with STOP_BEGUN."""
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    calls: dict[str, list] = {"cancel": [], "kill": [], "choice": [], "steer": []}
    cancelled = threading.Event()

    def cancel():
        calls["cancel"].append(1)
        cancelled.set()

    def on_steer(text: str) -> object:
        calls["steer"].append(text)
        return panel_mod.STOP_BEGUN

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("esc", master)
        assert _wait_for(lambda: CORRECT_LEGEND in out.getvalue()), out.getvalue()
        assert _answer(master, b"t", lambda: TELL_PROMPT in out.getvalue(), tty_in), out.getvalue()
        time.sleep(0.15)
        os.write(master, CORRECTION)
        assert cancelled.wait(15), out.getvalue()
        if second_press is not None:
            _press(second_press, master)
            assert _wait_for(lambda: calls["kill"]), out.getvalue()
            # The harness ignores the cancel: only the kill ends this.
            while True:
                time.sleep(0.05)
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=" winding down")
        yield AgentEvent(kind=EventKind.DONE)

    result = p.stream(
        events(),
        cancel=cancel,
        force_stop=lambda: calls["kill"].append(1),
        on_choice=lambda outcome, reason: calls["choice"].append((outcome, reason)),
        on_steer=on_steer,
        steer_label=STOP_AND_CORRECT_LABEL,
    )
    return result, out.getvalue(), calls


def test_stop_begun_cancels_once_prints_stopping_and_is_not_an_interruption(pty_pair):
    result, out, calls = _stop_to_correct_run(pty_pair)
    assert calls["steer"] == ["look at nvpmodel instead"]
    assert calls["choice"] == [("steer", "key")]
    assert calls["cancel"] == [1]
    assert calls["kill"] == []
    assert out.count(STOPPING) == 1
    # Rendering continued until the cancelled turn produced its own end.
    assert result.text == "working winding down"
    assert result.done is True
    assert result.stopped_to_correct is True
    assert result.interrupted is False
    assert "nvsh: interrupted" not in out


@pytest.mark.parametrize("second", ["esc", "sigint"])
def test_a_press_after_stop_begun_kills_once_and_interrupts(pty_pair, second):
    result, out, calls = _stop_to_correct_run(pty_pair, second_press=second)
    assert calls["cancel"] == [1]
    assert calls["kill"] == [1]
    # The choice prompt is never re-opened once a stop has begun.
    assert out.count(CORRECT_LEGEND) == 1
    assert result.stopped_to_correct is True
    assert result.interrupted is True


def test_stop_begun_on_an_already_finished_turn_cancels_nothing(pty_pair):
    """c36: the turn ended while only rendering was paused.

    The caller still answers ``STOP_BEGUN`` -- it cannot know the turn is
    over -- and the panel stops nothing: there is no running turn to cancel,
    so the correction simply becomes the caller's next request.
    """
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    calls: dict[str, list] = {"cancel": [], "kill": [], "choice": [], "steer": []}
    at_prompt = threading.Event()

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("esc", master)
        assert _wait_for(lambda: CORRECT_LEGEND in out.getvalue()), out.getvalue()
        at_prompt.set()
        yield AgentEvent(kind=EventKind.DONE)

    def typist():
        assert at_prompt.wait(15)
        _answer(master, b"t", lambda: TELL_PROMPT in out.getvalue(), tty_in)
        time.sleep(0.15)
        os.write(master, CORRECTION)

    thread = threading.Thread(target=typist, daemon=True)
    thread.start()
    try:
        result = p.stream(
            events(),
            cancel=lambda: calls["cancel"].append(1),
            force_stop=lambda: calls["kill"].append(1),
            on_choice=lambda outcome, reason: calls["choice"].append((outcome, reason)),
            on_steer=lambda text: (calls["steer"].append(text), panel_mod.STOP_BEGUN)[1],
            steer_label=STOP_AND_CORRECT_LABEL,
        )
    finally:
        thread.join(15)
    out = out.getvalue()
    assert calls["steer"] == ["look at nvpmodel instead"]
    assert calls["cancel"] == [] and calls["kill"] == []
    assert result.not_running is True
    assert result.stopped_to_correct is False
    assert result.interrupted is False
    assert result.text == "working"
    assert STOPPING not in out


# --- t8 / deviation d3, seam B: Panel.confirm, a one-key yes/no ------------

QUESTION = "stop the agent and send your correction as a new request?"


def _confirm_run(pty_pair, key: bytes | None, *, hang_up: bool = False):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    done = threading.Event()

    def typist():
        if not _wait_for(lambda: QUESTION in out.getvalue()):
            return
        time.sleep(0.15)
        if hang_up:
            os.close(master)
            return
        while key is not None and not done.is_set():
            try:
                os.write(master, key)
            except OSError:  # pragma: no cover - the pty went away
                return
            time.sleep(0.2)

    thread = threading.Thread(target=typist, daemon=True)
    thread.start()
    try:
        answer = p.confirm(QUESTION)
    finally:
        done.set()
        thread.join(15)
        _drain(tty_in)
    return answer, out.getvalue()


@pytest.mark.parametrize("key", [b"y", b"Y"])
def test_confirm_reads_yes(pty_pair, key):
    answer, out = _confirm_run(pty_pair, key)
    assert answer is True
    assert f"nvsh: {QUESTION} [y/N]" in out


@pytest.mark.parametrize(
    "key,hang_up",
    [(b"n", False), (b"N", False), (b"\x1b", False), (b"\x03", False), (None, True)],
    ids=["n", "N", "esc", "ctrl-c", "eof"],
)
def test_confirm_reads_no_for_everything_that_is_not_y(pty_pair, key, hang_up):
    answer, _out = _confirm_run(pty_pair, key, hang_up=hang_up)
    assert answer is False


def test_confirm_times_out_as_no(pty_pair, monkeypatch):
    monkeypatch.setattr(panel_mod, "STOP_PROMPT_TIMEOUT", 0.5)
    started = time.monotonic()
    answer, _out = _confirm_run(pty_pair, None)
    assert answer is False
    assert 0.4 < time.monotonic() - started < 30.0


def test_confirm_off_a_tty_is_no_at_once_and_asks_nothing():
    out = io.StringIO()
    # A stdin that would block forever if it were ever read.
    p = panel_mod.Panel(out=out, in_=io.StringIO(), env=TTY_ENV, isatty=False)
    started = time.monotonic()
    assert p.confirm(QUESTION) is False
    assert time.monotonic() - started < 1.0
    assert out.getvalue() == ""


def test_confirm_under_term_dumb_is_no_at_once(pty_pair):
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out, env={"TERM": "dumb", "NO_COLOR": "1"})
    os.write(master, b"y")  # even with a yes already typed
    started = time.monotonic()
    assert p.confirm(QUESTION) is False
    assert time.monotonic() - started < 1.0
    assert out.getvalue() == ""
    _drain(tty_in)


# --- t8b / plan risk r7: a Ctrl+C that lands during teardown ----------------
#
# Once the stream loop has ended, Panel.stream still has work to do: halt the
# feeder, close the key watcher, join the waiting ticker (up to 2s), put
# termios back and print "nvsh: interrupted". A press landing in that window
# used to reach Python's default handler -- the panel restored the previous
# SIGINT handler *first* -- so the operator got a traceback instead of their
# prompt. It is most reachable on a harness whose cancel() already ends the
# turn: the kill press lands at a client that is already tearing down.
#
# The rule: a press during teardown is recorded as an interrupt and nothing
# more. The turn is over, so nothing is sent to the harness; the panel's own
# handler is still gone by the time stream() returns, and the terminal is
# restored either way.


def _teardown_press_run(monkeypatch, pty_pair, where: str, presses: int = 1):
    """Stream one clean turn and press Ctrl+C during ``stream``'s teardown."""
    master, tty_in = pty_pair
    out = io.StringIO()
    p = _panel(tty_in, out)
    calls: dict[str, list] = {"cancel": [], "kill": [], "choice": []}
    tearing_down = threading.Event()
    released = threading.Event()

    def fire() -> None:
        for _ in range(presses):
            os.kill(os.getpid(), signal.SIGINT)

    original_halt = panel_mod._Feeder.halt

    def halt(self) -> None:
        # The first teardown step, and the one that runs right after the
        # panel used to hand SIGINT back to Python's default handler.
        tearing_down.set()
        if where == "halt":
            fire()
        original_halt(self)

    monkeypatch.setattr(panel_mod._Feeder, "halt", halt)

    if where == "join":
        # Hold the ticker alive so ticker.join(2.0) is a real, wide window,
        # and press from another thread while the main thread is inside it.
        def ticker(_self, _stop_waiting):
            released.wait(15)

        monkeypatch.setattr(panel_mod.Panel, "_wait_ticker", ticker)

        def presser():
            if tearing_down.wait(15):
                fire()
            released.set()

        thread = threading.Thread(target=presser, daemon=True)
    else:
        thread = None

    if where == "restore-termios":
        original_restore = panel_mod._restore_termios

        def restore(saved):
            # The last blocking step is done; only the terminal and the
            # closing lines are left.
            fire()
            original_restore(saved)

        monkeypatch.setattr(panel_mod, "_restore_termios", restore)

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        yield AgentEvent(kind=EventKind.DONE)

    before = signal.getsignal(signal.SIGINT)
    import termios as _termios

    attrs_before = _termios.tcgetattr(tty_in.fileno())
    if thread is not None:
        thread.start()
    try:
        result = p.stream(
            events(),
            cancel=lambda: calls["cancel"].append(1),
            force_stop=lambda: calls["kill"].append(1),
            on_choice=lambda outcome, reason: calls["choice"].append((outcome, reason)),
        )
    finally:
        released.set()
        if thread is not None:
            thread.join(15)
        _drain(tty_in)
    after = signal.getsignal(signal.SIGINT)
    attrs_after = _termios.tcgetattr(tty_in.fileno())
    return result, out.getvalue(), calls, (before, after), (attrs_before, attrs_after)


@pytest.mark.parametrize("where", ["halt", "join", "restore-termios"])
def test_a_press_during_teardown_never_escapes_stream(monkeypatch, pty_pair, where):
    result, out, calls, handlers, attrs = _teardown_press_run(monkeypatch, pty_pair, where)
    # No traceback: stream() returned, and it returned its result.
    assert result.done is True
    assert result.text == "working"
    # The press is recorded as an interrupt and nothing more.
    assert result.interrupted is True
    assert "nvsh: interrupted" in out
    assert calls["cancel"] == [] and calls["kill"] == []
    assert calls["choice"] == []
    # The panel's handler is gone and the terminal is back.
    before, after = handlers
    assert after is before
    assert attrs[1] == attrs[0]


def test_repeated_presses_during_teardown_are_all_absorbed(monkeypatch, pty_pair):
    result, out, calls, handlers, attrs = _teardown_press_run(
        monkeypatch, pty_pair, "halt", presses=3
    )
    assert result.interrupted is True
    assert out.count("nvsh: interrupted") == 1
    assert calls["cancel"] == [] and calls["kill"] == []
    assert handlers[1] is handlers[0]
    assert attrs[1] == attrs[0]


# --- an answer typed the instant the legend appears is never lost -----------


class _TypingOut(io.StringIO):
    """A panel ``out`` that types ``key`` the instant the legend is written.

    No polling, so there is no window to be lucky in: the key is on its way
    to the pty before ``write()`` has even returned to the panel. That is the
    fastest an operator could possibly answer, and it is what review finding
    1 says must never be discarded.
    """

    def __init__(self, master: int, key: bytes) -> None:
        super().__init__()
        self._master = master
        self._key = key
        self.typed = 0

    def write(self, text: str) -> int:
        written = super().write(text)
        if PAUSED_LEGEND in text:
            os.write(self._master, self._key)
            self.typed += 1
        return written


def _instant_answer_run(master: int, tty_in, key: bytes) -> list:
    """One press, one legend, one key typed as it appears. Returns the choices."""
    out = _TypingOut(master, key)
    p = _panel(tty_in, out)
    choices: list[tuple[str, str]] = []

    def events():
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        _press("esc", master)
        assert _wait_for(lambda: choices, timeout=20), out.getvalue()
        yield AgentEvent(kind=EventKind.DONE)

    try:
        p.stream(
            events(),
            cancel=lambda: None,
            force_stop=lambda: None,
            on_choice=lambda outcome, reason: choices.append((outcome, reason)),
        )
    finally:
        _drain(tty_in)
    assert out.typed == 1, out.getvalue()
    return choices


def test_a_key_typed_the_instant_the_legend_appears_always_answers(monkeypatch, pty_pair):
    """20 prompts, one write each, answered by the key every time.

    The prompt used to print its legend and only then discard typeahead, so
    an answer this fast could be thrown away and the prompt would sit there
    until the 30 s timeout dismissed it as "keep going". The short timeout
    here is what makes that failure visible in seconds instead of minutes:
    a lost key shows up as ``timeout``, never as ``key``.
    """
    monkeypatch.setattr(panel_mod, "STOP_PROMPT_TIMEOUT", 2.0)
    master, tty_in = pty_pair
    for _ in range(20):
        assert _instant_answer_run(master, tty_in, b"\x1b") == [("keep_going", "key")]
