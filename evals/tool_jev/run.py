"""Runner for the Tool-Jev DeepEval release gate (issue #64, task t17).

One run directory holds one gate run: the call ledger and response cache
(``ledger.py``), the run record (``run.json``), the billing record
(``billing.jsonl``), and -- once every call is answered -- the traces,
per-(subject, policy) metrics, permutation entries, judge results,
``result.json`` and ``report.md`` that ``report.py`` reads and writes. The
run directory is PRIVATE and must sit outside every git worktree (refused
otherwise); only the aggregate page may later be copied into ``docs/`` by
the operator. The runner never writes into the repository.

Run-dir layout::

    run.json                     run record: run id/date, scope (full or the smoke
                                 selection), manifest sha256 history, hosts that
                                 received case text, stops, capability entries,
                                 reservations, uncertain charges, batch failures
    billing.jsonl                one line per charge, priced when it was answered
    ledger.json cache/ events.jsonl .lock    the call ledger (ledger.py); .lock is
                                 the run lock, held for a whole pass
    traces/<subject>.jsonl       raw record + each policy's final decision
    metrics/<subject>__<policy>.json         metrics_bridge.compute() output
    deepeval/<subject>__<policy>/            deepeval's own per-case results
    .deepeval-judge/             deepeval state of the judge passes
    permutation.json judge_results.json manifest.json   what report.py reads
    result.json report.md        the result and the page (report.py)
    smoke.json                   smoke only: tokens, cost, projection, OK/CAPPED
    drive.log                    drive only: one line per step

Subject names are file-safe: ``<checkpoint>.<case set>`` for a candidate or
baseline, ``<provider>.<model>.<A|B>.<case set>`` for a reference.

Subjects: candidates and baselines replay their saved predictions (no GPU);
references answer Track A through the candidates' own ``LfmTier`` loop
(deviation d1, one batch per model, case set and round for OpenAI and
Anthropic) and Track B through one choice call per case; the judge panel's
two passes send free-text calls once every subject answer is final.

Money and stops (c41 c42 c45 h28 h29 h31 h32; plan risks r6, r10)
------------------------------------------------------------------
- **Every charge is billed once, at answer time** (``billing.jsonl``), and
  spend is never recomputed from the current manifest.
- **Every call reserves its estimated cost before it is sent** (prompt size
  plus its whole output budget). A provider sends nothing that would take
  spent + reserved + estimate past ``usd_cap``; a batch is split to fit. A
  reservation is released when its answer (or its failure) is recorded.
- A **sync call** is marked in flight (the reservation) before it is sent.
  A crash leaves the marker: the next pass resends the call and bills its
  estimate as an *uncertain charge*, logged (h32: continue loses at most the
  calls in flight).
- **Work is deduplicated by ledger key**, and each key is claimed under a
  lock just before it is sent.
- **money stop** (402 / insufficient credit or quota, also in a 429 body)
  stops one provider. ``continue`` retries it; ``drive`` probes it every
  ``recheck_seconds`` with ONE call and keeps the provider blocked until that
  probe's answer is recorded.
- **budget cap** stops one provider; **transient stops** (429, timeouts,
  network, lookup failures, unexpected adapter errors) pause one provider for
  the pass; a **rejected request** stops one model while the rejected request
  parameters are unchanged and becomes a capability entry.
- **failed batches** (expired / failed / cancelled -- never ``cancelling``):
  their completed, billed answers are kept; only unanswered keys are
  requeued, after a backoff of 2^n x the poll interval (at most 30 min);
  three consecutive failures stop the model for an operator decision.
- **truncation stop** (r10): ``[stops]`` thresholds stop one model.
- **stop and ask**: a batch lookup that cannot be resolved is never resubmitted.

Every host that received case text is recorded in ``run.json`` before the
call leaves (h27). Nothing here reads a key: adapters read their key from the
environment variable the manifest names, at call time.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import datetime
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import evals.tool_jev  # noqa: F401  (env guard before anything imports deepeval)
from nvsh.ops import table as ops_table

from . import permutation as perm
from . import request as contract
from . import runstate
from . import track_a_loop as loop
from .cases import Case
from .ledger import (
    DONE,
    INVALID,
    PENDING,
    SUBMITTED,
    CallSpec,
    Ledger,
    LedgerCorrupt,
    LedgerLocked,
    ledger_key,
)
from .manifest import CaseSet, Manifest, ManifestError, Reference, load_manifest
from .providers import anthropic as anthropic_mod
from .providers.base import BatchHandle, BatchLookupUnresolved, CallRequest, CallResult
from .providers.errors import (  # noqa: F401  (classify_transport re-exported)
    Classification,
    Outcome,
    classify_transport,
    stop_message,
)
from .runplan import (  # noqa: F401  (re-exported: evals.tool_jev.run is the public API)
    CHOICE_TARGET,
    DEEPEVAL_DIR,
    ENV_BASE_URL,
    ENV_PRIVATE,
    ENV_PRIVATE_ROOT,
    EXIT_ASK,
    EXIT_ENV,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_STOPPED,
    EXIT_USER,
    EXIT_WAITING,
    MAX_BACKOFF_SECONDS,
    MAX_BATCH_FAILURES,
    MAX_UNCERTAIN_ATTEMPTS,
    MONEY_REASONS,
    NONE_REASONING,
    PAGE_FILE,
    RESULT_FILE,
    STATUS_COMPLETE,
    STATUS_STOPPED,
    STATUS_WAITING,
    TRACK_A_PERMUTATION_REASON,
    TRUNCATION_DECISION,
    UNCERTAIN_REASONS,
    EnvError,
    Model,
    Plan,
    ProviderFactory,
    RunError,
    StopAndAsk,
    Subject,
    WorkItem,
    _classification_of,
    _sha256_file,
    backoff_delay,
    build_plan,
    choice_call,
    classify_exception,
    default_factory,
    judge_request_id,
    private_root,
    provider_host,
    provider_reasoning,
    refused_before_send,
    request_fingerprint,
    request_kind,
    resolve_private,
    shared_limiter,
    smoke_scope,
    spec_interface,
    subject_name,
    tokens_in,
    tokens_out,
)
from .runstatus import render_status, status  # noqa: F401  (re-exported)
from .trace import RawRecord, Trace, _is_inside_git_worktree, write_traces

RUN_FILE = runstate.RUN_FILE
SMOKE_FILE = runstate.SMOKE_FILE
BILLING_FILE = runstate.BILLING_FILE
DEFAULT_POLL_SECONDS = 60.0
DEFAULT_RECHECK_SECONDS = 1800.0


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


def _new_state(digest: str, run_id: str | None, date: str | None, scope: dict) -> dict:
    return {
        "schema": 2,
        "run_id": run_id or digest[:12],
        "date": date or datetime.date.today().isoformat(),
        "manifest_sha256": digest,
        "manifest_history": [digest],
        "scope": scope,
        "hosts": {},
        "stops": {},
        "capabilities": {},
        "reserved": {},
        "uncertain_charges": [],
        "batch_failures": {},
    }


def _open_ledger(run_dir: Path) -> Ledger:
    try:
        return Ledger(run_dir)
    except LedgerLocked as exc:
        raise RunError(f"{run_dir} is in use by another runner (a drive loop?): {exc}") from exc
    except LedgerCorrupt as exc:
        raise RunError(f"{run_dir}: the ledger cannot be trusted: {exc}") from exc


def _check_run_dir(run_dir: Path) -> None:
    if _is_inside_git_worktree(run_dir):
        raise RunError(f"{run_dir} is inside a git worktree; the run dir must be private")


def init_state(
    run_dir: Path,
    manifest_path: Path,
    *,
    run_id: str | None = None,
    date: str | None = None,
    scope: dict | None = None,
) -> dict:
    """Create ``run.json`` for a fresh run dir, under the run lock (or return the existing one)."""
    run_dir = Path(run_dir)
    _check_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    digest = _sha256_file(manifest_path)
    with _open_ledger(run_dir):
        state = runstate.load_state(run_dir)
        if not state:
            state = _new_state(digest, run_id, date, scope or {"mode": "full"})
            runstate.save_state(run_dir, state)
        return state


# ---------------------------------------------------------------------------
# The pass engine
# ---------------------------------------------------------------------------


#: A transient pause (rate limit, timeout, reserved budget still settling)
#: lasts this long, not the rest of the pass: a pass can run for hours while
#: a slow provider works through its round, and a brief 429 must not idle a
#: fast provider for all of it (full run 2026-09-26).
PAUSE_SECONDS = 60.0

#: A round of sync calls stops claiming new calls after this many seconds;
#: calls already in flight finish, the rest stay pending, and the pass loops
#: (polling batches, re-planning, resuming expired pauses), so the slowest
#: provider never holds the others behind one long round barrier.
ROUND_SECONDS = 300.0


@dataclass
class StepOutcome:
    status: str
    messages: list[str] = field(default_factory=list)
    money_stopped: set[str] = field(default_factory=set)
    exit_code: int = EXIT_OK


class Runner:
    """One pass over a run dir, under the run lock: resume, send, finalize when done."""

    def __init__(
        self,
        run_dir: Path,
        plan: Plan,
        ledger: Ledger,
        state: dict,
        *,
        retry_money: bool = True,
        retry_rejected: bool = False,
        out: Callable[[str], None] = print,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
        factory: ProviderFactory | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.factory = factory or default_factory
        self.plan = plan
        self.ledger = ledger
        self.state = state
        self.out = out
        self.env = os.environ if env is None else env
        self.clock = clock
        self.poll_seconds = poll_seconds
        self.recheck_seconds = recheck_seconds
        self.messages: list[str] = []
        self._lock = threading.RLock()
        self.billing = runstate.Billing(self.run_dir)
        for name in (
            "reserved",
            "batch_failures",
            "capabilities",
            "hosts",
            "uncertain_attempts",
            "fetch_backoff",
        ):
            self.state.setdefault(name, {})
        self.state.setdefault("uncertain_charges", [])
        self._spec_model: dict[tuple[str, str], Model] = {}
        for model in plan.models.values():
            self._spec_model[(model.provider.name, model.ref.model)] = model
            self._spec_model[(model.ref.provider, model.ref.model)] = model
        try:
            self.billing.repair(self.clock(), log=self.ledger._log)
            entries = self.billing.entries()
        except runstate.BillingTorn as exc:
            raise StopAndAsk(str(exc)) from exc
        self.billed: dict[str, float] = runstate.sums(entries, "provider")
        self._billed_keys = {e["key"] for e in entries if e.get("kind") == runstate.BILL_ANSWER}
        self._billed_attempts = {
            (e.get("kind"), e["attempt_id"]) for e in entries if e.get("attempt_id") is not None
        }
        self.ghosts: dict[str, Model] = {}
        self.answers: dict[str, int] = {}
        self.truncated: dict[str, int] = {}
        self.invalid: dict[str, int] = {}
        self.model_stops: dict[str, dict] = {}
        self.provider_stops: dict[str, dict] = {}
        self.paused: dict[str, str] = {}
        #: kind -> clock time a transient pause ends (PAUSE_SECONDS after it began).
        self.paused_until: dict[str, float] = {}
        self._judge_cache: tuple | None = None
        self._track_a: dict[tuple[str, str], loop.RoundResult] = {}
        # Stops first: every later persist writes them back from memory.
        self._restore_stops(retry_money, retry_rejected)
        self._build_ghosts()
        self._reconcile_reservations()
        self._scan_ledger()
        for kind in sorted({m.kind for m in plan.models.values()}):
            self._check_budget(kind)
        for label in sorted(plan.models):
            self._check_truncation(plan.models[label])

    # -- bookkeeping -------------------------------------------------------

    def say(self, message: str) -> None:
        with self._lock:
            if message in self.messages:
                return
            self.messages.append(message)
        self.out(message)

    def _model_of_spec(self, spec: Mapping[str, Any]) -> Model | None:
        return self._spec_model.get((spec.get("provider"), spec.get("model")))

    def _discount(self, kind: str) -> float:
        budget = self.plan.manifest.budget_for(kind)
        return budget.batch_discount if budget else 0.0

    def _judge_params(self, model: Model) -> dict[str, Any]:
        params: dict[str, Any] = {"max_output_tokens": self.plan.manifest.judging.max_output_tokens}
        effort = provider_reasoning(model.ref.provider, model.ref.reasoning)
        if effort:
            params["reasoning"] = effort
        return params

    def _current_fingerprint(self, model: Model, role: str, interface: str) -> str | None:
        """The parameters this model is sent now for one (role, interface) (codex item 11)."""
        if role == "judge":
            return request_fingerprint(self._judge_params(model))
        if interface == "tool_call":
            return request_fingerprint({**model.knobs, "tool_choice": loop.LOOP_TOOL_CHOICE})
        if interface == "choice":
            return request_fingerprint(model.knobs)
        return None

    def _current(self, model: Model, spec: Mapping[str, Any]) -> bool:
        """Whether *spec* was made with the model's current settings (stale ones never stop it)."""
        params = spec.get("params") or {}
        target = str(spec.get("target", ""))
        if spec.get("subject_role") == "judge":
            judging = self.plan.manifest.judging
            want = self._judge_params(model)
            return (
                params.get("rubric") == judging.rubric
                and params.get("max_output_tokens") == want["max_output_tokens"]
                and params.get("reasoning") == want.get("reasoning")
            )
        if target.startswith(loop.TARGET_PREFIX):
            return params == {**model.knobs, "tool_choice": loop.LOOP_TOOL_CHOICE}
        if target == CHOICE_TARGET:
            return params == model.knobs
        return False

    def _count(self, model: Model, key: str, spec: Mapping[str, Any], state: str) -> None:
        """Add one answered call to the model's answer / truncated / invalid counts."""
        if not self._current(model, spec):
            return
        label = model.label
        self.answers[label] = self.answers.get(label, 0) + 1
        cut = self.ledger.entry(key).reason == loop.TRUNCATED
        cached = self.ledger.cached(key)
        if cached is not None and not cut:
            cut = _cut(model, cached.raw)
        if cut:
            self.truncated[label] = self.truncated.get(label, 0) + 1
        if state == INVALID:
            self.invalid[label] = self.invalid.get(label, 0) + 1

    def _scan_ledger(self) -> None:
        for entry in self.ledger.entries():
            if entry.state not in (DONE, INVALID):
                continue
            model = self._model_of_spec(entry.spec)
            if model is not None:
                self._count(model, entry.key, entry.spec, entry.state)

    # -- money: billing and reservations -----------------------------------

    def reserved_sum(self, kind: str) -> float:
        return sum(r["usd"] for r in self.state["reserved"].values() if r["provider"] == kind)

    def spent(self, kind: str) -> float:
        return self.billed.get(kind, 0.0)

    def _bill(self, entry: dict) -> bool:
        """Append one charge unless this attempt is already billed; True when appended."""
        attempt = (entry["kind"], entry["attempt_id"])
        if attempt in self._billed_attempts:
            return False
        self.billing.append(entry)
        self._billed_attempts.add(attempt)
        self.billed[entry["provider"]] = self.billed.get(entry["provider"], 0.0) + entry["cost_usd"]
        if entry["kind"] == runstate.BILL_ANSWER:
            self._billed_keys.add(entry["key"])
        return True

    def _bill_answer(self, model: Model, key: str, result: CallResult) -> None:
        discount = self._discount(model.kind)
        self._bill(
            {
                "kind": runstate.BILL_ANSWER,
                "key": key,
                "attempt_id": f"{key}:{result.response_id}",
                "provider": model.kind,
                "label": model.label,
                "route": model.route,
                "usd_per_mtok_in": model.ref.usd_per_mtok_in,
                "usd_per_mtok_out": model.ref.usd_per_mtok_out,
                "discount": discount if model.route == "batch" else 0.0,
                "usage": dict(result.usage),
                "cost_usd": model.cost(result.usage, discount),
            }
        )

    def _release(self, *keys: str) -> None:
        with self._lock:
            changed = False
            for key in keys:
                changed |= self.state["reserved"].pop(key, None) is not None
            if changed:
                self._persist()

    def _reconcile_reservations(self) -> None:
        """Settle reservations a crash left behind (codex review P1-2, P1-4, items 15, 16)."""
        reserved = self.state["reserved"]
        for key in sorted(reserved):
            held = reserved[key]
            try:
                entry = self.ledger.entry(key)
            except Exception:  # noqa: BLE001 -- a key the ledger never knew: nothing to hold
                reserved.pop(key)
                continue
            if entry.state in (DONE, INVALID):
                reserved.pop(key)
                continue
            if held.get("route") == "batch":
                if entry.state == SUBMITTED or entry.submit_token:
                    continue  # still possibly running and billing
                reserved.pop(key)  # an orphaned reservation with no submitted work
                continue
            # A sync call in flight at a crash: it may have been billed. It is
            # resent (h32) and its estimate counts as an uncertain charge.
            if key in self._billed_keys:
                reserved.pop(key)
                continue
            self._uncertain(key, "in flight at a crash")
        self._persist()

    def _uncertain(self, key: str, why: str) -> None:
        """One uncertain sync attempt: bill it once (by its reservation id), count it, resend.

        The charge, the attempt count and the reservation's removal reach
        ``run.json`` in one save; the billing line is idempotent by attempt
        id, so a crash between the two never charges twice (codex item 15).
        """
        with self._lock:
            held = self.state["reserved"].pop(key)
            attempt_id = held.get("id") or f"legacy:{key}:{held.get('since')}"
            charge = {
                "kind": runstate.BILL_UNCERTAIN,
                "key": key,
                "attempt_id": attempt_id,
                "provider": held["provider"],
                "label": held["label"],
                "route": "sync",
                "cost_usd": held["usd"],
                "since": held.get("since"),
                "why": why,
            }
            self._bill(charge)
            charges = self.state["uncertain_charges"]
            if all(c.get("attempt_id") != attempt_id for c in charges):
                charges.append(charge)
                self.ledger._log(
                    "uncertain_resend", [key], provider=held["provider"], usd=held["usd"]
                )
            attempts = self.state["uncertain_attempts"]
            attempts[key] = attempts.get(key, 0) + 1
            message = (
                f"{held['label']}: call {key[:12]} was {why}; it may have been billed, so its "
                f"estimated ${held['usd']:.4f} counts as an uncertain charge and it is resent"
            )
            model = self.plan.models.get(held["label"]) or self.ghosts.get(held["label"])
            # The stop guards money: a free model (local, a free tier) is resent.
            paid = bool(
                model is not None
                and (model.ref.usd_per_mtok_in > 0 or model.ref.usd_per_mtok_out > 0)
            )
            if attempts[key] >= MAX_UNCERTAIN_ATTEMPTS and paid:
                message = (
                    f"{held['label']}: call {key[:12]} had {attempts[key]} uncertain attempts "
                    f"(sent, maybe billed, no answer); {self._model_remaining(model)} call(s) "
                    "left pending; needs operator decision: check the provider's usage, then "
                    "continue with --retry-rejected"
                )
                self.model_stops[held["label"]] = {
                    "kind": "uncertain_attempts",
                    "reason": "uncertain_attempts",
                    "message": message,
                    "key": key,
                }
            self._persist()
        self.say(message)

    def claim(self, model: Model, key: str, request: CallRequest) -> bool:
        """Claim *key* for sending: fresh, unblocked, within budget; reserve its cost.

        Called under the lock just before the call leaves, for sync and batch
        alike (codex review P1-3, P1-4). The reservation is persisted before
        the send, so it doubles as a sync call's in-flight marker (P1-2).
        """
        with self._lock:
            if self.blocked(model) or key in self.state["reserved"] or not self._fresh(key):
                return False
            budget = self.plan.manifest.budget_for(model.kind)
            estimate = model.estimate(request, self._discount(model.kind))
            if budget is not None:
                room = budget.usd_cap - self.spent(model.kind) - self.reserved_sum(model.kind)
                if estimate > room and estimate > 0:
                    self._refuse_budget(model.kind, estimate, room)
                    return False
            stop = self.provider_stops.get(model.kind)
            if stop is not None and stop.get("probe_open"):
                stop["probe_open"] = False
                stop["probe"] = key
            # Submission-time metadata: enough to rebuild the adapter and bill
            # the answer if the model leaves the manifest meanwhile (item 16).
            self.state["reserved"][key] = {
                "id": uuid.uuid4().hex,
                "provider": model.kind,
                "label": model.label,
                "usd": round(estimate, 8),
                "route": model.route,
                "since": self.clock(),
                "spec_provider": model.provider.name,
                "model": model.ref.model,
                "adapter": model.ref.provider,
                "api_key_env": model.ref.api_key_env,
                "capabilities": list(model.ref.capabilities),
                "reasoning": model.ref.reasoning,
                "usd_per_mtok_in": model.ref.usd_per_mtok_in,
                "usd_per_mtok_out": model.ref.usd_per_mtok_out,
            }
            self._persist()
            return True

    def _refuse_budget(self, kind: str, estimate: float, room: float) -> None:
        message = (
            f"{kind}: stopping cleanly (budget_cap_reached: the next call's estimated "
            f"${estimate:.4f} exceeds the ${max(room, 0.0):.4f} left after "
            f"${self.spent(kind):.4f} spent and ${self.reserved_sum(kind):.4f} reserved); "
            f"{self._provider_remaining(kind)} call(s) left pending; raise [budget.{kind}] "
            "usd_cap, or continue once the reserved calls settle"
        )
        if self.reserved_sum(kind) > 0:
            self._pause(kind, message)
        else:
            self.provider_stops[kind] = {
                "kind": "budget_cap",
                "reason": "budget_cap_reached",
                "message": message,
            }
        self.say(message)

    # -- stops ---------------------------------------------------------------

    def _restore_stops(self, retry_money: bool, retry_rejected: bool) -> None:
        """Carry stops over from the last pass.

        A rejected (or batch-failure) model stays stopped while the rejected
        request parameters are still sent, unless the operator retries it. A
        money stop is dropped by ``continue`` and probed by ``drive``: one
        call once ``recheck_seconds`` have passed, the provider blocked until
        that probe's answer is recorded (codex review P2-9). Budget-cap and
        truncation stops are recomputed.
        """
        stops = self.state.get("stops", {})
        for label, stop in stops.get("models", {}).items():
            model = self.plan.models.get(label)
            if model is None or retry_rejected:
                continue
            if stop.get("kind") in ("batch_failures", "uncertain_attempts"):
                self.model_stops[label] = stop
            elif stop.get("kind") == "rejected":
                sent = stop.get("request") or {}
                now_sent = self._current_fingerprint(
                    model, sent.get("role", ""), sent.get("interface", "")
                )
                if now_sent is not None and now_sent == sent.get("params"):
                    self.model_stops[label] = stop
        if retry_rejected:
            self.state["batch_failures"] = {}
            self.state["uncertain_attempts"] = {}
        now = self.clock()
        for kind, stop in stops.get("providers", {}).items():
            if stop.get("kind") != "money" or retry_money:
                continue
            stop = dict(stop)
            probe = stop.get("probe")
            if probe:
                try:
                    entry = self.ledger.entry(probe)
                except Exception:  # noqa: BLE001 -- unknown key: a dead probe
                    entry = None
                if entry is not None and entry.state in (DONE, INVALID):
                    self.say(f"{kind}: the probe call went through; its money stop is cleared")
                    continue
                alive = entry is not None and (
                    entry.state == SUBMITTED
                    or bool(entry.submit_token)
                    or probe in self.state["reserved"]
                )
                if alive:
                    self.provider_stops[kind] = stop
                    continue
                stop.pop("probe")
            if now - float(stop.get("since", 0.0)) >= self.recheck_seconds:
                stop["probe_open"] = True
                self.say(f"{kind}: probing the money stop with one call")
            self.provider_stops[kind] = stop

    def _persist(self) -> None:
        with self._lock:
            self._persist_locked()

    def _persist_locked(self) -> None:
        self.state["stops"] = {
            "models": dict(sorted(self.model_stops.items())),
            "providers": {
                k: {n: v for n, v in stop.items() if n != "probe_open"}
                for k, stop in sorted(self.provider_stops.items())
            },
        }
        self.state["models"] = {
            label: {
                "provider": model.kind,
                "host": model.host,
                "route": model.route,
                "spec_keys": sorted([list(k) for k, m in self._spec_model.items() if m is model]),
                "answers": self.answers.get(label, 0),
                "truncated": self.truncated.get(label, 0),
                "invalid": self.invalid.get(label, 0),
            }
            for label, model in sorted({**self.ghosts, **self.plan.models}.items())
        }
        self.state["budgets"] = {
            b.provider: {"usd_cap": b.usd_cap, "batch_discount": b.batch_discount}
            for b in self.plan.manifest.budgets
        }
        runstate.save_state(self.run_dir, self.state)

    def _note_host(self, model: Model) -> None:
        """Record *model*'s host as having received case text, before anything is sent."""
        with self._lock:
            names = self.state["hosts"].setdefault(model.host, [])
            if model.label not in names:
                names.append(model.label)
                names.sort()
                self._persist()

    def _remaining(self, keys_of: Callable[[Mapping[str, Any]], bool]) -> int:
        return sum(
            1
            for entry in self.ledger.entries()
            if entry.state in (PENDING, SUBMITTED) and keys_of(entry.spec)
        )

    def _model_remaining(self, model: Model) -> int:
        return self._remaining(lambda spec: self._model_of_spec(spec) is model)

    def _provider_remaining(self, kind: str) -> int:
        return self._remaining(
            lambda spec: (m := self._model_of_spec(spec)) is not None and m.kind == kind
        )

    def _pause(self, kind: str, message: str) -> None:
        self.paused[kind] = message
        self.paused_until[kind] = self.clock() + PAUSE_SECONDS

    def is_paused(self, kind: str) -> bool:
        """Whether *kind* is in a transient pause that has not run out yet."""
        with self._lock:
            if kind not in self.paused:
                return False
            if self.clock() < self.paused_until.get(kind, float("inf")):
                return True
            self.paused.pop(kind, None)
            self.paused_until.pop(kind, None)
            return False

    def blocked(self, model: Model) -> bool:
        if self.is_paused(model.kind):
            return True
        with self._lock:
            if model.label in self.model_stops:
                return True
            stop = self.provider_stops.get(model.kind)
            return stop is not None and not stop.get("probe_open")

    def apply_stop(
        self,
        model: Model,
        classification: Classification,
        params: Mapping[str, Any] | None = None,
        kind: tuple[str, str] = ("lookup", "none"),
    ) -> None:
        """Stop or pause for *classification*.

        A rejection is fingerprinted by the request that was rejected: its
        role, interface and parameters (codex item 11). A rejection with no
        request behind it (a batch lookup) matches nothing, so it is retried
        on the next pass.
        """
        with self._lock:
            if classification.rejected or not classification.retryable:
                role, interface = kind
                sent = {
                    "role": role,
                    "interface": interface,
                    "params": request_fingerprint(params or {}),
                }
                message = stop_message(model.label, classification, self._model_remaining(model))
                self.model_stops[model.label] = {
                    "kind": "rejected",
                    "reason": classification.reason,
                    "message": message,
                    "request": sent,
                }
                capabilities = self.state["capabilities"].setdefault(model.label, [])
                entry = {"request_rejected": classification.reason, "request": sent}
                if entry not in capabilities:
                    capabilities.append(entry)
            elif classification.reason in MONEY_REASONS:
                message = stop_message(
                    model.kind, classification, self._provider_remaining(model.kind)
                )
                self.provider_stops[model.kind] = {
                    "kind": "money",
                    "reason": classification.reason,
                    "message": message,
                    "since": self.clock(),
                }
            else:
                message = stop_message(
                    model.kind, classification, self._provider_remaining(model.kind)
                )
                self._pause(model.kind, message)
            self._persist()
        self.say(message)

    def _check_budget(self, kind: str) -> None:
        budget = self.plan.manifest.budget_for(kind)
        if budget is None:
            return
        spent = self.spent(kind)
        if spent >= budget.usd_cap and not (budget.usd_cap == 0 and spent == 0):
            message = (
                f"{kind}: stopping cleanly (budget_cap_reached: ${spent:.4f} of "
                f"${budget.usd_cap:.2f}); {self._provider_remaining(kind)} call(s) left "
                f"pending; raise [budget.{kind}] usd_cap, then continue"
            )
            with self._lock:
                self.provider_stops[kind] = {
                    "kind": "budget_cap",
                    "reason": "budget_cap_reached",
                    "message": message,
                }
            self.say(message)

    def _check_truncation(self, model: Model) -> None:
        rules = self.plan.manifest.stops
        answers = self.answers.get(model.label, 0)
        cut = self.truncated.get(model.label, 0)
        if answers < rules.min_answers or cut / answers < rules.max_truncated_share:
            return
        if self.model_stops.get(model.label, {}).get("kind") == "truncation":
            return
        message = (
            f"{model.label}: truncation stop ({cut} of {answers} answers cut at "
            f"max_output_tokens={model.knobs['max_output_tokens']}, "
            f"{self.invalid.get(model.label, 0)} invalid); "
            f"{self._model_remaining(model)} call(s) left pending; {TRUNCATION_DECISION}"
        )
        with self._lock:
            self.model_stops[model.label] = {
                "kind": "truncation",
                "reason": "truncated",
                "message": message,
                "truncated": cut,
                "answers": answers,
                "invalid": self.invalid.get(model.label, 0),
            }
        self.say(message)

    # -- recording ---------------------------------------------------------

    def record(self, model: Model, key: str, result: CallResult) -> None:
        """One answer: billed, then into the ledger, then its reservation released."""
        with self._lock:
            entry = self.ledger.entry(key)
            if entry.state not in (PENDING, SUBMITTED):
                self._release(key)
                return
            if result.outcome is Outcome.PENDING:
                self._release(key)
                self.apply_stop(
                    model,
                    _classification_of(result),
                    entry.spec.get("params"),
                    request_kind(entry.spec.get("subject_role", ""), spec_interface(entry.spec)),
                )
                return
            if key not in self._billed_keys:
                # Billed before the cache: a kill in between replays the fetch,
                # and this check keeps the replay from billing again (item 14).
                self._bill_answer(model, key, result)
            cached = loop.cached_response(result)
            if result.outcome is Outcome.OK:
                if entry.spec.get("subject_role") == "judge" and _cut(model, result.raw):
                    self.ledger.mark_invalid(key, loop.TRUNCATED, cached)
                else:
                    self.ledger.record_done(key, cached)
            else:
                self.ledger.mark_invalid(key, result.reason or "invalid", cached)
            self.state["reserved"].pop(key, None)
            stop = self.provider_stops.get(model.kind)
            if stop is not None and stop.get("probe") == key:
                del self.provider_stops[model.kind]
                self.say(f"{model.kind}: the probe call went through; its money stop is cleared")
            self._count(model, key, entry.spec, self.ledger.entry(key).state)
            self._persist()
            self._check_budget(model.kind)
            self._check_truncation(model)

    # -- batches -----------------------------------------------------------

    def resolve_orphans(self) -> None:
        """Submissions that may or may not have reached the provider: find them first."""
        for submit_ref, keys in sorted(self.ledger.continue_plan().orphans.items()):
            model = self._model_of_spec(self.ledger.entry(keys[0]).spec)
            if model is None:
                raise StopAndAsk(
                    f"submit ref {submit_ref} ({len(keys)} key(s)) belongs to no model in the "
                    "manifest; restore that model or ask before resending: " + ", ".join(keys)
                )
            if self.is_paused(model.kind):
                continue
            try:
                handle = model.provider.find_batch(submit_ref)
            except BatchLookupUnresolved as exc:
                raise StopAndAsk(
                    f"{model.label}: cannot tell whether batch submission {submit_ref} reached "
                    f"the provider ({exc}); it is NOT resubmitted. Check the provider's batch "
                    f"list, then ask the operator. Keys: {', '.join(keys)}"
                ) from exc
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 -- the orphan marker is kept (P2-7)
                classification = classify_exception(exc, model.kind)
                if classification is None:
                    raise
                self.apply_stop(model, classification)
                continue
            if handle is None:
                self.ledger.abandon_submit(keys)
                self._release(*keys)
            else:
                self.ledger.mark_submitted(keys, handle.batch_id)

    def poll_batches(self) -> bool:
        progress = False
        for batch_id, keys in sorted(self.live_batches().items()):
            model = self._model_of_spec(self.ledger.entry(keys[0]).spec)
            if self.is_paused(model.kind):
                continue
            backoff = self.state["fetch_backoff"].get(batch_id)
            if backoff and self.clock() < backoff["next_at"]:
                continue
            handle = BatchHandle(batch_id=batch_id, provider=model.provider.name)
            # Anthropic carries a long case id as a hash in custom_id; the
            # ledger's own specs say which case ids this batch holds.
            anthropic_mod.register_case_ids(
                str(self.ledger.entry(key).spec.get("case_id", "")) for key in keys
            )
            try:
                batch_status = model.provider.poll_batch(handle)
                if not batch_status.complete:
                    continue
                results = model.provider.fetch_batch(handle)
            except (KeyboardInterrupt, BatchLookupUnresolved):
                raise
            except Exception as exc:  # noqa: BLE001 -- classified or re-raised
                classification = classify_exception(exc, model.kind)
                if classification is None:
                    raise
                # An unreadable result never proves a request went unanswered:
                # the batch stays submitted and is fetched again (codex item 1).
                self._fetch_failed(model, batch_id, classification)
                continue
            self.state["fetch_backoff"].pop(batch_id, None)
            by_request = {self._request_id(key): key for key in keys}
            answered = 0
            for result in results:
                key = by_request.get(result.case_id)
                if key is not None:
                    self.record(model, key, result)
                    answered += result.outcome is not Outcome.PENDING
            requeued = self.ledger.requeue_batch(batch_id, "unanswered")
            self._release(*requeued)
            if batch_status.expired:
                self._batch_failed(model, batch_id, answered, len(requeued))
            else:
                self.state["batch_failures"].pop(model.label, None)
            progress = True
        return progress

    def _fetch_failed(self, model: Model, batch_id: str, classification: Classification) -> None:
        with self._lock:
            backoff = self.state["fetch_backoff"].setdefault(batch_id, {"count": 0})
            backoff["count"] += 1
            delay = backoff_delay(self.poll_seconds, backoff["count"])
            backoff["next_at"] = self.clock() + delay
            self._persist()
        self.say(
            f"{model.label}: batch {batch_id} ended but its results could not be read "
            f"({classification.reason}); it stays submitted and is fetched again in {delay:.0f} s"
        )
        if classification.reason in MONEY_REASONS or classification.rejected:
            self.apply_stop(model, classification)

    def _batch_failed(self, model: Model, batch_id: str, answered: int, requeued: int) -> None:
        """Count a failed batch; back off exponentially; stop after three in a row (P1-1)."""
        with self._lock:
            failures = self.state["batch_failures"].setdefault(model.label, {"count": 0})
            failures["count"] += 1
            delay = backoff_delay(self.poll_seconds, failures["count"])
            failures["next_submit_at"] = self.clock() + delay
            message = (
                f"{model.label}: batch {batch_id} expired, failed or was cancelled; "
                f"{answered} answered call(s) kept, {requeued} requeued; "
                f"failure {failures['count']} in a row, next submission in {delay:.0f} s"
            )
            if failures["count"] >= MAX_BATCH_FAILURES:
                message = (
                    f"{model.label}: {failures['count']} failed batches in a row (last "
                    f"{batch_id}); {self._model_remaining(model)} call(s) left pending; needs "
                    "operator decision: check the provider's batch errors, then continue "
                    "with --retry-rejected"
                )
                self.model_stops[model.label] = {
                    "kind": "batch_failures",
                    "reason": "batch_failures",
                    "message": message,
                }
            self._persist()
        self.say(message)

    def _build_ghosts(self) -> None:
        """Adapters for models that left the manifest while their work was outstanding.

        Rebuilt from each reservation's submission-time metadata so their
        batches are still polled, fetched, billed and released (codex items
        5, 16). A ghost never gets new work and scores nothing.
        """
        for key, held in sorted(self.state["reserved"].items()):
            if "model" not in held or (held.get("spec_provider"), held["model"]) in (
                self._spec_model
            ):
                continue
            ref = Reference(
                provider=held["adapter"],
                model=held["model"],
                reasoning=held.get("reasoning", "medium"),
                batch=held.get("route") == "batch",
                capabilities=tuple(held.get("capabilities", ())),
                api_key_env=held.get("api_key_env"),
                usd_per_mtok_in=held.get("usd_per_mtok_in", 0.0),
                usd_per_mtok_out=held.get("usd_per_mtok_out", 0.0),
            )
            try:
                provider = self.factory(ref, self.plan.manifest.budget_for(ref.provider), self.env)
            except Exception as exc:  # noqa: BLE001 -- reported; its reservation keeps counting
                self.say(f"{held['label']}: cannot rebuild its adapter to settle key {key}: {exc}")
                continue
            ghost = Model(
                ref=ref, provider=provider, host=provider_host(provider), route=held["route"]
            )
            self.ghosts[ghost.label] = ghost
            self._spec_model[(held["spec_provider"], held["model"])] = ghost
            self._spec_model[(ref.provider, ref.model)] = ghost
            self.say(f"{ghost.label}: no longer in the manifest; settling its outstanding calls")

    def live_batches(self) -> dict[str, list[str]]:
        """Submitted batches of models in this manifest (a removed model's are reported)."""
        live = {}
        for batch_id, keys in self.ledger.submitted_batches().items():
            if self._model_of_spec(self.ledger.entry(keys[0]).spec) is None:
                self.say(
                    f"batch {batch_id} ({len(keys)} call(s)) belongs to a model no longer in "
                    "the manifest; it is left alone and its reservation still counts"
                )
                continue
            live[batch_id] = keys
        return live

    def _request_id(self, key: str) -> str:
        spec = self.ledger.entry(key).spec
        if spec.get("subject_role") == "judge":
            return judge_request_id(key)
        return str(spec["case_id"])

    def submit_batches(self, model: Model, items: list[WorkItem]) -> bool:
        failures = self.state["batch_failures"].get(model.label)
        if failures and self.clock() < failures.get("next_submit_at", 0.0):
            self.say(
                f"{model.label}: backing off after a failed batch until "
                f"{failures['next_submit_at']:.0f}"
            )
            return False
        progress = False
        groups: dict[str, list[WorkItem]] = {}
        for item in items:
            groups.setdefault(item.group, []).append(item)
        for group in sorted(groups):
            chunk = [item for item in groups[group] if self.claim(model, item.key, item.request)]
            if not chunk:
                continue
            keys = [item.key for item in chunk]
            with self._lock:
                submit_ref = self.ledger.begin_submit(keys)
            self._note_host(model)
            try:
                handle = model.provider.submit_batch([item.request for item in chunk], submit_ref)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 -- left as an orphan, looked up next pass
                classification = classify_exception(exc, model.kind)
                if classification is None:
                    raise
                self.apply_stop(
                    model,
                    classification,
                    chunk[0].request.params,
                    request_kind("", chunk[0].request.interface),
                )
                continue
            with self._lock:
                self.ledger.mark_submitted(keys, handle.batch_id)
            progress = True
        return progress

    # -- sync --------------------------------------------------------------

    def run_sync(self, work: list[tuple[Model, WorkItem]]) -> bool:
        """Send *work* synchronously, each provider under its concurrency cap.

        A worker holds its provider's semaphore through the claim, the send,
        the billing and the ledger write, so with a cap of N at most N calls
        are in flight when a stop lands, and none starts after it.
        """
        if not work:
            return False
        semaphores: dict[str, threading.Semaphore] = {}
        workers = 0
        for model, _item in work:
            if model.kind not in semaphores:
                budget = self.plan.manifest.budget_for(model.kind)
                cap = budget.concurrency_cap if budget else 1
                semaphores[model.kind] = threading.Semaphore(cap)
                workers += cap

        deadline = time.monotonic() + ROUND_SECONDS

        def task(model: Model, item: WorkItem) -> bool:
            with semaphores[model.kind]:
                if time.monotonic() > deadline or self.blocked(model):
                    return False  # stays pending: the next round (or pass) sends it
                if not self.claim(model, item.key, item.request):
                    return False
                self._note_host(model)
                try:
                    result = model.provider.submit_sync(item.request)
                except (KeyboardInterrupt, BatchLookupUnresolved):
                    raise
                except Exception as exc:  # noqa: BLE001 -- classified or re-raised
                    classification = classify_exception(exc, model.kind)
                    if classification is None:
                        raise
                    if classification.reason in UNCERTAIN_REASONS and not refused_before_send(exc):
                        # Sent and maybe accepted, no answer: one uncertain charge (item 2).
                        self._uncertain(
                            item.key, f"sent but not answered ({classification.reason})"
                        )
                    else:
                        self._release(item.key)
                    self.apply_stop(
                        model,
                        classification,
                        item.request.params,
                        request_kind("", item.request.interface),
                    )
                    return False
                self.record(model, item.key, result)
                return True

        progress = False
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers))
        futures = [pool.submit(task, model, item) for model, item in work]
        failure: BaseException | None = None
        try:
            for future in concurrent.futures.as_completed(futures):
                if future.cancelled():
                    continue
                try:
                    progress |= future.result()
                except BaseException as exc:  # noqa: BLE001 -- re-raised after the drain
                    if failure is None:
                        failure = exc
                        for other in futures:
                            other.cancel()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if failure is not None:
            raise failure
        return progress

    # -- work --------------------------------------------------------------

    def _fresh(self, key: str) -> bool:
        entry = self.ledger.entry(key)
        return entry.state == PENDING and not entry.submit_token

    def track_a(self, model: Model, cs: CaseSet) -> loop.RoundResult:
        return loop.run_round(
            self.plan.cases[cs.name],
            provider=model.provider,
            model=model.ref.model,
            ledger=self.ledger,
            snapshot=self.plan.snapshot,
            platform=self.plan.platform,
            params=model.knobs,
        )

    def _choice_calls(self, model: Model, cs: CaseSet) -> list[tuple[str, CallSpec, CallRequest]]:
        calls = [choice_call(model, case) for case in self.plan.cases[cs.name]]
        keys = self.ledger.register_many([spec for spec, _request in calls])
        return [(key, spec, request) for key, (spec, request) in zip(keys, calls)]

    def subject_work(self) -> tuple[dict[str, list[WorkItem]], bool]:
        """Every subject call that can go out now, once per key; and whether all are final."""
        work: dict[str, list[WorkItem]] = {}
        seen: set[str] = set()
        final = True

        def add(model: Model, key: str, request: CallRequest, group: str) -> None:
            if key in seen or not self._fresh(key):
                return
            seen.add(key)
            work.setdefault(model.label, []).append(WorkItem(key, request, group))

        for subject in self.plan.references:
            model = self.plan.models[subject.model]
            cs = subject.case_set
            if subject.track == "A":
                round_result = self.track_a(model, cs)
                self._track_a[(model.label, cs.name)] = round_result
                if round_result.pending:
                    final = False
                for call in round_result.pending:
                    add(model, call.key, call.request, f"A:{cs.name}:r{call.round}")
            else:
                for key, _spec, request in self._choice_calls(model, cs):
                    if self.ledger.entry(key).state not in (DONE, INVALID):
                        final = False
                    add(model, key, request, f"B:{cs.name}")
        return work, final

    # -- records per subject -------------------------------------------------

    def choice_record(self, model: Model, case: Case, key: str, request: CallRequest) -> RawRecord:
        """A Track B answer as the scorer's own prediction mapping reads it."""
        cached = self.ledger.cached(key)
        entry = self.ledger.entry(key)

        def invalid(reason: str, result: CallResult | None = None) -> RawRecord:
            return RawRecord.from_provider_answer(
                provider=model.provider.name,
                model=model.ref.model,
                returned_model=result.returned_model if result else None,
                interface="choice",
                outcome="invalid",
                candidates=result.candidates if result else None,
                invalid_reason=reason,
            )

        if cached is None:
            return invalid(entry.reason or "invalid")
        result = model.provider.result_from_raw(request, cached.raw)
        if _cut(model, cached.raw):
            # A reply cut at the output budget is never an answer, even when a
            # valid letter came before the cut (codex review P2-13).
            return invalid(loop.TRUNCATED, result)
        labels = request.params["labels"]
        classification, name = contract.parse_choice(result.answer, labels)
        if result.outcome is not Outcome.OK or name is None:
            reason = result.reason if result.outcome is not Outcome.OK else classification.reason
            return invalid(reason, result)
        base = {
            "provider": model.provider.name,
            "model": model.ref.model,
            "returned_model": result.returned_model,
            "interface": "choice",
            "candidates": result.candidates,
        }
        if name == loop.lfm.EXPLAIN_TOOL:
            return RawRecord.from_provider_answer(outcome="explain", **base)
        if name == loop.lfm.ESCALATE_TOOL:
            return RawRecord.from_provider_answer(outcome="escalate", **base)
        operation = ops_table.get(name)
        grounded = (
            contract.scorer.ground_arguments(
                operation,
                contract._request_text(case),
                loop.measure.snapshot_runner(self.plan.snapshot),
            )
            if operation is not None
            else "unknown operation"
        )
        if isinstance(grounded, str):
            return RawRecord.from_provider_answer(
                outcome="invalid", invalid_reason="not_grounded", **base
            )
        return RawRecord.from_provider_answer(
            outcome="propose", operation=name, arguments=dict(grounded), **base
        )

    def subject_answers(self) -> dict[str, tuple[list[Trace], dict[str, str]]]:
        """Every subject's traces (case order) and explain texts; call only when final."""
        out: dict[str, tuple[list[Trace], dict[str, str]]] = {}
        for subject in self.plan.subjects:
            if subject.kind != "reference":
                saved = self.plan.saved[subject.name]
                out[subject.name] = (saved.traces, saved.explanations)
                continue
            model = self.plan.models[subject.model]
            cs = subject.case_set
            cases = self.plan.cases[cs.name]
            traces: list[Trace] = []
            explanations: dict[str, str] = {}
            if subject.track == "A":
                round_result = self._track_a.get((model.label, cs.name)) or self.track_a(model, cs)
                for case in cases:
                    traces.append(self._trace(subject, case, round_result.finished[case.id]))
                explanations = dict(round_result.explanations)
            else:
                for (key, _spec, request), case in zip(self._choice_calls(model, cs), cases):
                    record = self.choice_record(model, case, key, request)
                    traces.append(self._trace(subject, case, record))
            out[subject.name] = (traces, explanations)
        return out

    def _trace(self, subject: Subject, case: Case, record: RawRecord) -> Trace:
        return Trace(
            case_id=case.id,
            split=case.split,
            raw=record,
            ground_truth=dict(case.expect),
            subject=subject.name,
        )

    # -- judges --------------------------------------------------------------

    def judge_plan(self, answers: Mapping[str, tuple[list[Trace], dict[str, str]]]):
        from . import judge as judge_mod

        if self._judge_cache is not None:
            return self._judge_cache
        case_of = {case.id: case for cases in self.plan.cases.values() for case in cases}
        explain_answers = []
        for subject in self.plan.subjects:
            traces, explanations = answers[subject.name]
            model = self.plan.models.get(subject.model) if subject.model else None
            for trace in traces:
                if trace.raw.outcome != "explain":
                    continue
                case = case_of.get(trace.case_id)
                request_text = (
                    contract._request_text(case) if case is not None and case.text else None
                )
                explain_answers.append(
                    judge_mod.ExplainAnswer(
                        subject=subject.name,
                        policy="raw",
                        case_id=trace.case_id,
                        request_text=request_text,
                        explain_text=explanations.get(trace.case_id),
                        provider=model.ref.provider if model else None,
                        model=model.ref.model if model else None,
                    )
                )
        judging = self.plan.manifest.judging
        extra = [s.name for s in self.plan.subjects]
        extra += [e.name for e in (*self.plan.manifest.candidates, *self.plan.manifest.baselines)]
        plan = judge_mod.plan_panel(
            explain_answers,
            [judge_mod.JudgeId(m.ref.provider, m.ref.model) for m in self.plan.judges],
            seed=judging.seed,
            rubric=judge_mod.load_rubric(judging.rubric),
            extra_blind_terms=extra,
            params={"max_output_tokens": judging.max_output_tokens},
        )
        recorded = []
        for rec in judge_mod.record_prompts(plan, self.run_dir):
            model = self.plan.models[f"{rec.judge.provider}/{rec.judge.model}"]
            params = {**rec.spec.params, **self._judge_params(model)}
            recorded.append(
                dataclasses.replace(rec, spec=dataclasses.replace(rec.spec, params=params))
            )
        self.ledger.register_many(judge_mod.unique_call_specs(recorded))
        split_of = {case.id: case.split for cases in self.plan.cases.values() for case in cases}
        requests: dict[str, tuple[Model, CallRequest]] = {}
        for rec in recorded:
            key = ledger_key(rec.spec)
            if key in requests:
                continue
            model = self.plan.models[f"{rec.judge.provider}/{rec.judge.model}"]
            params = {
                name: value
                for name, value in rec.spec.params.items()
                if name in ("max_output_tokens", "reasoning")
            }
            requests[key] = (
                model,
                CallRequest(
                    case_id=judge_request_id(key),
                    split=split_of.get(rec.spec.case_id, "test"),
                    case_text=rec.prompt,
                    prompt="",
                    interface="text",
                    params=params,
                ),
            )
        self._judge_cache = (plan, recorded, requests)
        return self._judge_cache

    def judge_work(self, answers) -> tuple[dict[str, list[WorkItem]], bool]:
        if not self.plan.judges:
            return {}, True
        _plan, _recorded, requests = self.judge_plan(answers)
        work: dict[str, list[WorkItem]] = {}
        final = True
        for key, (model, request) in requests.items():
            if self.ledger.entry(key).state not in (DONE, INVALID):
                final = False
            if self._fresh(key):
                work.setdefault(model.label, []).append(WorkItem(key, request, "judge"))
        return work, final

    def judge_results(self, answers) -> dict | None:
        from . import judge as judge_mod

        if not self.plan.judges:
            return None
        plan, recorded, _requests = self.judge_plan(answers)
        replies: dict[str, dict[str, str]] = {}
        for rec in recorded:
            key = ledger_key(rec.spec)
            cached = self.ledger.cached(key)
            if cached is None:
                continue
            model = self.plan.models[f"{rec.judge.provider}/{rec.judge.model}"]
            text = ""
            if self.ledger.entry(key).state == DONE:
                text = model.provider.reply_text(cached.raw).text
            replies.setdefault(rec.judge.name, {})[rec.prompt_hash] = text
        outcome = judge_mod.replay_scores(plan, replies, self.run_dir)
        return judge_mod.aggregate(plan, outcome.scores)

    # -- the pass ------------------------------------------------------------

    def dispatch(self, work: Mapping[str, list[WorkItem]]) -> bool:
        progress = False
        sync: list[tuple[Model, WorkItem]] = []
        seen: set[str] = set()
        for label in sorted(work):
            model = self.plan.models[label]
            items = [item for item in work[label] if item.key not in seen]
            seen.update(item.key for item in items)
            if self.blocked(model):
                continue
            if model.route == "batch":
                progress |= self.submit_batches(model, items)
            else:
                sync.extend((model, item) for item in items)
        progress |= self.run_sync(sync)
        return progress

    def run_pass(self) -> StepOutcome:
        self.resolve_orphans()
        answers = None
        final = False
        while True:
            progress = self.poll_batches()
            work, final = self.subject_work()
            if final:
                answers = answers or self.subject_answers()
                judge_work, judges_final = self.judge_work(answers)
                for label, items in judge_work.items():
                    work.setdefault(label, []).extend(items)
                final = judges_final
            if final and not self.live_batches():
                break
            if work:
                progress |= self.dispatch(work)
            if not progress:
                break
        self._persist()
        if final and not self.live_batches():
            if self.plan.smoke:
                self.write_smoke()
            else:
                self.finalize(answers)
            return StepOutcome(STATUS_COMPLETE, list(self.messages), exit_code=EXIT_OK)
        money = {k for k, v in self.provider_stops.items() if v.get("kind") == "money"}
        waiting = self.live_batches()
        if waiting:
            self.say(
                f"waiting on {len(waiting)} submitted batch(es) "
                f"({sum(len(v) for v in waiting.values())} call(s)); continue later"
            )
            return StepOutcome(STATUS_WAITING, list(self.messages), money, EXIT_WAITING)
        if not self.messages:
            self.say("no call could be sent; see status")
        return StepOutcome(STATUS_STOPPED, list(self.messages), money, EXIT_STOPPED)

    # -- outputs -------------------------------------------------------------

    def finalize(self, answers) -> None:
        """Traces, metrics, permutation, judge results, result.json and the page."""
        from . import deepeval_layer, policies, report

        answers = answers or self.subject_answers()
        run_dir = self.run_dir
        loaded_policies = {
            name: policies.load_policy(policies.builtin_policy_path(name))
            for subject in self.plan.subjects
            for name in subject.policies
        }
        subjects_doc = []
        permutation: dict[str, dict] = {}
        for subject in self.plan.subjects:
            traces, _explanations = answers[subject.name]
            final_traces = []
            for trace in traces:
                offered = list(trace.raw.candidates or [])
                truncated = (
                    trace.raw.outcome == "invalid" and trace.raw.invalid_reason == loop.TRUNCATED
                )
                for policy in subject.policies:
                    if truncated:
                        # A cut reply stays invalid under every policy (codex item 13).
                        trace = trace.with_policy(policy, "invalid", loop.TRUNCATED)
                        continue
                    decision, reason, _n, _v = policies.apply(
                        loaded_policies[policy], trace.raw.to_dict(), offered
                    )
                    trace = trace.with_policy(policy, decision, reason or "")
                final_traces.append(trace)
            write_traces(report.traces_path(run_dir, subject.name), final_traces)
            for policy in subject.policies:
                folder = run_dir / DEEPEVAL_DIR / f"{subject.name}__{policy}"
                folder.mkdir(parents=True, exist_ok=True)
                # deepeval prints its own summary; keep stdout for the runner's lines.
                with open(folder / "deepeval.log", "w", encoding="utf-8") as sink:
                    with contextlib.redirect_stdout(sink):
                        outcome = deepeval_layer.evaluate_traces(
                            final_traces, policy, results_folder=folder
                        )
                runstate.write_json_durable(
                    report.metrics_path(run_dir, subject.name, policy), outcome.corpus_metrics
                )
            doc: dict[str, Any] = {
                "name": subject.name,
                "kind": subject.kind,
                "policies": list(subject.policies),
            }
            if subject.kind != "reference":
                doc["artifact"] = self.plan.saved[subject.name].artifact
            subjects_doc.append(doc)
            key = perm.permutation_key(subject.name, "raw")
            probe = (
                subject.entry.permutation_probes.get(subject.case_set.name)
                if subject.entry
                else None
            )
            if subject.track == "A":
                permutation[key] = perm.not_measurable_entry(TRACK_A_PERMUTATION_REASON)
            elif probe:
                loaded_probe = perm.load_probe_json(resolve_private(probe, self.env))
                permutation[key] = perm.probe_to_entry(loaded_probe, split=subject.case_set.split)
        if permutation:
            runstate.write_json_durable(
                run_dir / report.PERMUTATION_FILENAME, perm.build_permutation_file(permutation)
            )
        results = self.judge_results(answers)
        if results is not None:
            from . import judge as judge_mod

            judge_mod.write_judge_results(run_dir, results)
        runstate.write_json_durable(
            run_dir / report.MANIFEST_FILENAME,
            {"run_id": self.state["run_id"], "date": self.state["date"], "subjects": subjects_doc},
        )
        report.generate(
            run_dir, result_path=run_dir / RESULT_FILE, markdown_path=run_dir / PAGE_FILE
        )
        self.state["status"] = STATUS_COMPLETE
        self._persist()
        self.say(f"complete: {run_dir / RESULT_FILE} and {run_dir / PAGE_FILE}")

    def write_smoke(self) -> None:
        """Per-model tokens, cost, truncation and a projected full-run cost (smoke run)."""
        smoke_ids = {case.id for cases in self.plan.cases.values() for case in cases}
        smoke_count = len(smoke_ids)
        scale = self.plan.full_case_count / smoke_count if smoke_count else 0.0
        billed = {
            e["key"]: e for e in self.billing.entries() if e.get("kind") == runstate.BILL_ANSWER
        }
        rows: dict[str, dict[str, Any]] = {}
        for entry in self.ledger.entries():
            model = self._model_of_spec(entry.spec)
            if model is None or entry.spec.get("case_id") not in smoke_ids:
                continue
            if entry.state not in (DONE, INVALID) or not self._current(model, entry.spec):
                continue
            row = rows.setdefault(
                model.label,
                {
                    "provider": model.kind,
                    "calls": 0,
                    "invalid": 0,
                    "truncated": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            cached = self.ledger.cached(entry.key)
            row["calls"] += 1
            row["invalid"] += entry.state == INVALID
            usage = cached.usage if cached else {}
            row["input_tokens"] += tokens_in(usage)
            row["output_tokens"] += tokens_out(usage)
            row["reasoning_tokens"] += usage.get(
                "reasoning_tokens", usage.get("thinking_tokens", 0)
            )
            bill = billed.get(entry.key)
            row["cost_usd"] += (
                bill["cost_usd"] if bill else model.cost(usage, self._discount(model.kind))
            )
            cut = entry.reason == loop.TRUNCATED
            if cached is not None and not cut:
                cut = _cut(model, cached.raw)
            row["truncated"] += bool(cut)
        for label, row in rows.items():
            model = self.plan.models[label]
            row["cost_usd"] = round(row["cost_usd"], 6)
            row["projected_full_run_usd"] = round(row["cost_usd"] * scale, 4)
            row["max_output_tokens"] = model.knobs["max_output_tokens"]
            row["reasoning"] = model.ref.reasoning
            row["flag"] = "CAPPED" if row["truncated"] else "OK"
            stop = self.model_stops.get(label)
            if stop:
                row["stop"] = stop["kind"]
        providers: dict[str, dict[str, float]] = {}
        for row in rows.values():
            total = providers.setdefault(
                row["provider"], {"cost_usd": 0.0, "projected_full_run_usd": 0.0}
            )
            total["cost_usd"] = round(total["cost_usd"] + row["cost_usd"], 6)
            total["projected_full_run_usd"] = round(
                total["projected_full_run_usd"] + row["projected_full_run_usd"], 4
            )
        smoke = {
            "cases": smoke_count,
            "full_run_cases": self.plan.full_case_count,
            "models": dict(sorted(rows.items())),
            "providers": dict(sorted(providers.items())),
        }
        runstate.write_json_durable(self.run_dir / SMOKE_FILE, smoke)
        self.state["smoke"] = smoke
        self._persist()
        for label, row in sorted(rows.items()):
            self.out(
                f"smoke {label}: {row['flag']} calls={row['calls']} truncated={row['truncated']} "
                f"invalid={row['invalid']} tokens in/out/reasoning={row['input_tokens']}/"
                f"{row['output_tokens']}/{row['reasoning_tokens']} cost=${row['cost_usd']:.4f} "
                f"projected full run=${row['projected_full_run_usd']:.2f}"
            )


def _cut(model: Model, raw: bytes) -> bool:
    try:
        return model.provider.reply_text(raw).truncated
    except Exception:  # noqa: BLE001 -- an unreadable reply is not a cut one
        return False


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> Manifest:
    try:
        return load_manifest(manifest_path)
    except (OSError, ManifestError, ValueError) as exc:
        raise RunError(f"manifest {manifest_path}: {exc}") from exc


def step(
    run_dir: Path,
    manifest_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    factory: ProviderFactory | None = None,
    retry_money: bool = True,
    retry_rejected: bool = False,
    out: Callable[[str], None] = print,
    clock: Callable[[], float] = time.time,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
    _start: Mapping[str, Any] | None = None,
) -> StepOutcome:
    """One pass over *run_dir* (``continue``), holding the run lock from the first
    read of ``run.json`` to its last write (codex review P2-12).

    The run's persisted scope decides what is sent: a smoke run stays a smoke
    run however it is resumed (P1-6).
    """
    env = os.environ if env is None else env
    run_dir = Path(run_dir)
    _check_run_dir(run_dir)
    if _start is None and not (run_dir / RUN_FILE).exists():
        raise RunError(f"{run_dir} holds no run; start one with `run` (or `smoke`)")
    digest = _sha256_file(manifest_path)
    manifest = _load_manifest(manifest_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    with _open_ledger(run_dir) as ledger:
        state = runstate.load_state(run_dir)
        if _start is not None:
            state = _begin(state, manifest, env, digest, _start)
        elif not state:
            raise RunError(f"{run_dir} holds no run; start one with `run` (or `smoke`)")
        if digest != state.get("manifest_sha256"):
            state["manifest_sha256"] = digest
            state.setdefault("manifest_history", []).append(digest)
            out(f"manifest changed since the last pass (now {digest[:12]})")
        scope = state.get("scope") or {"mode": "full"}
        plan = build_plan(manifest, env, factory or default_factory, scope=scope)
        if _start is not None and scope.get("mode") == "full":
            state["started_full"] = True
        state["mode"] = scope.get("mode", "full")
        runstate.save_state(run_dir, state)
        runner = Runner(
            run_dir,
            plan,
            ledger,
            state,
            retry_money=retry_money,
            retry_rejected=retry_rejected,
            out=out,
            env=env,
            clock=clock,
            poll_seconds=poll_seconds,
            recheck_seconds=recheck_seconds,
            factory=factory or default_factory,
        )
        try:
            outcome = runner.run_pass()
        finally:
            runner._persist()
        state["status"] = outcome.status
        state["messages"] = outcome.messages
        runstate.save_state(run_dir, state)
    return outcome


def _begin(
    state: dict, manifest: Manifest, env: Mapping[str, str], digest: str, start: Mapping
) -> dict:
    """``run`` / ``smoke`` on *state* (under the lock): create it, or check the scope."""
    smoke_cases = start.get("smoke_cases")
    if not state:
        scope = smoke_scope(manifest, env, smoke_cases) if smoke_cases else {"mode": "full"}
        return _new_state(digest, start.get("run_id"), start.get("date"), scope)
    scope = state.get("scope") or {"mode": "full"}
    if smoke_cases:
        if scope.get("mode") != "smoke":
            raise RunError("this run dir holds a full run; a smoke run needs its own run dir")
        if len(scope.get("case_ids", [])) != smoke_cases:
            raise RunError(
                f"this run dir holds a {len(scope.get('case_ids', []))}-case smoke run; "
                "use the same --cases, or a new run dir"
            )
        return state
    if scope.get("mode") == "smoke":
        if not start.get("expand"):
            raise RunError(
                "this run dir holds a smoke run; expand it to the full run with "
                "`run --expand`, or start the full run in a new run dir"
            )
        state["scope"] = {"mode": "full"}
        state["expanded_from"] = scope
        return state
    if state.get("started_full"):
        raise RunError("this run dir already holds a run; use `continue`")
    return state


def start(
    run_dir: Path,
    manifest_path: Path,
    *,
    run_id: str | None = None,
    date: str | None = None,
    smoke_cases: int | None = None,
    expand: bool = False,
    **kwargs: Any,
) -> StepOutcome:
    """``run`` (and ``smoke``): initialize or check *run_dir*, then take a pass.

    Nothing is written until the plan has been built, so a configuration
    error can be fixed and ``run`` repeated.
    """
    start_args = {"run_id": run_id, "date": date, "smoke_cases": smoke_cases, "expand": expand}
    return step(Path(run_dir), Path(manifest_path), _start=start_args, **kwargs)
