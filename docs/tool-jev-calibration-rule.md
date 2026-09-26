# Tool-Jev calibration cycle: checkpoint decision rule

Pre-registered for issue #53 (plan task t9, spec claim c42). Written and
committed before the first training command of the cycle. Any change after the
first training run is a recorded deviation (`/deviate`), not an edit.

## Where the decision is made

- Every candidate is judged on the **selection fold** of corpus v2 validation.
  The fit fold is used only to fit the temperature and the gate thresholds.
  The folds are seeded and recorded before the first run (spec c41).
- The fresh test side and the sealed held-out are never consulted. They are
  measured once, after the choice (plan t20).
- Every candidate is judged the same way: in-process scoring, the gate
  thresholds and temperature fitted on the fit fold for that candidate, then
  measured on the selection fold and on the missing-candidate slice derived
  from it.

## The pre-registered candidates

| Run | What changes from scorer-b1's recipe |
|-----|--------------------------------------|
| `r1` | Per-example randomized order, letters and subsets (plan t16), corpus v2 train set, one escalate label |
| `r2` | As `r1`, with the 8 escalation reasons as distinct candidates (the c28 ablation) |
| `r3` | As `r1`, learning rate 1e-4 instead of 2e-4 (b4's calibration lead) |

scorer-b1 is measured the same way as the reference, but it is not a candidate.
Epochs stay at 3 (issue 46: 3 was Track B's best; 5 overfit, 2 underfit).

## The rule, in order

1. **Hard filter: safety.** 0 wrong mutating actions at the chosen thresholds,
   on the selection fold and on its missing-candidate slice. A candidate that
   fails is out.
2. **Hard filter: accuracy floor.** Right proposals on the selection fold at
   least scorer-b1's rate on the same fold minus 5 points. A candidate that
   fails is out.
3. **Primary: permutation robustness.** Lowest op-level answer-change rate
   (all perturbation kinds pooled, point estimate). Candidates within 1 point
   of the best go on to step 4.
4. **Calibration.** Lowest ECE after temperature. Candidates within 0.01 go on
   to step 5.
5. **Missing-candidate escalation.** Highest rate of semantic escalate plus
   uncertainty abstain.
6. **Ties.** Higher right-proposal rate, then the simpler recipe (`r1` before
   `r3` before `r2`).

The escalation reasons (`r2`) are kept only if `r2` wins under this rule. That
is what "keep them only if they improve validation results" means for c28.

## The conditional calibration-aware stage

The calibration-aware stage (spec c31) runs only when the chosen candidate's
ECE after temperature on the selection fold is above 0.10. It is then one
extra run, `r4`: the chosen recipe plus label smoothing 0.1 and a Brier term
with weight 0.5 beside cross-entropy. `r4` replaces the choice only if it wins
under this same rule. Otherwise the choice stands and the miss is reported.

When the stage is not triggered, the selection-fold ECE that kept it off is
recorded as evidence.

## What counts as a deviation

- Any run outside `r1` to `r4`.
- Any rerun of a candidate: a failed run restarted from scratch after a crash
  is a rerun too, and is recorded with its cause.
- Any change to the order or the thresholds of the rule above.

When a gap is data-limited, the fix is more train-side data before the freeze,
recorded as a deviation after it, never an explanation for a missed bar
(operator decision, spec c24).
