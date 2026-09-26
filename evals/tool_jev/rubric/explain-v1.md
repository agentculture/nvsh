# Explain rubric, version explain-v1

This rubric grades *explain text*: the prose an answer gives an operator
at an NVIDIA Jetson / DGX Spark / RTX Spark shell when it explains a failure
or a request instead of (or as well as) proposing a command. It is read by
`evals/tool_jev/judge.py`, which passes the numbered steps below to
DeepEval's G-Eval **verbatim** as fixed `evaluation_steps`, so G-Eval makes
exactly one scoring call per answer and never drafts its own steps.

The judged test case carries only the operator's request (G-Eval's "Input")
and the answer's explain text ("Actual Output"). The answer's author is
never named: model, provider and subject names are removed before a judge
sees the text.

Panel scores are reported next to the release gate, never inside it: no
explain score enters any release bar.

Changing a step changes every judge prompt, so any edit is a new version:
copy this file to `explain-v2.md` and edit the copy; never edit a
published version in place.

## Evaluation steps

1. Read the Input as an operator's request or a failed command at a Linux shell on an NVIDIA Jetson, DGX Spark or RTX Spark machine, and identify what the operator needs to understand.
2. Check that the Actual Output explains the likely cause or meaning correctly and does not state anything technically false about the command, the error or the platform.
3. Check that the explanation is grounded in the details the Input gives (command, exit status, error text) rather than generic advice that would fit any failure.
4. Check that any next step it suggests is concrete and actionable, and that a destructive or privileged step (deleting data, sudo, changing drivers or power modes) is clearly flagged as such rather than presented casually.
5. Check that it is honest about uncertainty: it says what it cannot know from the Input instead of inventing machine state, versions or outputs it was not given.
6. Penalise padding, repetition and irrelevant material; a short, correct, specific explanation should score higher than a long, vague one.
