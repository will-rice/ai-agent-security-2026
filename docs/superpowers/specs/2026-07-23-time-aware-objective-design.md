# Time-Aware Objective (green-seconds replay budget) — Design

**Date:** 2026-07-23
**Status:** Approved (pending spec review)
**Component:** `src/jed_attack/campaign/submission_score.py`, `config.py`, `optimize_prompts.py`, `prompts.toml`

## Goal

Make the local scorer **measure each submission's replay time** and **zero (with feedback) any submission that exceeds a green-seconds budget** calibrated to the T4 gateway's 9000s/model `INVALID_SUBMISSION` limit — so the optimizer stops building submissions that time out on the real board.

## Motivation

Even with faithful in-process scoring, the optimizer has no replay-time constraint, so it builds submissions that exceed T4's 9000s/model wall-clock budget → `INVALID_SUBMISSION` → 0 on the LB (memory `jed-t4-replay-time-budget`). The scorer now replays each candidate in-process, so it can **time** them. By zeroing over-budget submissions, the local `public` faithfully reproduces T4's validity verdict, and the optimizer avoids over-time submissions *because they score 0* — no separate knapsack needed.

## Key decisions (locked with the user)

1. **Zero-invalid + feedback (A).** Over-budget → `public = 0` (mirrors T4 `INVALID`), and the feedback states *why* (the overage + how many candidates to trim), so the proposer shrinks rather than getting a silent 0. Budget carries a **safety margin** (set below the true T4 cliff) so gpt_oss time-variance doesn't lose near-boundary winners.
2. **Green-seconds budget (B), measured and exposed.** Budget directly in green replay-seconds (NOT scaled to T4). The scorer measures + logs green-seconds so we calibrate the budget from *observed* pass/fail totals ("see how many green-seconds actually passes").

## Calibration (measured 2026-07-23, warm, all-8-hop)

- gpt_oss: **5.6 green-s/candidate** (time is stable run-to-run — 28.1s vs 28.2s — even though firing is non-deterministic).
- gemma: **0.54 green-s/candidate**.
- T4 fits ~23 gpt_oss candidates in 9000s (385s each); green does 23 in ~129 green-s. Initial budget with margin: **`GREEN_REPLAY_BUDGET_S = {"gpt_oss": 120.0, "gemma_4": 60.0}`** (gpt_oss binds at ~21 candidates; gemma never binds — 25 candidates = 13.5 green-s). Refine from the logged green-seconds vs T4 pass/fail.
- Sanity: 25×8-hop candidates = 140 green-s (gpt_oss) > 120 → correctly rejected (this is exactly the over-time class `MAX_SHIP_MESSAGES=25` was letting through).

## Architecture

### `submission_score.py`

- **Measure**: in `replay_trace`, wrap the model-touching block (`env.reset()` + `env.interact()`, already under the per-model lock) with `time.perf_counter()`; return the elapsed green-seconds alongside `(trace, predicates)`.
- **Accumulate**: `score_submission` sums elapsed green-seconds per model into `replay_seconds: dict[str, float]`.
- **Zero-invalid**: after the board math, if `replay_seconds[model] > config.GREEN_REPLAY_BUDGET_S[model]` for ANY model, set `public = 0.0` and mark the submission over-budget. Replay ALL candidates first (green is fast — no early-stop needed), then zero — matching T4's binary INVALID.
- **Feedback**: when zeroed, set a submission-level reason and surface it in the per-message `feedback` / a `SubmissionScore` field, e.g. `"OVER T4 REPLAY BUDGET: gpt_oss {secs:.0f} green-s > {budget:.0f} (trim ~{n} candidates)"`, so `optimize_prompts`/`victim_feedback` carry it to the proposer.
- **`SubmissionScore`** gains `replay_seconds: dict[str, float]` (and an over-budget indicator, e.g. via `public==0` + the feedback string, or an explicit bool — implementer's cleanest choice).

### `config.py`

- `GREEN_REPLAY_BUDGET_S: dict[str, float] = {"gpt_oss": 120.0, "gemma_4": 60.0}` — green replay-seconds per model, calibrated (see above). Env-overridable is optional; a plain constant is fine (a calibration knob like `MODEL_GPU`).

### `optimize_prompts.py` + `prompts.toml`

- **Expose**: log `replay_seconds` (per model) to wandb each generation, and include it in the blackboard record / generation log, so the budget is tunable from observation.
- **Proposer guidance**: add a `{{TIME_BUDGET}}` token to `prompts.toml` stating the green-seconds budget per model and the ~per-candidate cost (gpt_oss ~5.6 green-s), so the proposer self-limits candidate count. Substitute it in `submission_prompt` like `{{MAX_MESSAGES}}`.

## Determinism note

gpt_oss *firing* is non-deterministic (memory `jed-t4-replay-time-budget`), but its *replay time* is stable, so the time-invalid check is reliable. The budget margin absorbs the small time variance.

## Testing

- **Unit (no GPU):** stub `replay_trace` to return scripted `(trace, predicates, elapsed_seconds)`; assert (a) an under-budget submission scores normally and carries `replay_seconds`; (b) an over-budget submission (elapsed × count > budget) gets `public == 0` AND feedback naming the overage; (c) the per-model check zeroes when only ONE model is over budget.
- The existing no-dedup / resident-cache tests keep passing (the timing is additive to `replay_trace`'s return; update their stubs to the new return arity).

## Risks

- **Calibration drift:** green-seconds↔T4 mapping assumes a stable green/T4 speed ratio; if green hardware changes, recalibrate (the budget is one constant per model, and green-seconds are logged).
- **Ratio varies by message:** a candidate's green/T4 time ratio may not be exactly 69×; the empirical budget + margin absorbs this, and the logged green-seconds let us tighten it.
- **`replay_trace` return arity changes** (adds elapsed seconds) — every caller + test stub updates; contained to `submission_score.py` + its tests.

## Out of scope

- gpt_oss firing non-determinism / median-of-N averaging (separate; the score is faithful-but-noisy, the time is stable).
- A T4 re-verify to pin the exact budget (spend a slot later; the initial budget is calibrated from the green rate + known T4 fit).
