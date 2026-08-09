# Char-Model Objective + Drop Green Budgets — Design Spec

**Date:** 2026-08-09
**Status:** Draft for review
**Scope:** the jed_attack optimizer's objective + the green-budget constants

## Decision

Chars are a **better proxy than green wall-clock time** for T4 replay cost, so:
1. Make the deterministic **char-model projection** (`project_public_board`) the objective
   again.
2. **Delete** the sampled/measured-green machinery (`sampled_board`, its cache, the
   per-generation extra replays).
3. **Drop all green budgets** — remove `GREEN_REPLAY_BUDGET_S` entirely and everything that
   reads it.

## Why

- **Efficiency:** the char projection is pure arithmetic over data `score_submission`
  already produces (gen_chars, turns, severity, cells), so it costs **zero extra replays
  per submission** — versus the sampled objective's +80 replays each. Because it's that
  cheap, the ~6× objective-call sites no longer need memoization, so the caching layer
  (`sampled_board_by_model`) is deleted too. Net: each generation's cost drops back to just
  the shape replays.
- Green replay wall-clock is ~28× faster than the T4 gateway **and** noisy (a prior
  measured-time objective spiked when one shape glitch-measured at ~0.005 s). It is a
  worse, noisier proxy than the deterministic `gen_chars` (T4 generation time ≈ 0.0525·chars,
  plus a turn term) that the char model already fits. The recent switch to sampled-green
  timing was a regression.
- Making wandb equal the *absolute* real LB is a separate concern — a **calibration** of the
  proxy→T4 constant against real submission scores (40.4 / 53.4 / 67.2), NOT a matter of
  which proxy we measure. It is explicitly **out of scope** here; the char objective *ranks*
  correctly (fewer generated chars → more candidates fit → higher board), which is what the
  search needs. Recalibration can be a later, separate change.

## Design

The char projection is already implemented (`project_public_board`, currently used only as
telemetry). Rewire it back to the objective and delete the green path.

- `optimize_prompts._score_public_raw_per_gen_char(score)` →
  `0.0 if not score.valid else sum(project_public_board(score, config.FILL_BUDGET_CHARS,
  config.SHIP_CANDIDATE_CAP).values())` (SUM over models — unchanged from the read-exfil
  change; the private columns are independent).
- **Delete:** `submission_score.sampled_board`, `SubmissionScore.sampled_board_by_model`,
  `config.REPLAY_SAMPLE_SIZE`, `optimize_prompts._objective_messages`, the `sampled_board`
  import, and the `_score_batch` sampled computation (so `_score_batch` just returns
  `score_submission` results). No per-message `replay_seconds` field is added.
- **Drop `GREEN_REPLAY_BUDGET_S`** (`config.py`) and its readers:
  - `submission_score.py` — the sampled `budget_fit` reader (removed with `sampled_board`);
    fix the module docstring reference.
  - `optimize_prompts.py:934` — the `{{TIME_BUDGET}}` substitution: express the replay
    budget in the terms the objective now uses (the ~9000 T4-s / `FILL_BUDGET_CHARS`
    char budget per model) instead of green-seconds.
  - `prompts.toml` — update the `{{TIME_BUDGET}}` token doc accordingly.
- `blackboard.objective_scheme_name` base: `..._sum_sampled_v15` → `..._sum_v16` (retire the
  sampled pool; back to the char projection under a fresh tag).
- **Log the turn count.** The objective already penalizes turns (`TURN_COST_WEIGHT *
  agent_turns`), but the count isn't logged. Add a per-model `batch_mean_turns_{model}`
  wandb metric alongside `batch_mean_gen_chars_bottleneck`, so turn minimization (e.g. the
  post-tool wrap-up collapsing) is visible.

## Kept (unchanged)
The char cost model — `cost = gen_chars + TURN_COST_WEIGHT * agent_turns` (`TURN_COST_WEIGHT
= 55`), so the objective minimizes BOTH generated characters AND agent turns (a turn ≈ 55
chars of fixed per-hop cost; fewer hops — e.g. collapsing the post-tool wrap-up that
generates but scores nothing — lowers cost and raises the board). Also `FILL_BUDGET_CHARS`,
`project_public_board`, `fill` (distinct-shape authoring + URL-swap expansion + unique-domain
novelty), `MAX_SHIP_MESSAGES = 30`, `MAX_SCORE_BATCH = 1`, read-exfil / deputy / gate-guardrail
semantics, the `attack.py` grade-time trim contract.

## Edge cases / risks
- Small blast radius: this reverts a recent change and deletes constants; fill/ship/novelty
  untouched.
- The objective's absolute magnitude changes (green board → char board) — handled by the
  scheme bump (fresh pool), so a stale-scale row can't freeze the champion.
- Watch for any other `GREEN_REPLAY_BUDGET_S` reader outside the grep (scripts, tests).

## Testing
- Objective sums the char `project_public_board` over models; invalid → 0.
- Scheme is `..._sum_v16`.
- No reference to `GREEN_REPLAY_BUDGET_S`, `sampled_board`, `REPLAY_SAMPLE_SIZE` remains in
  `src/` (grep clean).
- Existing `project_public_board` / fill / ship tests stay green; sampled tests deleted.

## Resolved decisions
1. `{{TIME_BUDGET}}` is phrased as "~9000 T4-s per model; minimize generated characters"
   (the real constraint the chars proxy), not as the char-budget figure.
2. Recalibrating the char→T4 constant against real LB scores (to make wandb ≈ the absolute
   real board) is a SEPARATE, later change. This spec leaves the objective a consistent
   scaled proxy — correct for ranking, which is what the search needs.
