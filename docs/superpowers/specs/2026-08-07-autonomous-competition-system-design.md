# Autonomous Competition System — Architecture Design

**Goal:** A fully autonomous system that wins the `ai-agent-security-multi-step-tool-attacks`
Kaggle competition — it searches, calibrates against the real T4, decides when to spend a
submission slot, and submits, with a human only setting policy once.

**Locked decisions (2026-08-07):**
- **Autonomy:** auto-submit within a policy (margin + slots/day + budget). No human in the
  per-submission loop.
- **Objective:** maximize public `min(gpt_pub, gemma_pub)` **plus** a private-robustness
  hedge (final rank is on the blind private board).
- **Horizon:** weeks — invest in the closed loop, self-recalibration, predicate expansion,
  and self-repair.

This is an **architecture spec** covering five coupled subsystems. Only the keystone is
detailed to implementation depth here; the other four are scoped and each gets its own
spec (see Build Order). This document is the parent all five reference.

---

## Background: why this system exists

Three findings from the 2026-08-07 investigation motivate the whole design:

1. **The local objective does not predict the real board.** `best_objective` is a projection
   (`≈ 0.09 × budget/cost` from local gen_chars). Across schemes v7–v11 the raw replayed
   board stayed flat at 0.72 while `best_objective` swung 57→100.8 — pure re-projection.
   team-90234a climbed 76→100.8 with a *static* attack. The projection is decoupled from T4
   throughput; local scoring provably cannot rank T4 per-candidate speed.
2. **The search mode-collapses.** Showing the champion's verbatim shape enables cross-gen
   hill-climbing (v7's climb) but collapses the search onto one shape when the champion is
   unreproducible (v10's forge ghost). Objective loopholes (mean-over-models) let a
   lopsided shape that fires on one model win.
3. **Only EXFIL fires today.** CONFUSED_DEPUTY is reachable and **additive** but untapped;
   the other two predicates are structurally blocked by the guardrail. Calib kernels give
   **free** T4 signal (no slot); submissions are the only ground truth (5/day); **private
   columns are blind until the competition ends.**

The system's core bet: **make the fast local optimizer *correct* by continuously
recalibrating it against free T4 measurements**, rather than trusting a proxy or throttling
the search to kernel speed.

---

## Architecture

```
        ┌──────────────── Autonomy controller (budget, self-repair, telemetry) ───────────────┐
        │                                                                                       │
   Search engine ──candidates──▶ Calibration service ──real T4 signal──▶ Transfer/recalibrator
   (propose + local score)           (free calib kernels)               (fit local→T4, fix objective)
        ▲                                                                            │
        │ grounded objective params                                                 ▼
   Private-robustness hedge ◀───────────────────────────────── Submission policy (auto-submit, ingest LB)
```

Data flows one loop: search proposes and scores locally under a **grounded** objective; the
calibration service measures the champion + challengers on the real T4; the transfer model
refits the objective's constants and per-family corrections from those measurements; the
submission policy spends a slot when the *calibrated* estimate clears the bar and re-anchors
the scale from the real result; the controller supervises budgets and restarts.

---

## Keystone — Calibration service + Transfer model

Closes the local↔T4 gap. Nothing else is worth building until `best_objective` predicts the
real board.

### Calibration service
- **Input:** current champion + top-K local challengers, deduped by templatized shape
  (`fill.templatize`).
- **Action:** builds **one** free T4 calib kernel (wrapping `scripts/build_calib_kernel.py`)
  that probes each shape on **both** models, measuring per `shape × model`:
  `fires?`, `t4_seconds_per_candidate`, `board_contribution`.
- **Execution:** push + poll in the background (no submission slot). Cadence: on every new
  local champion **plus** a periodic floor (default every few hours) to catch drift.
- **Interface:** `calibrate(shapes: list[str]) -> list[CalibResult]` where
  `CalibResult = {shape_templatized, model, fires: bool, t4_s_per_cand: float,
  board: float, kernel_ref: str, ts: float}`.

### Calib store
Append-only JSONL of `CalibResult` rows — the single source of measured T4 truth. Read by
the recalibrator; never mutated.

### Transfer / recalibrator
A **pure refit** over `(calib store, submission anchors)` producing `objective_params`, at
two levels:
- **Global constants:** fit `T4_s ≈ a·gen_chars + b·turns` and the effective per-model T4
  budget → rewrites `FILL_BUDGET_CHARS` and the cost weights so projected
  `N = budget/cost` matches measured T4 throughput.
- **Per-family residual:** a multiplier per shape-family, capturing "fires locally but not
  on T4" (→ ~0) and "local gen_chars mispredicts T4 speed" (speed correction).
  `project_public_board` gains a **per-family multiplier input**.
- **Anchoring:** real **submission** results set the global *scale* (generalizes today's
  `ARTIFACT_LB_REFERENCE`); calib refines *between* submissions.

**Interface:** `refit(calib_store, submission_anchors) -> objective_params`, where
`objective_params = {budget_by_model, cost_a, cost_b, family_multiplier: dict[str, float]}`,
written to a params file the scorer reads each generation (hot-reload, no restart).

### Data flow
search → champion+challengers → calib kernel → calib store → `refit` writes `objective_params`
→ scorer reads them → championing/submits grounded → each completed submission re-anchors the
scale and yields a **transfer prediction error**; large error forces an immediate recalibration.

### Error handling (bad signal is structurally harmless)
- Calib kernel fails/times out → keep last-good params, retry with backoff.
- Too few points / degenerate fit → fall back to submission-anchored constants (today's
  behavior).
- **Stale calib** (no fresh data in a window) → down-weight *unverified* families so an
  unmeasured shape **cannot become champion or be auto-submitted**. This is the core safety
  rule.
- Clamp per-refit constant swings so one noisy kernel can't move the objective wildly.

### Testing
- Transfer fit recovers known constants from synthetic calib data.
- A non-firing-on-T4 family gets multiplier → 0 and is demoted in championing.
- Recalibrated projected board matches a held-out calib measurement within tolerance.
- Integration: local and T4 disagree (fires locally, not on T4) → the T4 signal wins the
  championing decision.

---

## Subsystem 2 — Submission policy engine

Depends on the keystone's calibrated estimate. **Gate to spend a slot:**
`calibrated_est > best_real_LB + margin` **and** `slots_left_today > reserve` **and** within
budget **and** the candidate differs from the last submit by more than a **family threshold**
(slots probe distinct hypotheses, not near-duplicates). On fire: build artifact → verify
(parse / fill / isolation contract) → push kernel → `competition_submit_code` → poll → ingest
all four column scores → re-anchor. **Adaptive:** if a submit underperforms its estimate by
> X, auto-tighten `margin`. Human sets `{margin, slots/day reserve, min-hours-between,
budget cap}` once. Its own spec.

## Subsystem 3 — Search robustness + predicate expansion

The 2026-08-07 exploration fixes (min-over-models objective, incumbent-shape hill-climbing,
forge demoted, deputy enabled) **plus** a **mode-collapse detector** (entropy of recent
proposal families) that, on collapse, injects exploration (raise temperature / withhold
incumbent for a few gens / seed fresh families) — self-repair for the search. **Predicate
expansion:** each reachable predicate is a shape-family with its own `fill.py` stamping
support (as deputy now has); only EXFIL + CONFUSED_DEPUTY are reachable, so this means
harvesting deputy fully and probing for anything newly reachable — not chasing the two
structurally-blocked predicates. Its own spec.

## Subsystem 4 — Private-robustness hedge

**Private is blind until the competition ends** (all `privateScore` blank in the submission
history), so this hedge is **proxy-only and unlearnable** — a human-set weight, not optimized.
Objective term `+ λ_priv · robustness_hedge`, where the hedge = min board across **guardrail
perturbations** (existing robustness-sweep tool) and firing-family **diversity** (one fragile
private quirk cannot blank the whole fill). Periodic sweep on the champion; a champion that
blanks under perturbation is demoted. Its own spec.

## Subsystem 5 — Autonomy controller

Supervises the long-running processes (search, calibration, recalibrator, submission gate,
robustness sweep) with **self-repair** — restart-from-persistent-state on crash (blackboard,
calib store, params, anchors are all durable). A **budget ledger** (API $, GPU, 5 slots/day,
kernel quota) that throttles when a limit nears (e.g. pause proposers if API $ low).
Telemetry to wandb (`best_objective`, `calibrated_est`, `real_LB` anchor, transfer error,
slots/budget used) and push-notification alerts on: new best real LB, submit failure, budget
threshold, prolonged collapse. Global pause + per-subsystem enable flags. Its own spec.

---

## Global constraints (bind every subsystem)

- **Isolation contract:** the shipped `attack.py` imports only `aicomp_sdk` + stdlib; the
  candidate list ships as embedded JSON. Calib/submission builders must preserve this.
- **Do not modify** `harness/models.py`, `harness/runner.py`, or `vendor/`.
- **Submissions spend a daily slot (5/day) and are externally visible.** The policy engine is
  the *only* component that may submit; it must respect the slot ledger and never
  double-submit.
- **Calib kernels are free** (no slot) but consume Kaggle kernel quota and T4 queue time —
  budget them.
- **Private columns are unobservable** until competition end — never gate or learn on private.
- **`KAGGLE_API_TOKEN` and `.env` secrets** are never printed or logged.
- **Two-GPU green box:** gpt_oss on device 0 (RTX 3090), gemma_4 on device 1 (RTX 6000 Ada),
  scoring in parallel. Calibration/sweeps share these GPUs — the controller schedules them so
  they don't starve the search.
- **`uv run` for everything; `pre-commit` green before commit; no `# type: ignore`/`# noqa`.**

---

## Decomposition & build order

Each is its own spec + plan after this parent is approved:

1. **Keystone — calibration + transfer** (this doc, detailed). Foundational; unblocks all.
2. **Submission policy engine.** Needs the calibrated estimate from (1).
3. **Search robustness + predicate expansion.** Partly landed 2026-08-07; extend.
4. **Private-robustness hedge.** Objective term + periodic sweep.
5. **Autonomy controller.** Budget ledger, self-repair, telemetry — ties (1)–(4) together.

## Success criteria

- **Keystone:** `calibrated_est` predicts each completed submission's public score within a
  stated tolerance (transfer error tracked in wandb and trending down).
- **System:** best **real** public LB strictly exceeds the current best (87.48) and climbs
  across submissions with no human in the submission loop; no wasted slots on
  near-duplicate or unverified candidates; survives crashes without losing state.
