# Per-Model Optimization — Design Spec

**Date:** 2026-08-05
**Status:** Draft for review
**Goal:** Optimize `gpt_oss` and `gemma_4` as independent leaderboard columns instead of collapsing them into one mean-objective champion, so each victim column is maximized by the shape that is actually fastest for *that* model.

---

## 1. Motivation

The public LB scores four independent columns (`gpt_oss_public`, `gemma_public`, and the two private mirrors); the public score is the **mean of the two public rows**. The two victims want *different* attack shapes, confirmed by real-T4 measurement (`scripts/build_calib_kernel.py`, 2026-08-04):

| template | gpt_oss T4 | gemma T4 |
|---|---|---|
| harmony injection (`discovered_0`) | **13.35 s** (optimal) | 17.09 s |
| `bare_ok` / `run_reply_ok` | 30–33 s | **~16 s** (optimal) |

The harmony analysis-forge collapses gpt_oss's chain-of-thought (2.4× win) but does nothing for gemma, which has no heavy reasoning channel and is roughly flat ~16–24 s across templates — where the injection is actually *slightly slower* than a plain `bare_ok`.

Today the whole search collapses this asymmetry (see §3). A candidate that is excellent for gpt_oss but mediocre for gemma is averaged down and discarded, when it should be *kept as gpt_oss's champion*. Independent columns deserve an independent search.

---

## 2. The hard constraint (shapes the entire design)

**You submit exactly ONE `attack.py`. The gateway runs it once per model — fresh instance, fresh `env` bound to that model — but never tells the attack which model it is attacking.** Verified against the installed SDK and gateway excerpts:

- `aicomp_sdk.attacks.contracts.AttackRunConfig` has only `time_budget_s`, `max_steps`, `max_tool_hops` — **no model field**.
- The gateway's init message to the attack is `{cmd:init, budget_s, seed}` — **no `model_name`**. `AICOMP_MODEL_NAMES` lists *all* models, not the current one.
- Consequence: the artifact's existing `_GPT_TEMPLATE_ORDER` / `_GEMMA_TEMPLATE_ORDER` name-branching (`_model_hint`) is **dead code at grade time** — it only fires in unit tests that inject `{"model": ...}`.

Therefore per-model behavior is achievable **only** through two channels, and this design uses both:

1. **Offline (proposer/search):** find per-model-best shapes and keep a champion per model.
2. **Grade-time (shipped attack):** the one artifact carries *both* champions' shapes and **probes its live env**, selecting the per-env-fastest by **measured latency** (it runs separately on each model's env, so it can adapt without knowing the name).

Shipping two files, or branching on model name, is impossible and out of scope.

---

## 3. Current architecture (what collapses the columns)

- **Objective** (`optimize_prompts.py`): `_score_public_raw_per_replay_s` = `_robust_public(score) / _replay_s_cost(score)`, where the numerator is `mean_over_models(public_by_model)` (at `ROBUSTNESS_LAMBDA=0`) and the denominator is `max_over_models(replay_seconds)` (the bottleneck model). Net: **mean-raw ÷ bottleneck-replay-second — a single collapsed scalar.**
- **Champion** (`blackboard.py`): one global `best_objective()` ranked by that scalar; `append()` reships one `attack.py` from that champion's messages.
- **Already per-model (reuse, don't rebuild):** `SubmissionScore.public_by_model`, `.replay_seconds`, `.gen_chars`, `.severity_by_model` are all per-model dicts; `GREEN_REPLAY_BUDGET_S` and `MODEL_GPU` are per-model; the scorer already replays both models concurrently; `_score_fires_model` exists; `reship_champions` already demonstrates writing two artifacts.
- **The only true collapse points:** the `mean` at `submission_score.py:467` (`SubmissionScore.public`), the bottleneck-`max` in `_replay_s_cost`/`_gen_chars_cost`, the mean numerator in `_robust_public`, and the single scalar `Record.objective` + single `best_objective()`.

---

## 4. Design

### 4.1 Objective becomes a per-model vector

Replace the collapsed scalar with a per-model objective computed independently for each model:

```
objective_by_model[m] = public_by_model[m] / replay_seconds[m]      (floored as today)
```

No mean numerator, no bottleneck-max denominator. Each candidate carries one objective per model.

**Faithfulness note.** The proposer scores on the local GPU (green-seconds). Green mis-predicts *absolute* T4 candidate count (§ calibration work), but **within a single model** green replay-seconds preserves the *ranking* of shapes (injection < natural on both green and T4). The bug being fixed here is the cross-model *mean*, not within-model ranking — so a per-model green objective is a real, sound improvement. Absolute T4 faithfulness is handled where it matters: the grade-time probe (§4.4) ranks on measured *T4* latency directly.

### 4.2 Two independent champions

Keep the blackboard as one pool of records; every scored record carries `objective_by_model` for *both* models. A champion is just an argmax:

```
champion(m) = argmax_records objective_by_model[m]      (among valid, firing-on-m, current-scheme rows)
```

The two champions may be the same record or different records. No record duplication or per-model segregation of the pool.

- Firing must be **per-model** (`_score_fires_model`, already exists): a record can be champion for gpt_oss even if it does not fire on gemma.
- Validity/budget is per-model: an over-`GREEN_REPLAY_BUDGET_S[m]` candidate zeros only model `m`'s column, not the whole record.

### 4.3 Data-model changes

- `Record`: add `objective_by_model: dict[str, float]` and `replay_seconds_by_model: dict[str, float]` (the latter is **not** persisted today and is required to recompute the per-model objective). Keep the legacy scalar `objective` populated (e.g. with the better of the two, or the gpt_oss column) for backward-compatible tooling, but it stops driving champion selection.
- `objective_name` / scheme tag: bump to a **per-model scheme** (e.g. `per_model_raw_per_replay_s_v3`). This is the existing mechanism (`_objective_key` prefers current-scheme rows) and it cleanly quarantines old single-scalar rows so they cannot pollute per-model champion selection. Old records lack `objective_by_model` → treated as stale scheme (never champion until re-scored), exactly as prior scheme bumps behaved.
- `SubmissionScore`: stop reading scalar `.public` as "the score." Downstream consumes `public_by_model` / `replay_seconds` / firing per model. `.public` (mean) may remain for telemetry only.

### 4.4 Shipped artifact: carry both champions, select per-env by measured latency

`assemble.build` currently emits one attack from one champion's messages. Change to emit **both champions' shapes** into the discovered template pool (deduped if identical). Then fix grade-time selection so the probe picks correctly *per env*:

- **Rank the probe by measured latency, not blind gen_chars.** Today `_effective_cost` ranks on `gen_chars_lists` (SDK analysis-blind), which is proven wrong on hardware (it cannot see reasoning: `inj_analysis_empty` gen=31 beats `call_reply_ok` gen=2 despite being 2.4× faster; for gemma the gen=1 injection wrongly beats the faster gen=2 `bare_ok`). Rank by `median(success_latencies[i])` — which the artifact **already tracks and already uses for fill sizing** — gated by fire-rate, with gen_chars as a deterministic tiebreaker only.
- **Noise handling:** modestly raise `_PROBE_REPS` (e.g. 2 → 4) so the per-template median is stable. This is safe: on gpt_oss the injection's 2.4× lead is far above probe noise; on gemma the fast cluster (~16 s) all beats the injection (~17 s), so any pick in that cluster is an improvement.
- **No model fingerprinting.** Selection is purely empirical latency on the live env, so it auto-specializes: the gpt_oss env ranks the injection first, the gemma env ranks its plain template first. (`_GPT_TEMPLATE_ORDER`/`_GEMMA_TEMPLATE_ORDER`/`_model_hint` remain, but only as a probe pre-ordering; correctness no longer depends on them. They may be deleted in a follow-up.)

### 4.5 Proposer: surface both incumbents

The proposer prompt/incumbent block (`submission_prompt` / `_render_incumbent`) shows one global champion today. Show **both per-model incumbents** with their per-model objectives so the proposer can specialize (e.g. "gpt_oss best = injection @ X; gemma best = bare_ok @ Y"). Every candidate is still scored on both models (the scorer already replays both), so both champions update from the same batch — **no lane→model assignment needed** for the minimal version. The refine loop keeps a candidate if it advances **either** model's column (Pareto-style), rather than only if the mean improves.

---

## 5. Component change list (grounded in the maps)

| Area | File | Change |
|---|---|---|
| Per-model objective helpers | `optimize_prompts.py` | `_replay_s_cost`, `_gen_chars_cost`, `_robust_public`, `_score_public_raw_per_replay_s`, `_batch_refine_objective` → take a `model` arg / return per-model |
| Record construction | `optimize_prompts.py` `make_record` | populate `objective_by_model`, `replay_seconds_by_model`; set per-model scheme tag |
| Refine accept | `optimize_prompts.py` `_refine_batch` | accept if any model's column improves |
| Champions | `blackboard.py` | `best_objective(model)`, per-model `_objective_key`; `append` reships if *either* champion changed |
| Stop the mean | `submission_score.py:467` | keep `public_by_model` as the unit; `.public` mean → telemetry only; per-model validity/firing |
| Two-champion ship | `assemble.py` `build` | emit both champions' discovered shapes; rank probe by `success_latencies` median not `gen_chars` |
| Probe stability | `assemble.py` `_TEMPLATE` | `_PROBE_REPS` 2→4; `_effective_cost` ranking input → latency |
| Prompt | `prompts.toml`, `_render_incumbent` | show both per-model incumbents |
| Telemetry | `optimize_prompts.py` wandb | `best_objective_gpt_oss`, `best_objective_gemma_4`, per-model champion tags |
| Config | `config.py` | scheme name constant; optionally per-model build dirs if two artifacts are ever needed (not required — one artifact carries both) |

**Explicitly out of scope / do not build:** two submittable files; model-name branching; the `ROBUSTNESS_LAMBDA` min-blend (its cross-model-min semantics are the *opposite* of per-model specialization — leave at 0.0 and do not wire it into the per-model path).

---

## 6. Risks & mitigations

- **Hot-path regression (produces our 87.48).** The gpt_oss champion stays the injection, which is what currently wins; per-model only *adds* the gemma-specialized pick + carries both shapes. Mitigation: the artifact still ships the injection for the gpt_oss env; the change is additive. Gate behind a scheme bump so a bad per-model champion cannot silently replace the working one, and re-score via the both-models artifact score before any Kaggle submission.
- **Latency-ranking noise at grade time.** Mitigated by `_PROBE_REPS` bump + median + fire-rate gate; and the within-model latency gaps that matter (gpt_oss 13 vs 32; gemma's ~16 cluster vs 17 injection) are resolvable at 4 reps.
- **Probe latency ≠ scoring-replay latency** (competitor-flagged). The gpt_oss injection lead is so large it survives this; the gemma margin is small — if a both-models artifact score shows no gemma gain, ship gpt_oss-only selection and drop the gemma specialization. Fail-safe, not fail-open.
- **Blackboard migration.** Old rows lack `objective_by_model`; the scheme-tag gate quarantines them (same pattern as prior scheme bumps). No destructive migration.

---

## 7. Testing

- Update `test_robust_public_blends_mean_and_worst_model`, `test_robustness_lambda_stamps_distinct_objective_scheme`, and the objective/scheme-name assertions for the new per-model scheme.
- Update `test_assembled_attack_*` selection tests: ranking now by latency, not gen_chars; assert the gpt_oss-fast probe wins on a gpt_oss-like env and a plain template wins on a gemma-like env (extend the existing `test_assembled_attack_orders_probe_templates_by_model_hint`, but drive it by *probe latency*, not the dead model-hint).
- New: `best_objective(model)` returns different records when the columns disagree; `append` reships when either champion changes; per-model firing (a record championing gpt_oss while not firing on gemma).
- `test_score_submission_*`: assert `.public` mean no longer gates per-model validity.
- Blackboard scheme constant assertion → new per-model scheme name.

---

## 8. Rollout

1. Land the per-model objective + two-champion blackboard behind the new scheme tag (search keeps running; champions repopulate on the new scheme).
2. Land the artifact latency-ranking + two-champion ship.
3. Run the both-models artifact score (`scripts/score_artifact.py` / a both-models calib push) → confirm gpt_oss row holds and gemma row improves.
4. Only then submit (spends a slot; requires explicit human direction).

---

## 9. Open decisions for review

1. **Refine-accept semantics:** Pareto ("keep if any column improves") vs a scalarization. Recommend Pareto for the minimal version.
2. **Grade-time selection:** pure latency-probe (recommended, no fingerprinting) vs explicit model fingerprinting from probe behavior. Recommend latency-probe; revisit only if the empirical pick proves unstable.
3. **Legacy `Record.objective`:** keep populated for tooling vs remove. Recommend keep (cheap, avoids breaking telemetry).
4. **Lane→model targeting:** not needed for the minimal version (both champions update from every batch). Add only if per-model *proposer prompting* proves insufficient to specialize.
