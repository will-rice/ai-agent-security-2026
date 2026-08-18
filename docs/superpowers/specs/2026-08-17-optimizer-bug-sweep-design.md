# Optimizer Bug Sweep — Design Spec

**Date:** 2026-08-17 · **Branch:** `replay-speed-investigation` · **Status:** for review

## 1. Problem

A 4-agent audit of the two-pool per-model optimizer found a cluster of bugs that all
**starve the gpt_oss column**. The observed symptom: over a ~12h run the champion froze at
`objective≈61` while the per-batch objective drifted **45→25** (codex 200-OK throughout).
Two were already fixed this session (router probe misdetection `3480b7b`; frontier ship
crowding `2aa45b5`). This spec covers the rest.

## 2. Root-cause model (the shared understanding)

Every archive elite is a **single-model specialist** — `score_pools` scores each pool only
on its own victim, so a gpt shape's gemma axes are 0 and vice-versa. Consequences we rely on:

- **The Pareto archive already protects the good gpt shape.** A gpt (forge) shape's 4-D
  vector is `{gpt_tput:T, gemma_tput:0, gpt_sev:16, gemma_sev:0}`. A gemma shape (gpt_tput:0)
  can never dominate it — orthogonal axes. Only a **leaner gpt shape (fewer `gen_tokens` →
  higher throughput)** dominates it. So forge falls off the frontier *only* when a genuinely
  better gpt shape appears — never to noise, never to gemma. **The objective can't discard
  the forge without eating the output regression itself.**
- **Therefore the drift is NOT the objective discarding forge.** The archive kept 269 gpt
  elites the whole run. The drift is that the **proposer never *saw* them**: the exemplar
  table (`_render_opro_table`) and `parents()` rank the frontier by *summed* board-density and
  truncate globally; gemma-plain shapes are uniformly denser than gpt-forge, so gpt exemplars
  and parents vanish. Starved of gpt exemplars — and told by the prompt to coast ("a weak
  column only costs that pool's half of the mean") — the proposer authors plain-gpt shapes.
  Those fire but gpt then reasons → gpt board ≈ 0. Under **MEAN** (blind to a dead column) the
  batch scores gemma/2 ≈ 25, never beats the frozen balanced champion → the reported symptom.

## 3. Key realization: MEAN already rewards balance; the fix is proposer *visibility*

MEAN is **not** blind to the collapse in the way that matters: `mean(gpt 40, gemma 50) = 45`
strictly beats `mean(gpt 0, gemma 50) = 25`. So MEAN *rewards* getting both columns high, and
the champion correctly froze on the early balanced-45 build. What declined was the proposer's
*new batches* — because the exemplar signal it learns from (`_render_opro_table`, `parents()`)
is gemma-skewed (global summed-density truncation, gemma denser), so the proposer never *saw*
that a balanced submission beats a gemma-only one, and drifted toward gemma-only (mean 25).

**Therefore the entire drift fix is: make the proposer see the gpt shapes.** No objective
change, no forge gate, no schema surgery — MEAN does the rest once the proposer can see that
balance wins.

## 4. Scope — minimal fix now, everything else deferred

**DO NOW — make the proposer's WHOLE context serve both-model optimization.** Of the five
context blocks it sees (optimize_prompts.py:1221-1225), `{{INCUMBENT}}` is already per-model
(both pools + per-pool feedback — leave it). The other three "what's winning across the
search" blocks show a gemma-dominated world; fix all three + the prompt:
- **B1. `{{OPRO}}` per-model.** `_render_opro_table` sorts the frontier by *summed*
  board-density, top-`OPRO_TABLE_ROWS` → 100% gemma. Rank each model's firing elites by *that
  model's* density and interleave, so gpt shapes appear in the scored landscape.
- **B2. `{{PARENTS}}` per-model + explore.** `parents(k)` returns `front[:k]` (fixed prefix,
  gemma-heavy). Return both models' shapes (interleave per-model) and sample under-filled cells
  (wire the dead fallback) so gpt variants get *explored*, not only gemma.
- **B3. `{{TEAM}}` best-messages per-victim-model.** `top_messages(mtype, k)` is a global
  top-k tagged by *proposer lane*, not victim — victim-blind. Tag each feedback entry with its
  victim model (`gpt_oss`/`gemma_4`) in `make_record`, and render top messages balanced across
  victims so the proposer sees the best gpt-shaped AND gemma-shaped messages.
- **Prompt de-coasting.** Delete the lines that license a dead column (prompts.toml "a weak
  column only costs half"; incumbent block "never penalized for the other pool's weakness") —
  they directly tell the proposer *not* to optimize both. (This also stops the reasoning digest
  from recirculating coasting.)

Shared helper: a per-model `board_density` for an elite (`archive.model_density`) used by B1,
B2, and the already-shipped `_frontier_map`. Rank purely on that density (no `input_chars`).

That's it for the drift. Validate by watching the per-model boards after restart (§5).

**DEFERRED BACKLOG (real, audited bugs — do only if the boards show they matter):** these are
NOT needed to fix the drift and add scope/risk we don't need now.
- A3 token-based submission objective (mild gemma bias; within-model ranking unaffected).
- C1 never-ship-an-empty-live-pool guard; C2 probe `max(ps)` + more reps (router robustness).
- Schema: `url_scheme` → declarative `Field(pattern=...)` (fixes the `{h}`-in-path novelty
  hole, puts the rule in the schema); drop redundant `{{SCHEMA}}` from the prompt; fix the
  `Message.type` exfil description; bound `diagnoses`.
- Delete dead `Archive.ship_set()` + retire `champion_by_board_density` (loaded guns).
- Drop `input_chars` as a ranking tiebreak (prefill mirage).
- Minors: diagnosis misattribution → `""`; per-model `_APPROX_CHARS_PER_TOKEN`; `throughput`
  non-firing guard; `score_pools` `config.MODELS` order.

**KEEP `FIXED`** (measured T4 per-candidate floor; within-model ranking is unaffected by it,
and the per-model difference reflects a real cost). Revisit by re-measuring only if we distrust
the value — do not drop.

## 5. Success criteria

- **Correctness:** each fix has a regression test that fails on the pre-fix code. Full suite +
  `pre-commit` green (optimizer stopped, so the 2 GPU-gated scheme tests pass).
- **Behavioral (post-restart, the real test of the design):** with the optimizer restarted,
  the **per-model boards `board_gpt_oss` and `board_gemma_4` both stay > 0 and roughly track
  each other** — gpt no longer collapses — and the batch objective does not monotonically
  drift down. gpt exemplars appear in the OPRO table. This is the signal that MEAN-with-fixes
  holds; if gpt still lags, escalate to the mean/min blend (§3 deferred).

## 6. Non-goals / risks

- **Non-goals:** switching the objective to min/blend now; any forge gate or shape-family
  constraint; changing the LB metric; the char↔token telemetry beyond the objective path.
- **Risk:** under pure MEAN, even with correct exemplars + token costing, the proposer *could*
  still under-invest in the harder gpt column if gemma's ceiling is higher. Mitigation: watch
  the per-model boards; the blend is one wired constant away.
- **Risk:** dropping `{{SCHEMA}}` from the prompt assumes the strict schema's descriptions
  reach the model. Mitigation: it's reversible; if proposer quality drops, re-add a concise
  prose rule summary (not the full JSON dump).
