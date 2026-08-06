# Portfolio-as-Unit Optimization — Design Spec

**Date:** 2026-08-06 (rev 2)
**Status:** Draft for review
**Goal:** Make the **batch the optimized unit** — the proposer already emits a batch of N submissions, so score/refine/ship that batch *as one portfolio*, optimized for the real 4-column LB (`gpt_oss/gemma × public/private`) rather than public throughput of one shape.

---

## 1. What already exists (and the actual gap)

The machinery is mostly here already:

- **The proposer emits a structured batch of N submissions** (`{"submissions": [...]}`).
- **A batch-level objective already exists** — `_batch_refine_objective` pools raw/gen-char across the batch, and the refine loop hill-climbs the *whole batch* as a unit.

So "score the submission as a unit" is **not** new plumbing. The gaps are narrow:

1. **The batch is dissolved, not shipped.** After scoring, each of the N is appended as an **independent Record** and the champion is the *single best shape*. The batch stops being a unit the moment it's scored.
2. **The objective is public-only.** `_batch_refine_objective` optimizes pooled public throughput. There is **no private dimension** anywhere — no `SECRET_MARKER`-payload survival test, nothing for the board that sets final ranking.
3. **The shipped submission is a *different* set.** Today it's the top-K distinct shapes pulled from the *whole blackboard* (a post-hoc heuristic), then live-filled into ~2000 candidates. It is not the batch the objective refined.

The fix: keep the **batch as the unit end-to-end** — refine it, score it on *both* guardrails, keep the best *batch* as the champion portfolio, and ship exactly that. This reuses the existing batch scoring at **no extra search cost**; the only new work is a private-proxy guardrail (shared by any design) and extending the batch objective to include private coverage.

---

## 2. The two objectives a portfolio must trade

- **Public throughput** (have it): each firing candidate scores the same ~18 raw, so the row is `~0.09 × candidates`, maximized by the *leanest* shapes. Shape diversity is *neutral* here. `_batch_refine_objective` already proxies this (pooled raw / pooled gen-char, per model).
- **Private survival** (missing): the private board is a **static replay of the returned candidates against a held-out, undisclosed guardrail**. A monoculture survives-or-dies as a unit; a diverse batch scores whatever fraction of its distinct shapes the held-out defense fails to block. **Diversity is only valuable here.**

The batch objective must trade these: spend a little public throughput (slightly-less-lean shapes) to raise expected private coverage. That trade is unmeasurable without a private score — hence the prerequisite.

---

## 3. Prerequisite: a private-proxy guardrail (shared by every design)

The real private guardrail is held out. Add a **proxy** to `GATE_GUARDRAILS` alongside `optimal` and replay each shape against it:

- **Candidates** (the competition says "stricter"; the forum speculates an LLM-judge):
  1. `RulesGuardrail` baseline (`aicomp_sdk.attacks.baselines.guardrail_rules`) — inspects the `http.post` `data` payload and denies `secret_exfil`.
  2. An LLM-judge-as-guardrail (reuse the existing judge service for a verdict).
- **What its first measurement decides (load-bearing):** if the proxy blocks by inspecting the `SECRET_MARKER` payload, then **every exfil shape scores 0 on private regardless of framing** — framing diversity is worthless and the private lever is **predicate** diversity (the benign `email.send` **deputy**, no secret; `lb-lever-plan` lever 2). If it blocks by framing/heuristics, framing diversity is the lever. **This single result tells us what the batch should diversify over.**

Cost: `score_submission` already loops guardrails, so this is a config addition (a second guardrail = 2× replay per shape) — the same added cost for *any* design.

---

## 4. Design: the batch IS the portfolio (Option A)

Make the batch the unit through the whole pipeline:

1. **Author** — the proposer already emits N shapes; the diverse prompt already makes them structurally distinct. Unchanged.
2. **Score** — each shape is scored on **both** guardrails (public + private-proxy), as now, per model. Unchanged plumbing, one added guardrail.
3. **Refine** — the batch hill-climb (`_refine_batch`) already re-authors and keeps the better *batch*; swap its accept criterion to the **portfolio objective** (§5). Reuses the existing loop.
4. **Champion = the best-scoring BATCH.** Rank whole *batches* by the portfolio objective (§5) and keep the single best batch as the champion — exactly analogous to today's `best_objective`, but the unit is a batch (a set of shapes), not one shape. No cross-batch accumulation, no curation of a top-K from the whole blackboard.
5. **Ship the champion batch, verbatim.** `assemble.build` already accepts a list of shapes and the fill already round-robins across them (the diverse-fill work just landed). Feed it the champion batch's shapes directly. What was scored is what ships.

**Why this is the better design (and free):** the batch already IS a portfolio; we're just (a) scoring it on private too, (b) ranking whole batches by a portfolio objective, and (c) shipping the winning batch instead of dissolving it into per-shape records and re-curating. There's **no extra search cost** — we already score N shapes per batch — and it exactly matches "what you optimize is what ships." The post-hoc top-K curation (`top_distinct_shapes`, `SHIP_TOP_K`) becomes unnecessary and is removed: the shipped set is the champion batch.

(The earlier draft's "Option B — curate a top-K from the whole blackboard" is now redundant: it existed only because the batch was being dissolved. Keeping the batch whole is strictly cleaner.)

---

## 5. The portfolio (batch) objective

```
score(batch) = public_throughput(batch) + λ · private_coverage(batch)
```

- **`public_throughput(batch)`**: the fill round-robins across the batch's firing shapes, so throughput ≈ `budget / mean_shape(cost)` per model, meaned over models — essentially the current `_batch_refine_objective` (pooled per-model raw/gen-char). Extend it to consume the batch's shapes as the fill set.
- **`private_coverage(batch)`**: number of **distinct** shapes in the batch that still fire the scored predicate under the private proxy (submodular — a shape that survives where others don't is worth more than a redundant one). If the proxy is payload-based (exfil all die), coverage is carried by the **deputy** shape, so the objective naturally pulls a deputy into the batch instead of the fixed `_DEPUTY_FRAC`.
- **`λ`**: the public↔private weight. `λ=0` is today's public-only batch objective (a clean A/B and safe default); raise it as private-proxy fidelity is validated.

---

## 6. Components / touch points

| Area | File | Change |
|---|---|---|
| Private proxy | `guardrails.py` | add a `private` entry to `GATE_GUARDRAILS` (RulesGuardrail or judge-as-guardrail) |
| Per-shape private score | `submission_score.py` | already loops guardrails; surface per-guardrail per-model board + firing |
| Batch objective | `optimize_prompts.py` `_batch_refine_objective` | add `λ · private_coverage`; consume the batch's shapes as the fill set for the public term |
| Refine accept | `optimize_prompts.py` `_refine_batch` | accept on the portfolio objective (already batch-level) |
| Champion = best batch | `blackboard.py` | persist scored **batches** (a set of shapes + the batch's portfolio objective) and select the best batch; the per-shape record path stays for feedback/telemetry |
| Ship path | `blackboard.py` `append`/`reship_best` | ship the champion batch's shapes; **remove** `top_distinct_shapes`/`SHIP_TOP_K` curation |
| Scheme tag | `blackboard.py` | bump when `λ>0` changes the ship objective |
| Config | `config.py` | `PORTFOLIO_LAMBDA`, private-proxy id (the proposer's batch size is the portfolio size) |
| Fill | `assemble.py` | unchanged (already round-robins the shipped set) |

---

## 7. Risks & mitigations

- **Private-proxy ≠ real private guardrail.** It's a guess. Keep `λ` low until a real private submission validates its direction; never let private optimization regress public (floor the public term). The load-bearing signal is "does exfil survive at all?", not exact scores.
- **Best-batch coarseness.** The champion is one batch, so its quality is bounded by what the proposer put in that single batch (vs. cherry-picking across all batches). Mitigation: show the champion batch as the incumbent so the proposer *improves on it* each generation, and let the refine loop hill-climb within a batch — the best batch converges to a strong portfolio without cross-batch curation.
- **Scoring cost (2nd guardrail).** 2× replay per shape — the same for any design; cache per-shape private verdicts on the Record.
- **The exfil-payload trap.** If the proxy shows all exfil dies on private, a framing-diverse exfil-only batch does NOT hedge — the objective must then weight the deputy predicate, which the `private_coverage` term does automatically.

---

## 8. Testing

- `_batch_refine_objective` with `λ>0` rewards a batch that adds a distinct private-surviving shape over one that adds a redundant public-lean shape; `λ=0` equals today's public-only batch objective.
- Private-proxy guardrail: an exfil shape carrying `SECRET_MARKER` scores 0 under a payload-inspecting proxy; a deputy shape (no secret) survives.
- Champion is a *set* of shapes; `append`/`reship_best` ship that set; round-trips through the blackboard.
- Scheme-tag bump when `λ>0`; champion pool resets.

---

## 9. Rollout

1. Land the private-proxy guardrail at `λ=0` (pure telemetry). **Its first result decides framing-vs-predicate diversity.**
2. Track the best *batch* + ship it (replacing `top_distinct_shapes`), still `λ=0` — behavior-equivalent to today, now batch-as-unit.
3. Raise `λ` only after step 1 shows private survival is real and which predicate carries it.
4. Validate on Kaggle before trusting the private direction.

---

## 10. Open decisions for review

1. **Proxy choice:** `RulesGuardrail` (deterministic, cheap, directly tests the payload hypothesis) vs LLM-judge-as-guardrail. Recommend `RulesGuardrail` first.
2. **`λ` schedule:** fixed small constant vs adaptive. Recommend fixed-small first.
3. **Batch/champion granularity:** RESOLVED — ship the best-scoring *batch* as the portfolio (analogous to `best_objective`, unit = batch). The champion batch is shown as the incumbent so the proposer improves on it each generation; no cross-batch accumulation or curation.
4. **Deputy as a measured batch slot** (via `private_coverage`) vs the fixed `_DEPUTY_FRAC`. Recommend folding it into the objective.
