# Batch-as-Unit Portfolio Optimization — Design Spec

**Date:** 2026-08-06 (rev 4)
**Status:** Draft for review
**Goal:** Make the **batch the optimized and shipped unit** — the proposer already emits a batch of N submissions, so score/refine/ship that batch *as one portfolio*, optimizing **public throughput + shape diversity**. Diversity is measured **structurally** (distinct shapes) as a blind private hedge; the private-*proxy guardrail* (scoring actual survival) is left out for now (§8).

---

## 1. What already exists (and the gap)

The machinery is mostly here:

- **The proposer emits a structured batch of N submissions** (`{"submissions": [...]}`).
- **A batch-level objective already exists** — `_batch_refine_objective` pools raw/gen-char across the batch, and the refine loop hill-climbs the *whole batch* as a unit.

So making the batch the unit is **not** new plumbing. The gaps are narrow:

1. **The batch is dissolved, not shipped.** After scoring, each of the N is appended as an **independent Record** and the champion is the *single best shape*. The batch stops being a unit the moment it's scored, and its diversity is thrown away.
2. **The shipped submission is a *different* set.** Today it's a post-hoc top-K from the *whole blackboard* (`top_distinct_shapes`/`SHIP_TOP_K`), live-filled into ~2000 candidates. What ships is not the batch the objective refined.

The fix: keep the **batch whole end-to-end** — score it as a portfolio (public + diversity), keep the best *batch* as the champion, and ship exactly that batch.

---

## 2. Why diversity, without a private guardrail

The private board is a static replay against a **held-out, undisclosed** guardrail. A monoculture (one shape × many URLs) survives-or-dies as a unit; a diverse portfolio scores whatever fraction the defense fails to block. So **diversity is an intrinsic hedge against an unknown defense** — valuable *a priori*, without modeling the guardrail.

That means we don't need a private-proxy guardrail to *value* diversity — we can value it **structurally**: reward a batch for carrying more **distinct** firing shapes. And it's nearly free on public: each firing candidate scores the same ~18 raw regardless of shape, so a diverse-lean batch and a lean monoculture have the *same* public throughput. Diversity costs public only to the extent diverse framings are slightly less lean.

The private-proxy guardrail (§8) is a *future refinement* of the diversity measure — from "count distinct shapes" to "count shapes that actually survive the held-out defense." Same design; better diversity signal. Deferred.

---

## 3. Design: the batch IS the portfolio

1. **Author** — the proposer already emits N structurally-distinct shapes (diverse prompt). Unchanged.
2. **Score** — each shape scored per model on the public guardrail, as now. Unchanged.
3. **Refine** — the batch hill-climb (`_refine_batch`) already keeps the better *batch*; it stays, on the portfolio objective (§4) — so it can't erode the batch to a monoculture (diversity is in the objective).
4. **Champion = the best-scoring BATCH.** Rank whole *batches* by the objective and keep the single best — analogous to today's `best_objective`, unit = batch. No cross-batch accumulation or curation.
5. **Ship the champion batch, verbatim.** `assemble.build` already accepts a list of shapes and the fill already round-robins across them. Feed it the champion batch's shapes. What was scored is what ships; `top_distinct_shapes`/`SHIP_TOP_K` are removed.

---

## 4. The portfolio (batch) objective

```
score(batch) = public_throughput(batch) + λ · diversity(batch)
```

- **`public_throughput(batch)`**: the fill round-robins across the batch's firing shapes, so throughput ≈ `budget / mean_shape(cost)` per model, meaned over models — essentially today's `_batch_refine_objective`, but consuming the batch's shapes as the fill set so scored throughput == shipped throughput.
- **`diversity(batch)`**: the count of **distinct** firing shapes in the batch (deduped by templatized form) — a structural, guardrail-free proxy for how well the portfolio hedges an unknown private defense. (Optionally normalized to [0,1] against the batch size.)
- **`λ`**: the public↔diversity weight. Small — public is the known board and diversity is nearly free, so `λ` only needs to break ties toward more distinct shapes and stop refinement collapsing to a monoculture. `λ=0` reproduces pure-public (and would trend monoculture — the reason `λ>0` matters).

---

## 5. Components / touch points

| Area | File | Change |
|---|---|---|
| Batch objective | `optimize_prompts.py` `_batch_refine_objective` | `public_throughput + λ·diversity`; consume the batch's shapes as the fill set (scored == shipped) |
| Refine accept | `optimize_prompts.py` `_refine_batch` | unchanged mechanism; now on the portfolio objective (keeps diversity) |
| Champion = best batch | `blackboard.py` | persist scored **batches** (shapes + batch objective), select the best batch; per-shape records stay for feedback/telemetry |
| Ship path | `blackboard.py` `append`/`reship_best` | ship the champion batch's shapes; **remove** `top_distinct_shapes`/`SHIP_TOP_K` |
| Incumbent | `optimize_prompts.py` prompt block | show the champion *batch* as the incumbent so the proposer improves on it |
| Config | `config.py` | `PORTFOLIO_LAMBDA` (the public↔diversity weight) |
| Fill | `assemble.py` | unchanged (already round-robins the shipped set) |

---

## 6. Risks & mitigations

- **`λ` too high starves public.** Keep it small (diversity is nearly free, so it only needs to tip ties and prevent monoculture collapse). Floor the public term so a batch can never drop public below the lean-monoculture minus a small band.
- **Best-batch coarseness.** The champion is one batch, bounded by what the proposer put in it. Mitigation: show it as the incumbent so the proposer improves on it; the refine loop hill-climbs within a batch.
- **Structural diversity ≠ true survival.** Counting distinct shapes assumes distinct framings hedge the unknown guardrail; if the real private defense is payload-based, framing diversity won't help and the deputy predicate is the lever. That's exactly what the deferred private-proxy (§8) would reveal — until then, structural diversity is a reasonable blind hedge.

---

## 7. Testing

- `_batch_refine_objective`: a batch with more distinct firing shapes outscores an equal-throughput monoculture at `λ>0`; `λ=0` is pure public throughput.
- Champion is a *set* of shapes; `append`/`reship_best` ship that set; round-trips through the blackboard.
- Removing `top_distinct_shapes`/`SHIP_TOP_K` doesn't break the ship path; the shipped batch's distinct-shape count matches the champion batch.

---

## 8. Deferred: private-proxy guardrail (future refinement)

Same design; upgrades the diversity signal. When ready: add a `private` guardrail to `GATE_GUARDRAILS` (`RulesGuardrail` baseline, or judge-as-guardrail) and replace `diversity = distinct-shape count` with `diversity = distinct shapes that SURVIVE the proxy`. Its first measurement decides whether **framing** or **predicate (deputy)** diversity is the real private lever (if the proxy inspects the `SECRET_MARKER` payload, all exfil dies on private and the deputy is the only private scorer). No structural change to the batch-as-unit design — just a better `diversity` term.

---

## 9. Open decisions for review

1. **`λ` magnitude / normalization:** small fixed constant on a [0,1]-normalized distinct-shape count. Recommend fixed-small.
2. **Batch/champion granularity:** ship the best-scoring *batch* (unit = batch), shown as incumbent so the proposer improves on it. (Recommended, matches the "optimize what ships" intent.)
