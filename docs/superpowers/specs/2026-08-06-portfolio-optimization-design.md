# Batch-as-Unit Optimization (Public First) — Design Spec

**Date:** 2026-08-06 (rev 3)
**Status:** Draft for review
**Goal:** Make the **batch the optimized and shipped unit** — the proposer already emits a batch of N submissions, so score/refine/ship that batch *as one submission*, with an objective faithful to the actual grade-time fill. **Public board only**; the private board is deferred to a later phase (§8).

---

## 1. What already exists (and the gap)

The machinery is mostly here:

- **The proposer emits a structured batch of N submissions** (`{"submissions": [...]}`).
- **A batch-level objective already exists** — `_batch_refine_objective` pools raw/gen-char across the batch, and the refine loop hill-climbs the *whole batch* as a unit.

So making the batch the unit is **not** new plumbing. The gaps are narrow:

1. **The batch is dissolved, not shipped.** After scoring, each of the N is appended as an **independent Record** and the champion is the *single best shape*. The batch stops being a unit the moment it's scored.
2. **The shipped submission is a *different* set.** Today it's the top-K distinct shapes pulled from the *whole blackboard* (a post-hoc heuristic, `top_distinct_shapes`/`SHIP_TOP_K`), then live-filled into ~2000 candidates. What ships is not the batch the objective refined.

The fix: keep the **batch whole end-to-end** — refine it, keep the best *batch* as the champion, and ship exactly that batch. What's optimized is what ships.

---

## 2. Honest scope: what this does and does not buy

Because this phase is **public-only**, be clear-eyed about the value:

- **Public is diversity-neutral.** Each firing candidate scores the same ~18 raw regardless of shape, so a batch optimized for public alone converges toward the **leanest shapes**. This design will **not** produce or preserve diversity — that's the private board's job (deferred).
- **The public gain is modest.** The per-shape objective we already run (`mean_m(public_m / gen_chars_m)`) is already a good proxy for the fill's public throughput. Making the batch the unit mainly buys **faithfulness** (the scored throughput is the shipped throughput) and a **clean architecture** (optimize-what-ships) — the foundation the private phase plugs into — not a large score jump.
- **Interim private hedge stays.** The top-K diverse fill just shipped remains as a cheap private hedge until the private phase lands; this refactor would replace it (public trends lean), so sequencing matters — don't land this until the private phase is ready to restore diversity, OR accept the temporary loss of the private hedge.

If the goal is simply "raise the public score now," the higher-leverage work is the **search + prompt tuning** already in flight, not this refactor. This spec is the architectural groundwork that makes the private phase clean.

---

## 3. Design: the batch IS the submission

1. **Author** — the proposer already emits N shapes. Unchanged.
2. **Score** — each shape scored per model on the public guardrail, as now. Unchanged.
3. **Refine** — the batch hill-climb (`_refine_batch`) already keeps the better *batch*; it stays, on the faithful public throughput objective (§4).
4. **Champion = the best-scoring BATCH.** Rank whole *batches* by the objective and keep the single best — analogous to today's `best_objective`, but the unit is a batch (a set of shapes). No cross-batch accumulation or curation.
5. **Ship the champion batch, verbatim.** `assemble.build` already accepts a list of shapes and the fill already round-robins across them. Feed it the champion batch's shapes. What was scored is what ships; `top_distinct_shapes`/`SHIP_TOP_K` are removed.

---

## 4. The batch objective (public)

```
score(batch) = public_throughput(batch)
```

The fill round-robins across the batch's firing shapes, so throughput ≈ `budget / mean_shape(cost)` per model, meaned over models — essentially today's `_batch_refine_objective`, but consuming the batch's shapes as the fill set so the scored throughput matches the shipped one. Mean public stays a tie-breaker. No private term, no `λ`.

---

## 5. Components / touch points

| Area | File | Change |
|---|---|---|
| Batch objective | `optimize_prompts.py` `_batch_refine_objective` | keep public throughput; consume the batch's shapes as the fill set so scored == shipped |
| Refine accept | `optimize_prompts.py` `_refine_batch` | unchanged (already batch-level) |
| Champion = best batch | `blackboard.py` | persist scored **batches** (shapes + batch objective), select the best batch; the per-shape record path stays for feedback/telemetry |
| Ship path | `blackboard.py` `append`/`reship_best` | ship the champion batch's shapes; **remove** `top_distinct_shapes`/`SHIP_TOP_K` |
| Incumbent | `optimize_prompts.py` prompt block | show the champion *batch* as the incumbent so the proposer improves on it |
| Fill | `assemble.py` | unchanged (already round-robins the shipped set) |

---

## 6. Risks & mitigations

- **Loses the interim diversity hedge.** Public-only ships lean → the top-K diverse fill is superseded. Mitigation: land this together with (or after) the private phase, or knowingly accept the temporary loss.
- **Best-batch coarseness.** The champion is one batch; its quality is bounded by that batch. Mitigation: show it as the incumbent so the proposer improves on it; the refine loop hill-climbs within a batch.
- **Small payoff.** If the faithfulness gain proves negligible in a re-score, this is pure architecture — fine as private-phase groundwork, but don't expect a public jump.

---

## 7. Testing

- `_batch_refine_objective` scores a batch by its round-robin fill throughput; scored throughput matches what the shipped fill produces on the same shapes.
- Champion is a *set* of shapes; `append`/`reship_best` ship that set; round-trips through the blackboard.
- Removing `top_distinct_shapes`/`SHIP_TOP_K` does not break the ship path.

---

## 8. Deferred: the private board (future phase)

Out of scope here, but the reason batch-as-unit matters. When ready:
- Add a **private-proxy guardrail** to `GATE_GUARDRAILS` (`RulesGuardrail` baseline, or the judge-as-guardrail). Its first measurement decides whether **framing** diversity or **predicate** (deputy) diversity is the private lever — if the proxy inspects the `SECRET_MARKER` payload, all exfil dies on private and the deputy is the only private scorer.
- Extend the batch objective to `public_throughput + λ · private_coverage` (distinct shapes surviving the proxy; submodular). `λ=0` == this public-only design.
- This is where the batch-as-unit pays off: diversity becomes valuable, and the batch is optimized for both columns at once.

---

## 9. Open decisions for review

1. **Is the public-only faithfulness win worth building now**, or should the batch-as-unit wait and land *with* the private phase (so it doesn't temporarily drop the diversity hedge)? Recommend waiting for private, unless a quick re-score shows a real public gain from faithful batch scoring.
2. **Batch/champion granularity:** ship the best-scoring *batch* (recommended, unit = batch); shown as incumbent so the proposer improves on it.
