# Portfolio-as-Unit Optimization — Design Spec

**Date:** 2026-08-06
**Status:** Draft for review
**Goal:** Optimize the whole **submission** (a portfolio of shapes) as the scored unit — so what the optimizer maximizes is what actually ships — and make the **private board** a first-class objective, not a heuristic afterthought.

---

## 1. The gap this closes

Today the optimizer and the artifact are **two disconnected layers** (confirmed against the code):

- **Optimizer** scores individual *shapes* — each unit is a single-message candidate, ranked by `mean_m(public_m / gen_chars_m)` (public throughput only).
- **Artifact** (`attack.py`) is a live-fill *algorithm* that, at grade time, multiplies shapes into ~2000 candidates. It just shipped a diverse top-K set (`SHIP_TOP_K`, `_FILL_DIVERSITY_K`) as a heuristic private hedge — but that top-K is chosen by the *public* objective and a naive dedup, **not optimized for private survival**.

So two things are missing:
1. **The unit of optimization is a shape, not the submission.** The private-board value of a *portfolio* (diversity → survival against a held-out guardrail) is never scored, so the search can't trade public throughput for private robustness intelligently.
2. **The private board — which sets the final ranking — has no objective at all.** `GATE_GUARDRAILS` scores only the public `OptimalGuardrail`; the private column is invisible to the search.

This spec makes the **portfolio** the unit and adds a **private-proxy** objective, so the search optimizes the real 4-column LB (`gpt_oss/gemma × public/private`), not just public throughput of one shape.

---

## 2. The two objectives a portfolio must trade

- **Public throughput** (have it): each firing candidate scores the same ~18 raw, so the row is `~0.09 × candidates`, maximized by the *leanest* shapes. Shape diversity is *neutral* here. We already score this per shape.
- **Private survival** (missing): the private board is a **static replay of the returned candidates against a held-out, undisclosed guardrail**. A monoculture survives-or-dies as a unit; a diverse portfolio scores whatever fraction of its distinct shapes the held-out defense fails to block. **Diversity is only valuable here.**

The portfolio objective is therefore a trade: spend a little public throughput (slightly-less-lean shapes) to raise the expected private coverage. You cannot optimize that trade without *measuring* private survival — hence the prerequisite below.

---

## 3. Prerequisite: a private-proxy guardrail

The real private guardrail is held out. To optimize private we replay each shape against a **proxy** for it and score survival. Add it to `GATE_GUARDRAILS` alongside `optimal`:

- **Candidate proxies** (the competition says "stricter"; the forum speculates an LLM-judge):
  1. The SDK's `RulesGuardrail` baseline (`aicomp_sdk.attacks.baselines.guardrail_rules`) — inspects the `http.post` `data` payload and denies `secret_exfil`.
  2. An LLM-judge over the trace (we already run a judge service; repurpose it as a guardrail-style verdict).
- **What it reveals (critical):** if the proxy blocks by inspecting the `SECRET_MARKER` payload, then **every exfil shape scores 0 on private regardless of framing** — framing diversity is worthless and the private lever is **predicate** diversity (the benign `email.send` **deputy**, no secret; already flagged in `lb-lever-plan` lever 2). If the proxy blocks by framing/heuristics, framing diversity is the lever. **The proxy is what tells us which diversity matters** — so it must land first.

Scoring cost: adding a second guardrail doubles per-shape replay work. `score_submission` already loops guardrails, so this is a config addition, not new plumbing.

---

## 4. Architecture: curation, not full re-search (recommended)

Two ways to make the portfolio the unit:

**Option A — full portfolio scoring (the literal mental model).** The proposer emits a SET of K shapes; the scorer estimates the *whole submission's* public throughput + private coverage; the objective is the combined 4-column LB estimate; the champion portfolio IS the artifact.
- Pro: exactly optimizes what ships.
- Con: **K× scoring cost** per unit, a fill-throughput model to estimate candidate count, and a much larger proposer output. Heavy.

**Option B — keep the fast per-shape search, add a portfolio CURATOR (recommended).** The per-shape search stays as-is (cheap, fast, finds diverse lean shapes on public). A new **curation** step selects the shipped top-K to **maximize a portfolio objective over BOTH guardrails**, replacing the naive `top_distinct_shapes` dedup:
- `curate(k)` = pick the K shapes that maximize `public_throughput(portfolio)` + `λ · private_coverage(portfolio)`, where `private_coverage` = count of distinct shapes surviving the private proxy (a set-cover / greedy submodular pick, so redundant shapes aren't chosen).
- The artifact ships the curated set; the fill round-robins across it (already built).
- Pro: reuses the fast search; the *portfolio* is optimized (for both columns) even though *shapes* are searched. Curation is cheap (scores an existing pool, not a live search).
- Con: not a single monolithic unit — but it produces the same shipped artifact Option A would, at a fraction of the cost.

**Recommendation: Option B.** It closes the gap (the shipped portfolio is optimized, private included) without the K× search blow-up, and it degrades gracefully — with `λ=0` it's today's public-only top-K.

---

## 5. The portfolio objective (Option B)

```
score(portfolio) = public_throughput(portfolio) + λ · private_coverage(portfolio)
```

- **`public_throughput`**: the fill is round-robin across K shapes, so throughput ≈ `budget / mean_k(cost_shape)` per model, meaned over models (reuse the per-model rate machinery). The top-K are all lean, so this is ~flat vs the champion.
- **`private_coverage`**: replay each candidate shape under the private proxy; count DISTINCT shapes that still fire the scored predicate. Greedy submodular selection maximizes distinct survivors (adding a shape that survives where others don't beats a redundant one). If the proxy is payload-based (exfil all die), coverage is dominated by the **deputy** shape — so curation automatically reserves the deputy slice, tuned by measured survival instead of the fixed `_DEPUTY_FRAC=0.10`.
- **`λ`**: the public↔private weight. Start conservative (public dominates, since public is the known board), raise as private-proxy fidelity grows. `λ=0` reproduces today's behavior — a safe default and a clean A/B.

---

## 6. Components / touch points

| Area | File | Change |
|---|---|---|
| Private proxy | `guardrails.py` | add a `private` entry to `GATE_GUARDRAILS` (RulesGuardrail or judge-as-guardrail) |
| Per-shape private score | `submission_score.py` | already loops guardrails; expose per-guardrail per-model board + firing so curation can read private survival |
| Record persistence | `blackboard.py` `Record` | persist `private_by_model` (per-model private-proxy board) alongside `public_by_model` |
| Curator | `blackboard.py` | replace `top_distinct_shapes(k)` with `curate_portfolio(k, λ)` — greedy public+private submodular pick over the current-scheme firing pool |
| Ship path | `blackboard.py` `append`/`reship_best` | ship `curate_portfolio` output |
| Objective scale tag | `blackboard.py` | bump scheme when `λ>0` changes the ship objective (as with prior objective changes) |
| Config | `config.py` | `PORTFOLIO_LAMBDA`, private-proxy id |
| Fill | `assemble.py` | unchanged (already round-robins the shipped set) |

Explicitly **not** Option A: no proposer-emits-sets, no live portfolio search — the per-shape search is retained.

---

## 7. Risks & mitigations

- **Private-proxy ≠ real private guardrail.** The proxy is a guess. Mitigation: keep `λ` low until a real private submission validates the proxy's direction; never let private optimization *regress* public (cap the public cost of curation). Treat the proxy's biggest signal — "does exfil survive at all?" — as the load-bearing bit, not exact scores.
- **Curation over-fits to the proxy.** Greedy submodular coverage with a hard public-throughput floor; the portfolio must never drop public below the champion-monoculture minus a small band.
- **Scoring cost (2nd guardrail).** Doubles replay; acceptable (curation scores an existing pool, not a re-search). Cache per-shape private verdicts on the Record so curation is cheap.
- **The exfil-payload trap.** If the proxy shows all exfil dies on private, do NOT ship a framing-diverse exfil-only portfolio thinking it hedges — it doesn't. Curation must then weight the deputy predicate, which is the actual private scorer under a payload-inspecting guardrail.

---

## 8. Testing

- `curate_portfolio` picks distinct-surviving shapes (greedy coverage), respects the public floor, and with `λ=0` equals `top_distinct_shapes`.
- Private-proxy guardrail: an exfil shape carrying `SECRET_MARKER` scores 0 under a payload-inspecting proxy; a deputy shape (no secret) survives.
- `Record` round-trips `private_by_model`.
- Scheme-tag bump when `λ>0`; champion pool resets.
- `score_submission` returns per-guardrail per-model boards.

---

## 9. Rollout

1. Land the private-proxy guardrail (`λ=0`, pure telemetry) — see whether exfil survives it at all. **This single measurement decides framing-vs-predicate diversity.**
2. Persist `private_by_model`; add `curate_portfolio` behind `λ=0` (== current top-K).
3. Raise `λ` only after step 1 tells us private survival is real and which predicate carries it.
4. Validate on Kaggle before trusting the private direction (the proxy is a guess).

---

## 10. Open decisions for review

1. **Proxy choice:** `RulesGuardrail` baseline vs the LLM-judge-as-guardrail. Recommend starting with `RulesGuardrail` (deterministic, cheap, and it directly tests the payload-inspection hypothesis).
2. **Option B vs A:** curation (recommended) vs full portfolio search. Recommend B unless a real private score shows curation leaves large gains on the table.
3. **`λ` schedule:** fixed small constant vs adaptive (raise as proxy fidelity is validated). Recommend fixed-small first.
4. **Deputy as a first-class portfolio slot** vs the fixed `_DEPUTY_FRAC`: fold it into `curate_portfolio` so its share is measured, not hard-coded. Recommend folding in.
