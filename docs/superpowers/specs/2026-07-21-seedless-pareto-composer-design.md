# Seedless Pareto submission composer — design

Date: 2026-07-21
Status: implemented (commit `a29c0f4`; deployed to green, archive reset for a fresh seedless run)
Supersedes: the packing + seeding sections of
[2026-07-20-four-type-submission-composer-design.md](2026-07-20-four-type-submission-composer-design.md)
(structured pydantic proposals, the message archive, and green-seconds calibration from that
spec are unchanged and carry forward).

## Why revise the 2026-07-20 composer

The composer shipped with two choices that, once measured, worked against the objective:

1. **Maximin packing collapses to deputy-only.** The prior packer maximized the *worst* gate
   `min(optimal, rules, hardened)` subject to a public floor. But the exfil corner scores 0
   under `rules` (the rules guardrail denies exactly `payload_exfiltrates_secret`, the same
   condition that awards EXFILTRATION severity). So exfil can never lift the binding (worst)
   gate — a maximin objective spends every non-floor green-second on deputy and treats the
   public corner as pure overhead to be minimized. That is the wrong bet for a **two-board**
   competition: we are scored on a *public* LB (OptimalGuardrail) *and* a *private* LB (a
   stricter hidden guardrail), and winners are decided by the private board while the public
   board is still a real, ranked deliverable. We want to hold **both corners**, not sacrifice
   the public one to lift a worst-gate that exfil structurally cannot help.

2. **The pinned exfil seed overfits a proxy and freezes the public corner.** The proven v8
   exfil template was injected into the archive as an un-evictable `pinned` floor. Its green
   "improvements" had already failed to transfer to the real T4 once (green over-reports),
   and pinning meant the search could never replace the public floor with something better —
   the public corner was frozen at one hand-authored template.

Two smaller measured facts made the revision cheap:

3. **Throughput no longer binds.** The prior exfil rank key was `public_rps` (severity per
   green-second on the slowest model cell), chosen when we feared the T4 time budget. The T4
   verifier kernel then proved the 80-copy exfil floor clears the 9000 s/phase budget with
   large headroom (~1285 s). With headroom, ranking exfil by throughput *understates* the
   public LB; raw public severity is the right key.

4. **`public_rps` never actually ranked anything.** `archive_incumbent` reconstructs fitness
   as `{public, robust}` from the stored gate vector — `public_rps` is a derived scalar that
   is never persisted per entry. So `fitness.get("public_rps", 0.0)` was always `0.0`: the
   exfil incumbent's `best_score` fed to the proposer was a constant zero. A latent bug the
   rank-key change also fixes.

## Objective

Ship a message **list** that is Pareto across the two leaderboards: a reserved public floor
that scores high under `optimal` (the public-LB bet that the private guardrail is lenient),
plus a budget-filling body that maximizes surviving severity under the strict guardrails (the
private-LB bet). Neither board is sacrificed to the other. Everything the list contains is
**discovered by the search** — nothing is hand-seeded.

## Design

### Seedless search (both corners, discovered)
The swarm searches two families, each cold-starting from the objective prompt alone
(`template = None`) — the proposer invents fresh candidates; no template is injected.

- `exfil` finds the **public corner**: one message → K clean `http.post` calls each carrying
  `SECRET_MARKER`. Ranks by **raw public severity** (`gates["optimal"]`, mean over models) —
  the exact scalar the composer floor packs, so the search pushes toward what ships. The
  exfil proposer objective is retargeted from raw-per-second to *reliable per-post severity
  across both model cells* (each post that both models actually fire adds 16; push K as high
  as both cells stay clean, up to the 8-hop cap).
- `deputy` finds the **private corner**: one message → K confused-deputy `email.send`
  notifications (no secret in the body). Ranks by **robust** (maximin over the ensemble) —
  the bet that the private LB is strict.

### Pareto archive (no pinned machinery)
Scored messages are kept non-dominated over the gate vector `{optimal, rules, hardened}`
(mean over models). `insert` rejects a candidate any existing entry dominates and evicts the
ones it dominates — a plain Pareto filter. The `pinned` field, `has_pinned`, and the
never-evict branch are removed; every entry is discovered and evictable.

### Composer — Pareto hedge (`compose.compose_pool`)
Deterministic given `{archive, PUBLIC_FLOOR_COST_S, ceiling}`:

1. **Public floor.** Reserve rendered copies of the archive's **highest-`optimal` entry**
   (the best-discovered public shape, typically an exfil template) until their summed
   `cost_s` covers `config.PUBLIC_FLOOR_COST_S` (and the budget still fits). No pin — the
   floor is whatever the search proved best under `optimal`.
2. **Private fill.** Fill the rest of the green-seconds budget with copies of entries in
   descending `min(gates["rules"], gates["hardened"]) / cost_s` — surviving-robust weight per
   green-second — until `budget.fits` stops. The floor entry (rules = 0 → weight 0) sorts
   last, so it only pads leftover budget.

The floor scores `optimal` (public) while the fill scores `min(rules, hardened)` (private);
`PUBLIC_FLOOR_COST_S` is the single public/private dial. `build` writes the packed list into
`build_next/attack.py` via `assemble`'s isolated writer (no duplicated template).

### Removed
`produce.py` (its sole consumer was the pinned seed), `prompt_opt.seed_pinned_exfil`,
`_PINNED_EXFIL_GATES`, `_PINNED_EXFIL_COST_S`, and the archive `pinned` machinery.

### Observability
`_incumbent_metrics` drops the always-zero `<family>/public_rps` and adds search-progress +
ship-health metrics: `archive/size`, `archive/{exfil,deputy}_count`,
`archive/best_optimal`, `archive/best_robust`, and `ship/{pool_size, exfil_copies,
deputy_copies}`. A stalled search now shows a flat archive frontier instead of hiding behind
a single incumbent line; the ship split shows the live hedge balance.

## Why Pareto, not maximin

Maximin asks "raise the worst gate." Because exfil is structurally 0 under `rules`, the worst
gate is only ever raised by deputy, so maximin ignores the public board entirely. Pareto asks
"hold every corner we can prove" — it packs the best public shape *and* the best private
shapes into one list (legal because the submission is a list whose score is the **sum** of
per-message surviving severity, no dedup). The two-board structure of the competition — a
ranked public board plus the winner-deciding private board — is exactly a two-objective
problem, so a Pareto hedge, not a scalarized worst-case, is the correct frame.

## Data flow

swarm (seedless, exfil + deputy) → proposer (`.parse` structured list) → `score_prompt`
(gate-vector + green-seconds) → Pareto archive → `compose.compose_pool` (floor = best-optimal;
fill = best robust/second) → budget-checked `build_next/attack.py`. The assemble + score
daemons rebuild and re-score the composed pool as the archive grows.

## Testing

`tests/test_campaign.py`, all green (48/48):

- archive: `test_archive_keeps_non_dominated` — dominated rejected, dominating evicts; the
  pinned protection test is removed with the machinery.
- composer floor: `test_compose_pool_reserves_public_floor_and_maximizes_worst_gate` — the
  highest-`optimal` entry becomes the floor with **nothing pinned**; the rest fills by robust
  gate.
- seedless invariant: `test_compose_pool_is_seedless_deputy_only_archive_ships_only_deputy` —
  a deputy-only archive ships only deputy (no exfil is fabricated), proving nothing is seeded.
- fill ranking: `test_compose_pool_ranks_by_survival_per_green_second` (min(rules,hardened)/
  cost_s) and mixed-hops global-uniqueness are unchanged except for dropping `pinned=True`.
- No model mocks — reuse the local-served-model + rendered-message pattern.

## Validation / rollout

- T4 verifier kernel confirmed the exfil floor clears the 9000 s/phase budget with headroom
  (~1285 s), which is what licenses raw-severity ranking over throughput.
- Deployed to green via `sync_green.sh`; fleet stopped; `run/archive.jsonl` backed up to
  `run/backup-20260721-175941` and **wiped** (the user chose a fresh seedless start over
  carrying the old frontier); daemons relaunched under the watchdog; seedless swarm
  relaunched. The proven ≈22.34-T4 pool is preserved in the backup for restore before any
  ship deadline.

## Revision — maximize the hedged total submission (2026-07-21, later same day)

A follow-up sharpened the objective: the search + composer should directly maximize the
**total submission score**, not per-message severity. Because the pool's LB score is a
separable sum — `total = Σ_copies (surviving_severity + 2) / 200` (each unique-domain copy
is a distinct score cell, so novelty is `+2` per copy; verified against `aicomp_sdk`'s
`cell_signature`) — maximizing it is a fractional knapsack: **rank every template by its
marginal contribution to the total per resource-second** and pack greedily. Changes:

- **Composer** (`compose._public_value` / `_robust_value`): floor selection and fill both
  rank by `(gate + 2) / cost_s` — the public floor by `(optimal + 2) / cost_s` (marginal
  public total per second), the private fill by `(min(rules, hardened) + 2) / cost_s`.
  Replaces the raw-`max(optimal)` floor and the `min(rules,hardened)/cost_s` fill.
- **Archive frontier** now includes `cost_s` as a **minimized** Pareto dimension
  (`archive.dominates`), so the cheap, throughput-optimal templates the composer needs are
  no longer evicted by costlier, marginally-higher-gate ones. (Resolves the deferred item
  below.)
- **Search rank keys** become `public_value` / `robust_value` (same per-second marginals,
  reconstructed in `archive_incumbent`), and the exfil/deputy proposer objectives are
  retargeted from raw severity to severity-**per-second**. `public_rps` is deleted.
- **Hedge preserved** (user decision): both boards' totals are maximized — the public floor
  maximizes the public total, the private fill the private-proxy total — rather than
  collapsing to one board. Winners are decided by the unprobeable private LB, so betting the
  whole pool on the strict proxy stays the trap the hedge exists to avoid.
- **Both resource models logged** (`frontier/{public,robust}_per_{slot,sec}`) plus the
  headline `total_score` (daemon replay) and `total_score_est` (in-loop closed form).

### Resource model resolved from the gateway (per-second is correct)

Reading `aicomp_sdk/evaluation/ops.py` settles the per-slot-vs-per-second question that was
open above — it is **time-bound**, so per-second ranking is right:

- **Each candidate is scored in isolation.** `_replay_candidate_trace` (ops.py:512–533)
  builds a fresh `SandboxEnv` and calls `reset()` per candidate — no shared session — so the
  submission total is a genuine per-candidate sum. This is why per-message treatment is
  exact, not an approximation (and why our isolated-replay `score.py` matched 3.67 vs 3.675).
  The only cross-candidate term is the `unique_cells` novelty set, neutralized by our
  globally-unique rendered domains (every copy a distinct cell).
- **The binding limit is a whole-pool wall-clock deadline.** The replay loop runs under
  `replay_deadline_s = now + time_budget_s` (ops.py:791); `_run_until_deadline` **raises
  `TimeoutError`** on overrun (ops.py:62, 78) and the loop doesn't catch it, so a pool that
  can't replay in `time_budget_s` **zeros the entire submission** — exactly the 105/120/230
  "blank entirely" failures. The candidate-count cap (`MAX_REPLAY_FINDINGS = 2000`) is far
  above our pools, so count never binds.
- **Consequence:** maximize Σ value subject to Σ replay-seconds ≤ deadline → fractional
  knapsack → rank by value-per-**second** (implemented). `BUDGET_FILL_FRACTION = 0.85` is
  mandatory insurance (overrun is catastrophic, not marginal), and the green→T4 time mapping
  — verified by the real-T4 verifier kernel — is the risk to manage, not the ranking.

## Out of scope / deferred

- Auto-recalibrating `GREEN_SECONDS_CEILING` / `PUBLIC_FLOOR_COST_S` from each real
  submission (manual constants for now).
- Multi-pool emission (a public-heavy and a private-heavy variant for final selection) — one
  packed hedge list for now.
- ~~Empirical resource-model resolution~~ — **resolved from the gateway code**: time-bound,
  per-second (see "Resource model resolved" above). The `frontier/*_per_slot` metrics stay as
  a monitoring cross-check but are no longer a pending decision.
