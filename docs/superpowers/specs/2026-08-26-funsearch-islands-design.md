# FunSearch Islands for the Attack-Shape Optimizer — Design

**Date:** 2026-08-26
**Status:** approved (design), pending spec review
**Research basis:** `docs/research/2026-08-26-improving-the-search.md`, lever #4.

## Goal

Replace the optimizer's single shared MAP-Elites archive with **N parallel island
archives**, so independent lineages evolve concurrently instead of the search collapsing
onto (and endlessly re-refining) one champion. This is the canonical FunSearch/CMA-ME cure
for the monoculture we've been hitting.

## Why

The current loop: one shared `archive.Archive`, and every generation the proposer refines
`board.best_objective()` (the global champion). It reliably rediscovers the champion and
rarely finds a structurally different shape. Islands break this by (a) evolving several
lineages in parallel, (b) hard-resetting stalled lineages to *random* elites so they can't
converge, and (c) dedicating one island to structural *novelty* rather than throughput.

## Non-goals

- No change to the replay/scoring path (`score_pools`, `replay_trace`, the objective).
- No change to the token cost model (v23), the ablation post-pass, or the shipped-cut format.
- No new proposer model or lane; islands ride the existing codex lane via `replicas`.

## Architecture

### Topology — N parallel workers, one island each

- The `Blackboard` holds an **`IslandSet`**: `list[archive.Archive]` of length N, plus
  per-island bookkeeping (best-objective seen, generation of last local improvement).
- `optimize_team` spawns **N workers on the codex lane** (`JED_PROPOSER_REPLICAS = N`;
  the codex ChatGPT token supports N concurrent workers — user-confirmed 2026-08-26).
  Each worker is pinned to one island index `i` (its `worker_id`).
- Worker `i` owns island `i`: every generation it proposes from *island i's* local
  incumbent + parents, scores (unchanged path), and inserts into *island i's* archive.
  No round-robin rotation — the parallelism is the workers.
- **Island 0 is the novelty island**; islands 1…N−1 are quality islands (current Pareto
  behaviour). N = 4 by default (1 novelty + 3 quality).

### Components

**`IslandSet`** (new, in `blackboard.py` or a sibling module):
- `archives: list[archive.Archive]` — one per island.
- `local_best[i]`, `stall[i]` — per-island stall tracking (updated by worker i).
- `global_best()` — the best-objective elite/record across all islands (for shipping).
- `insert(i, elite) -> bool` — insert into island i (novelty i=0 uses novelty rule).
- `reset(i, seed_elites)` — clear island i and reseed it.
- `to_jsonl(dir)` / `from_jsonl(dir)` — persist/restore N archives (island_<i>.jsonl).
- Mutations are serialized by the Blackboard's existing `asyncio.Lock` (each worker
  mostly touches only its own island, so contention is rare; migration/global-best/reship
  touch shared state and take the lock).

**Quality island (i≥1):** exactly today's archive — Pareto over per-model
throughput+severity, `input_char_bucket` diversity axis, `shape_family` cells. Reuse
`archive.Archive` unchanged.

**Novelty island (i=0):** same `archive.Archive` container, but selection/retention prefers
structural **novelty** over throughput. Novelty(elite) = mean descriptor-space distance to
its k=3 nearest archived elites, where the descriptor is `(shape_family, input_char_bucket)`
(the v23 structural key). On insert, when a cell is contested, keep the more novel elite
(not the higher-throughput one); parents are sampled to maximise structural spread. Firing
is still required (a non-firing shape is never kept).

### The three FunSearch mechanisms (under the shared lock)

- **Migration** — every `M = 12` generations (a global counter), copy the current
  `global_best()` elite into a **random quality island** (spreads good genes without
  collapsing a lineage). The novelty island receives migrants too but keeps selecting on
  novelty, so it explores *around* good shapes rather than converging on them.
- **Stagnation reset** — worker i tracks `stall[i]` (generations since island i's local
  best improved). At `K = 10`, worker i **hard-resets island i**: clear its archive and
  reseed from a **random elite of the best island** (novelty island reseeds from a
  structurally-distinct random elite). Reset never touches the global best or the shipped
  artifact.
- **Shipping** — `attack.py` ships the **global** best across all islands (via the existing
  per-model router); `_ablate_champion` post-passes the global best. Because resets and
  novelty churn only ever *add* candidates and shipping reads the global best, a burst can
  discover a champion but never ship a worse one.

## Data flow

```
worker i, each generation:
  incumbent  = islandset.local_best(i)          # this island's champion, not global
  parents    = islandset.archives[i].parents(k) # (novelty island: novel parents)
  batch      = propose(incumbent, parents, ...)  # UNCHANGED proposer path
  scores     = score_pools(batch)                # UNCHANGED replay path
  for elite in _shape_elites(batch, scores):
      islandset.insert(i, elite)                 # island i only
  if islandset.global_best_changed():
      reship(global_best); ablate(global_best)   # UNCHANGED ship path
  islandset.tick(i)                              # stall/reset + periodic migration
```

## Config (env-overridable, in `config.py`)

- `ISLAND_COUNT = int(env "JED_ISLANDS", 4)` — N.
- `ISLAND_STAGNATION_GENERATIONS = int(env "JED_ISLAND_STALL", 10)` — K.
- `ISLAND_MIGRATION_GENERATIONS = int(env "JED_ISLAND_MIGRATE", 12)` — M.
- `NOVELTY_NEIGHBORS = int(env "JED_NOVELTY_K", 3)` — novelty k.
- `run_optimizer.sh` sets `JED_PROPOSER_REPLICAS=4` for the codex lane.
- 0 for K or M disables that mechanism (islands still evolve independently).

## Persistence & rollout

- N per-island archive JSONLs (`run/island_<i>.jsonl`) alongside `blackboard.jsonl`
  (which still logs every record). `Blackboard.load` restores the IslandSet.
- **Scheme bump v23 → v24** (island selection changes what ships), so the board
  cold-starts. Reseed the 108 champion into a quality island (island 1) via the existing
  reseed path.

## Testing

All decision logic is pure and GPU-free; the replay path is untouched.

- `IslandSet.insert` / `global_best` / `reset` — with fake elites.
- Novelty distance + novelty-island retention — a contested cell keeps the more novel elite.
- Stall/reset trigger — reaches K, resets, reseeds from best island.
- Migration cadence — fires every M, copies global best into a quality (not novelty) island.
- Worker→island pinning — worker i touches only archive i (plus shared global/migration).
- Persistence round-trip — N archives save/restore.
- Existing scoring/objective/ablation tests unchanged and still green.

## Risks

- **Concurrency:** N workers mutate shared IslandSet state. Mitigated by keeping per-island
  mutation lock-free-by-ownership (worker i is the only writer of island i) and taking the
  Blackboard lock only for global-best/migration/reship. Verify no cross-island races.
- **Quota:** N=4 concurrent codex requests → ~4× token spend. Accepted (token supports it).
- **Shipping regressions:** guarded by shipping the global best only; a smoke test asserts
  the shipped pool equals the global-best across islands.

## Decisions (locked)

N=4 (1 novelty + 3 quality) · K=10 stalled gens · M=12 gens · novelty k=3 · reset reseeds
from a random elite of the best island · ship global best · scheme v24 · reseed 108 into a
quality island.
