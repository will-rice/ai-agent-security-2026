> **SUPERSEDED (2026-08-18):** Task 1 (weight recalibration) shipped as planned. Task 2
> (per-cell scalar collapse) was DROPPED -- it ships fat shapes across cost buckets. The
> Pareto frontier was kept; instead firing was made intrinsic to the reward + an explicit
> input tiebreak was added. See the spec's OUTCOME section.

# Archive Per-Pool Scalar — Implementation Plan

> Executes docs/superpowers/specs/2026-08-18-archive-per-pool-scalar-design.md.

**Goal:** Recalibrate the input-prefill weight to its measured value and collapse the
cross-model 4-D Pareto archive to a per-model scalar MAP-Elites grid (one densest elite
per `(model, family, bucket)` cell).

**Tech:** Python, dataclasses, pytest. `uv run pre-commit run -a` must stay green.

## Global Constraints
- Cost stays token-based; input stays in `throughput` at `INPUT_PREFILL_WEIGHT = 0.022`.
- Shipping path `_frontier_map` (already per-model `model_density` over `frontier()`) is
  UNCHANGED — the collapse only changes what `frontier()` contains.
- No `input_chars` anywhere; no `dominates`.

---

### Task 1: Recalibrate input weight

**Files:** Modify `src/jed_attack/campaign/config.py:293`; Test `tests/test_campaign.py:8581`.

- [ ] config: `INPUT_PREFILL_WEIGHT = 0.2` -> `0.022`, comment cites prefill/decode
      ms-per-token ratio (0.166/7.445) measured 2026-08-18.
- [ ] Update the throughput input test: assert (a) input lowers throughput, (b) a
      1-gen-token saving outranks a +10-input-token increase (output dominance at 0.022),
      (c) backward compat `input_tokens=0 == omitted`.
- [ ] `uv run pytest tests/test_campaign.py -k throughput` green; commit.

### Task 2: Per-model scalar MAP-Elites (drop Pareto)

**Files:** Modify `src/jed_attack/campaign/archive.py`,
`src/jed_attack/campaign/optimize_prompts.py` (`_shape_elites` Elite ctor);
Test `tests/test_campaign.py` (rewrite the Pareto-contract tests).

**New contract (archive.py):**
- `Elite`: drop `input_chars`.
- Delete `dominates`.
- `_cells: dict[(model,family,bucket), Elite]` — ONE densest elite per cell.
- `insert(elite)`: for each `model` with `throughput[model] > 0`, key
  `(model, family, bucket)`; store iff empty or strictly greater `model_density(elite,
  model)` than the occupant. Return whether it won >=1 cell.
- `frontier()`: unique cell occupants (dedup by identity, insertion order).
- `ship_set()`: `rank_by_model_density(frontier())[:ARCHIVE_FRONTIER_CAP]` (drop the
  summed-density + `-input_chars` sort).
- `parents(k)`: `rank_by_model_density(frontier())[:k]` (frontier == all held elites now,
  so the under-filled-cell top-up is moot — remove it).
- `elite_board_density` / `model_density` / `rank_by_model_density`: unchanged.
- `from_jsonl`: keep the "no `severity` key -> stale, discard whole file" guard; filter
  each row to `Elite` field names so a persisted `input_chars` key loads harmlessly.

**optimize_prompts `_shape_elites`:** drop `input_chars=len(message.text)` from the Elite
ctor; keep the `input_tokens` threading into `throughput`.

**Test rewrites (old Pareto -> new scalar-cell contract):**
- Two-pool specialists (≈1309, 2815): gpt-only + gemma-only land in different-model cells
  -> both in `frontier()`/ship; drop `dominates` asserts.
- Same-cell eviction (≈1733, 2900): denser shape evicts the weaker from ITS cell; assert
  frontier holds only the denser. Replace `dominates(strong,weak)` asserts.
- Identical vectors (≈1464): two equal-density elites in the SAME cell -> one kept (not
  strictly greater); in DIFFERENT cells -> both kept.
- Gemma trade-off set (≈2860): under one-per-cell, gemma elites sharing a
  `(gemma,family,bucket)` cell collapse to the densest — rewrite the expected count to the
  distinct-cell count, not "all 25".
- `dominates` unit tests (≈8614, 8686): delete or convert to `insert`/`frontier`
  eviction assertions.
- Any `insert`+`frontier` test (1185,1409,1868,3826,5395): re-derive expected frontier
  under one-per-cell; fix counts.

- [ ] Rewrite tests first (red), implement archive.py + `_shape_elites` (green).
- [ ] Full `uv run pytest tests/test_campaign.py` + `uv run pre-commit run -a` green; commit.

## Terminal step
Merge to `replay-speed-investigation`; cold-restart the optimizer.
