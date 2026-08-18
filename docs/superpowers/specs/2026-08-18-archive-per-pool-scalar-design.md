# Archive Simplification: Output-Only Cost + Per-Pool Scalar Selection

**Status:** proposed (2026-08-18)
**Supersedes the live archive design** in `two-pool-per-model-search-live` memory (cross-model
4-D Pareto) and reverts the input-cost objective (`f07d3aad`).

## Goal

Replace the archive's cross-model 4-D Pareto with a per-pool scalar MAP-Elites grid, and drop the
input-length cost term, so the search selects each pool's shapes by the one quantity that actually
drives replay cost: **fewest output (generated) tokens among firing shapes.**

## Why (measured, not inferred)

Direct micro-benchmark on the resident gemma GGUF (`scratchpad/measure_prefill_vs_decode.py`,
2026-08-18): prefill **0.166 ms/token**, decode **7.445 ms/token** → a decode token costs **~45×**
a prefill (input) token. The plain-vs-forge input gap is ~10 tokens ≈ 1.7 ms; output is ~30 tokens
≈ 223 ms. Input length is ~0.7 % of output — negligible. End-to-end: a 725-char-longer user message
replayed *faster* because it generated fewer tokens. So:

1. The `INPUT_PREFILL_WEIGHT` term (`f07d3aad`) optimizes a ~1 % rounding error — **revert it.**
2. The env cannot be made faster for us: `build_attack_env` is grader-fixed, and the ~16.9K-char
   system+tool-schema prefix is KV-prefix-reused across candidates (llama_cpp 0.3.34
   `Llama.generate` longest-common-prefix reuse) → paid once, not per candidate.
3. On the public board severity is constant (single exfil = 16 for every firing shape, 0 for duds),
   so `board_density = (16+2)/200 × 1/(gen_tokens+FIXED)` is monotonic in `−gen_tokens`. The
   cross-model Pareto over (throughput, severity) is therefore ≈ a per-model scalar ranking on
   output tokens. It buys complexity (and a junk-axis pollution surface) for a diversity benefit the
   MAP-Elites cell grid already provides.

## Design

### Cost (revert input)

- `throughput(gen_tokens, model) = 1/(gen_tokens + FIXED_TOKENS[model])`, `inf → 0`. Remove the
  `input_tokens` param and `INPUT_PREFILL_WEIGHT` (delete the config constant).
- `board_density(severity, gen_tokens, model)` — remove `input_tokens`. `severity ≤ 0 → 0`.
- `_shape_elites` — remove `input_tokens` computation and pass-through; remove `input_chars` set.

### Selection (per-pool scalar MAP-Elites, no Pareto)

- Delete `dominates()` and all cross-model Pareto logic.
- Each scored shape is a **single-model specialist**: in the two-pool design a message is authored
  into one pool and scored on one victim, so its other-model throughput/severity is 0. An Elite that
  fires on both models (same text authored to both pools) occupies one cell per firing model.
- **Cell key = `(model, family, bucket)`.** Keep the **single highest `model_density` elite per
  cell** (standard MAP-Elites one-per-cell). Structural diversity comes from the family×bucket grid;
  quality from best-per-cell. `insert(elite)`: for each model the elite fires on
  (`throughput[model] > 0`), place into `(model, family, bucket)`, replacing the occupant iff strictly
  denser by `model_density(·, model)`.
- `frontier()` → union of all cell occupants. `ship_set()` / `parents(k)` rank per model by
  `model_density` and interleave via `rank_by_model_density` (unchanged), so neither model crowds the
  other. `_frontier_map`'s split by `throughput[model] > 0` is unchanged.
- `elite_board_density` / `model_density` keep the `(sev+NOVELTY_PER_CELL)/200 × throughput`
  identity (now with output-only throughput).

### Removals

`dominates`; `Elite.input_chars`; `config.INPUT_PREFILL_WEIGHT`; the `input_tokens` params on
`throughput`/`board_density`; input-cost tests. The `ship_set` `−input_chars` tiebreak goes with
`input_chars`.

### Persistence / backward-compat

- `Elite.from_jsonl`: rows still load; a persisted `input_chars` key is ignored (not a field).
  Existing behavior of discarding pre-severity rows is unchanged.
- No live board to migrate: the input-cost restart is the current cold board and will be re-archived
  under this cold-start anyway.

## Testing

- `throughput`/`board_density` output-only: shorter-output shape ranks higher; input no longer
  appears in any signature.
- `insert` keeps the leaner-output firing shape per `(model, family, bucket)` cell and evicts a
  heavier one; a non-firing shape (throughput 0) never occupies a cell.
- A gpt-only and a gemma-only specialist both survive (different models → different cells) and both
  ship — the per-model-specialist property the Pareto used to guarantee.
- `rank_by_model_density` interleave unchanged (regression already covered).
- Full suite + `uv run pre-commit run -a` green.

## Risks

- **Fewer distinct shapes on the frontier** (one-per-cell vs Pareto sets). Acceptable: the submission
  fills each shape-template across many URLs (`candidate_chains`), so a handful of dense shapes per
  model fills the whole 2000-candidate board; the archive's job is to find the leanest firing shape
  per model, not thousands.
- **Severity assumption.** If we later pursue varying severity (multi-predicate / private-deputy
  board), scalar selection under-weights a high-severity-low-throughput shape. Out of scope now
  (public single-exfil); revisit by re-introducing severity as a second per-model axis *then*.

## Terminal step

After green: cold-restart the optimizer on the output-only per-pool-scalar archive.
