# Archive Simplification: Correct-Weighted Cost + Per-Pool Scalar Selection

**Status:** proposed (2026-08-18)
**Supersedes the live archive design** in `two-pool-per-model-search-live` memory (cross-model
4-D Pareto) and **recalibrates** the input-cost objective (`f07d3aad`) rather than reverting it.

## Goal

Replace the archive's cross-model 4-D Pareto with a per-pool scalar MAP-Elites grid, and set the
input-length cost term to its **measured** weight so the search selects each pool's shapes by the
true replay cost: mostly output (generated) tokens, plus a small, correctly-weighted input term.

## Why (measured, not inferred)

Direct micro-benchmark on the resident gemma GGUF (`scratchpad/measure_prefill_vs_decode.py`,
2026-08-18): prefill **0.166 ms/token**, decode **7.445 ms/token** -> a decode token costs **~45x**
an input (prefill) token. Input is NOT free (the user-message tokens are the differing suffix,
freshly prefilled every candidate -- only the ~16.9K system+tool-schema prefix is KV-reused), it is
just small: ~10-token input delta ~= 1.7 ms vs ~30-token output ~= 223 ms. So:

1. Keep input in the cost, but at the **measured** weight. The replay cost in decode-token-equivalents
   is `gen_tokens + (0.166/7.445)*input_tokens + FIXED`, i.e. **`INPUT_PREFILL_WEIGHT = 0.022`** (was a
   guessed `0.2`, ~9x too heavy). At 0.022 the term minimizes input for free -- a 1-output-token
   saving (~7.4 ms) dwarfs tens of input tokens, so output leanness is never traded away, yet a large
   input saving that genuinely beats a tiny output increase is still captured.
2. The env cannot be made faster for us: `build_attack_env` is grader-fixed, and the ~16.9K-char
   system+tool-schema prefix is KV-prefix-reused across candidates (llama_cpp 0.3.34
   `Llama.generate` longest-common-prefix reuse) -> paid once, not per candidate.
3. On the public board severity is constant (single exfil = 16 for every firing shape, 0 for duds),
   so `board_density = (16+2)/200 * throughput` is monotonic in `throughput`. The cross-model Pareto
   over (throughput, severity) is therefore ~= a per-model scalar ranking. It buys complexity (and a
   junk-axis pollution surface) for a diversity benefit the MAP-Elites cell grid already provides.

## Design

### Cost (recalibrate, do not remove input)

- `throughput(gen_tokens, model, input_tokens=0.0) = 1/(gen_tokens + INPUT_PREFILL_WEIGHT*input_tokens
  + FIXED_TOKENS[model])`, `inf -> 0`. Unchanged shape; only `config.INPUT_PREFILL_WEIGHT` changes
  `0.2 -> 0.022` (a measured constant; document the source as prefill/decode ms-per-token ratio).
- `board_density(severity, gen_tokens, model, input_tokens=0.0)` -- unchanged; `severity <= 0 -> 0`.
- `_shape_elites` -- unchanged input threading (`input_tokens = len(text)/CHARS_PER_TOKEN[model]`,
  passed only for firing models). Input is now correctly encoded inside `throughput`.

### Selection (per-pool scalar MAP-Elites, no Pareto)

- Delete `dominates()` and all cross-model Pareto logic.
- Each scored shape is a **single-model specialist**: in the two-pool design a message is authored
  into one pool and scored on one victim, so its other-model throughput/severity is 0. An Elite that
  fires on both models (same text authored to both pools) occupies one cell per firing model.
- **Cell key = `(model, family, bucket)`.** Keep the **single highest `model_density` elite per
  cell** (standard MAP-Elites one-per-cell). Structural diversity comes from the family x bucket grid;
  quality from best-per-cell. `insert(elite)`: for each model the elite fires on
  (`throughput[model] > 0`), place into `(model, family, bucket)`, replacing the occupant iff strictly
  denser by `model_density(., model)`.
- `frontier()` -> union of all cell occupants. `ship_set()` / `parents(k)` rank per model by
  `model_density` and interleave via `rank_by_model_density` (unchanged), so neither model crowds the
  other. `_frontier_map`'s split by `throughput[model] > 0` is unchanged.
- `elite_board_density` / `model_density` keep the `(sev+NOVELTY_PER_CELL)/200 * throughput`
  identity -- `throughput` already carries the correctly-weighted input cost, so input needs no
  separate ship tiebreaker.

### Removals

`dominates`; `Elite.input_chars` and the `ship_set` `-input_chars` tiebreak (both now redundant --
input lives in `throughput`); the cross-model Pareto path in `insert`/`frontier`. **Keep**
`INPUT_PREFILL_WEIGHT` (recalibrated) and the `input_tokens` params.

### Persistence / backward-compat

- `Elite.from_jsonl`: rows still load; a persisted `input_chars` key is ignored (no longer a field).
  Existing behavior of discarding pre-severity rows is unchanged.
- No live board to migrate: the input-cost restart is the current cold board and re-archives under
  this cold-start anyway.

## Testing

- `throughput`/`board_density` with `INPUT_PREFILL_WEIGHT=0.022`: a 1-output-token saving outranks
  any realistic input increase (output dominance), AND a large input saving at equal output ranks
  higher (input still minimized).
- `insert` keeps the denser firing shape per `(model, family, bucket)` cell and evicts a heavier one;
  a non-firing shape (throughput 0) never occupies a cell.
- A gpt-only and a gemma-only specialist both survive (different models -> different cells) and both
  ship -- the per-model-specialist property the Pareto used to guarantee.
- `rank_by_model_density` interleave unchanged (regression already covered).
- Full suite + `uv run pre-commit run -a` green.

## Risks

- **Fewer distinct shapes on the frontier** (one-per-cell vs Pareto sets). Acceptable: the submission
  fills each shape-template across many URLs (`candidate_chains`), so a handful of dense shapes per
  model fills the whole 2000-candidate board; the archive's job is to find the leanest firing shape
  per model, not thousands.
- **Weight calibration.** `0.022` is the local GGUF prefill/decode ms ratio; the real grader could
  differ, but input is ~1% of replay cost either way, so the exact value is low-stakes.
- **Severity assumption.** If we later pursue varying severity (multi-predicate / private-deputy
  board), scalar selection under-weights a high-severity-low-throughput shape. Out of scope now
  (public single-exfil); revisit by re-introducing severity as a second per-model axis then.

## Terminal step

After green: cold-restart the optimizer on the per-pool-scalar archive with the recalibrated cost.

## OUTCOME (2026-08-18) — collapse DROPPED, Pareto kept

Implementing the collapse surfaced that "one densest elite per (model, family, bucket)
cell, ship all cells" ships FAT shapes: ``bucket`` is a cost axis, so a fat shape in a
high-cost bucket occupies its own cell and rides the frontier (the cold-start test caught
it). Deciding what SHIPS requires comparing across cost buckets to drop dominated fat
shapes -- which the global Pareto frontier does and a per-cell scalar cannot. Also, the
junk-protection this collapse aimed to remove came only from ``input_chars`` as a Pareto
AXIS, which was never in play (input lives in ``throughput``). So the collapse fixed no
live bug and introduced one; it was reverted.

WHAT SHIPPED instead (all green, reviewed):
- ``INPUT_PREFILL_WEIGHT`` 0.2 -> measured 0.022 (input minimized at the right magnitude).
- Firing is intrinsic to the reward: ``model_density``/``elite_board_density`` and the
  ``rank_by_model_density`` / ``parents()`` selection all return/contribute 0 unless
  ``severity > 0`` (not merely ``throughput > 0``) -- a generate-but-never-fire shape can
  neither ship nor parent.
- ``_frontier_map`` ships only firing shapes and breaks density ties toward shorter input.
- The Pareto frontier is KEPT: it correctly excludes fat-firing and non-firing shapes
  across cost buckets, which is load-bearing.
