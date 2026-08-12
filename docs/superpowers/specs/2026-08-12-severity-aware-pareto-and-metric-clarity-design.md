# Severity-aware 4-D Pareto + metric clarity

**Date:** 2026-08-12
**Status:** approved (design)

## Motivation

The QD-Pareto proposer (live main@982a963) plateaued at `board_min_models ≈ 53–54`
within ~5 generations. wandb run `team-982a96` shows why:

1. **The Pareto vector is throughput-only** — `throughput_m = 1/(gen_chars_m + FIXED[m])`,
   leanness only. Every firing forge shape is the same ~175-char length, so all 28→140
   frontier elites share ONE throughput vector `(0.0058, 0.0072)`. Pareto has nothing to
   select on; the archive is a dedup bag, not a quality-diversity frontier.
2. **Severity — the signal that actually moves the score — is invisible to the vector.**
   `batch_severity` swings 404 → 1832 across batches and tracks the objective, yet the
   frontier froze. Two shapes of equal length but different severity are treated as ties.
3. **We are gemma-bound.** Per-model, `board_gpt_oss ≈ 66–84` while `board_gemma_4 ≈ 54`;
   the MIN-over-models objective (`_score_public_raw_per_gen_char`) lands on gemma every
   time. gemma severity is the lever, and a per-model vector is how the search sees it.

Separately, the wandb metric names are misleading — `batch_mean_objective` is a mean over
the *batch* (size 1) of a per-submission *min-over-models*, and a code comment at
`optimize_prompts.py:548-554` wrongly claims `best_objective` is the *mean* of the two
columns when it is the *min*. This obscured exactly the gemma-bound diagnosis above.

## Approved decisions

- **4-D Pareto** over independent per-model axes `(throughput_gpt, throughput_gemma,
  severity_gpt, severity_gemma)` for the ARCHIVE, dominance, and parent sampling — chosen
  over collapsing to a scalar board-density, to preserve exploratory diversity (a
  lean-but-weaker shape stays on the frontier as crossover material).
- **Ship by board-density.** `ship_set` ranks the frontier by the shape's actual board
  contribution and ships the top `ARCHIVE_FRONTIER_CAP`, so exploration stays in the
  archive but the *shipped* pool is the highest-scoring subset (never diluted by weak
  tradeoffs).
- **Fold the metric renames + stale-comment fix + a new `board_mean_models` into the same
  build** (one restart cuts everything over).

## Architecture

### Score vector (archive.py, submission_score.py, optimize_prompts.py)

`Elite` gains `severity: dict[str, float]` alongside the existing `throughput: dict[str,
float]`. Both keyed by `config.MODELS`.

- `throughput_m` unchanged: `1/(gen_chars_m + FIXED_CHARS[m])`, `0.0` when non-firing.
- `severity_m` = `severity_by_model[config.GATE_GUARDRAIL_NAME][m]` — the raw
  `_SEVERITY_W`-weighted severity the board itself uses (verified weighted at
  `submission_score.py:528`; NO re-weighting). `0.0` when non-firing (severity 0), which
  is the same firing gate the throughput axis uses, so a fully non-firing shape is
  `(0,0,0,0)` and dominated by anything.

`dominates(a, b)` becomes Pareto over all `2 * len(config.MODELS)` components: `a`
dominates `b` iff `a` ≥ `b` on every component AND `>` on at least one. Cells
`(shape_family, gen_char_bucket)` and the `frontier()`/`insert()` logic are unchanged —
only the per-element comparison widens.

### Board-density (the ship/rank key)

Add `submission_score.board_density(severity: float, gen_chars: float, model: str) ->
float`:

    board_density = 0.0 if severity <= 0 else
        (severity + config.NOVELTY_PER_CELL) / 200.0 / (gen_chars + FIXED_CHARS[model])

This is `_firing_templates`' own per-candidate board over its per-candidate cost — the
LB's value-per-char. The `/200` (competition normalization,
`_NORMALIZATION_RAW_DENOMINATOR/_NORMALIZATION_SCALE`) is a constant scale that does not
change ordering, kept only so the number reads as real board-per-char.

`Archive.ship_set()` ranks the frontier by `sum(board_density_m over config.MODELS)`
descending, returns the top `ARCHIVE_FRONTIER_CAP`. The logging champion
(`champion_by_mean_throughput`, Task 7) is renamed/retargeted to the same
board-density ranking so the reported champion is the best-shipping shape.

### Generation prompt (prompts.toml render helpers)

`_render_opro_table` and `_render_parents` show severity next to throughput per model, so
the proposer sees BOTH axes (e.g. `gpt_oss(thru=0.0058, sev=976) | gemma_4(...)`). No
scalar leaks. The OPRO row sort may key on `sum(board_density)` (display order only).

### wandb metric clarity (optimize_prompts.py logging block ~538-581)

Rename to make the two aggregation dimensions explicit (`_min_models`/`_mean_models` =
over victims; `batch_`/`best_`/`champion_` = over batch vs champion) and name units
(`board` = LB pts):

| old | new |
|---|---|
| `gpt_oss_objective` / `gemma_4_objective` | `board_gpt_oss` / `board_gemma_4` |
| `best_objective` | `best_board_min_models` |
| `batch_mean_objective` | `batch_mean_board_min_models` |
| `batch_severity_gpt_oss` / `_gemma_4` | `batch_severity_raw_gpt_oss` / `_gemma_4` |
| `best_gen_chars_bottleneck` | `champion_bottleneck_gen_chars` |
| `n_shapes` | `champion_n_shapes` |
| `refine_objective_gain` | `refine_board_gain` |
| `replay_s_{m}` | `replay_seconds_{m}` |

Additions/fixes:
- **Add `board_mean_models`** = mean of `board_gpt_oss`, `board_gemma_4` (the LB-display
  metric), logged alongside `best_board_min_models` so the coverage-bound objective and
  the display number sit side by side.
- **Add archive gauges**: `frontier_size`, `frontier_families` (distinct family count),
  `frontier_distinct_throughput`, `frontier_distinct_severity` — so the tie/monoculture
  collapse is directly visible next time.
- **Fix the stale comment** at `optimize_prompts.py:548-554`: `best_objective` is the MIN
  over columns, not the mean.

Renaming starts fresh wandb series; acceptable because the severity change forces a
restart anyway. `best_objective_name`, judge/shadow, and `_batch_score_metrics` keys that
are already clear stay as-is (rename only the misleading ones above).

## Components

- **`archive.py`** — `Elite.severity` field; `dominates` widened to 4-D; `ship_set`
  ranks by board-density; persistence round-trips the new field.
- **`submission_score.py`** — `board_density(...)`; docstrings.
- **`optimize_prompts.py`** — `_shape_elites` stores severity; ship/champion ranking;
  render helpers show severity; the wandb logging renames + additions + comment fix.
- **`prompts.toml`** — OPRO/parents framing mentions both axes (throughput + severity).
- **tests** — dominance 4-D truth table; non-firing → (0,0,0,0) dominated; ship_set
  ordered by board-density (a lean-but-weak shape stays on the frontier but ships BELOW a
  balanced high-severity shape); render shows severity; metric-rename presence.

## Migration / deploy

The live `run/blackboard.archive.jsonl` (140 throughput-only elites, no `severity` field)
is stale under the 4-D vector. `Elite(**json.loads(line))` would raise on the missing
field, and stale elites would pollute the frontier regardless. Deploy step: **delete
`run/blackboard.archive.jsonl` before restart** so the loop cold-start-re-seeds from the
incumbent with 4-D vectors. `from_jsonl` should also tolerate a missing `severity` key
(treat as all-zero → dominated) so a mixed file never crashes the loop.

## Out of scope

- Changing the blackboard MIN objective itself (still the logged coverage scalar; the
  archive Pareto path is what ships).
- Novelty as a separate axis (folded into board-density only, as the board defines it).
- The `batch_mean` naming on `_batch_score_metrics` internals beyond the table above.

## Risks

- **Ship-density vs the archive:** if `ship_set` ranks wrong, the shipped pool diverges
  from the frontier's best — covered by the ordered-ship test.
- **Persistence compatibility:** a mixed old/new archive file must not crash — covered by
  the `from_jsonl` missing-`severity` tolerance + the delete-on-deploy step.
- **gemma still binds:** 4-D exposes gemma severity but the proposer must actually raise
  it; if the frontier stays forge-monoculture after this, the next lever is proposer
  diversity pressure, not the vector.
