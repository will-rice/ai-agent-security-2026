# QD-Pareto-Reflective proposer

**Date:** 2026-08-12
**Status:** approved (design)

## Motivation

The proposer/optimizer is a blind greedy hill-climber: one incumbent, a scalar objective, a
raw-generation text sample, and strict-improvement acceptance. It has converged and
re-derives the same shape. Two compounding problems cause the plateau:

1. **The search is degenerate.** No population, no diversity, no directed edit signal. Every
   modern prompt-optimization method (OPRO, ProTeGi/APO, TextGrad, EvoPrompt, Promptbreeder,
   PromptAgent, GEPA, MAP-Elites/QD) adds at least one ingredient we lack: scored-trajectory
   memory, a textual "gradient," or a diverse population. See
   `~/Documents/PromptOptimization_Research_20260812/`.
2. **The objective is a miscalibrated proxy.** Backed out from config's own LB reference
   (958.5 candidates -> 86.265 public), the model bills ~11.87 s/candidate vs the real
   ~9.39 s, and uses `REPLAY_MARGIN_S=7500` vs the real 9000 -- so the projected board
   (~49) is ~2/3 of the real board (~86). A search that commits to this one scalar converges
   on the proxy's optimum, not reality's.

This design replaces the greedy loop with a **quality-diversity evolutionary loop**: the LLM
is the variation operator, reflection is the fitness feedback, and Pareto + behavioral
diversity is the selection. Pareto over the two *raw* per-model columns (never the scalar) is
what makes the search robust to the proxy bias.

## Approved decisions

- **Full Pareto over both model columns**, end to end (selection *and* shipping) -- not the
  MIN scalar.
- **Reflection on every candidate**, folded into the generation call (diagnose-then-author in
  one structured request) to bound gpt-5.5 load.
- **Built as one integrated system** (not phased).

## Architecture

### Unit and score vector

The unit of evolution is a **shape** (a `Message` template). Each scored shape carries a 2-D
score vector `(throughput_gpt_oss, throughput_gemma_4)` where

    throughput_m = 1 / (gen_chars_m + FIXED_CHARS[m])     # its per-model leanness

measured from the deterministic local replay (higher = leaner = more candidates fit that
model's budget). A shape must fire on both victims to have a finite vector (non-firing on a
model -> that component is 0, dominated). Pareto operates on these two raw columns.

### Archive (MAP-Elites + Pareto)

A behavioral grid keyed by descriptor `(family, gen_char_bucket)`:

- `family` -- a coarse structural class of the shape, one of `{plain, forge, verb_variant,
  injection_variant, deputy}`, computed by a pure classifier in `submission.py` from the
  text (presence of the harmony forge tokens, the leading verb, deputy vs exfil type, etc.).
- `gen_char_bucket` -- a quantized bucket of `max(gen_chars_gpt_oss, gen_chars_gemma_4)`
  (e.g. 25-char bins), so cells span the cost/leanness axis.

Each cell keeps the **Pareto-non-dominated** shapes for that cell (a small local frontier).
The **global Pareto frontier** across all cells is the elite set. `Record.objective` /
single-champion selection in `blackboard.py` is replaced by this archive.

Dominance: shape A dominates B iff `A.throughput_gpt >= B.throughput_gpt` and
`A.throughput_gemma >= B.throughput_gemma`, with strict inequality on at least one.

### Generation (OPRO + EvoPrompt)

Each proposer call:

1. **Sample parents** -- 1-2 elites, biased toward the global frontier and toward
   under-filled cells (novelty pressure).
2. **Build the prompt** with three new blocks (in `submission_prompt` / `prompts.toml`):
   - **OPRO scored-trajectory table** -- recent shapes sorted, each row
     `shape | gen_chars(gpt,gemma) | board(gpt,gemma)`, so the LLM optimizes against the
     landscape, not one incumbent.
   - **Parents + their cached reflections** (the GEPA diagnosis lines).
   - **EvoPrompt crossover/mutation instruction** -- "recombine parent A's forge injection
     with parent B's terser verb; or mutate parent toward its diagnosis."
3. The proposer authors N new shapes as structured output.

### Reflection (every candidate, folded into generation)

The generation structured output includes, per parent, a one-line **diagnosis** field the
model writes *before* authoring -- it reads the parent's per-model raw-generation samples and
`(board_gpt, board_gemma)` and states why a column is weak and what to trim ("gemma echoes the
harmony tokens; drop them for its shapes"). This is GEPA-style reflective feedback without a
separate round-trip, so gpt-5.5 call volume stays ~flat vs today. Diagnoses are cached on the
shape and surfaced when it is later sampled as a parent.

### Selection and shipping (Pareto end to end)

- New shapes are Pareto-inserted into their cell; the global frontier is updated.
- **Shipped pool = the global Pareto frontier.** The fill blends the frontier's shapes
  round-robin (as today), keeping both columns strong.
- The reported champion (for logging / the LB-metric view) is the frontier point maximizing
  the mean of the two columns, but the whole frontier ships.

### Companion fix: recalibrate the cost proxy

Independently of the search, recalibrate `T4_FIXED_S` / `REPLAY_MARGIN_S` (and thus
`FIXED_CHARS` / `FILL_BUDGET_CHARS`) against the LB back-out so the *local* board tracks the
*real* board. This shrinks the ~2/3 gap so the Pareto vectors reflect reality. Kept in-scope
because the search consumes these constants; a wrong proxy still misranks even under Pareto.

## Components

- **`archive.py` (new)** -- the MAP-Elites+Pareto archive: descriptor computation dispatch,
  `dominates`, `insert`, `frontier`, `ship_set`, persistence (JSONL like the blackboard).
- **`submission.py`** -- a pure `shape_family(text, type)` classifier and a
  `gen_char_bucket(gen_chars)` helper.
- **`blackboard.py`** -- champion selection delegates to `archive` (frontier/ship-set);
  Record gains the 2-D score vector + cached reflection fields; scheme tag bumps.
- **`submission_score.py`** -- expose per-shape per-model `throughput` from the existing
  `gen_chars_by_model` + `FIXED_CHARS` (no new replay).
- **`optimize_prompts.py`** -- `worker_loop` reads/writes the archive; `_score_batch`
  unchanged; new parent-sampling + OPRO-table rendering; `submission_prompt` gains the OPRO
  table, parents+reflections, and crossover instruction; the generation structured output
  gains the per-parent diagnosis field; objective/Pareto selection helpers.
- **`prompts.toml`** -- crossover/mutation and reflection prompt sections; OPRO-table framing.
- **`config.py`** -- recalibrated `T4_FIXED_S` / `REPLAY_MARGIN_S` from the LB back-out;
  archive descriptor constants (family set, bucket size, frontier caps).
- **`SubmissionBatch` schema** -- extended so a proposer reply carries the per-parent
  diagnosis + the new shapes; the strict-schema/`type_to_response_format_param` path and the
  single-shared-pool structure from the prior refactor are preserved.

## Data flow

    archive (persisted) --sample parents--> proposer prompt
      (OPRO table + parents + reflections + crossover) --author--> new shapes
      --score locally (score_submission, both models)--> 2-D vectors + raw-gen samples
      --reflect (folded in next gen)--> diagnosis
      --Pareto-insert--> archive
      --frontier--> shipped pool (assemble.build, unchanged flat single-pool shipping)

## Testing

- `archive`: `dominates` truth table; `insert` keeps only non-dominated per cell; `frontier`
  is globally non-dominated; `ship_set` returns the frontier; persistence round-trips.
- `shape_family` classifier: forge/plain/verb/injection/deputy each classify correctly;
  `gen_char_bucket` quantizes as specified.
- OPRO table renders sorted with gen_chars + both boards; no scalar leakage.
- Pareto selection: a lopsided-but-lean shape and a balanced shape both stay on the frontier;
  a strictly-dominated shape is dropped.
- Reflection: the generation structured output validates with the diagnosis field; a missing
  diagnosis degrades gracefully.
- Cost recalibration: `FIXED_CHARS`/`FILL_BUDGET_CHARS` recompute from the new constants;
  projected board on a fixture moves toward the LB back-out.
- End-to-end: generate -> score -> reflect -> insert -> ship a flat single-pool artifact;
  `uv run pre-commit run -a` green.

## Out of scope

- MCTS/beam over shapes against the local score (a later, cheap-to-add enhancement now that
  the archive exists).
- Promptbreeder self-referential mutation-prompt evolution (second-order; add once the
  population loop is stable).
- Changes to the shipped `attack.py` skeleton (the latency-precise trim already landed).
- The live margin-sweep submissions (already in flight).

## Risks

- **gpt-5.5 quota:** every-candidate reflection folded into generation keeps call volume
  flat, but the prompt grows (OPRO table + parents + reflections) -> more tokens per call.
  Mitigate with caps on table rows / parent count.
- **Pareto vs the LB metric:** shipping the whole frontier assumes the fill blend keeps the
  mean high; validate against a local projection before a real submission.
- **Archive cold-start:** seed from the current single-pool incumbents so the frontier is
  non-empty on first run.
