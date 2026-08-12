# Revert to a single shared candidate pool

**Date:** 2026-08-12
**Status:** approved (design)

## Motivation

The competition grades each victim model's LB column by running the *same* shipped
`attack.py` against that model, through an **opaque** env (`aicomp_sdk` exposes only
`seed/reset/interact/export_trace_dict`; neither the env nor `AttackRunConfig` reveals
which victim is being graded). The artifact therefore **cannot route candidates per
model** — it ships one candidate list and live-probes every candidate on whichever victim
is grading, keeping the ones that fire until the wall-clock replay budget.

The two-pool data model (commit `8bbab36`) splits authoring into a `gpt_oss` pool and a
`gemma_4` pool and **projects each pool's board in isolation** (gpt_oss board from the
gpt_oss pool only, no cross-replay). But the shipped artifact ships the **union** of both
pools, and each model-column grading probes the union. Because every probe costs a full
`reset+interact` regardless of whether it fires, each column pays to probe the pool
authored for the *other* model. The projection assumes isolation; the shipping does not.
This mismatch makes the two-pool projection (~48.9) diverge from real board behavior, and
the two-pool artifact was **never submitted**. The best *submitted* artifact — a single
flat pool of plain shapes — scored **79** on the public LB.

The pre-`8bbab36` single shared pool has no such mismatch: one pool, scored on both models,
shipped as one flat list. Every shape counts toward both columns because a plain imperative
fires on both models and a forged shape still fires on gemma (the harmony tokens are
harmless literal text there). Nothing is wasted. **This revert restores that structure.**

## Target structure (restore pre-`8bbab36`)

- **Data model** — `Submission.messages: list[Message]` (one shared pool, `min_length`/
  `max_length` = `MIN_SHIP_MESSAGES`/`MAX_SHIP_MESSAGES`). Drop the `gpt_oss`/`gemma_4`
  fields. `blackboard.Record` carries a single `messages` list again.
- **Scoring & objective** — score the one shared pool on **both** models; per-model board
  via `project_public_board(pool, model)`; the whole pool counts toward both columns (no
  per-pool isolation). **Objective = MIN over the two per-model boards, then SUM as the
  tiebreak** (the pre-migration objective, restored — a shape must fire on BOTH models to
  score; covering only one is bounded by the dead column).
- **Prompt/schema** — restore the pre-migration single-pool prompt guidance: author forge
  shapes for gpt_oss speed **and** plain shapes, all in one pool, every shape firing on
  both models; MIN-objective framing. Schema `Submission` exposes the single `messages`
  field with the pre-migration Field description.
- **Shipping** — drop `_candidates_union`; `assemble` embeds and ships the single flat
  candidate list. The shipped `attack.py` live-probe/trim logic is unchanged (it already
  probes a flat list). This removes the projection↔shipping mismatch.
- **Warm start** — archive the current two-pool `blackboard.jsonl`; cold-start the
  single-pool era seeded with the plain **79** shape as the incumbent.

## Preserved from this session (pool-agnostic, do NOT revert)

These are bug fixes / infra unrelated to the pool split and must survive the revert:

- **Strict schema built from the model via the SDK** (`type_to_response_format_param(
  SubmissionBatch)`), fixing the raw-`model_json_schema` `$ref`-with-sibling-`description`
  bug so constrained decoding actually enforces the `type` enum.
- **Dynamic schema injection** at prompt-assembly time (no cached module constant).
- **`JED_JUDGE_MODE=off`** in `run_optimizer.sh` (optimizer scores in-process, no dylan
  judge dependency).
- **Robust streaming parse** in `propose_batch_async` (raw-chunk accumulate + single final
  `model_validate_json`; not the SDK incremental parser).

## Files to change (revert surface)

Source (11): `submission.py`, `submission_score.py`, `blackboard.py`, `optimize_prompts.py`,
`assemble.py`, `victim_feedback.py`, `judge.py`, `judge_policy.py`, `judge_study.py`,
`agentic_proposer.py`, `artifact_sweep.py`; plus `scripts/cut_submission.py` and
`src/jed_attack/campaign/prompts.toml` (single-pool prompt). Tests: `tests/test_campaign.py`.

`config.MODELS` stays two-valued (both models are still scored). `MIN/MAX_SHIP_MESSAGES`,
`SHIP_CANDIDATE_CAP`, `FIXED_CHARS`, `FILL_BUDGET_CHARS` are unchanged.

## Approach

Forward-refactor, **not** a `git revert`: `8bbab36` has 14 commits on top of it (the
mean-objective flip, per-pool prompt rewrites, this session's schema fixes), so a revert
conflicts everywhere and would drop the preserved fixes. Instead, collapse the two fields
to one shared `messages` field and update each consumer, using the pre-`8bbab36` file
versions (`git show 8bbab36^:<path>`) as the reference for the single-pool shape while
keeping the preserved fixes above.

## Testing

- Unit: `Submission`/`Message` validators construct a valid single-pool submission and
  reject invariant violations; `SubmissionBatch.model_json_schema()` and the strict
  transform build; `project_public_board` on a single pool; `_batch_refine_objective`
  returns the MIN-then-SUM tuple.
- Prompt: `submission_prompt` assembles with the single `messages` schema, MIN-objective
  language, no `gpt_oss`/`gemma_4` pool references.
- Shipping: `assemble.build` embeds a flat candidate list (no per-model map, no
  `_candidates_union`); the built `attack.py` parses.
- Full `uv run pre-commit run -a` green (ruff, ty, pytest).

## Out of scope

- No change to the shipped `attack.py` probe/trim algorithm.
- No new firing mechanism or leaner-shape work (separate follow-up).
- The char-budget ceiling search (a later step, on the restored single-pool submission).
