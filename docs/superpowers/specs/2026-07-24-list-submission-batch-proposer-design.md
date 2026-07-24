# list[Submission] Batch Proposer + Batch Refinement + Curation Ship (Design)

**Date:** 2026-07-24
**Status:** design, pending user review

## Goal

Change each proposer generation from authoring ONE `Submission` to authoring a
`list[Submission]` (a batch) in a single, large API request — because
cheapestinference caps **concurrency per key** (one in-flight request per key) but not
request **size**. A fat request yields N submissions for the price of one in-flight
slot, where N thin requests would be serialized by the cap. Score every submission in
the batch, adapt the per-generation refinement to batch level, and ship a pool curated
across all candidates (wiring in the previously-deferred curation).

## Why

- Under the per-key concurrency cap (confirmed in `jed-model-authored-pivot`), the
  proposer is the throughput bottleneck: one submission per in-flight call. A big
  `list[Submission]` request amortizes that single slot over N submissions.
- Batches produce many diverse candidates across submissions; that is exactly the
  candidate pool `select_pool` (built in the dylan-judge feature) was designed to curate
  — so the ship path becomes `curate_from_blackboard` (novelty gate + severity rank),
  finishing the deferred wiring.

## Decisions (locked with the user)

- **Batch size N: open-ended.** The prompt asks for "as many diverse, high-quality
  submissions as you can"; N is bounded by the token budget, not a target number.
- **`_PROPOSER_MAX_TOKENS` → the model's max**, per model (a high ceiling; a model with
  a lower max gets a per-model override). Await the FULL response (higher latency
  accepted); do NOT design around truncation.
- **Refinement: adapt to batch level** — re-author the WHOLE batch against per-submission
  scores + feedback to improve the weak submissions, re-score, keep the better batch, up
  to `REFINE_MAX_ROUNDS`.
- **Scoring load: not optimized now** — score all N submissions each generation.

## Components

### 1. Batch schema (`submission.py`)

```python
class SubmissionBatch(pydantic.BaseModel):
    """A batch of independent candidate submissions authored in one request."""

    submissions: list[Submission]
```

Used as the proposer's structured-output `response_format` (its JSON schema wraps the
existing `Submission` schema in a list). The existing `Submission` (typed messages,
`MAX_SHIP_MESSAGES` cap, hop validators) is unchanged.

### 2. Batch proposer (`optimize_prompts.py`)

- `propose_batch_async(...) -> tuple[list[Submission], str]` mirrors
  `propose_submission_async` but parses a `SubmissionBatch`: structured `.parse` with
  `response_format=SubmissionBatch` first, tolerant JSON fallback (parse the array, keep
  every submission that validates). Returns the list + the backend's reasoning.
- **max_tokens = the model's max.** Add a per-model max to `providers.Provider`
  (e.g. `max_tokens: int` with a high default like 65536); the proposer passes it. A
  model that rejects the value gets its real max in the provider entry.
- Await the full streamed response — the existing streaming already has only an IDLE
  timeout (no wall-clock cap), so a long generation is never cut off; only a genuine
  stall rotates the model. No truncation-salvage reliance (kept only as a safety net).
- Prompt: `submission_prompt` gains batch framing — "author AS MANY diverse,
  high-quality submissions as fit; each is a complete `Submission`; diversity across
  submissions (different tools/framings/targets) is rewarded by the novelty judge." The
  `{{SCHEMA}}` becomes the `SubmissionBatch` schema.

### 3. `worker_loop` restructure

Per generation:
1. Read the incumbent/team digest (unchanged).
2. `propose_batch_async` → `list[Submission]` (round 0).
3. Score EVERY submission (`score_submission` per submission, off-thread).
4. **Batch refinement** (up to `REFINE_MAX_ROUNDS`): re-author the whole batch against
   each submission's per-message score + feedback, re-score, keep the better batch. The
   hill-climb metric is the batch's **mean public** over its submissions (a replay-only
   metric — no dylan dependency in the tight loop); stop at the first round that doesn't
   strictly improve it. Diversity is preserved by the prompt (asks for diverse
   submissions), then enforced at ship time by the novelty curation.
5. **Append** every submission in the kept batch to the blackboard as its own `Record`
   (the blackboard is the curation candidate pool).
6. **Ship via curation:** call `curate_from_blackboard(board, out_dir, run)` (novelty
   gate + severity rank over the blackboard's firing candidates) to rebuild the shipped
   `attack.py`. See the fallback below.

### 4. Curation ship path + fallback

`curate_from_blackboard` calls the dylan judges over HTTP. Make it **best-effort**: on
any judge/transport error, log and **skip the reship this generation** (the last-good
`attack.py` stays), so a dylan outage never stalls the optimizer. A `curate_pool`
config flag (default on) lets us disable curation and fall back to the old
`blackboard.best()` ship if needed. (The novelty/severity thresholds live in config from
the dylan feature: `NOVELTY_ADMIT_THRESHOLD`, `MAX_SHIP_MESSAGES` as the pool cap.)

### 5. wandb

Per generation add: `batch_n` (submissions authored), `batch_mean_public`,
`refine_rounds`/`refine_gain` (now batch-level), and (from curation) `pool_size` /
novelty rejects. Keep the one-run-per-team model.

## Config additions (`config.py`)

- Per-model max tokens (via `providers.Provider.max_tokens`, high default 65536).
- `CURATE_POOL = True` (ship via curation; False = legacy best-submission ship).

## Design tension (flagged, addressed)

Batch refinement optimizes **mean public** (firing) — which alone would pull the batch
toward the same high-firing shape, working against diversity. Mitigation: (a) the
proposer prompt explicitly asks for DIVERSE submissions at authoring; (b) the novelty
judge gates the SHIP pool, so even a firing-homogeneous batch is filtered for diversity
before shipping. So refine drives quality, curation drives diversity — decoupled, no
single objective conflates them. If batches still collapse to sameness, a follow-up can
add a diversity term to the refine metric.

## Non-goals

- Not optimizing scoring load (score all N; revisit if green becomes the bottleneck).
- Not changing `Submission`/`Message` schemas, `score_submission`, or the dylan judges.
- Not changing curation's `select_pool` core (already built; this only calls it).

## Testing

- `SubmissionBatch`: pydantic round-trip (a list of valid `Submission`s parses).
- `propose_batch_async`: with a stubbed stream returning a `SubmissionBatch` JSON,
  returns the list + reasoning; a tolerant-fallback array parses; a batch with one
  invalid submission drops just that one.
- `worker_loop` (async, stubbed proposer + `score_submission`): one generation scores all
  submissions, batch-refine appends the kept batch's records, and ships via a stubbed
  `curate_from_blackboard`; a stubbed curation FAILURE skips reship without crashing.
- Live end-to-end is a green step, not CI.

## Files

- Modify: `src/jed_attack/campaign/submission.py` (`SubmissionBatch`)
- Modify: `src/jed_attack/campaign/providers.py` (`Provider.max_tokens`)
- Modify: `src/jed_attack/campaign/optimize_prompts.py` (`propose_batch_async`,
  `worker_loop` restructure, batch refine, curation ship, max_tokens, wandb)
- Modify: `src/jed_attack/campaign/prompts.toml` (batch framing)
- Modify: `src/jed_attack/campaign/config.py` (`CURATE_POOL`)
- Modify: `tests/test_campaign.py`

## Self-review

- **Placeholder scan:** none — concrete schema, function signatures, loop steps.
- **Consistency:** `SubmissionBatch.submissions: list[Submission]` feeds `score_submission`
  per element; kept batch's records feed `curate_from_blackboard` → `select_pool`
  (passed-in candidates, the reuse designed earlier); ship reuses `assemble.build`.
- **Scope:** batch proposer + batch refine + score-all + curation ship — matches the
  locked decisions. Scoring-load optimization explicitly deferred.
- **Ambiguity:** refine metric = mean public (replay-only, no dylan coupling in the tight
  loop); diversity handled by prompt + novelty ship gate; curation is best-effort so a
  dylan outage never stalls the loop.
