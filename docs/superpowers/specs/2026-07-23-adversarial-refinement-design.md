# Adversarial Refinement Loop — Design

**Date:** 2026-07-23
**Status:** Approved (pending spec review)
**Component:** `src/jed_attack/campaign/optimize_prompts.py` (`worker_loop`)

## Goal

Add a per-generation, real-score hill-climb to the async optimizer: after a lane
proposes and scores a draft submission, refine the whole submission against its own
real per-message score + guardrail trace, re-score, and repeat while it strictly
improves (hard cap 4 refine rounds). Record the best submission of the rounds. This
trades search breadth (distinct round-0 proposals) for depth (a polished candidate)
exactly when depth is paying off.

## Motivation

At the current plateau (~23.48), ~87% of authored messages fire but a large fraction
score 0 — wrong verb, blocked target, missing marker — and the loop only learns from
that feedback on the *next* generation (via the global incumbent). A lane pays the full
proposal + scoring cost, gets rich per-message feedback back, and then throws the draft
away. Refining the draft in place, on the ground-truth score, extracts more value from
each proposal before moving on.

## Key decisions (locked with the user)

1. **Feedback signal = the real score in the loop.** No proxy. Each round re-scores with
   the real `score_submission` replay (the fidelity oracle). Rejected: cheap victim-probe
   (`introspect`) refinement — a single-turn chat is not the guardrailed multi-hop replay,
   so it reintroduces the est-vs-real gap we removed when the score daemon was retired.
2. **Rewrite scope = the whole submission, fed per-message feedback.** Not a mechanical
   freeze-winners merge. The model sees the whole scored draft + each message's severity
   and guardrail trace and re-authors freely. Safe because of keep-best (below): a
   regressing rewrite is discarded, never shipped, so no winner can be lost. This also
   reuses the existing `submission_prompt` incumbent-refine path verbatim.
3. **Round budget = up to 4 refine rounds while strictly improving; early-stop.** The
   first non-improving round breaks. `REFINE_MAX_ROUNDS = 4` → at most 5 scorings per
   generation, and the 5th only happens if rounds 1–4 all improved.
4. **Refine rounds include the team digest** (`top_messages` / `reasoning`) — a mid-climb
   round may pull in a teammate's winning verb; cross-pollination is the point of the team.

## Architecture

The only code change is inside `worker_loop` in `optimize_prompts.py`. Today a generation
is propose → score → record. It becomes:

```python
# Round 0 — initial proposal from the GLOBAL incumbent (unchanged from today).
prompt  = submission_prompt(board.best(), best.feedback, {}, top_messages, reasoning)
sub, rz = await propose_submission_async(prompt, provider, timeout_s)
score   = await asyncio.to_thread(score_submission, sub.messages)
best    = make_record(sub, score, rz, provider.model or provider.kind, worker_id)

# Rounds 1..REFINE_MAX_ROUNDS — refine the LOCAL draft on its own real score.
for _ in range(config.REFINE_MAX_ROUNDS):              # cap 4
    prompt      = submission_prompt(best, best.feedback, {}, top_messages, reasoning)
    r_sub, r_rz = await propose_submission_async(prompt, provider, timeout_s)
    r_score     = await asyncio.to_thread(score_submission, r_sub.messages)
    if r_score.public <= best.public:                  # strict-improvement early stop
        break
    best = make_record(r_sub, r_score, r_rz, provider.model or provider.kind, worker_id)

await board.append(best, out_dir)                      # record the best of the rounds
```

**Why this reuses everything:**

- The refine step is the *same* `submission_prompt(incumbent, feedback, introspection,
  top_messages, reasoning)` call the loop already makes across generations. The only
  difference: rounds 1+ pass the generation's LOCAL draft as the incumbent instead of the
  global `board.best()`.
- `make_record(submission, score, reasoning, model, worker)` is the adapter that turns a
  scored draft into a `blackboard.Record` whose `.feedback` is the per-message
  `[{message, type, severity, feedback}]` list (`feedback` = the guardrail trace summary).
  It is called on the draft to build the local incumbent, and again on each accepted
  refinement.
- `introspection` stays `{}` — the feedback comes from the real score's per-message trace
  (already in `make_record`'s output), not a victim probe. `introspect_worst` remains
  dormant.

**Concurrency:** rounds 1+ use the LOCAL draft as incumbent, so the climb is isolated from
another lane appending a better global best mid-climb. On completion we append the local
best; `board.best()` then reflects it iff it is globally best. No new race.

## Acceptance & stopping

- **Keep-best:** `best` advances only on strict improvement (`r_score.public > best.public`).
  A regressing whole-rewrite is discarded; the previous best stands. Winners cannot be lost.
- **Early stop:** the first non-improving round breaks the loop.
- **Cap:** `REFINE_MAX_ROUNDS = 4` → ≤5 scorings per generation.
- The appended record is the local best, which is always ≥ the round-0 draft.

## Config

- `REFINE_MAX_ROUNDS = 4` — a plain static constant in `config.py`, alongside `EVAL_HOPS`
  (a calibration knob, not hot-reloadable; changing it is a worker restart).
- `REFINE_MAX_ROUNDS = 0` disables the loop entirely — the loop body reduces to today's
  propose → score → record. This doubles as the ablation switch.

## Observability

Extend the per-generation `_log_wandb` metrics dict (currently `public`, `best_public`,
`total_hops`, `model`, `worker`) with:

- `refine_rounds` — number of refine rounds actually run this generation (0..4).
- `refine_gain` — `best.public - round0_public` (the score the climb added; 0 when no
  round improved).

These let us confirm from data whether refinement earns its extra scoring cost, and tune
`REFINE_MAX_ROUNDS` accordingly.

## Error handling

- A refine round whose proposer times out or salvages empty (`propose_submission_async`
  raises or yields an empty submission), or whose scoring raises, is caught and treated as
  "no improvement" → break, ship best-so-far. A failed refinement never costs the round-0
  draft.
- Round-0 failure is handled as today by the loop's existing per-generation try/except
  (log, sleep `_GENERATION_RETRY_S`, continue).

## Testing

Pure control-flow unit tests of the refine logic with a fake proposer (a scripted sequence
of submissions) and a fake scorer (scripted public scores) — no real models, fast and
deterministic:

1. **Monotone-improving** sequence → runs to the cap, records the last (highest).
2. **Improve-then-regress** → stops at the regression, records the peak (regression
   discarded).
3. **Round-0 already best** (round 1 scores lower) → one refine round, records round 0.
4. **Refine round raises** → records best-so-far, no crash.
5. **`REFINE_MAX_ROUNDS = 0`** → no refine rounds; behaves exactly as propose→score→record.

To keep these tests possible, the refine loop should be factored so the fake proposer/
scorer can be injected (e.g. the per-generation body extracted into a helper that takes the
propose and score callables), rather than reaching global functions directly.

## Out of scope

- Victim-probe / introspect-based refinement (rejected; proxy signal).
- Mechanical freeze-winners merge (rejected; keep-best + whole-rewrite is simpler and
  strictly safe).
- Hot-reloading `REFINE_MAX_ROUNDS` (static calibration constant, consistent with the
  caps refactor in `ce9b27b`).
