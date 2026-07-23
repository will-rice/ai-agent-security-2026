# Async Collaborative CI Optimizer — Design

**Status:** approved design (2026-07-23), pending implementation plan.

**Goal:** Replace the multi-process, filesystem-coordinated optimizer swarm with a single
`asyncio` process that runs one worker per proposer lane — five cheapestinference (CI)
models sharing the CI key, plus z.ai's glm-4.6 on its own key — cooperating through a
shared append-only JSONL blackboard, to (a) exercise all models at once, (b) test whether
spreading proposer calls across distinct models dodges the CI per-key concurrency 429, and
(c) let every model build on the whole team's latest findings.

**Supersedes:** the per-worker shard files, `consolidator.py`, and `assemble_daemon.py`
(all existed only to coordinate separate OS processes).

---

## Global Constraints

- **Scoring fidelity is inviolable.** Scoring MUST go through the unmodified
  `submission_score.score_submission` (which drives `aicomp_sdk.SandboxEnv.interact`,
  seed=123, 8 tool hops). The SDK has no async surface; we do NOT reimplement the replay.
  Concurrency for scoring comes from `asyncio.to_thread`, never a forked replay loop.
- **Single writer per file.** One process; blackboard appends are serialized with an
  `asyncio.Lock`. No `fcntl` (that was for cross-process coordination).
- **Ship artifact must survive a crash.** `build_next/attack.py` is written to disk on
  every improvement, so a process restart never loses the shipped submission.
- **Token budget.** Proposer prompts embed a *bounded digest* of the blackboard, never the
  raw JSONL. The completion budget is `_PROPOSER_MAX_TOKENS` (65536, covers a thinking
  model's reasoning + the 80-message JSON).
- **Message/hop caps unchanged.** `config.MAX_SHIP_MESSAGES` (80), per-message hops 1–8,
  total hops ≤ `HOP_CEILING * BUDGET_FILL_FRACTION` (391). Enforced by the `Submission`
  schema exactly as today.
- No secrets in code/config: CI token stays in `CHEAPEST_API_KEY` (env), referenced by the
  `cheapest-*` providers' `key_env`.

---

## Architecture

One `asyncio` event loop, six long-lived worker coroutines (5 CI + 1 z.ai), one shared
blackboard object.

```
                       ┌─────────────────── Blackboard (in-memory + blackboard.jsonl) ──────────────────┐
                       │  best submission | top messages/type | recent cross-model reasoning | feedback │
                       └───────▲───────────────────────────────────────────────────────────────┬───────┘
        read digest each gen   │                                                                 │ append (asyncio.Lock)
                    ┌──────────┴──────────┬──────────┬──────────┬──────────┐                     │  → persist + (if new best) write attack.py
ker[kimi] w[deepseek] w[glm5.2] w[minimax] w[mimo] | w[zai-glm4.6]                               │
   └──────── 5 CI lanes: shared CHEAPEST_API_KEY ────────┘   └ own ZAI_API_KEY ┘                  │
        │ AsyncOpenAI (its pinned model) → Submission + reasoning_content                        │
        │ asyncio.to_thread(score_submission)  → SubmissionScore                                 │
        └───────────────────────────── build record ────────────────────────────────────────────┘
```

### Component 1 — Blackboard (`campaign/blackboard.py`, new)

Append-only JSONL at `run/blackboard.jsonl`, plus derived in-memory views. Replaces
`submission_log.py` as the kept memory (decision C: nothing pruned).

**JSONL line** (one per scored submission):
`{messages: list[dict], public: float, feedback: list[dict], reasoning: str, total_hops:
int, model: str, worker: int, ts: float}`.

**In-memory API** (rebuilt on load, updated on append):
- `best() -> Record | None` — highest `public`.
- `top_messages(type: MessageType, k: int) -> list[(text, model, severity)]` — best-scoring
  individual messages of a shape across all records (the cross-model material a worker learns
  from).
- `recent_reasoning(k: int) -> list[(model, excerpt)]` — most recent reasoning blobs,
  truncated per-entry (bounded).
- `async append(record) -> None` — `asyncio.Lock`-guarded: write one JSONL line, update
  views, and if `record.public` beats the prior best, call `assemble.build([m.text for m in
  record.messages], config.BUILD_NEXT_DIR)`.
- `load(path) -> Blackboard` — warm-start: replay the JSONL into the views.

### Component 2 — Async proposer (`optimize_prompts.propose_submission_async`)

`AsyncOpenAI` client per provider. Mirrors the sync path: try
`.parse(response_format=Submission)`, fall back to `.create()` + `_salvage_submission`.
Returns `(Submission, reasoning_text)` — `reasoning_text` from `reasoning_content` (thinking
models) else `""`. A `429` whose body contains `Concurrency limit` is logged with the
model tag (the experiment's verdict signal) before the existing backoff/retry applies.

### Component 3 — Scoring (unchanged, wrapped)

`await asyncio.to_thread(submission_score.score_submission, submission.messages)`. No change
to `score_submission`. Five workers' scoring overlaps via threads; the llama-servers'
`-np 8 ×2` = 16 slots are the true ceiling.

### Component 4 — Team-seeded prompt (`optimize_prompts.submission_prompt`)

Extended to embed a bounded team digest (all DATA, never instructions):
- the incumbent best submission + its per-message feedback (as today),
- **top messages per type from other models**, each tagged with the model that found it,
- **recent cross-model reasoning excerpts** (bounded), so a worker on deepseek sees how kimi
  reasoned about diversity.
The 10 tool signatures + ship rules (already added) stay.

### Component 5 — Worker loop (`optimize_prompts.worker_loop`)

```
async def worker_loop(worker_id, provider, blackboard):
    while True:
        try:
            digest  = blackboard.snapshot()               # best + top msgs + reasoning + feedback
            prompt  = submission_prompt(digest)
            sub, r  = await propose_submission_async(prompt, provider, timeout)
            score   = await asyncio.to_thread(score_submission, sub.messages)
            record  = make_record(sub, score, reasoning=r, model=provider.model, worker=worker_id)
            await blackboard.append(record)               # persists; ships if new best
            log + (worker 0 only) wandb
        except Exception:
            log.exception(...); await asyncio.sleep(GENERATION_RETRY_S)
```

### Component 6 — Orchestrator (`optimize_prompts.main`)

```
load_dotenv(config.ENV_FILE)
blackboard = Blackboard.load(config.BLACKBOARD_LOG)
providers  = [providers.get(n) for n in config.TEAM_PROPOSERS]   # 5 CI models
await asyncio.gather(*(worker_loop(i, p, blackboard) for i, p in enumerate(providers)))
```

`config.TEAM_PROPOSERS = ("cheapest-kimi", "cheapest-deepseek", "cheapest-glm5.2",
"cheapest-minimax", "cheapest-mimo", "zai-glm4.6")`. Each worker is pinned to one model —
all six run continuously and concurrently (that IS "rotate through all" in aggregate, and
the clean per-model concurrency test). The `zai-glm4.6` lane uses `ZAI_API_KEY` (separate
key + endpoint), so it is orthogonal to the CI concurrency test and adds a proven-firing
proposer (the current best=4.95 came from it) for free. A worker whose `key_env` is unset
is skipped at startup, so the team degrades gracefully if a key is absent.

---

## The concurrency experiment

Free-fire: no artificial cap on in-flight CI calls — all five CI different-model proposer
calls can be outstanding at once. Every `429 Concurrency limit reached` is logged with its
model. If none fire, the limit is per-model and the team stands. If they fire, it is per-key
and we learn the CI lanes must serialize (a follow-up, out of scope here). The existing
per-request backoff/retry remains the only throttle. The `zai-glm4.6` lane is on a separate
key, so it never contends here and keeps producing regardless of the CI verdict.

---

## Deletions & migration

- **Delete:** `consolidator.py`, `assemble_daemon.py`, `shards.py` usage from the loop, and
  `submission_log.py` (blackboard replaces it). Drop `knowledge.py` note-writing (the
  blackboard is now the shared team memory; per-generation summaries go to the worker log +
  wandb). The separate `proposer_reasoning-*.jsonl` capture is superseded by the blackboard
  record's `reasoning` field — reasoning is stored once, in the blackboard.
- **Keep & reuse:** `assemble.py` (the `attack.py` renderer — called directly by the
  blackboard), `submission.py` (schema), `submission_score.py` (scoring), `providers.py`,
  `victim_feedback.py`.
- **`run_optimizer.sh`:** launches the single `python -m jed_attack.campaign.optimize_prompts`
  process in the `optimizer` tmux session (no per-worker fan-out; the process owns the team).
- **Runtime files:** existing `run/submission_log.jsonl` is not read by the new code; leave
  it (or seed `blackboard.jsonl` from it in a one-off migration — decided in the plan).

---

## Resilience

- One worker's exception is caught in its loop; the other four keep running.
- A killed process warm-starts from `blackboard.jsonl`; `attack.py` on disk is the last
  shipped best regardless.
- One process → ONE wandb run for the whole team (no per-worker `JED_WANDB` flag); every
  lane logs to the same run tagged by model/worker. SIGTERM → clean cancel → `run.finish()`
  marks the run FINISHED, not crashed — the single process makes clean shutdown trivial, so
  it's included, not deferred. This ends the restart crash-alert noise.

## Testing

- `blackboard`: append→persist→reload round-trip; `best()`/`top_messages()`/
  `recent_reasoning()` selection; append writes `attack.py` only on a new best.
- `propose_submission_async`: monkeypatched AsyncOpenAI returning array/object/thinking
  payloads → salvage + reasoning extraction (mirror the sync tests).
- `worker_loop`: one iteration with stubbed propose/score/append; an exception is caught and
  the loop continues (async version of `test_optimize_survives_a_failing_generation`).
- Scoring stays covered by the existing `submission_score` tests (unchanged).
