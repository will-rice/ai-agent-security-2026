# Per-Model Candidate Pools — Implementation Plan

**Goal:** Replace the single shared-submission maximin objective with two independent
per-model candidate pools, a mean objective, and a deterministic per-model character
cost model — because the gateway scores each model in its own 9000s replay phase and
`attack.py` already serves a separate pool per model.

**Architecture:** The proposer's `Submission` becomes two named per-model message lists.
`score_submission` replays each list on *its own* model, producing two independent
columns. The objective is their mean (the LB metric). Ship sizing is per-model and
gemma-safe against the hard replay deadline.

**Tech:** pydantic v2, in-process llama-cpp replay, existing fill-on-dump ship path.

## Global constraints

- **Cost stays in characters.** `raw_gen_chars` (reasoning-inclusive, deterministic) is
  the x-variable. Seconds appear only to *derive* char constants. No wall-clock in the
  objective.
- **Pinned T4 constants** (`t4-cost-model-per-model` memory): gpt_oss `(a=3.9, b=0.0546)`,
  gemma_4 `(a=3.4, b=0.1052)` vs `raw_gen_chars`.
- **Hard replay deadline** (`gateway-replay-hard-fail` memory): 9000s per (model,
  guardrail); overrun ⇒ `INVALID_SUBMISSION`, no partial credit. Every ship-sizing path
  must assert projected replay < budget, target **7500s** margin.
- **Do not modify** `vendor/`, harness, `models.py`, `runner.py`.
- **Sequencing:** `min` objective stays until the two-pool schema + proposer loop land;
  flipping to mean earlier reintroduces lopsidedness.
- Standard: `uv run pre-commit run -a` green before each commit; no `# type: ignore` /
  `# noqa`; `MIN_SHIP_MESSAGES` env-seam preserved for tests.

---

### Task 1 — Per-model char constants

**Files:** Modify `src/jed_attack/campaign/config.py`

**Interfaces produced:**
- `T4_RATE_S_PER_CHAR: dict[str, float]` and `T4_FIXED_S: dict[str, float]` (reference,
  for re-derivation)
- `FIXED_CHARS: dict[str, float]` = `{gpt_oss: 71, gemma_4: 32}`
- `REPLAY_MARGIN_S: float = 7500.0`
- `FILL_BUDGET_CHARS` re-set to `{gpt_oss: 137000.0, gemma_4: 71000.0}` (`REPLAY_MARGIN_S / b`)

- [ ] Add the constants with a comment showing the derivation `FIXED_CHARS = a/b`,
  `FILL_BUDGET_CHARS = REPLAY_MARGIN_S / b`, citing the 2026-08-11 T4 calib.
- [ ] Assert-guard: `FILL_BUDGET_CHARS` keys == `MODELS`; `FIXED_CHARS` keys == `MODELS`.
- [ ] Test `tests/test_campaign.py::test_char_constants_derive_from_pinned_rates`:
  `FILL_BUDGET_CHARS[m] == pytest.approx(REPLAY_MARGIN_S / T4_RATE_S_PER_CHAR[m], rel=.02)`
  and `FIXED_CHARS[m] == pytest.approx(T4_FIXED_S[m]/T4_RATE_S_PER_CHAR[m], rel=.05)`.
- [ ] `uv run pre-commit run -a`; commit.

---

### Task 2 — Swap the per-candidate cost term to per-model chars

**Files:** Modify `src/jed_attack/campaign/submission_score.py` (`_firing_templates`)

**Interfaces consumed:** `config.FIXED_CHARS`

- [ ] In `_firing_templates`, change `cost = gen_chars + TURN_COST_WEIGHT*turns` to
  `cost = raw_gen_chars + config.FIXED_CHARS[model]`. (`raw_gen_chars` is already the
  per-message `gen_chars_by_model` value — reasoning-inclusive.)
- [ ] Remove `TURN_COST_WEIGHT` from the cost path (keep the constant only if still logged;
  otherwise delete and its config entry).
- [ ] Test `test_firing_template_cost_is_raw_chars_plus_fixed`: a stubbed MessageScore with
  `gen_chars_by_model={gpt_oss: 145}` yields cost `145 + 71`; gemma `123 + 32`.
- [ ] Confirm `project_public_board` (already per-model `budget_chars`) now round-robins on
  the new cost with no other change. Update its docstring.
- [ ] `pre-commit`; commit. *(Objective still `min` here — safe, char-only recalibration.)*

---

### Task 3 — Proposer schema: two named per-model pools

**Files:** Modify `src/jed_attack/campaign/submission.py`, `submission_score.py`

**Interfaces produced:**
- `Submission { gpt_oss: list[Message], gemma_4: list[Message] }`, each
  `Field(min_length=config.MIN_SHIP_MESSAGES, max_length=config.MAX_SHIP_MESSAGES)`
- `Submission.pool(model) -> list[Message]`
- `Submission.to_shipped_json(caps: dict[str,int]) -> dict[str, str]` (per-model filled JSON)

**Design notes:**
- Two *named* fields (not a dict) so `model_json_schema()` emits `minItems`/`maxItems` per
  pool and constrained decoding gets a concrete per-pool target. "Both pools non-empty"
  becomes structural — the runtime both-non-empty gate disappears.
- `SubmissionBatch { submissions: list[Submission] }` unchanged in shape; each Submission
  now carries both pools.
- Field names must equal `config.MODELS` members exactly (`gpt_oss`, `gemma_4`) so
  `pool(model)` and per-model scoring can key by model string.

- [ ] Rewrite `Submission`: two lists, `ConfigDict(extra="forbid")`, per-list length Fields.
- [ ] Add `pool(self, model)` and rework `template_texts`/`_fill_templates`/
  `candidate_chains`/`to_shipped_json` to operate per pool and return per-model results.
- [ ] Update the shipped `attack.py` embed path: it now embeds a `{model: [candidates]}`
  map and serves `pool` by `_model_hint`. (Fill logic unchanged per pool.)
- [ ] Tests: schema has `minItems` on both `gpt_oss` and `gemma_4`; a Submission missing
  either pool (or under floor) fails validation; `to_shipped_json` returns both keys with
  the right caps. Update `tests/conftest.py` helpers that build single-list submissions.
- [ ] Update every test fixture constructing `Submission(messages=...)` → two lists.
- [ ] `pre-commit`; commit.

---

### Task 4 — Per-model scoring + proposer loop

**Files:** Modify `submission_score.py`, `optimize_prompts.py`, `blackboard.py`,
`prompts.toml`

**Interfaces produced:**
- `score_submission` replays each pool on *its own* model only → `public_by_model` from
  per-pool columns (no cross-replay).
- Blackboard tracks a champion per model column; record carries both pools.

- [ ] `score_submission`: iterate `for model in MODELS: replay Submission.pool(model) on
  model`. Each pool's messages score only their model's column. Drop the both-models
  replay of a single list.
- [ ] `prompts.toml`: fork the incumbent + team + feedback blocks per pool
  (`{{INCUMBENT_GPT}}`, `{{INCUMBENT_GEMMA}}` …); the schema (two lists) drives authoring.
  Keep the "follow the schema / push toward the cap" framing per pool.
- [ ] `blackboard.py`: per-model diversity term; per-column best tracking.
- [ ] Tests: a Submission whose `gpt_oss` pool fires and `gemma_4` pool is nonfiring scores
  a high gpt column and ~0 gemma column (no cross-credit); diversity is per-pool.
- [ ] `pre-commit`; commit. *(Objective still `min` — flips next task.)*

---

### Task 5 — Mean objective (flip from min)

**Files:** Modify `blackboard.py` (`_objective_key`, scheme name), `optimize_prompts.py`
(wandb)

- [ ] `_objective_key`: primary = `mean(gpt_col, gemma_col)` (= the LB). Bump
  `objective_scheme_name` (e.g. `permodel_mean_v18`). Both-non-empty is structural now.
- [ ] wandb: log `gpt_oss_column`, `gemma_4_column`, and the mean; drop the maximin metric.
- [ ] Tests: `test_objective_is_mean_over_model_columns`; a record strong on gpt and weak
  on gemma now out-ranks one that is mediocre on both (min would have tied them).
- [ ] `pre-commit`; commit. **Restart the optimizer** on the new scheme.

---

### Task 6 — Gemma-safe ship sizing + hard-deadline assertion

**Files:** Modify `run/calib/attack.py` + shipped `attack.py`, `submission.py`

- [ ] Ship count per pool = `FILL_BUDGET_CHARS[model] / cost_chars(slowest firing template
  in pool)`, i.e. sized to the pool's own budget and slowest shape.
- [ ] Assert `projected_replay_s(pool, model) < config.REPLAY_MARGIN_S` before ship; raise
  a concise error otherwise (no silent over-ship).
- [ ] Test: a pool of slow templates caps the ship count so projected replay < 7500s; a
  fast pool ships more.
- [ ] `pre-commit`; commit.

---

### Task 7 — LB validation (provisional-budget close-out, Decision C)

**Files:** `scripts/score_artifact.py` (read-only analysis)

- [ ] After the next rerun-**surviving** submission, back out `fired = LB_col × 200 / 18`
  per column; compare to the projected fired-count. Log the ratio per model.
- [ ] If reruns still fail or projection diverges >10%, adjust `REPLAY_MARGIN_S` and
  re-derive `FILL_BUDGET_CHARS`. Record the surviving-budget number in
  `t4-cost-model-per-model` memory.

---

## Self-review checklist

- [ ] Schema (Task 3) precedes proposer loop (4) precedes mean flip (5) — dependency held.
- [ ] `min`→mean happens exactly once, after two pools exist.
- [ ] Every ship path asserts projected replay < margin (hard-fail guard).
- [ ] No `FILL_BUDGET_CHARS` value stated in candidates anywhere; all in chars/seconds.
- [ ] Pinned constants live in one place (`config.py`) + memory; not duplicated.

## Open (Decision C)

Real surviving replay budget + margin — provisional 7500s until Task 7 measures it.
