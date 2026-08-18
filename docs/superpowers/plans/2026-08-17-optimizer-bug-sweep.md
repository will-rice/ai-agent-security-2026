# Optimizer Bug Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fix the cluster of gpt-column-starving bugs a 4-agent audit found in the two-pool per-model optimizer, so the MEAN objective is *safe* (the proposer can no longer abandon the gpt pool). Keep MEAN (user decision) and close every collapse path around it.

**Architecture:** `jed_attack.campaign` two-pool optimizer. Fixes span `submission.py` (schema/validators), `submission_score.py` (token projection), `optimize_prompts.py` (objective/exemplars/prompt), `archive.py`/`blackboard.py` (per-model ranking/ship), `prompts.toml`. The unifying theme: **every per-model ranking, cost, and exemplar must be per-model, and the gpt pool must structurally carry the forge.**

**Tech Stack:** Python 3, pydantic v2, `uv run`, pytest, ruff/ty. The optimizer is STOPPED (no live process to disturb). Branch `replay-speed-investigation`.

## Global Constraints

- **Keep the MEAN objective** (`_score_public_raw_per_gen_char` stays `sum(boards.values())/len(boards)`). Do NOT switch to min/blend. Make MEAN safe via the structural fixes below.
- Two victims: `config.MODELS = ("gpt_oss", "gemma_4")`. gpt needs the harmony forge (`<|channel|>analysis` tail) to stay lean; gemma runs plain. Cost scales with generated TOKENS, not chars.
- `uv run` for everything. `uv run pre-commit run -a` must end green (ruff, ruff-format, ty — fix type errors, no `# type: ignore`; comments ≤88 cols). The 2 GGUF/scheme tests (`test_score_measures_rendered_scheme`, `test_short_scheme_ships_and_fires_16_both`) fail only when a GPU process holds the card — the optimizer is stopped, so they should pass; if they fail, confirm no stray GPU process before treating them as a regression.
- Every fix ships with a regression test that would FAIL on the pre-fix code. Don't gut a test to pass.
- Do not mention Claude in commits.

---

### Task 1: Token objective + prompt de-coasting (the collapse mechanism)

**Files:** `src/jed_attack/campaign/submission_score.py`, `src/jed_attack/campaign/optimize_prompts.py`, `src/jed_attack/campaign/prompts.toml`, `src/jed_attack/campaign/submission.py`; Test: `tests/test_campaign.py`

**Interfaces produced:** a token-based board projection consumed by `_project_boards`.

**DELIBERATELY NOT DOING — no forge gate.** An earlier draft required the `<|channel|>analysis` forge tail in gpt_oss shapes. REJECTED: hardcoding forge bakes the current best-known solution into the schema and forecloses the search finding a leaner non-forge gpt shape. The collapse is prevented instead by A3 (correct token costing makes a heavy gpt shape — forge or not — score low) + B1/B2 (proposer sees gpt exemplars) + prompt de-coasting. No shape family is named or required anywhere.

- [ ] **A3 — Token-based submission objective (the primary collapse fix).** `_project_boards` (optimize_prompts.py) calls `project_public_board(score, config.FILL_BUDGET_CHARS, …)` which projects on CHARS (`_firing_templates` uses `gen_chars_by_model` + `FIXED_CHARS`). Add a token variant (or parameterize) so the SUBMISSION-selection objective projects on `gen_tokens_by_model` + `config.FIXED_TOKENS` and `config.FILL_BUDGET_TOKENS` — matching the archive axes (`board_density`) and the real per-token replay cost. This removes the systematic gemma bias (gpt-forge is char-heavy but token-light: special tokens). Keep the char path only if still used as pure telemetry; the champion/refine ranking must use tokens. Update the now-correct docstrings. Test: two shapes equal in the token model but different in chars (forge special-tokens) rank by the token cost, not chars.
- [ ] **Prompt de-coasting.** In `prompts.toml`, delete the lines that license a dead column ("a weak column only costs that pool's half of the mean, not the whole score" and similar, ~lines 43-45). In `optimize_prompts.py` incumbent block (~:1277), delete "a pool strong on its own victim is never penalized for the other pool's weakness." Replace with framing that BOTH pools must fire well (mean rewards each, and a dead column halves the score — state it as a cost to avoid, not a license).
- [ ] **type-desc fix.** `Message.type` EXFIL description (submission.py ~:64) says "one clean short http://<name>.co URL target" but the validator requires a `{u}` placeholder (no literal URL). Reword the description to reference `{u}`.
- [ ] **Drop `{{SCHEMA}}` from the prompt (rely on API structured output).** The schema is passed BOTH as `{{SCHEMA}}` text (prompts.toml:105) AND as the strict structured-output schema via the API (`codex_proposer._STRICT_SCHEMA`/`"strict": True`; `optimize_prompts` `response_format`/`type_to_response_format_param`). For the OpenAI/codex lane, constrained decoding ENFORCES structure and the strict schema CARRIES the Field descriptions (verified: descriptions survive `to_strict_json_schema` verbatim), so the prompt's JSON dump is redundant. Remove the `{{SCHEMA}}` token + its framing lines from prompts.toml (KEEP the prose objective/per-model framing) and drop the `_submission_schema_json()` substitution in optimize_prompts.py (~:1221). Keep `response_format`/`_STRICT_SCHEMA` untouched. This leans the ~13.7K-char prompt substantially. Note: the sole lane is codex-gpt55 (OpenAI Responses, strict). If a non-OpenAI "soft-following" lane is ever re-added it would need a concise prose rule summary — out of scope now; leave a one-line comment. Verify the proposer still assembles + a schema round-trips.
- [ ] Run the new tests, then `uv run pytest tests/test_campaign.py -q` (fix any fixture that now must carry the forge in its gpt_oss pool), then commit.

---

### Task 2: Per-model exemplars + delete dead global-density helpers

**Files:** `src/jed_attack/campaign/optimize_prompts.py` (`_render_opro_table`), `src/jed_attack/campaign/archive.py` (`parents`, `ship_set`), `src/jed_attack/campaign/blackboard.py` (`champion_by_board_density`); Test: `tests/test_campaign.py`

- [ ] **B1 — Per-model OPRO table.** `_render_opro_table` (optimize_prompts.py ~:1432) sorts the frontier by summed `elite_board_density` and truncates to `OPRO_TABLE_ROWS=20` → all gemma (gemma-denser), gpt exemplars vanish. Render PER MODEL: take each model's firing elites ranked by that model's `_model_density` (reuse `blackboard._model_density` or add an archive helper) and interleave/round-robin so BOTH models always appear (e.g. 10 gpt + 10 gemma). Test: a frontier with 25 gemma + 10 gpt specialists renders ≥1 gpt row.
- [ ] **B2 — Per-model parents.** `archive.parents(k)` (archive.py:123) returns `front[:k]` (fixed prefix, no rotation; the `under` under-filled fallback is dead). Return a per-model-balanced set: rank each model's frontier specialists by `_model_density` and alternate models so the proposer gets BOTH gpt and gemma parents. Either wire the `under`-filled fallback into the normal path or delete it + the false docstring claim. Test: parents from a gemma-heavy frontier still include a gpt specialist.
- [ ] **B4 — Delete dead `ship_set()`.** Repo-grep confirms no production caller (only tests + docstrings). Remove `Archive.ship_set` and its tests. It still carries the exact summed-density truncation bug `_frontier_map` was written to replace — a loaded gun.
- [ ] **B3 — `champion_by_board_density` docstring/per-model.** It's logging-only and always reports gemma (`max(frontier, key=elite_board_density)`) with a stale docstring claiming it matches the ship order. Either report per-model representatives or correct the docstring to "gemma-biased logging representative, not the ship order." (Minimal: fix the docstring.)
- [ ] Run tests + commit.

---

### Task 3: Ship robustness + schema url_scheme

**Files:** `src/jed_attack/campaign/blackboard.py` (`_ship_pools`/`reship_frontier`), `src/jed_attack/campaign/assemble.py` (`_PERMODEL_TEMPLATE` probe), `src/jed_attack/campaign/submission.py` (`_URL_SCHEME_RE`); Test: `tests/test_campaign.py`

- [ ] **C1 — Refuse to ship an empty live pool.** `_frontier_map` can yield `out["gpt_oss"]=[]` (no gpt-firing frontier elite); `_ship_pools` only logs and ships → gpt column 0 (and the router's `_FORGE` exception fallback can zero both if forge is the empty pool). In `reship_frontier`/`_ship_pools`: if any model's pool is empty, DO NOT ship the frontier — fall back to `_champion_map` (its pools are `min_length`-guaranteed non-empty), or skip the reship and keep the prior artifact. Keep the warning. Test: a frontier with only gemma-firing elites triggers the champion-fallback (or no-ship), never a `build_permodel` call with `forge=[]`.
- [ ] **C2 — Probe: `plain_t=max(ps)` + more reps.** In `assemble.py` `_PERMODEL_TEMPLATE`: `forge_t=min(fs)` biases the ratio up (safe for gpt), but `plain_t=min(ps)` biases it DOWN toward the gpt→"gemma" crater. Change to `plain_t=max(ps)` so both reductions push toward the safe forge default; a spuriously-fast plain sample can't flip gpt. Bump `_PROBE_REPS` 2→4. Update the build_permodel test's marker assertions if they pin `min(ps)`. Rebuild `run/submission_cuts/permodel_2000/attack.py`. Test: the generated template contains `plain_t = max` and `_PROBE_REPS = 4`.
- [ ] **schema — make `url_scheme` a DECLARATIVE `Field(pattern=...)`, delete the hand-rolled regex validator.** submission.py:30 `_URL_SCHEME_RE` is invoked imperatively inside `_validate_url_scheme` / `_valid_and_consistent`, so the constraint is invisible to the JSON schema (proposer only post-rejected) AND it has a real hole: `^[a-z][a-z0-9]{0,5}://.*\{h\}.*$` admits `{h}` in the PATH (`ab://host/{h}`) → all filled candidates share one domain → novelty collapses to ~1 cell → ~0 board. FIX declaratively: put the constraint ON the field — `url_scheme: str = Field(..., pattern=r"^[a-z][a-z0-9]{0,5}://[^/]*\{h\}[^/]*(/.*)?$")` (host-only `{h}`). Fold the "exactly one `{h}`" and the `_DIRTY` forbidden-substring checks into the pattern where clean (e.g. exactly-one via `[^/]*` around a single `\{h\}` in the host and no `\{h\}` elsewhere); if a residue can't be a pattern, keep only that residue in the validator. DELETE `_URL_SCHEME_RE` + the format/`{h}`-count checks in `_validate_url_scheme`; `_valid_and_consistent` keeps only genuinely CROSS-FIELD rules (text↔`{u}`, `{u}`-count==hops, deputy terms). VERIFY BOTH schema builds still construct without error: `to_strict_json_schema(SubmissionBatch)` (codex path, codex_proposer.py) and `type_to_response_format_param(SubmissionBatch)` (chat path). OpenAI STRICT mode may reject `pattern` as an unsupported keyword — if `to_strict_json_schema` raises, keep the `Field(pattern=...)` for pydantic enforcement but strip `pattern` from the strict export (whatever minimal shim the codex path needs), and note it. Test: `s://{h}` and `ab://{h}.co` pass; `ab://host/{h}` and `x://p/{h}/q` are rejected; and both schema-build calls succeed.
- [ ] **Also apply the declarative-first lens to other single-field rules** in submission.py while here: any constraint that is a pure single-field format (not cross-field) should be a `Field(...)` constraint (bounds/pattern/enum) so it lands in the schema, not an imperative check. Report which moved and which had to stay in `_valid_and_consistent` (cross-field).
- [ ] Run tests + commit.

---

### Task 4: Minor correctness cleanups

**Files:** `src/jed_attack/campaign/optimize_prompts.py`, `src/jed_attack/campaign/submission_score.py`, `src/jed_attack/campaign/submission.py`; Test: `tests/test_campaign.py`

- [ ] **E1 — Diagnosis misattribution.** `_shape_elites` (optimize_prompts.py:309) attaches `diagnoses[shape_index]` (parent-indexed diagnoses) to child shapes by global message position → a shape inherits an unrelated parent's diagnosis, resurfacing as its own in `_render_parents`. Stop attaching parent diagnoses to child shapes (store `""`), or attach only via a correct parent→child map. Test: a shape's stored diagnosis is not a mismatched parent's text.
- [ ] **E2 — Per-model `_APPROX_CHARS_PER_TOKEN`.** submission_score.py:68/372: the chars→tokens fallback divides by a global `4.0`, ignoring `config.CHARS_PER_TOKEN` (gpt 4.70, gemma 3.84). Pass `model` into `_trace_gen_tokens` and divide by `config.CHARS_PER_TOKEN[model]` (and/or make a 0 token-count on a real replay a hard error). Test: the fallback uses the per-model ratio.
- [ ] **E3 — `throughput` non-firing guard.** submission_score.py:436 returns 0 only on `inf`; a non-firing shape with finite tokens is ranked fast. Mirror `board_density`: add a severity/`fires` arg and return 0 when non-positive. Update the sole caller. Test: `throughput(finite_tokens, model, severity=0)` → 0.
- [ ] **E4 — `score_pools` per_message order.** submission_score.py:753: concatenate in `config.MODELS` order (not the `models` param) OR assert `models == config.MODELS`, so a non-default `models` arg can't silently misalign per_message vs `all_messages()`.
- [ ] **`diagnoses` bound.** submission.py: `SubmissionBatch.diagnoses: list[str]` is unbounded — add a `max_length` (e.g. = MAX submissions × a few) so the field is fully constrained.
- [ ] Run full suite + `uv run pre-commit run -a` green + commit.

---

## Post-plan (controller)
- Whole-branch review across all 4 tasks.
- Rebuild the router cut; the optimizer is stopped — restart is a separate user decision (its search now can't collapse gpt, but shapes are still floor-bound; hold for the router score first).
