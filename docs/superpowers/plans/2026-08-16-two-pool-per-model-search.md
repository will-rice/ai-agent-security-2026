# Two-Pool Per-Model Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the two-pool per-model *search* (proposer authors a `gpt_oss` forge pool and a `gemma_4` plain pool independently, each scored only on its own victim) on top of the current code, so the gemma search is unconstrained by the shared-pool "must fire on both" rule.

**Architecture:** Re-widen the data model (`Submission`, `blackboard.Record`, `prompts.toml`) back to two named `list[Message]` pools; re-insert `score_pools` (scores each pool on its own model, merges public=mean) as the batch scorer; ship two per-model ship-sets through the start-probe **router** (canonicalized from `scratchpad/assemble_permodel.py`). The already-per-model 4-D Pareto archive and the token cost model are KEPT unchanged — they already speak per-model, so a gpt-only elite and a gemma-only elite never dominate each other and both survive.

**Tech Stack:** Python 3, pydantic v2 (schema is the single source of truth), `uv run`, pytest, ruff/ty via pre-commit. Reference commits: `fdd9bb5^` (pre-revert two-pool model), `03d1229` (deleted `build_permodel`).

## Global Constraints

- **KEEP these four post-revert improvements intact; the port layers on top of them:**
  - `url_scheme` authoring: `Message.url_scheme` (submission.py:105), `_URL_SCHEME_RE`, `_validate_url_scheme`, EXFIL-only check in `_valid_and_consistent`. It is a per-Message field, orthogonal to pool — thread it through per-pool `candidate_chains`.
  - Token cost model: `submission_score.throughput`/`board_density` use `gen_tokens + FIXED_TOKENS`; `config.CHARS_PER_TOKEN`/`FIXED_TOKENS`. The per-pool scorer must propagate `gen_tokens` per model, not just `gen_chars`.
  - `KaggleGemma4ToolCallParser` (src/jed_attack/harness/kaggle_parsers.py) + gemma factory in harness/models.py — untouched by this port.
  - Per-model 4-D Pareto archive (archive.py: `Elite.throughput`/`severity` dicts, `dominates` compares per model). DO NOT change `Elite` or `dominates`.
- `Message.hops` stays `le=4` (multi-post is a separate concern; keep the current bound, not pre-revert `le=1`).
- Keep `SubmissionBatch.diagnoses` (newer than pre-revert, pool-agnostic).
- Keep the current `url_scheme` guidance paragraph in prompts.toml (pool-agnostic).
- `uv run pre-commit run -a` must stay green (ruff format, ruff, ty). Fix type errors, never `# type: ignore`. Comments ≤ the project's line length (E501).
- Tests: `uv run pytest tests/test_campaign.py` green. Functional pytest style, minimal mocks.
- The optimizer process (PID confirmed at plan time) MUST be stopped before Task 1 begins — it writes the blackboard/archive under the old schema. Restart it only after the whole plan is green.
- Do not mention Claude in commits; the branch is `replay-speed-investigation` (already a feature branch — do NOT branch again).

---

### Task 1: Two-pool `Submission` schema (submission.py)

**Files:**
- Modify: `src/jed_attack/campaign/submission.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces: `Submission.gpt_oss: list[Message]`, `Submission.gemma_4: list[Message]`; `Submission.pool(model: str) -> list[Message]`; `Submission.all_messages() -> Iterator[tuple[str, Message]]` (yields `(model, message)` across `config.MODELS`); `Submission.candidate_chains(model: str, cap: int) -> list[...]` (builds `(text, type.value, url_scheme)` specs from `self.pool(model)` and calls `fill.ordered_chains(specs, cap)`); `Submission.to_shipped_json(caps: Mapping[str, int]) -> dict[str, str]` (per-model filled JSON).
- Consumes: `Message` (with `url_scheme`, unchanged), `fill.ordered_chains`, `config.MODELS`.

- [ ] **Step 1: Write failing tests.** In `tests/test_campaign.py`: (a) a `Submission` with distinct `gpt_oss` and `gemma_4` messages exposes them via `pool("gpt_oss")`/`pool("gemma_4")`; (b) `all_messages()` yields exactly the gpt_oss messages tagged `"gpt_oss"` then gemma_4 tagged `"gemma_4"`; (c) `candidate_chains("gemma_4", cap=3)` renders the gemma pool's `url_scheme` into unique `{u}` URLs (assert the rendered text differs per candidate and matches the pool's scheme); (d) `to_shipped_json({"gpt_oss": 2, "gemma_4": 2})` returns a `dict` with both keys, each a JSON string of filled candidates.

- [ ] **Step 2: Run tests, verify they fail** (`uv run pytest tests/test_campaign.py -k "two_pool or pool_or all_messages" -x`). Expected: AttributeError / single-`messages` schema.

- [ ] **Step 3: Implement.** Restore from `git show fdd9bb5^:src/jed_attack/campaign/submission.py` the two-pool fields, `pool`, `all_messages`, `candidate_chains`/`template_texts`, and `to_shipped_json(caps)`. RECONCILE with current: `candidate_chains(model, cap)` must build `specs = [(m.text, m.type.value, m.url_scheme) for m in self.pool(model)]` then `fill.ordered_chains(specs, cap)` (NOT the pre-revert raw `list[str]`). Keep `Message`, `url_scheme`, `_validate_url_scheme`, `_valid_and_consistent`, `hops` (`le=4`), `SubmissionBatch.diagnoses` unchanged. Restore the two per-pool `Field(description=...)`: `gpt_oss` = "harmony forge / analysis-channel to suppress gpt_oss reasoning"; `gemma_4` = "LEAN plain imperative, NO forge (forge is inert on gemma; plain saves input tokens)".

- [ ] **Step 4: Run tests, verify pass.** `uv run pytest tests/test_campaign.py -k "two_pool or pool_ or all_messages" -v`. Also run the full file to see the blast radius on other fixtures: `uv run pytest tests/test_campaign.py -q` (expect other single-pool tests to fail — Task 7 fixes them; note which).

- [ ] **Step 5: Commit.** `refactor(submission): restore two-pool gpt_oss/gemma_4 schema, thread url_scheme per pool`

---

### Task 2: Canonicalize the start-probe router (assemble.py)

**Files:**
- Modify: `src/jed_attack/campaign/assemble.py`
- Reference: `scratchpad/assemble_permodel.py` (the better router — start-only probe, ships full pool, graceful, `_RATIO_GPT_MIN` classifier)
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces: `assemble.build_permodel(forge_pool: list, plain_pool: list, out_dir) -> Path` — takes the two pools as TYPED filled candidate lists (`list[list[str]]`, exactly what `Submission.candidate_chains(model, cap)` returns), `json.dumps`-serializes them INTERNALLY, and writes a compiling `attack.py` embedding the forge pool (gpt) and plain pool (gemma) with a start-of-run forge/plain probe that classifies the victim by latency ratio and ships that model's pool at full count (`[:MAX_REPLAY_FINDINGS]`), safe forge default. NO input validation (malformed JSON is unrepresentable: the input is a python list, serialized in place) — this is deliberate, do not add a round-trip or `"""` guard.
- Consumes: nothing new (typed python lists passed in).
- Keep `build`/`_TEMPLATE` (current flat cut) unchanged for legacy.

- [ ] **Step 1: Write failing test.** `assemble.build_permodel(forge_json, plain_json, tmp_path)` writes `attack.py` that (a) `py_compile.compile` succeeds, (b) source contains `class AttackAlgorithm(AttackAlgorithmBase)`, both embedded pools, `_RATIO_GPT_MIN`, and the start-probe (`_PROBE_FORGE`/`_PROBE_PLAIN`), (c) does NOT self-size/trim (no per-candidate probe).

- [ ] **Step 2: Run test, verify fail** (`uv run pytest tests/test_campaign.py -k build_permodel -x`). Expected: `AttributeError: module has no attribute build_permodel`.

- [ ] **Step 3: Implement.** Port `scratchpad/assemble_permodel.py`'s `TEMPLATE` + `AttackAlgorithm` into `assemble.py` as `build_permodel(forge_json, plain_json, out_dir)`. Replace the scratchpad's file-reads (`FORGE`/`PLAIN` from `scratchpad/*.json`) with the two JSON strings passed in (`.replace("__FORGE_JSON__", forge_json).replace("__PLAIN_JSON__", plain_json)`). Assert no `"""` in the embedded JSON. Write + `py_compile` check.

- [ ] **Step 4: Run test, verify pass.** `uv run pytest tests/test_campaign.py -k build_permodel -v`.

- [ ] **Step 5: Commit.** `feat(assemble): canonicalize the start-probe per-model router as build_permodel`

---

### Task 3: Per-pool scorer `score_pools` (submission_score.py)

**Files:**
- Modify: `src/jed_attack/campaign/submission_score.py`
- Test: `tests/test_campaign.py` (there is a dormant `score_pools` test near line 2468 — restore/enable it)

**Interfaces:**
- Produces: `score_pools(submission, models=config.MODELS, guardrails=...) -> SubmissionScore` — for each model, calls `score_submission(submission.pool(model), (model,), ...)` concurrently; merges: `public = mean(public_by_model)`, `per_message` = both pools' rows concatenated in `config.MODELS` order, `public_by_model[m] = per_model[m].public_by_model[m]`, AND propagates `gen_tokens={m: per_model[m]...gen_tokens.get(m, 0.0)}` per message (NOT just gen_chars).
- Consumes: `score_submission` (unchanged; already records `gen_tokens_by_model` + `severity_by_model` per replayed model), `Submission.pool`.

- [ ] **Step 1: Write/restore failing test.** Restore the dormant test at ~test_campaign.py:2468 ("gives each per_message row exactly one model's column"): a two-pool submission scored by `score_pools` yields per_message rows where a gpt_oss-pool row has only the `gpt_oss` entry in `gen_tokens_by_model`/`severity_by_model` populated (gemma 0.0/absent) and vice versa. Add an assertion that `gen_tokens_by_model` (not only chars) is populated for the scored model.

- [ ] **Step 2: Run test, verify fail** (`uv run pytest tests/test_campaign.py -k score_pools -x`). Expected: `score_pools` not imported/defined.

- [ ] **Step 3: Implement.** Restore `score_pools` from `git show fdd9bb5^:src/jed_attack/campaign/submission_score.py` (~:598-676). RECONCILE with tokens: extend the per-message merge to carry `gen_tokens` per model exactly as it carries `gen_chars` (add `gen_tokens_by_model` propagation), so the token objective survives. Do NOT change `throughput`/`board_density`/`score_submission`.

- [ ] **Step 4: Run test, verify pass.** `uv run pytest tests/test_campaign.py -k "score_pools" -v`.

- [ ] **Step 5: Commit.** `feat(scoring): restore score_pools (per-pool, own-model-only) with token propagation`

---

### Task 4: Two-pool `Record` + router ship (blackboard.py)

**Files:**
- Modify: `src/jed_attack/campaign/blackboard.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `Submission` two-pool (Task 1), `assemble.build_permodel` typed (Task 2).
- Produces: `Record` is now a pydantic `BaseModel` (was frozen dataclass), `model_config = ConfigDict(frozen=True)`, storing the authored `submission: Submission` ONCE (pools live only on Submission — NO `gpt_oss`/`gemma_4` fields on Record). `Record.messages` back-compat property = concat of `submission.gpt_oss + submission.gemma_4` dicts. A `@model_validator(mode="before")` maps legacy flat `{"messages":[...]}` rows to `submission={"gpt_oss": <those>, "gemma_4": []}`. Persistence uses `model_dump()`/`model_validate()` (native pydantic nesting — no asdict). Ship functions (`reship`, `reship_frontier`, `_champion_candidates` callers) call `assemble.build_permodel(record.submission.candidate_chains("gpt_oss", cap), record.submission.candidate_chains("gemma_4", cap), out_dir)`. `dataclasses.replace(record, …)` → `record.model_copy(update=…)`; `asdict(record)` → `record.model_dump()`.

- [ ] **Step 1: Write failing tests.** (a) A `Record` with distinct `gpt_oss`/`gemma_4` round-trips through `to_json`/`from_json`; (b) a LEGACY flat `{"messages": [...]}` JSON loads into `gpt_oss` (gemma_4 empty) via `from_json`; (c) `Record.messages` returns the concatenation; (d) a ship call writes an `attack.py` (via `build_permodel`) embedding both pools — assert it compiles and contains both.

- [ ] **Step 2: Run tests, verify fail** (`uv run pytest tests/test_campaign.py -k "record_two_pool or legacy or reship" -x`).

- [ ] **Step 3: Implement.** Restore from `git show fdd9bb5^:src/jed_attack/campaign/blackboard.py`: `Record.gpt_oss/gemma_4`, `pool_messages`, `messages` property (concat), `from_json` legacy branch, `_champion_map(record) -> dict`. RECONCILE ship: replace flat `assemble.build(_champion_candidates(record), out_dir)` calls with `assemble.build_permodel(forge_pool, plain_pool, out_dir)` where `forge_pool` / `plain_pool` are the FILLED python candidate lists for each model — obtain them by filling each pool (the same way `Submission.candidate_chains(model, cap)` fills: `(text, type, url_scheme)` specs → `fill.ordered_chains`). build_permodel now takes TYPED lists, NOT JSON strings, and serializes internally (Task 2 was reworked to this). If `Record`/`Submission.to_shipped_json` (the dict[str,str] JSON serializer from Task 1) is now unused by the ship path, prefer calling the fill directly and REMOVE `to_shipped_json` (DRY — don't keep a stringify nobody calls); if a legacy caller still needs it, keep it. Update `reship`/`reship_frontier`.

- [ ] **Step 4: Run tests, verify pass.** `uv run pytest tests/test_campaign.py -k "record or reship or champion_map or legacy" -v`.

- [ ] **Step 5: Commit.** `refactor(blackboard): two-pool Record + ship via build_permodel router; legacy flat loads into gpt_oss`

---

### Task 5: Feed the per-model archive from two-pool scores (optimize_prompts.py)

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `score_pools` (Task 3), `blackboard.Record` two-pool (Task 4), `Submission.all_messages` (Task 1), existing `archive.Elite`/`_shape_elites` (UNCHANGED per Global Constraints).
- Produces: `_score_batch` uses `score_pools(s)`; `make_record` writes `gpt_oss=…, gemma_4=…`; `_shape_elites` iterates `submission.all_messages()` across both pools.

- [ ] **Step 1: Write failing test.** From a two-pool `Submission` (a lean gemma-plain message that fires only gemma + a forge message that fires only gpt), after `score_pools` + `_shape_elites`, BOTH a gemma-only elite (gpt throughput 0) and a gpt-only elite (gemma throughput 0) enter the archive and neither dominates the other (assert both in `archive.frontier()`). This is the core "unconstrained gemma" behavior.

- [ ] **Step 2: Run test, verify fail** (`uv run pytest tests/test_campaign.py -k "unconstrained or per_model_elite or score_batch" -x`).

- [ ] **Step 3: Implement.** Import `score_pools` (replace `score_submission` in `_score_batch`: `return [await asyncio.to_thread(score_pools, s) ...]`). Change `make_record` to construct `Record(submission=submission, ...)` (Record now stores the Submission once — Task 4). Change `_shape_elites` to iterate `submission.all_messages()` (each message already tagged by the pool/model that scored it, so the unscored axis reads 0.0). DO NOT modify `archive.Elite` or `dominates` — the port map confirms none is needed; the `.get(model, 0.0)` fallback already yields the non-firing axis.

- [ ] **Step 4: Run test, verify pass.** `uv run pytest tests/test_campaign.py -k "unconstrained or per_model_elite or score_batch or shape_elites" -v`.

- [ ] **Step 5: Commit.** `feat(optimizer): score batches per-pool, feed gpt-only + gemma-only specialists to the archive`

---

### Task 6: Per-model proposer objective (prompts.toml)

**Files:**
- Modify: `src/jed_attack/campaign/prompts.toml`

**Interfaces:** none (prompt text; the JSON SCHEMA from `SubmissionBatch` — now two-pool — drives the hard rules).

- [ ] **Step 1: Read the current objective block** (`src/jed_attack/campaign/prompts.toml` ~lines 8, 24, 34-46: "ONE shared pool … BOTH models", "OBJECTIVE: the MIN over the two per-model boards …").

- [ ] **Step 2: Replace** those lines with the pre-revert per-model framing from `git show fdd9bb5^:src/jed_attack/campaign/prompts.toml` (~lines 19-49): "Each Submission has TWO named pools -- `gpt_oss` and `gemma_4` … replays each pool ONLY on its own model (no cross-replay). Author each pool separately for its own victim." and "OBJECTIVE: … averaged over the two pools". Update the final output-format line to `{"submissions": [ {"gpt_oss": [...], "gemma_4": [...]}, ... ]}`. KEEP the current `url_scheme`/`{u}` guidance paragraph verbatim (pool-agnostic).

- [ ] **Step 3: Smoke-test the prompt loads + schema injects.** `uv run python -c "from jed_attack.campaign import optimize_prompts as o; print('prompt+schema OK')"` (imports the module that builds the prompt from prompts.toml + the two-pool SubmissionBatch schema). No traceback = pass.

- [ ] **Step 4: Commit.** `feat(prompts): per-model objective — author gpt_oss forge + gemma_4 plain pools independently`

---

### Task 7: Reconcile secondary callers + remaining tests

**Files:**
- Modify (as needed): `src/jed_attack/campaign/codex_proposer.py`, `src/jed_attack/campaign/providers.py`, `src/jed_attack/campaign/artifact_sweep.py`, `scripts/cut_submission.py`, `scripts/run_optimizer.sh`
- Test: `tests/test_campaign.py` (convert remaining single-pool fixtures)

**Interfaces:** consumes everything above; produces a fully green `pre-commit` + `pytest`.

- [ ] **Step 1: Sweep for single-`messages` assumptions.** `grep -rnE "\.messages\b|to_shipped_json\(|score_submission\(" src/jed_attack scripts` and reconcile each: `cut_submission.py` (`champion.messages` → `record.messages` concat property still works, or per-pool if it builds ship JSON → move to `caps`/`build_permodel`); `run_optimizer.sh` comment `score_submission`→`score_pools`; `providers.py` stale ":111 two-pool" comment becomes correct — verify no single-`messages` assumption in the Responses `text.format` path; `codex_proposer.py` validates via `SubmissionBatch.model_validate_json` (auto per-pool once Task 1 lands) — update any `{"submissions":[…]}` example to the two-pool object; `artifact_sweep.py` — verify no positional `to_shipped_json(cap)`.

- [ ] **Step 2: Convert remaining test fixtures.** Update single-pool `_mk_sub`/`_exfil` fixtures in `tests/test_campaign.py` (lines flagged by the port map: ~527, 868, 1024, 1330, 1343, 1471, 3902) to author both `gpt_oss` and `gemma_4`. Keep the `_exfil(..., url_scheme=...)` helper.

- [ ] **Step 3: Full green.** `uv run pytest tests/test_campaign.py -q` then `uv run pre-commit run -a`. Fix every failure (type errors included, no ignores).

- [ ] **Step 4: Commit.** `refactor(campaign): reconcile secondary callers + tests to the two-pool model`

---

## Post-plan (controller, not a task)

- **Archive the stale blackboard BEFORE restart (deploy blocker).** The existing `run/blackboard.jsonl` is 560 rows, 482 (86%) pre-`{u}` concrete-URL records that fail the new `Message` validation → `Blackboard.load` raises (`>0.5` malformed, blackboard.py:390). Rename `run/blackboard.jsonl` (and the elite `run/blackboard.archive.jsonl`) to `*.pre-twopool-bak` so the optimizer starts on a clean board (the v20 scheme bump already intends a clean champion pool; the harvest json is preserved separately). Confirm this with the user at deploy — losing warm-start history vs. clean start is their call.
- Rebuild the router artifact from the new two-pool ship path and sanity-check it compiles.
- Restart the optimizer (`scripts/run_optimizer.sh` env, `uv run python -m jed_attack.campaign.optimize_prompts`) so the per-model search runs.
- The pending router submission (55557959) and the plain-input diagnostic remain the empirical check on whether per-model pays off.
