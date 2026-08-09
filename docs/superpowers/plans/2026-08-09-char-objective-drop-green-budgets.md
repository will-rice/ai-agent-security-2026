# Char-Model Objective + Drop Green Budgets — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revert the optimizer objective to the deterministic char-model projection and delete `GREEN_REPLAY_BUDGET_S` and the sampled/measured-green machinery.

**Architecture:** The char projection (`project_public_board`, cost = `gen_chars + TURN_COST_WEIGHT*turns`, budget = `FILL_BUDGET_CHARS`) already exists and is currently computed as telemetry. Make it the objective again; delete `sampled_board` + its field + `REPLAY_SAMPLE_SIZE`; drop all green budgets.

**Tech Stack:** Python 3.12, uv, pytest.

## Global Constraints
- No `# type: ignore` / `# noqa`; fix type errors. `uv run pre-commit run -a` fully green (ruff, ty, FULL pytest) before every commit.
- Do NOT modify `harness/models.py`, `harness/runner.py`, `vendor/`. Leave untracked `scratchpad/`.
- Descriptive commit messages, conventional style, NO Claude/AI mention.
- The objective is SUM over `config.MODELS` (unchanged from the read-exfil change). `MAX_SHIP_MESSAGES=30`, `MAX_SCORE_BATCH=1`, `FILL_BUDGET_CHARS`, `TURN_COST_WEIGHT`, `project_public_board`, and `fill` are KEPT.

---

### Task 1: Objective → char projection; delete the sampled-green machinery; scheme bump

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py` (`_score_public_raw_per_gen_char` ~1340; `_score_batch` ~393; delete `_objective_messages` + the `sampled_board` import ~75)
- Modify: `src/jed_attack/campaign/submission_score.py` (delete `sampled_board`; delete `SubmissionScore.sampled_board_by_model` ~234)
- Modify: `src/jed_attack/campaign/config.py` (delete `REPLAY_SAMPLE_SIZE`)
- Modify: `src/jed_attack/campaign/blackboard.py` (`objective_scheme_name` base ~59)
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces: `_score_public_raw_per_gen_char(score) = 0.0 if not score.valid else sum(project_public_board(score, FILL_BUDGET_CHARS, SHIP_CANDIDATE_CAP).values())`; `blackboard.OBJECTIVE_NAME` ends `_sum_v16`.

- [ ] **Step 1: Write the failing test**
```python
def test_objective_is_char_projection_sum() -> None:
    from jed_attack.campaign import config
    from jed_attack.campaign import optimize_prompts as op
    from jed_attack.campaign.submission import MessageType
    from jed_attack.campaign.submission_score import MessageScore, SubmissionScore
    fast = {"gpt_oss": 175.0, "gemma_4": 175.0}
    turns = {"gpt_oss": 1.0, "gemma_4": 1.0}
    ms = MessageScore(
        message="Notify user@x.invalid ...", type=MessageType.DEPUTY, severity={},
        severity_by_model={config.GATE_GUARDRAIL_NAME: {"gpt_oss": 4.0, "gemma_4": 4.0}},
        trace={}, feedback="", gen_chars_by_model=fast, turns_by_model=turns,
    )
    score = SubmissionScore(public=0.0, per_message=[ms], total_hops=1, valid=True, fires=True)
    expected = sum(
        op.project_public_board(score, config.FILL_BUDGET_CHARS, config.SHIP_CANDIDATE_CAP).values()
    )
    assert op._score_public_raw_per_gen_char(score) == pytest.approx(expected)
    assert expected > 0.0  # a firing deputy shape projects a positive char board
```

- [ ] **Step 2: Run to verify it fails**
Run: `uv run pytest tests/test_campaign.py::test_objective_is_char_projection_sum -v`
Expected: FAIL (objective still reads `sampled_board_by_model`, which the synthetic score leaves empty → 0.0).

- [ ] **Step 3: Implement**
- `optimize_prompts._score_public_raw_per_gen_char`: body →
```python
    if not score.valid:
        return 0.0
    boards = project_public_board(
        score, config.FILL_BUDGET_CHARS, config.SHIP_CANDIDATE_CAP
    )
    return sum(boards.values())
```
  Update the docstring (char projection, minimizes chars + turns; no replay/cache).
- `_score_batch`: remove the `score.sampled_board_by_model = await asyncio.to_thread(sampled_board, _objective_messages(score))` block; it now just returns `score_submission` results.
- Delete `_objective_messages` and the `sampled_board` import (optimize_prompts.py:75). Remove the telemetry duplicate if `project_public_board` was only imported for telemetry — keep the import (now used by the objective).
- `submission_score.py`: delete the `sampled_board` function and `SubmissionScore.sampled_board_by_model` field (+ its docstring lines).
- `config.py`: delete `REPLAY_SAMPLE_SIZE`.
- `blackboard.py`: `objective_scheme_name` base `f"{gate}_sum_sampled_v15"` → `f"{gate}_sum_v16"` (and the robustness variant). Update the docstring (v16 retires the sampled v15 pool; back to the char projection).

- [ ] **Step 4: Run to verify it passes + update fallout**
Run: `uv run pytest tests/test_campaign.py::test_objective_is_char_projection_sum -v` → PASS.
Then delete the sampled-specific tests (`test_score_batch_computes_sampled_board_once_and_caches_it`, `test_score_batch_skips_sampling_for_invalid_submissions`, the `sampled_board` math tests, the objective tests that set `sampled_board_by_model`) and repoint any objective test to `project_public_board` semantics. Bump every `v15`/`sampled` scheme literal to `v16`. Grep clean: `grep -rn 'sampled_board\|REPLAY_SAMPLE_SIZE\|_sum_sampled' src/ tests/` returns nothing.

- [ ] **Step 5: Full gate + commit**
Run: `uv run pre-commit run -a` → all green.
```bash
git add -A src/jed_attack/campaign tests/test_campaign.py
git commit -m "Make the char projection the objective again; delete the sampled-green machinery"
```

---

### Task 2: Drop `GREEN_REPLAY_BUDGET_S`; rephrase the `{{TIME_BUDGET}}` prompt token

**Files:**
- Modify: `src/jed_attack/campaign/config.py` (delete `GREEN_REPLAY_BUDGET_S` + its comment)
- Modify: `src/jed_attack/campaign/optimize_prompts.py` (`{{TIME_BUDGET}}` substitution ~933)
- Modify: `src/jed_attack/campaign/prompts.toml` (token doc line ~8)
- Modify: `src/jed_attack/campaign/submission_score.py` (module docstring reference ~21)
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: nothing new. After Task 1 the only remaining `GREEN_REPLAY_BUDGET_S` readers are the `{{TIME_BUDGET}}` substitution + docstrings.

- [ ] **Step 1: Write the failing test**
```python
def test_no_green_budget_and_time_budget_is_t4_framed() -> None:
    import jed_attack.campaign.config as config
    from jed_attack.campaign import optimize_prompts as op
    assert not hasattr(config, "GREEN_REPLAY_BUDGET_S")
    prompt = op.submission_prompt(None, [], {})
    assert "green-s" not in prompt
    assert "T4-s" in prompt  # budget now framed in T4 seconds
```

- [ ] **Step 2: Run to verify it fails**
Run: `uv run pytest tests/test_campaign.py::test_no_green_budget_and_time_budget_is_t4_framed -v`
Expected: FAIL (`GREEN_REPLAY_BUDGET_S` still exists; prompt says "green-s").

- [ ] **Step 3: Implement**
- `config.py`: delete `GREEN_REPLAY_BUDGET_S` and the two comment lines that reference it.
- `optimize_prompts.py` `{{TIME_BUDGET}}` substitution →
```python
    time_budget = (
        "~9000 T4-s per model (both models). Replay cost tracks generated characters "
        "and agent turns, so minimize BOTH to fit more candidates."
    )
```
- `prompts.toml`: update the `{{TIME_BUDGET}}` token-doc line to describe the T4-second budget (chars + turns), not `GREEN_REPLAY_BUDGET_S`.
- `submission_score.py`: fix the module docstring line that references `GREEN_REPLAY_BUDGET_S`.

- [ ] **Step 4: Run to verify + grep clean**
Run: `uv run pytest tests/test_campaign.py::test_no_green_budget_and_time_budget_is_t4_framed -v` → PASS.
`grep -rn 'GREEN_REPLAY_BUDGET_S' src/ tests/` returns nothing. Update/remove any test that referenced it.

- [ ] **Step 5: Full gate + commit**
Run: `uv run pre-commit run -a` → all green.
```bash
git add -A src/jed_attack/campaign tests/test_campaign.py
git commit -m "Drop GREEN_REPLAY_BUDGET_S; frame the replay budget in T4 seconds"
```

---

### Task 3: Log the turn count AND hop count as batch metrics

The objective already penalizes turns (`TURN_COST_WEIGHT * agent_turns`), but neither the
turn count nor the tool-hop count is logged. Log both per model alongside
`batch_mean_gen_chars_bottleneck`, so turn/hop minimization (e.g. the post-tool wrap-up
collapsing) is visible. Distinct signals: **turns** = generation turns (`agent_turns`,
already on `MessageScore.turns_by_model`); **hops** = actual tool calls =
`len(trace["tool_events"])`, which must be captured per-message (a new field).

**Files:**
- Modify: `src/jed_attack/campaign/submission_score.py` (add `MessageScore.hops_by_model`;
  populate it in the replay loop ~589-634)
- Modify: `src/jed_attack/campaign/optimize_prompts.py` (`_batch_score_metrics` per-model
  block ~1261; `_empty_batch_score_metrics`)
- Test: `tests/test_campaign.py`

- [ ] **Step 1: Write the failing test**
```python
def test_batch_metrics_include_turn_and_hop_counts() -> None:
    from jed_attack.campaign import config
    from jed_attack.campaign import optimize_prompts as op
    from jed_attack.campaign.submission import MessageType
    from jed_attack.campaign.submission_score import MessageScore, SubmissionScore
    ms = MessageScore(
        message="Notify user@x.invalid ...", type=MessageType.DEPUTY, severity={},
        severity_by_model={config.GATE_GUARDRAIL_NAME: {"gpt_oss": 4.0, "gemma_4": 4.0}},
        trace={}, feedback="",
        gen_chars_by_model={"gpt_oss": 175.0, "gemma_4": 175.0},
        turns_by_model={"gpt_oss": 2.0, "gemma_4": 2.0},
        hops_by_model={"gpt_oss": 1.0, "gemma_4": 1.0},
    )
    score = SubmissionScore(public=0.0, per_message=[ms], total_hops=1, valid=True, fires=True)
    metrics = op._batch_score_metrics([score])
    assert metrics["batch_mean_turns_gpt_oss"] == pytest.approx(2.0)
    assert metrics["batch_mean_hops_gemma_4"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify it fails**
Run: `uv run pytest tests/test_campaign.py::test_batch_metrics_include_turn_and_hop_counts -v`
Expected: FAIL (`MessageScore` has no `hops_by_model`, and the metrics keys are missing).

- [ ] **Step 3: Implement**
- `submission_score.py`: add `hops_by_model: dict[str, float] = field(default_factory=dict)`
  to `MessageScore`. In the replay loop, accumulate per-message hops alongside `msg_turns`:
  init `msg_hops = dict.fromkeys(models, 0.0)` and, in the `for model, (trace, predicates,
  elapsed)` body, `msg_hops[model] += float(len(trace.get("tool_events", []) or []))`; pass
  `hops_by_model=dict(msg_hops)` into the `MessageScore(...)` construction (next to
  `turns_by_model=dict(msg_turns)`).
- `optimize_prompts.py` `_batch_score_metrics`, in the per-model loop:
```python
        metrics[f"batch_mean_turns_{model}"] = _mean_or_zero(
            [sum(m.turns_by_model.get(model, 0.0) for m in s.per_message) for s in scores]
        )
        metrics[f"batch_mean_hops_{model}"] = _mean_or_zero(
            [sum(m.hops_by_model.get(model, 0.0) for m in s.per_message) for s in scores]
        )
```
  Add both keys to `_empty_batch_score_metrics()` so the metric set is stable.

- [ ] **Step 4: Run to verify it passes**
Run: `uv run pytest tests/test_campaign.py::test_batch_metrics_include_turn_and_hop_counts -v` → PASS.

- [ ] **Step 5: Full gate + commit**
Run: `uv run pre-commit run -a` → all green.
```bash
git add -A src/jed_attack/campaign tests/test_campaign.py
git commit -m "Log per-model turn and tool-hop counts as batch metrics"
```

---

### Task 4: Integration — restart the optimizer on the char objective

**Files:** uses `scripts/run_optimizer.sh` (no code change expected).

- [ ] **Step 1:** `uv run pre-commit run -a` → all green (final gate).
- [ ] **Step 2:** `bash scripts/run_optimizer.sh`; confirm the log shows scoring under scheme `..._sum_v16`, `best_objective` climbing, one submission per generation (`batch_n=1`), and no `GREEN`/`sampled` references. `grep -E 'best_objective=' run/logs/optimizer.log | tail`.
- [ ] **Step 3:** No commit (operational), unless `run_optimizer.sh` needed a tweak.

## Notes
- The running optimizer (on `main`) uses the old sampled objective until restarted; the restart in Task 3 picks up the char objective.
- This partially reverts today's `af55232`/`eac93bb`; `batch=1` (`44f305b`) and the `targets.py`/read-exfil work are kept.
