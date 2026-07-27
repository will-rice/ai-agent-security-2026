# Campaign Loop Realignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the campaign loop optimize and report the live-fill/shipping objective instead of treating static public score as the only champion.

**Architecture:** Add persisted objective metrics to campaign records, rank incumbents by the throughput objective when judges are shadow/off, and expose W&B/log fields that separate static public from optimizer objective. Keep assembled `attack.py` live-validation-fill behavior unchanged, but make the campaign selector and observability align with it.

**Tech Stack:** Python stdlib, pytest, existing `jed_attack.campaign` modules, no new dependencies.

## Global Constraints

- Do not import harness modules from `src/jed_attack/submission/` or generated `dist/attack.py`.
- Keep judges shadow unless explicitly enabled by config.
- Pre-commit must pass before commit.
- Do not change SDK vendoring, `harness/runner.py`, or `harness/models.py`.

---

### Task 1: Persist optimizer objective metrics

**Files:**
- Modify: `src/jed_attack/campaign/blackboard.py`
- Modify: `src/jed_attack/campaign/optimize_prompts.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces: `Record.objective: float`, `Record.objective_tiebreaker: float`, `Record.objective_name: str`
- Consumes: existing `SubmissionScore` values from `make_record(...)`

- [ ] **Step 1: Write the failing test**

```python
def test_make_record_persists_public_throughput_objective() -> None:
    message = campaign_submission.Message(type=MessageType.EXFIL, text="x", hops=1)
    score = SubmissionScore(
        public=2.0,
        private=0.0,
        valid=True,
        fires=True,
        total_hops=1,
        public_by_model={"gpt_oss": 2.0, "gemma_4": 2.0},
        private_by_model={},
        replay_seconds={"gpt_oss": 10.0, "gemma_4": 30.0},
        per_message=[],
        feedback=[],
    )
    record = make_record(
        campaign_submission.Submission(messages=[message]),
        score,
        reasoning="",
        model="unit",
        worker=0,
    )
    assert record.objective_name == "public_raw_per_replay_s"
    assert record.objective == pytest.approx(10.0)
    assert record.objective_tiebreaker == pytest.approx(2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_campaign.py -k make_record_persists_public_throughput_objective -q`
Expected: FAIL because `Record` does not persist objective fields.

- [ ] **Step 3: Write minimal implementation**

Add the fields to `Record`, JSON serialization/deserialization, and `make_record(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_campaign.py -k make_record_persists_public_throughput_objective -q`
Expected: PASS.

### Task 2: Rank campaign incumbents by optimizer objective

**Files:**
- Modify: `src/jed_attack/campaign/blackboard.py`
- Modify: `src/jed_attack/campaign/optimize_prompts.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `Record.objective`, `Record.objective_tiebreaker`
- Produces: `Blackboard.best_objective() -> Record | None`

- [ ] **Step 1: Write the failing test**

```python
def test_blackboard_best_prefers_objective_over_static_public(tmp_path: Path) -> None:
    board = Blackboard(tmp_path / "board.jsonl")
    old = Record(public=8.0, objective=1.0, objective_tiebreaker=8.0, ...)
    new = Record(public=3.0, objective=12.0, objective_tiebreaker=3.0, ...)
    board._records.extend([old, new])
    assert board.best_objective() is new
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_campaign.py -k best_prefers_objective_over_static_public -q`
Expected: FAIL because `best_objective()` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add `Blackboard.best_objective()` and use it as the incumbent/champion when `JUDGE_MODE != "active"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_campaign.py -k best_prefers_objective_over_static_public -q`
Expected: PASS.

### Task 3: Realign W&B/log metrics

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces W&B metrics: `best_objective`, `best_objective_public`, `refine_objective_gain`, `refine_public_gain`, `batch_objective_raw_per_replay_s`

- [ ] **Step 1: Write the failing test**

```python
def test_worker_logs_objective_gain_separately_from_public_gain() -> None:
    metrics = _worker_metrics_for_test(...)
    assert metrics["refine_objective_gain"] == pytest.approx(...)
    assert metrics["refine_public_gain"] == pytest.approx(...)
    assert "refine_gain" not in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_campaign.py -k objective_gain_separately -q`
Expected: FAIL because only `refine_gain` exists.

- [ ] **Step 3: Write minimal implementation**

Compute round-0/final objective tuples and log objective/public gains separately.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_campaign.py -k objective_gain_separately -q`
Expected: PASS.

### Task 4: Validate campaign and submission isolation

**Files:**
- Modify as needed from Tasks 1-3
- Test: campaign tests, build-submission test, pre-commit

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_campaign.py -q`
Expected: PASS.

- [ ] **Step 2: Run build/submission isolation tests**

Run: `uv run pytest tests/test_build_submission.py tests/test_submission_isolated.py -q`
Expected: PASS.

- [ ] **Step 3: Run full pre-commit**

Run: `uv run pre-commit run -a`
Expected: PASS.
