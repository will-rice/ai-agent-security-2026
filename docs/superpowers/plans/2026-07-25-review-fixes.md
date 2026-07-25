# Operational Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the five review findings covering optimizer shutdown, vLLM
structured output and model propagation, whole-batch refinement, and forbidden
inline lint/type suppressions.

**Architecture:** Keep the fixes at their existing boundaries: shell launchers own
process lifetime and environment propagation, the correlation client mirrors the
judge service's proven OpenAI request, and the optimizer renders every scored batch
member into the refinement prompt while continuing to select batches by mean public
score. Regression tests execute shell artifacts with controlled fake commands and
exercise Python behavior through public call boundaries.

**Tech Stack:** Bash, Python 3.12, asyncio, OpenAI Python client, Pydantic, pytest,
ruff, ty.

## Global Constraints

- Do not modify SDK vendoring, `harness/runner.py`, or `harness/models.py`.
- Never add `# type: ignore` or `# noqa` suppressions.
- Preserve all unrelated tracked and untracked workspace changes.
- Run `uv run python -m pytest -q` after code changes.
- Run `uv run pre-commit run -a` before any commit; do not create a commit unless
  the user asks for one.

---

### Task 1: Graceful optimizer shutdown

**Files:**
- Modify: `scripts/run_optimizer.sh`
- Create: `tests/test_operational_scripts.py`

**Interfaces:**
- Consumes: `JED_OPTIMIZER_STOP_TIMEOUT_S`, defaulting above the 300-second scoring
  budget.
- Produces: a TERM/wait/KILL sequence that does not kill a scoring call which exits
  after the former 15-second limit.

- [ ] **Step 1: Write the failing test**

Create fake `pkill`, `pgrep`, `sleep`, and `tmux` executables in a temporary `PATH`.
Make `pgrep` report the optimizer alive for 16 checks and gone on the 17th, execute
`scripts/run_optimizer.sh`, and assert the fake `pkill` log contains `-TERM` but not
`-KILL`.

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_operational_scripts.py::test_optimizer_waits_for_inflight_score_before_kill -q`

Expected: FAIL because the current 15-iteration loop reaches `pkill -KILL`.

- [ ] **Step 3: Write minimal implementation**

Set:

```bash
STOP_TIMEOUT_S="${JED_OPTIMIZER_STOP_TIMEOUT_S:-330}"
for _ in $(seq 1 "$STOP_TIMEOUT_S"); do
```

Keep the existing final KILL fallback for a genuinely stuck process.

- [ ] **Step 4: Run test to verify it passes**

Run the focused pytest node from Step 2. Expected: PASS.

### Task 2: Direct vLLM structured output and optional imports

**Files:**
- Modify: `scripts/judge_correlation.py`
- Modify: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `severity_messages(...)` and `SeverityScore.model_json_schema()`.
- Produces: an OpenAI chat request with
  `response_format={"type": "json_schema", "json_schema": {"name": "judge",
  "schema": ...}}`.

- [ ] **Step 1: Write the failing test**

Load `scripts/judge_correlation.py`, replace its `OpenAI` constructor with a fake
client returning valid severity JSON, call `_judge_severity`, and assert the captured
request contains the same `response_format` contract as `judge_service._vllm_json`.

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_campaign.py::test_correlation_judge_uses_vllm_json_schema -q`

Expected: FAIL because the current request sends `extra_body.guided_json`.

- [ ] **Step 3: Write minimal implementation**

Replace `extra_body` with the standard `response_format` payload and type the messages
as `list[ChatCompletionMessageParam]`. Load optional matplotlib modules through
`importlib.import_module`, cast them to small protocols defining only `use`, plotting,
label, title, and `savefig`, and remove both inline type suppressions.

- [ ] **Step 4: Replace the forbidden lambda**

In `test_select_pool_gates_novelty_and_ranks_severity`, replace the `c = lambda ...`
assignment and `# noqa: E731` with a named nested `candidate(...) -> Candidate`
helper.

- [ ] **Step 5: Run focused tests and checks**

Run the new test, `uv run ruff check scripts/judge_correlation.py
tests/test_campaign.py`, and `uv run ty check scripts/judge_correlation.py
tests/test_campaign.py`. Expected: all pass with no cited suppressions.

### Task 3: vLLM model propagation to the judge service

**Files:**
- Modify: `scripts/serve_dylan_judges.sh`
- Modify: `tests/test_operational_scripts.py`

**Interfaces:**
- Consumes: caller-provided `VLLM_MODEL` or the existing default model.
- Produces: a generated judge-service launcher which exports the exact model served
  by the generated vLLM launcher, independent of tmux's environment.

- [ ] **Step 1: Write the failing test**

Run `serve_dylan_judges.sh` with a temporary `HOME`, `TMPDIR`, fake tmux/sleep
commands, and `VLLM_MODEL=example/override`. Then execute the generated judge-service
launcher in an environment where `VLLM_MODEL` is absent, using a fake `uv` executable
that records the variable. Assert the recorded value is `example/override`.

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_operational_scripts.py::test_judge_service_launcher_exports_selected_model -q`

Expected: FAIL because the generated launcher currently depends on inherited tmux
state.

- [ ] **Step 3: Write minimal implementation**

Honor `${TMPDIR:-/tmp}` for generated launchers/logs and embed:

```bash
export VLLM_MODEL="$MODEL"
```

in the judge-service launcher before `exec uv run uvicorn ...`.

- [ ] **Step 4: Run test to verify it passes**

Run the focused pytest node from Step 2. Expected: PASS.

### Task 4: Whole-batch refinement context

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py`
- Modify: `tests/test_campaign.py`

**Interfaces:**
- Consumes: every `(Submission, SubmissionScore)` pair in the currently kept batch.
- Produces: a refinement prompt containing every submission's public score,
  per-message feedback, and typed messages; batch acceptance remains based on mean
  public score.

- [ ] **Step 1: Write the failing test**

Call `_refine_batch` with two distinct submissions and scores, capture the first
refinement prompt through a fake `propose_batch_async`, and assert both submission
texts and both feedback strings are present.

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_campaign.py::test_refine_prompt_contains_entire_batch -q`

Expected: FAIL because only the highest-scoring submission is currently rendered.

- [ ] **Step 3: Write minimal implementation**

Add an optional `incumbent_batch: list[blackboard.Record]` input to
`submission_prompt`. Render it with a focused `_render_incumbent_batch` helper which
labels the mean and each member, delegates each member's details to the existing
incumbent renderer, and asks the proposer to improve the whole batch. In
`_refine_batch`, construct records for every current batch member and pass all of
them on each round.

- [ ] **Step 4: Run refinement tests**

Run:
`uv run pytest tests/test_campaign.py -k 'refine or submission_prompt' -q`

Expected: PASS.

### Task 5: Repository verification

**Files:**
- Verify all files changed above.

**Interfaces:**
- Consumes: completed fixes.
- Produces: evidence that targeted behavior, the full suite, formatting, lint, and
  type checking all pass.

- [ ] **Step 1: Run focused regressions**

Run:
`uv run pytest tests/test_operational_scripts.py tests/test_campaign.py -q`

- [ ] **Step 2: Run the full mechanics suite**

Run: `uv run python -m pytest -q`

- [ ] **Step 3: Run the repository gate**

Run: `uv run pre-commit run -a`

Expected: ruff, ty, and pytest all green.
