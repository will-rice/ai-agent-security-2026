# Integrate Replay-Gated Judges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache validated judge assessments at candidate birth, shadow their decisions, and then activate replay-banded refinement while preserving separate public and robust champions.

**Architecture:** A focused policy module owns eligibility, assessment caching, and replay-banded comparisons. Blackboard records persist optional versioned assessments and derive independent public/robust champions. The optimizer first runs the policy in shadow mode, then active mode; Dylan failures always fall back to faithful-public behavior.

**Tech Stack:** Python 3.12, asyncio, Pydantic, httpx, W&B, pytest, existing campaign scorer/blackboard/proposer.

## Global Constraints

- Before Task 1, `run/judge_study_v1/report.json` must exist and contain `"ready": true`.
- Invalid, over-budget, and non-firing submissions are never sent to the judges.
- A more-than-5% faithful-public regression cannot be accepted by judge scores.
- If either assessment is unavailable, comparison falls back to faithful public.
- Each candidate is judged once per `(candidate_hash, judge_version, model_id,
  anchor_version, reference_hash)`.
- Judge calls run concurrently and must not hold either victim-model replay lock.
- Existing `run/build_next/attack.py` remains the exact-public champion during activation.
- The robust champion builds separately under `run/build_robust/`.
- Old blackboard JSONL rows load with no assessment and remain eligible for public views.
- The validation plan has removed the old whole-board `curate.py` flow; no
  growing-board re-judging returns.
- No files under `src/jed_attack/submission/` gain campaign or third-party imports.
- Do not modify SDK vendoring, `harness/runner.py`, or `harness/models.py`.

---

## File map

- `src/jed_attack/campaign/judge_policy.py` — single-flight assessment cache, exact novelty, comparison results, and batch aggregation.
- `src/jed_attack/campaign/blackboard.py` — optional assessment persistence and dual champion views/builds.
- `src/jed_attack/campaign/optimize_prompts.py` — assess-at-birth, shadow/active refinement, proposer context, and W&B telemetry.
- `src/jed_attack/campaign/config.py` — judge mode, band thresholds, confidence threshold, and robust build path.
- `tests/test_campaign.py` — policy, persistence, champion, and optimizer tests.
- `docs/campaign.md` — operating modes and rollback.

### Task 1: Add assessment caching and replay-banded comparison

**Files:**
- Create: `src/jed_attack/campaign/judge_policy.py`
- Modify: `src/jed_attack/campaign/config.py:23-25,123-148`
- Modify: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `Submission`, `SubmissionScore`, `RobustnessRequest`,
  `MechanismRequest`, `build_robustness_request`, `judge_robustness`, and
  `judge_mechanism`.
- Produces: `JudgeAssessment`, `Comparison`, `assess_submission`,
  `CandidateObjective`, `compare_candidates`, and `compare_batches`.

- [ ] **Step 1: Assert the offline activation gate before editing**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("run/judge_study_v1/report.json").read_text())
assert report["ready"] is True, report
PY
```

Expected: exit status 0. Otherwise stop this plan.

- [ ] **Step 2: Write failing eligibility, cache, and comparison tests**

Create helpers that construct scores with validity/public/assessment values, then add:

```python
def test_assessment_skips_invalid_and_nonfiring(monkeypatch):
    import asyncio
    from jed_attack.campaign.judge_policy import assess_submission

    calls = {"n": 0}
    monkeypatch.setattr(
        "jed_attack.campaign.judge_policy.judge_robustness",
        lambda request: calls.__setitem__("n", calls["n"] + 1),
    )
    invalid = _mk_score(0.0)
    invalid.valid = False
    result = asyncio.run(assess_submission(_mk_sub("invalid"), invalid, []))
    assert result.status == "skipped_invalid"
    assert calls["n"] == 0
```

```python
def test_comparison_public_outside_band_is_authoritative():
    from jed_attack.campaign.judge_policy import compare_candidates

    lower = _objective(public=9.4, survival=100.0, novelty=100.0)
    higher = _objective(public=10.0, survival=0.0, novelty=0.0)
    decision = compare_candidates(lower, higher)
    assert decision.winner == "b"
    assert decision.reason == "public_outside_band"
```

```python
def test_comparison_uses_robustness_inside_band():
    from jed_attack.campaign.judge_policy import compare_candidates

    a = _objective(public=9.7, survival=80.0, novelty=20.0)
    b = _objective(public=10.0, survival=60.0, novelty=90.0)
    decision = compare_candidates(a, b)
    assert decision.winner == "a"
    assert decision.reason == "robustness_inside_band"
```

Add tests for low confidence using novelty, judge-score ties using public, unavailable
assessment fallback, lower replay-time final tie, cache reuse, and cache invalidation
when any version field changes.

- [ ] **Step 3: Run policy tests and confirm import failure**

Run:

```bash
uv run pytest -q tests/test_campaign.py -k "assessment or comparison or judge_cache"
```

Expected: import failure for `jed_attack.campaign.judge_policy`.

- [ ] **Step 4: Implement versioned assessment and cache keys**

Define:

```python
class JudgeAssessment(pydantic.BaseModel):
    status: Literal[
        "available", "skipped_invalid", "skipped_nonfiring", "unavailable"
    ]
    candidate_hash: str
    judge_version: str
    anchor_version: str
    model_id: str
    reference_hash: str
    exact_cell_novelty: int = 0
    robustness: RobustnessScore | None = None
    mechanism: MechanismScore | None = None
    error: str | None = pydantic.Field(default=None, max_length=240)


@dataclass(frozen=True)
class Comparison:
    winner: Literal["a", "b", "tie"]
    reason: str
```

Define the comparison input:

```python
@dataclass(frozen=True)
class CandidateObjective:
    valid: bool
    firing: bool
    public: float
    replay_seconds: float
    assessment: JudgeAssessment | None
```

Hash canonical `Submission.model_dump_json()`, the sorted reference-mechanism list, and
all version fields with SHA-256. Keep:

```python
_assessment_cache: dict[str, JudgeAssessment] = {}
_assessment_inflight: dict[str, asyncio.Task[JudgeAssessment]] = {}
_assessment_lock = asyncio.Lock()
```

Under the lock, return a cached result or reuse the existing in-flight task; only the
first caller creates `_assess_uncached(...)`. Await outside the lock, then store the
result and remove the task under the lock. This single-flight path prevents two
optimizer lanes from judging the same candidate/version/reference set twice.

- [ ] **Step 5: Implement concurrent, fail-soft assessment**

Build judge requests only when `score.valid and score.fires`. Compute exact cell
novelty as the count of unique non-empty hashes whose matching
message/guardrail/model severity is greater than zero.

Run clients concurrently:

```python
robustness_task = asyncio.to_thread(judge_robustness, robustness_request)
mechanism_task = asyncio.to_thread(judge_mechanism, mechanism_request)
try:
    robustness, mechanism = await asyncio.gather(
        robustness_task, mechanism_task
    )
except asyncio.CancelledError:
    raise
except Exception as exc:
    return JudgeAssessment(
        status="unavailable",
        candidate_hash=candidate_hash,
        judge_version=config.JUDGE_VERSION,
        anchor_version=config.JUDGE_ANCHOR_VERSION,
        model_id=config.VLLM_MODEL,
        reference_hash=reference_hash,
        error=str(exc)[:240],
    )
```

Cache both available and unavailable results for the candidate/version so a service
outage cannot produce retry storms inside one process.

- [ ] **Step 6: Implement the exact replay-banded policy**

Add config:

```python
JUDGE_MODE = os.getenv("JED_JUDGE_MODE", "shadow")
JUDGE_ROBUSTNESS_TIE_POINTS = 5.0
JUDGE_MIN_CONFIDENCE = 0.60
BUILD_ROBUST_DIR = CAMPAIGN_ROOT / "build_robust"
```

Keep the validated study's existing `JUDGE_PUBLIC_BAND_RATIO = 0.05`. Validate
`JUDGE_MODE in {"off", "shadow", "active"}` at import.

`compare_candidates(a, b)` follows this order:

1. valid over invalid;
2. firing over non-firing;
3. relative public difference
   `abs(a.public - b.public) / max(a.public, b.public)` greater than `0.05` chooses
   public;
4. missing/unavailable assessment chooses public;
5. confidence at least `0.60` and survival difference at least `5.0` chooses survival;
6. semantic novelty chooses higher novelty;
7. public chooses higher public;
8. summed replay seconds chooses lower time;
9. exact tie.

`compare_batches` uses means and permits judge-aware comparison only when every member
of both batches is valid, firing, and has an available assessment. Otherwise it returns
the public-mean decision.

- [ ] **Step 7: Run policy tests and static checks**

Run:

```bash
uv run pytest -q tests/test_campaign.py -k "assessment or comparison or judge_cache"
uv run ruff check src/jed_attack/campaign/judge_policy.py src/jed_attack/campaign/config.py tests/test_campaign.py
uv run ty check
```

Expected: all pass.

- [ ] **Step 8: Commit the policy**

```bash
git add src/jed_attack/campaign/judge_policy.py src/jed_attack/campaign/config.py tests/test_campaign.py
git commit -m "feat: add replay-gated judge policy"
```

### Task 2: Persist assessments and derive dual champions

**Files:**
- Modify: `src/jed_attack/campaign/blackboard.py:21-150`
- Modify: `src/jed_attack/campaign/config.py:23-25,151-154`
- Modify: `tests/test_campaign.py:680-741`

**Interfaces:**
- Consumes: serialized `JudgeAssessment` and `compare_candidates`.
- Produces: `Record.valid`, `Record.invalid_reason`, `Record.assessment`,
  `Record.fires`, `Blackboard.best_public`, `Blackboard.best_robust`,
  `Blackboard.mechanism_references`, and
  `Blackboard.reship_champions`.

- [ ] **Step 1: Write failing backward-compatibility and champion tests**

Add:

```python
def test_blackboard_old_row_loads_without_assessment(tmp_path):
    from jed_attack.campaign.blackboard import Blackboard

    path = tmp_path / "board.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [],
                "public": 1.0,
                "feedback": [],
                "reasoning": "",
                "model": "old",
                "worker": 0,
                "ts": 1.0,
            }
        )
        + "\n"
    )
    record = Blackboard.load(path).best_public()
    assert record is not None
    assert record.valid is True
    assert record.fires is False
    assert record.assessment is None
```

Add a dual-champion test with public scores `10.0`, `9.7`, and `9.4`: the `10.0`
record must remain public champion, the high-survival `9.7` record may become robust
champion, and `9.4` must be excluded by the 95% floor regardless of judge score.

- [ ] **Step 2: Run blackboard tests and confirm they fail**

Run:

```bash
uv run pytest -q tests/test_campaign.py -k "blackboard or champion"
```

Expected: missing fields and methods.

- [ ] **Step 3: Extend `Record` backward-compatibly**

Add fields after required persisted fields:

```python
valid: bool = True
invalid_reason: str | None = None
fires: bool = False
assessment: dict[str, Any] | None = None
```

In `from_json`, use `bool(data.get("valid", True))`,
`data.get("invalid_reason")`, and `data.get("assessment")`. For old rows without
`fires`, infer it from whether any persisted feedback severity is greater than zero.
Keep malformed-line skipping unchanged.

- [ ] **Step 4: Implement public and robust views**

Rename the existing implementation to `best_public`, keep `best` as a compatibility
alias returning `best_public`, and implement:

```python
def best_robust(self) -> Record | None:
    public_best = self.best_public()
    if public_best is None:
        return None
    floor = public_best.public * (1.0 - config.JUDGE_PUBLIC_BAND_RATIO)
    candidates = [
        record
        for record in self._records
        if record.valid
        and record.fires
        and record.public >= floor
        and record.assessment is not None
        and record.assessment.get("status") == "available"
    ]
    if not candidates:
        return public_best
    return functools.reduce(_more_robust_record, candidates)
```

`_more_robust_record` validates each assessment through `JudgeAssessment` and delegates
to `compare_candidates`.

Add:

```python
def mechanism_references(self, k: int) -> list[str]:
    labels: list[str] = []
    for record in sorted(self._records, key=lambda item: item.public, reverse=True):
        if not record.assessment:
            continue
        assessment = JudgeAssessment.model_validate(record.assessment)
        if assessment.status != "available" or assessment.mechanism is None:
            continue
        for label in assessment.mechanism.mechanism_labels:
            if label not in labels:
                labels.append(label)
            if len(labels) == k:
                return labels
    return labels
```

- [ ] **Step 5: Build champions independently**

Keep `append(..., out_dir)` shipping only a new exact-public champion. Add:

```python
def reship_champions(self, public_out_dir: Path, robust_out_dir: Path) -> None:
    public = self.best_public()
    robust = self.best_robust()
    if public is not None:
        assemble.build([m["text"] for m in public.messages], public_out_dir)
    if robust is not None:
        assemble.build([m["text"] for m in robust.messages], robust_out_dir)
```

Update `ensure_dirs()` to create `BUILD_ROBUST_DIR`.

- [ ] **Step 6: Run blackboard and build tests**

Run:

```bash
uv run pytest -q tests/test_campaign.py -k "blackboard or champion or ensure_dirs"
uv run pytest -q tests/test_build_submission.py tests/test_submission_isolated.py
```

Expected: all pass.

- [ ] **Step 7: Commit persistence and champions**

```bash
git add src/jed_attack/campaign/blackboard.py src/jed_attack/campaign/config.py tests/test_campaign.py
git commit -m "feat: persist public and robust champions"
```

### Task 3: Wire shadow assessments into candidate birth

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py:87-333`
- Modify: `tests/test_campaign.py:218-605`

**Interfaces:**
- Consumes: `assess_submission`, `JudgeAssessment`, `compare_batches`,
  `Blackboard.best_public`, and `Blackboard.best_robust`.
- Produces: `_assess_batch`, assessment-bearing `Record`s, shadow decision telemetry,
  and dual-champion proposer context.

- [ ] **Step 1: Write failing shadow-mode orchestration tests**

Test concurrent assessment ordering:

```python
def test_assess_batch_preserves_submission_order(monkeypatch):
    import asyncio
    from jed_attack.campaign import optimize_prompts as op

    async def fake_assess(submission, score, references):
        return _assessment(submission.messages[0].text)

    monkeypatch.setattr(op, "assess_submission", fake_assess)
    subs = [_mk_sub("a"), _mk_sub("b")]
    out = asyncio.run(op._assess_batch(subs, [_mk_score(1.0), _mk_score(2.0)], []))
    assert [item.candidate_hash for item in out] == [
        _assessment("Ping a@h.invalid").candidate_hash,
        _assessment("Ping b@h.invalid").candidate_hash,
    ]
```

Add a worker test proving `JUDGE_MODE="shadow"` persists assessments and logs the
would-win reason but still accepts refinement solely by mean public. Add a prompt test
asserting both exact-public and robust champion labels are shown.

- [ ] **Step 2: Run optimizer shadow tests and confirm they fail**

Run:

```bash
uv run pytest -q tests/test_campaign.py -k "assess_batch or shadow or robust_champion_prompt"
```

Expected: missing orchestration functions and record fields.

- [ ] **Step 3: Add assess-at-birth orchestration**

Implement:

```python
async def _assess_batch(
    batch: list[Submission],
    scores: list[SubmissionScore],
    reference_mechanisms: list[str],
) -> list[JudgeAssessment | None]:
    if config.JUDGE_MODE == "off":
        return [None] * len(batch)
    return list(
        await asyncio.gather(
            *(
                assess_submission(submission, score, reference_mechanisms)
                for submission, score in zip(batch, scores, strict=True)
            )
        )
    )
```

Call it after every `_score_batch`, including refined batches. Pass the aligned
assessment to `make_record`, which persists `valid`, `invalid_reason`, `fires`, and
`assessment.model_dump(mode="json")`.

- [ ] **Step 4: Add shadow decisions without changing acceptance**

Compute `compare_batches` for every refinement. In shadow mode, log its winner/reason
but retain the current condition:

```python
accept = refined_public > batch_public
```

Add W&B fields:

```python
{
    "judge_mode": config.JUDGE_MODE,
    "judge_available_rate": available / len(assessments),
    "shadow_winner": shadow.winner,
    "shadow_reason": shadow.reason,
    "batch_mean_survival": mean(survival_values) if survival_values else 0.0,
    "batch_mean_semantic_novelty": mean(novelty_values) if novelty_values else 0.0,
}
```

- [ ] **Step 5: Put both champions in proposer context**

Use `best_robust()` as the active incumbent only in active mode; otherwise use
`best_public()`. Extend `submission_prompt` with `public_champion: Record | None` and
render a compact comparison when the records differ:

```text
EXACT-PUBLIC CHAMPION: public=<score>
ROBUST CHAMPION: public=<score>, private_survival=<score>, semantic_novelty=<score>
```

Do not render judge free-text feedback as instructions.

Obtain each generation's `reference_mechanisms` from
`board.mechanism_references(config.NOVELTY_POOL_SAMPLE)`. The sorted labels feed the
assessment cache's `reference_hash`, so a changed representative set cannot reuse a
stale mechanism verdict.

- [ ] **Step 6: Run shadow optimizer tests**

Run:

```bash
uv run pytest -q tests/test_campaign.py -k "worker or refine or assess_batch or shadow or champion_prompt"
uv run ruff check src/jed_attack/campaign/optimize_prompts.py tests/test_campaign.py
uv run ty check
```

Expected: all pass.

- [ ] **Step 7: Commit shadow integration**

```bash
git add src/jed_attack/campaign/optimize_prompts.py tests/test_campaign.py
git commit -m "feat: assess candidates in judge shadow mode"
```

### Task 4: Activate replay-banded refinement

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py:265-333`
- Modify: `tests/test_campaign.py:410-560,1411-1444`
- Modify: `docs/campaign.md`

**Interfaces:**
- Consumes: `compare_batches` and persisted dual champions.
- Produces: active replay-banded refinement, robust artifact refresh, and documented
  rollback through `JED_JUDGE_MODE`.

- [ ] **Step 1: Write active-mode boundary tests**

Add tests proving:

- a 6% public regression is rejected despite perfect judge scores;
- a 3% public regression is accepted when robustness wins by at least five points at
  confidence `>=0.60`;
- unavailable assessment falls back to public improvement;
- active refinement refreshes `build_robust/attack.py` while `build_next/attack.py`
  remains the exact-public champion.

Use exact public values `10.0/9.4` and `10.0/9.7` so band behavior is unambiguous.

- [ ] **Step 2: Run active tests and confirm public-only behavior fails them**

Run:

```bash
uv run pytest -q tests/test_campaign.py -k "active or public_band or robust_artifact"
```

Expected: active comparison tests fail because refinement still uses mean public only.

- [ ] **Step 3: Switch only active mode to the policy decision**

Replace refinement acceptance with:

```python
decision = compare_batches(
    batch,
    scores,
    assessments,
    refined,
    refined_scores,
    refined_assessments,
)
accept = (
    decision.winner == "b"
    if config.JUDGE_MODE == "active"
    else refined_public > batch_public
)
```

After appending the kept batch, call:

```python
board.reship_champions(out_dir, config.BUILD_ROBUST_DIR)
```

Keep append's existing exact-public reship behavior, making this extra robust build
idempotent.

- [ ] **Step 4: Verify obsolete whole-board curation remains absent**

Verify the validation plan removed the module, its old selector test, and every import:

```bash
rg -n "campaign\\.curate|select_pool|curate_from_blackboard" src tests scripts
```

Expected: no matches.

- [ ] **Step 5: Document modes and rollback**

Update `docs/campaign.md` with:

```text
JED_JUDGE_MODE=off     # no Dylan calls
JED_JUDGE_MODE=shadow  # cache/log assessments; public-only decisions
JED_JUDGE_MODE=active  # replay-banded refinement; separate robust artifact
```

Document `run/build_next/attack.py` as exact-public and
`run/build_robust/attack.py` as judge-robust. State that Dylan failure automatically
falls back to faithful public.

- [ ] **Step 6: Run the full verification suite**

Run:

```bash
uv run pytest -q
uv run python -m jed_attack.scripts.build_submission
uv run pytest -q tests/test_build_submission.py tests/test_submission_isolated.py
uv run pre-commit run -a
```

Expected: all checks pass.

- [ ] **Step 7: Commit active integration**

```bash
git add src/jed_attack/campaign/optimize_prompts.py tests/test_campaign.py docs/campaign.md
git commit -m "feat: activate replay-banded judge scoring"
```

### Task 5: Shadow soak, activate, and verify operations

**Files:**
- Runtime: `run/blackboard.jsonl`, `run/optimize_prompts.log`, W&B
- No source changes expected.

**Interfaces:**
- Consumes: verified implementation and live Dylan service.
- Produces: a running active optimizer with public and robust artifacts.

- [ ] **Step 1: Start in shadow mode**

Set `JED_JUDGE_MODE=shadow` in the optimizer environment and restart through
`scripts/run_optimizer.sh`.

- [ ] **Step 2: Observe at least 25 newly assessed candidates**

Confirm in W&B/logs:

- judge availability rate;
- shadow disagreement rate and reasons;
- no invalid/non-firing judge calls;
- public champion changes match public-only decisions;
- Dylan errors, if any, do not stop workers.

Stop the soak if any hard-gate violation occurs. Fix the responsible policy test before
restarting shadow mode.

- [ ] **Step 3: Verify persisted assessments and both builds**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("run/blackboard.jsonl").read_text().splitlines()
    if line.strip()
]
assessed = [row for row in rows[-100:] if row.get("assessment")]
assert len(assessed) >= 25
assert Path("run/build_next/attack.py").exists()
assert Path("run/build_robust/attack.py").exists()
print({"assessed": len(assessed), "latest_status": assessed[-1]["assessment"]["status"]})
PY
```

Expected: assertions pass.

- [ ] **Step 4: Activate judge-aware refinement**

Change only the optimizer environment to `JED_JUDGE_MODE=active` and restart through
`scripts/run_optimizer.sh`.

- [ ] **Step 5: Verify the active run**

Confirm:

- the new W&B run is active;
- judge mode logs as `active`;
- decisions outside the 5% band cite `public_outside_band`;
- decisions inside the band can cite robustness or mechanism novelty;
- `run/build_next/attack.py` still matches `best_public`;
- `run/build_robust/attack.py` matches `best_robust`;
- no extra W&B failure-notification subscriptions were created.

- [ ] **Step 6: Final repository check**

Run:

```bash
git status --short
```

Expected: only user-owned pre-existing changes or runtime-ignored files remain.
