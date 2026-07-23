# Adversarial Refinement Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-generation real-score hill-climb to the async optimizer's `worker_loop`: after proposing and scoring a draft, refine the whole submission against its own real per-message score, re-score, and repeat while it strictly improves (cap 4 rounds), recording the best.

**Architecture:** The change is confined to `worker_loop` in `optimize_prompts.py`. Round 0 is today's propose→score (from the global incumbent). Rounds 1..N re-run the *same* `submission_prompt` with the generation's local draft as the incumbent (via `make_record`), keeping the best on strict improvement and early-stopping otherwise. One record is appended per generation, as today.

**Tech Stack:** Python 3, asyncio, pydantic, pytest. Design spec: `docs/superpowers/specs/2026-07-23-adversarial-refinement-design.md`.

## Global Constraints

- **Real-score signal only.** Every round re-scores with `score_submission` (the real replay). No proxy, no victim probe; `introspection` stays `{}`.
- **Whole-submission rewrite, reusing `submission_prompt`.** Rounds 1+ pass the local draft `Record` (from `make_record`) as the incumbent. No new prompt, no mechanical merge.
- **Keep-best, strict improvement.** The local best advances only when `r_score.public > local_best.public`. Regressing rounds are discarded.
- **Cap 4, early-stop.** `for _ in range(config.REFINE_MAX_ROUNDS)`; the first non-improving round `break`s. At most 5 scorings/generation.
- **Team digest included** in refine rounds (`top_messages` / `reasoning`).
- **`REFINE_MAX_ROUNDS` is a plain static constant** in `config.py` (like `EVAL_HOPS`); `0` disables refinement (loop reduces to today's propose→score→record).
- **`CancelledError` must always propagate** — both the outer per-generation handler and the new inner refine-round handler must re-raise it (only non-cancel exceptions are swallowed).
- **One append per generation** (the local best) — blackboard growth unchanged.
- **wandb** gains `refine_rounds` and `refine_gain`; `public`/`best_public`/`total_hops`/`model`/`worker` stay.
- Pre-commit (`uv run pre-commit run -a`) must pass: ruff, ty, pytest.

---

## File Structure

- `src/jed_attack/campaign/config.py` — add the `REFINE_MAX_ROUNDS` constant next to `EVAL_HOPS`.
- `src/jed_attack/campaign/optimize_prompts.py` — replace the body of `worker_loop`'s `try:` block with the round-0 + refine-loop version. Signature unchanged.
- `tests/test_campaign.py` — add refine-loop control-flow tests using the existing `monkeypatch.setattr(op, ...)` injection pattern.

This is one cohesive change to one function plus its constant and tests, so it is a single task with a TDD step sequence.

---

### Task 1: Adversarial refinement loop in `worker_loop`

**Files:**
- Modify: `src/jed_attack/campaign/config.py` (add `REFINE_MAX_ROUNDS`)
- Modify: `src/jed_attack/campaign/optimize_prompts.py:155-205` (`worker_loop` body)
- Test: `tests/test_campaign.py` (add 5 tests after `test_worker_loop_appends_then_survives_failure`)

**Interfaces:**
- Consumes (all existing, unchanged signatures):
  - `submission_prompt(incumbent, feedback, introspection, top_messages, reasoning) -> str`
  - `propose_submission_async(prompt, provider, idle_timeout_s) -> tuple[Submission, str]`
  - `score_submission(messages, models=config.MODELS) -> SubmissionScore` (called via `asyncio.to_thread`; module global so `monkeypatch.setattr(op, "score_submission", ...)` works)
  - `make_record(submission, score, reasoning, model, worker) -> blackboard.Record` (Record has `.public: float`, `.feedback: list[dict]`, `.messages: list[dict]`)
  - `board.best() -> Record | None`, `board.top_messages(type, k)`, `board.recent_reasoning(k)`, `board.append(record, out_dir)`
  - `_log_wandb(run, metrics)`, `_TEAM_TOP_K`, `_TEAM_REASONING_K`, `_GENERATION_RETRY_S`, `SubmissionScore` (fields `.public`, `.total_hops`, `.per_message`)
- Produces: `worker_loop` with the same signature; a new `config.REFINE_MAX_ROUNDS: int`.

---

- [ ] **Step 1: Add the config constant**

In `src/jed_attack/campaign/config.py`, immediately after the `EVAL_HOPS = 8` block (around line 52), add:

```python
# Adversarial refinement: max per-generation hill-climb rounds. After proposing and
# scoring a draft, the lane re-authors the whole submission against its own real
# per-message score + guardrail trace, re-scores, and repeats while the public score
# strictly improves, up to this many rounds (=> at most REFINE_MAX_ROUNDS + 1 scorings
# per generation). A static calibration knob (like EVAL_HOPS), not hot-reloadable.
# Set to 0 to disable refinement entirely (propose -> score -> record).
REFINE_MAX_ROUNDS = 4
```

- [ ] **Step 2: Write the first failing test (monotone-improving runs to the cap)**

In `tests/test_campaign.py`, add these helpers and the first test after `test_worker_loop_appends_then_survives_failure` (line ~254). The helpers are reused by all refine tests.

```python
def _mk_score(public: float) -> "SubmissionScore":
    from jed_attack.campaign.submission import MessageType
    from jed_attack.campaign.submission_score import MessageScore, SubmissionScore

    return SubmissionScore(
        public=public,
        total_hops=1,
        per_message=[
            MessageScore(
                message="m",
                type=MessageType.DEPUTY,
                severity={"optimal": public},
                trace={},
                feedback="",
            )
        ],
    )


def _mk_sub(tag: str) -> "Submission":
    from jed_attack.campaign.submission import Message, MessageType, Submission

    return Submission(
        messages=[Message(type=MessageType.DEPUTY, text=f"Ping {tag}@h.invalid", hops=1)]
    )


def _run_refine_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    subs: list["Submission"],
    publics: list[float],
) -> "blackboard.Record | None":
    """Drive one worker with a scripted propose/score sequence; return board.best().

    ``subs``/``publics`` are consumed one per successful propose/score. When ``subs``
    is exhausted the next propose raises CancelledError, ending the loop after the
    generation(s) the script covers.
    """
    import asyncio

    from jed_attack.campaign import blackboard as bb
    from jed_attack.campaign import config
    from jed_attack.campaign import optimize_prompts as op

    sub_it = iter(subs)
    pub_it = iter(publics)

    async def fake_propose(prompt: str, provider: object, timeout_s: float):
        try:
            return next(sub_it), "rz"
        except StopIteration:
            raise asyncio.CancelledError

    monkeypatch.setattr(op, "propose_submission_async", fake_propose)
    monkeypatch.setattr(
        op, "score_submission", lambda m, models=config.MODELS: _mk_score(next(pub_it))
    )
    monkeypatch.setattr(op, "_GENERATION_RETRY_S", 0.0)

    board = bb.Blackboard.load(tmp_path / "bb.jsonl")
    prov = providers.get("cheapest-kimi")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(op.worker_loop(0, [prov], board, tmp_path / "out", timeout_s=1.0))
    return board.best()


def test_refine_runs_to_cap_when_every_round_improves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4 improving refine rounds (the cap) -> records the peak; not cancelled mid-climb."""
    # round0=1.0, refine1..4 = 2..5 (all improve). 5th propose (gen1 round0) cancels.
    subs = [_mk_sub(f"s{i}") for i in range(5)]
    publics = [1.0, 2.0, 3.0, 4.0, 5.0]
    best = _run_refine_worker(monkeypatch, tmp_path, subs, publics)
    assert best is not None and best.public == 5.0
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_campaign.py::test_refine_runs_to_cap_when_every_round_improves -v`
Expected: FAIL — today's `worker_loop` scores once and appends `1.0` (or the run is cancelled at the 2nd propose before appending), so `best.public != 5.0`.

- [ ] **Step 4: Implement the refine loop in `worker_loop`**

In `src/jed_attack/campaign/optimize_prompts.py`, replace the body of the `try:` block inside `worker_loop` (lines 158-199, from `incumbent = board.best()` through the `_log_wandb(...)` call) with:

```python
            team = {t: board.top_messages(t, k=_TEAM_TOP_K) for t in MessageType}
            reasoning_digest = board.recent_reasoning(k=_TEAM_REASONING_K)
            model = provider.model or provider.kind

            # Round 0: propose from the GLOBAL incumbent, score, adopt as the local best.
            incumbent = board.best()
            prompt = submission_prompt(
                incumbent,
                incumbent.feedback if incumbent else [],
                {},
                top_messages=team,
                reasoning=reasoning_digest,
            )
            submission, reasoning = await propose_submission_async(
                prompt, provider, timeout_s
            )
            score = await asyncio.to_thread(score_submission, submission.messages)
            local_best = make_record(submission, score, reasoning, model, worker_id)
            local_best_score = score
            round0_public = score.public

            # Rounds 1..REFINE_MAX_ROUNDS: hill-climb the local draft on its own real
            # score. Whole-submission rewrite via the same submission_prompt, with the
            # draft as the incumbent; keep the best, stop at the first non-improving
            # round. A round's proposer/score failure ends the climb with the best kept.
            refine_rounds = 0
            for _ in range(config.REFINE_MAX_ROUNDS):
                try:
                    prompt = submission_prompt(
                        local_best,
                        local_best.feedback,
                        {},
                        top_messages=team,
                        reasoning=reasoning_digest,
                    )
                    refined, refined_reasoning = await propose_submission_async(
                        prompt, provider, timeout_s
                    )
                    refined_score = await asyncio.to_thread(
                        score_submission, refined.messages
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _log.warning(
                        "worker %d refine round failed; keeping best",
                        worker_id,
                        exc_info=True,
                    )
                    break
                if refined_score.public <= local_best.public:
                    break  # no improvement -> stop the climb
                local_best = make_record(
                    refined, refined_score, refined_reasoning, model, worker_id
                )
                local_best_score = refined_score
                refine_rounds += 1

            await board.append(local_best, out_dir)
            best = board.best()
            assert best is not None  # just appended -> the board is non-empty
            _log.info(
                "worker %d (%s): public=%g (+%g over %d refine rounds) best=%g",
                worker_id,
                provider.model,
                local_best.public,
                local_best.public - round0_public,
                refine_rounds,
                best.public,
            )
            _log_wandb(  # one shared run; tag by lane so models are comparable
                run,
                {
                    "public": local_best.public,
                    "best_public": best.public,
                    "total_hops": float(local_best_score.total_hops),
                    "refine_rounds": refine_rounds,
                    "refine_gain": local_best.public - round0_public,
                    "model": provider.model,
                    "worker": worker_id,
                },
            )
```

Leave the `except asyncio.CancelledError: raise` / `except Exception: ... _GENERATION_RETRY_S` outer handler and `gen += 1` unchanged.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_campaign.py::test_refine_runs_to_cap_when_every_round_improves -v`
Expected: PASS.

- [ ] **Step 6: Add the remaining control-flow tests**

Append after the first refine test:

```python
def test_refine_keeps_peak_and_discards_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """improve then regress -> records the peak; the regressing rewrite is discarded."""
    # round0=3.0, refine1=5.0 (accept), refine2=4.0 (<=5.0 -> stop). Then gen1 cancels.
    subs = [_mk_sub(f"s{i}") for i in range(3)]
    best = _run_refine_worker(monkeypatch, tmp_path, subs, [3.0, 5.0, 4.0])
    assert best is not None and best.public == 5.0


def test_refine_stops_when_round0_already_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refine round that doesn't beat round 0 -> records round 0, one refine attempt."""
    subs = [_mk_sub("s0"), _mk_sub("s1")]
    best = _run_refine_worker(monkeypatch, tmp_path, subs, [5.0, 3.0])
    assert best is not None and best.public == 5.0


def test_refine_round_failure_keeps_best_so_far(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refine round raising a non-cancel error -> break, still append round-0 best."""
    import asyncio

    from jed_attack.campaign import blackboard as bb
    from jed_attack.campaign import config
    from jed_attack.campaign import optimize_prompts as op

    calls = {"n": 0}

    async def fake_propose(prompt: str, provider: object, timeout_s: float):
        calls["n"] += 1
        if calls["n"] == 1:
            return _mk_sub("s0"), "rz"  # round 0 succeeds
        if calls["n"] == 2:
            raise RuntimeError("refine blip")  # refine round 1 fails -> break
        raise asyncio.CancelledError  # gen1 round 0 -> end the loop

    monkeypatch.setattr(op, "propose_submission_async", fake_propose)
    monkeypatch.setattr(
        op, "score_submission", lambda m, models=config.MODELS: _mk_score(3.0)
    )
    monkeypatch.setattr(op, "_GENERATION_RETRY_S", 0.0)

    board = bb.Blackboard.load(tmp_path / "bb.jsonl")
    prov = providers.get("cheapest-kimi")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(op.worker_loop(0, [prov], board, tmp_path / "out", timeout_s=1.0))
    best = board.best()
    assert best is not None and best.public == 3.0  # round-0 best still appended
    assert calls["n"] == 3  # round0, failed refine (caught), gen1 cancel


def test_refine_disabled_when_max_rounds_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REFINE_MAX_ROUNDS=0 -> no refine rounds; pure propose->score->record."""
    from jed_attack.campaign import config
    from jed_attack.campaign import optimize_prompts as op

    monkeypatch.setattr(config, "REFINE_MAX_ROUNDS", 0)
    # Only round-0 propose consumes a sub; the next propose (gen1) cancels.
    best = _run_refine_worker(monkeypatch, tmp_path, [_mk_sub("s0")], [3.0])
    assert best is not None and best.public == 3.0
```

- [ ] **Step 7: Run the full campaign test module**

Run: `uv run pytest tests/test_campaign.py -q`
Expected: all pass (the 4 new tests + the existing suite). The existing `test_worker_loop_appends_then_survives_failure` still passes: round 0 appends `3.0`; with the default cap its single-message draft is refined against `_mk_score`-equivalent stubs — but that test scripts propose to raise on call 2, which the inner refine handler now catches (break) rather than the outer handler. Verify it still asserts `best.public == 3.0` and `calls["n"] == 3`; if the refine loop changes its call count, update that test's comment/assertion to reflect that call 2 is now a caught refine failure (the first generation still appends `3.0`, and the third call still cancels).

- [ ] **Step 8: Pre-commit and commit**

Run: `uv run pre-commit run -a`
Expected: ruff, ty, pytest all pass.

```bash
git add src/jed_attack/campaign/config.py src/jed_attack/campaign/optimize_prompts.py tests/test_campaign.py
git commit -m "feat: per-generation adversarial refinement (real-score hill-climb)"
```

---

## Cross-cutting verification (after Task 1)

- `REFINE_MAX_ROUNDS = 0` reproduces today's behavior exactly (append round-0 draft) — covered by `test_refine_disabled_when_max_rounds_zero`.
- `CancelledError` still tears the team down cleanly from inside a refine round (the inner handler re-raises) — exercised by every `_run_refine_worker` test (the script ends via `CancelledError`).
- No new blackboard growth: one `board.append` per generation.
- Deploy note (not part of the commit): this needs a green worker restart. Bundle the `sync_green.sh` + `run_optimizer.sh` restart with the static-caps commit `ce9b27b` already awaiting deploy; back up green's `.env` around the sync per project convention.

## Self-Review

- **Spec coverage:** real-score signal (Global Constraints + Step 4 re-scores each round); whole-submission rewrite reusing `submission_prompt` (Step 4); keep-best strict improvement + cap 4 early-stop (Step 4 loop); team digest in refine rounds (Step 4 passes `team`/`reasoning_digest`); `REFINE_MAX_ROUNDS` static, 0=disabled (Steps 1, 6-test); wandb `refine_rounds`/`refine_gain` (Step 4); inner error handling re-raising cancel (Step 4). All spec sections map to a step.
- **Placeholder scan:** none — every step has concrete code and commands.
- **Type consistency:** `local_best` is a `Record` (`.public`, `.feedback`, `.messages`); `local_best_score` is a `SubmissionScore` (`.total_hops`); `make_record`/`submission_prompt`/`propose_submission_async` used with their real signatures; `_mk_score`/`_mk_sub` build valid `SubmissionScore`/`Submission`.
