# Async Collaborative CI Optimizer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the multi-process optimizer swarm with one `asyncio` process running one
worker per proposer lane (5 cheapestinference models + z.ai glm-4.6), cooperating through an
append-only JSONL blackboard.

**Architecture:** Single event loop; N worker coroutines each pinned to one provider; a
shared in-memory blackboard (persisted to `run/blackboard.jsonl`) that every worker reads
each generation and appends to; the async proposer is `AsyncOpenAI`, scoring stays the SDK
replay wrapped in `asyncio.to_thread`; the best submission is written directly to
`build_next/attack.py`.

**Tech Stack:** Python 3.13, `asyncio`, `openai` (`AsyncOpenAI`), `pydantic`, `aicomp_sdk`.

**Design:** `docs/superpowers/specs/2026-07-23-async-collaborative-optimizer-design.md`.

## Global Constraints

- Scoring MUST call the unmodified `submission_score.score_submission` (SDK replay, seed=123,
  8 hops). Concurrency for scoring comes ONLY from `asyncio.to_thread`. Never reimplement the
  replay.
- One process → blackboard appends serialized by an `asyncio.Lock`. No `fcntl`.
- `build_next/attack.py` is (re)written on every new best, so a crash never loses the ship
  artifact. Reuse `assemble.build(messages: list[str], out_dir)` — do NOT re-render.
- Message/hop caps are the `submission.Submission` schema (≤ `MAX_SHIP_MESSAGES`=80 messages,
  per-message hops 1–8, summed hops ≤ `HOP_CEILING*BUDGET_FILL_FRACTION`=391). Unchanged.
- Proposer completion budget `_PROPOSER_MAX_TOKENS`=65536 (already committed). Prompts embed a
  BOUNDED blackboard digest, never the raw JSONL.
- CI token in `CHEAPEST_API_KEY`, z.ai token in `ZAI_API_KEY` — env only, via each provider's
  `key_env`. A provider whose `key_env` is unset is skipped at startup.
- Run `uv run pre-commit run -a` (ruff, ruff-format, ty, pytest) green before each commit.

**Starting-base note:** the working tree has two small uncommitted edits on the *sync* path
(`config.REASONING_LOG`, `optimize_prompts._record_reasoning`). They are superseded by the
blackboard's `reasoning` field; Task 5 removes them. Do not build on them.

---

## Task 1: Blackboard module

**Files:**
- Create: `src/jed_attack/campaign/blackboard.py`
- Modify: `src/jed_attack/campaign/config.py` (add `BLACKBOARD_LOG`)
- Test: `tests/test_campaign.py` (append new tests)

**Interfaces:**
- Consumes: `assemble.build(messages: list[str], out_dir: Path) -> Path`;
  `submission.MessageType`.
- Produces:
  - `Record` frozen dataclass: `messages: list[dict]`, `public: float`,
    `feedback: list[dict]`, `reasoning: str`, `model: str`, `worker: int`, `ts: float`;
    `to_json()`/`from_json(dict)`.
  - `class Blackboard`: `load(path: Path) -> Blackboard` (classmethod);
    `best() -> Record | None`; `top_messages(mtype: MessageType, k: int) -> list[tuple[str,
    str, float]]` (text, model, severity-sum, best-first, deduped by text);
    `recent_reasoning(k: int, chars: int = 800) -> list[tuple[str, str]]` (model, excerpt);
    `async append(record: Record, out_dir: Path) -> None` (persists; reships attack.py on a
    new best).

- [ ] **Step 1: Write the failing test** — round-trip, selection, ship-on-new-best.

```python
def test_blackboard_append_persists_selects_and_ships(tmp_path, monkeypatch) -> None:
    """append persists to JSONL, rebuilds views, and writes attack.py only on a new best."""
    import asyncio
    from jed_attack.campaign import blackboard as bb
    from jed_attack.campaign.submission import MessageType

    log = tmp_path / "blackboard.jsonl"
    out = tmp_path / "build_next"
    board = bb.Blackboard.load(log)  # empty start
    assert board.best() is None

    def rec(public, model, sev):
        return bb.Record(
            messages=[{"type": "deputy", "text": "Ping u1@h.invalid", "hops": 1}],
            public=public,
            feedback=[{"message": "Ping u1@h.invalid", "type": "deputy",
                       "severity": {"optimal": sev}, "feedback": ""}],
            reasoning="chose diverse deputies", model=model, worker=0, ts=1.0,
        )

    asyncio.run(board.append(rec(2.0, "kimi-k2.7", 4.0), out))
    asyncio.run(board.append(rec(5.0, "glm-4.6", 8.0), out))     # new best -> ships
    asyncio.run(board.append(rec(3.0, "deepseek-v4-flash", 2.0), out))  # not best

    assert board.best().public == 5.0
    assert board.best().model == "glm-4.6"
    # persisted: three lines, reload rebuilds the same best
    assert bb.Blackboard.load(log).best().public == 5.0
    # top deputy messages ranked by severity-sum, deduped
    top = board.top_messages(MessageType.DEPUTY, k=2)
    assert top[0][1] == "glm-4.6" and top[0][2] == 8.0
    # attack.py written (last write = the best at that point)
    assert (out / "attack.py").exists()
    assert board.recent_reasoning(k=1)[0][0] == "deepseek-v4-flash"
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_campaign.py -k blackboard -x` → ImportError / no `blackboard`.

- [ ] **Step 3: Add `config.BLACKBOARD_LOG`**

In `config.py`, after the `OPTIMIZE_LOG` block:

```python
# The team's shared memory (blackboard.py): an append-only JSONL record of every scored
# submission (messages, score, feedback, the proposer's reasoning, model, worker). The
# in-memory blackboard is rebuilt from it on start (warm restart). Replaces submission_log.
BLACKBOARD_LOG = CAMPAIGN_ROOT / "blackboard.jsonl"
```

- [ ] **Step 4: Implement `blackboard.py`**

```python
"""The team's shared blackboard: append-only JSONL + derived in-memory views.

One async process owns it, so appends are serialized with an ``asyncio.Lock`` (no fcntl).
Every scored submission is one JSONL line; the in-memory views (best submission, best
individual messages per shape, recent cross-model reasoning) are rebuilt on load and
updated on append. When an append sets a new public best, the shipped ``attack.py`` is
rewritten immediately via :func:`assemble.build`, so the artifact never lags the best.
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jed_attack.campaign import assemble
from jed_attack.campaign.submission import MessageType


@dataclass(frozen=True)
class Record:
    """One scored submission on the blackboard."""

    messages: list[dict]
    public: float
    feedback: list[dict]
    reasoning: str
    model: str
    worker: int
    ts: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Record":
        return cls(
            messages=list(data["messages"]),
            public=float(data["public"]),
            feedback=list(data["feedback"]),
            reasoning=str(data.get("reasoning", "")),
            model=str(data.get("model", "")),
            worker=int(data.get("worker", 0)),
            ts=float(data["ts"]),
        )


def _severity_sum(entry: dict) -> float:
    """Total severity of one feedback entry across guardrails."""
    return float(sum(entry.get("severity", {}).values()))


class Blackboard:
    """In-memory team memory backed by an append-only JSONL file."""

    def __init__(self, path: Path, records: list[Record]) -> None:
        self._path = path
        self._records = records
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, path: Path) -> "Blackboard":
        """Warm-start: replay the JSONL into memory (skips malformed lines)."""
        records: list[Record] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(Record.from_json(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return cls(path, records)

    def best(self) -> Record | None:
        """The highest-``public`` record, or ``None`` if empty."""
        return max(self._records, default=None, key=lambda r: r.public)

    def top_messages(
        self, mtype: MessageType, k: int
    ) -> list[tuple[str, str, float]]:
        """Best-scoring individual messages of a shape: ``(text, model, severity)``.

        Ranked by severity-sum, deduped by text, best first. This is the cross-model
        material a worker on one model learns from another's wins.
        """
        best_by_text: dict[str, tuple[str, float]] = {}
        for record in self._records:
            for entry in record.feedback:
                if entry.get("type") != mtype.value:
                    continue
                sev = _severity_sum(entry)
                text = entry.get("message", "")
                if sev <= 0 or not text:
                    continue
                if text not in best_by_text or sev > best_by_text[text][1]:
                    best_by_text[text] = (record.model, sev)
        ranked = sorted(
            ((t, m, s) for t, (m, s) in best_by_text.items()),
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:k]

    def recent_reasoning(self, k: int, chars: int = 800) -> list[tuple[str, str]]:
        """The most recent non-empty reasoning blobs: ``(model, excerpt)`` (bounded)."""
        out: list[tuple[str, str]] = []
        for record in reversed(self._records):
            if record.reasoning:
                out.append((record.model, record.reasoning[:chars]))
            if len(out) >= k:
                break
        return out

    async def append(self, record: Record, out_dir: Path) -> None:
        """Append a record, persist it, and reship ``attack.py`` if it is a new best."""
        async with self._lock:
            prior_best = self.best()
            self._records.append(record)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
            if prior_best is None or record.public > prior_best.public:
                assemble.build([m["text"] for m in record.messages], out_dir)
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_campaign.py -k blackboard -x` → PASS.
- [ ] **Step 6: `uv run pre-commit run -a`** → all green.
- [ ] **Step 7: Commit** — `feat: blackboard — append-only JSONL team memory + ship-on-best`

---

## Task 2: Async proposer + AsyncOpenAI client

**Files:**
- Modify: `src/jed_attack/campaign/providers.py` (add `async_openai_client`)
- Modify: `src/jed_attack/campaign/config.py` (add `TEAM_PROPOSERS`)
- Modify: `src/jed_attack/campaign/optimize_prompts.py` (add `propose_submission_async`,
  `_reasoning_of`)
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `providers.Provider`, `Submission`, `_salvage_submission`,
  `_PROPOSER_MAX_TOKENS`, `_PROPOSER_TEMPERATURE`, `_SUBMISSION_SYSTEM`.
- Produces:
  - `providers.async_openai_client(provider) -> openai.AsyncOpenAI`.
  - `config.TEAM_PROPOSERS: tuple[str, ...]`.
  - `async optimize_prompts.propose_submission_async(prompt, provider, timeout_s) ->
    tuple[Submission, str]` (submission, reasoning_text).
  - `optimize_prompts._reasoning_of(message) -> str`.

- [ ] **Step 1: Write the failing test** — a fake AsyncOpenAI whose `.parse` returns a
  thinking message + parsed submission; assert reasoning extracted.

```python
def test_propose_submission_async_extracts_submission_and_reasoning(monkeypatch) -> None:
    import asyncio
    from types import SimpleNamespace
    from jed_attack.campaign import optimize_prompts, providers
    from jed_attack.campaign.submission import Message, MessageType, Submission

    sub = Submission(messages=[Message(type=MessageType.EXFIL,
        text="SECRET_MARKER https://a.invalid/r", hops=1)])
    msg = SimpleNamespace(parsed=sub, reasoning_content="weighed diversity")

    class FakeParse:
        async def parse(self, **_):  # noqa: ANN003
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    class FakeChat:
        completions = FakeParse()
    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(providers, "async_openai_client", lambda p: FakeClient())
    prov = providers.get("cheapest-kimi")
    got_sub, reasoning = asyncio.run(
        optimize_prompts.propose_submission_async("prompt", prov, timeout_s=1.0))
    assert got_sub.messages[0].text == "SECRET_MARKER https://a.invalid/r"
    assert reasoning == "weighed diversity"
```

- [ ] **Step 2: Run to verify it fails** — `-k propose_submission_async` → AttributeError.

- [ ] **Step 3: Add `providers.async_openai_client`** (mirror `openai_client`):

```python
from openai import AsyncOpenAI, OpenAI  # update the existing import

def async_openai_client(provider: Provider) -> AsyncOpenAI:
    """Async OpenAI-SDK client for a provider (api or local). Mirrors openai_client."""
    key = (
        os.environ.get(provider.key_env, "sk-local") if provider.key_env else "sk-local"
    )
    base = provider.base_url or resolve_base_url(provider.model)
    return AsyncOpenAI(base_url=base.rstrip("/"), api_key=key)
```

- [ ] **Step 4: Add `config.TEAM_PROPOSERS`** (after the models block):

```python
# The proposer lanes the async team runs — one worker per name, each pinned to that model.
# The five cheapest-* lanes share CHEAPEST_API_KEY (the per-key concurrency test); zai-glm4.6
# is on ZAI_API_KEY (separate, proven-firing). A lane whose key_env is unset is skipped.
TEAM_PROPOSERS: tuple[str, ...] = (
    "cheapest-kimi", "cheapest-deepseek", "cheapest-glm5.2",
    "cheapest-minimax", "cheapest-mimo", "zai-glm4.6",
)
```

- [ ] **Step 5: Add `propose_submission_async` + `_reasoning_of`** to `optimize_prompts.py`.
  Mirror the sync `propose_submission` (parse → create+salvage), but async and returning the
  reasoning. Log a `429`-with-`Concurrency limit` distinctly (the experiment signal).

```python
def _reasoning_of(message: Any) -> str:
    """A thinking backend's reasoning_content (or reasoning), else ''. SDK exposes extras."""
    return getattr(message, "reasoning_content", None) or getattr(
        message, "reasoning", None
    ) or ""


async def propose_submission_async(
    prompt: str, provider: providers.Provider, timeout_s: float
) -> tuple[Submission, str]:
    """Author one submission on ``provider`` via AsyncOpenAI; return (submission, reasoning).

    Tries structured ``.parse(response_format=Submission)`` first, else ``.create()`` +
    ``_salvage_submission``. Logs a CI concurrency 429 distinctly for the per-key test.
    """
    client = providers.async_openai_client(provider)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SUBMISSION_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        response = await client.chat.completions.parse(
            model=provider.model, messages=messages, response_format=Submission,
            max_completion_tokens=_PROPOSER_MAX_TOKENS,
            temperature=_PROPOSER_TEMPERATURE, timeout=timeout_s,
        )
        message = response.choices[0].message
        if message.parsed is not None:
            return message.parsed, _reasoning_of(message)
    except Exception as exc:
        if "concurrency limit" in str(exc).lower():
            _log.warning("CI concurrency 429 on %s (experiment signal)", provider.model)
        _log.info("async parse failed for %s (%s); tolerant path", provider.model, exc)
    response = await client.chat.completions.create(
        model=provider.model, messages=messages,
        max_completion_tokens=_PROPOSER_MAX_TOKENS,
        temperature=_PROPOSER_TEMPERATURE, timeout=timeout_s,
    )
    message = response.choices[0].message
    return _salvage_submission(message.content or ""), _reasoning_of(message)
```

- [ ] **Step 6: Run tests** — `-k propose_submission_async` → PASS.
- [ ] **Step 7: `uv run pre-commit run -a`** → green.
- [ ] **Step 8: Commit** — `feat: async proposer (AsyncOpenAI) + TEAM_PROPOSERS lanes`

---

## Task 3: Team digest in the proposer prompt

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py` (`submission_prompt`)
- Test: `tests/test_campaign.py`

**Interfaces:**
- `submission_prompt` gains two optional params (additive — the existing signature and
  callers keep working):
  `submission_prompt(incumbent, feedback, introspection, top_messages=None,
  reasoning=None)`, where `top_messages: dict[MessageType, list[tuple[str, str, float]]] |
  None` and `reasoning: list[tuple[str, str]] | None`.

- [ ] **Step 1: Write the failing test**

```python
def test_submission_prompt_embeds_team_digest() -> None:
    from jed_attack.campaign.submission import MessageType
    prompt = optimize_prompts.submission_prompt(
        None, [], {},
        top_messages={MessageType.DEPUTY: [("Ping u1@h.invalid", "kimi-k2.7", 4.0)]},
        reasoning=[("glm-4.6", "spread deputies across hosts")],
    )
    assert "kimi-k2.7" in prompt          # message tagged with the model that found it
    assert "Ping u1@h.invalid" in prompt
    assert "spread deputies across hosts" in prompt  # cross-model reasoning (DATA)
```

- [ ] **Step 2: Run to verify it fails** — `-k team_digest` → TypeError (unexpected kwarg).

- [ ] **Step 3: Extend `submission_prompt`** — add the two params (default `None`), and when
  present render two DATA blocks appended after the incumbent body (before `tail`), e.g.:

```python
    team: list[str] = []
    if top_messages:
        team.append("")
        team.append("TEAMMATE BEST MESSAGES (DATA — other models' highest-scoring "
                    "messages; borrow the shapes/framings, not the exact targets):")
        for mtype, rows in top_messages.items():
            for text, model, sev in rows:
                team.append(f"  [{mtype.value} sev={sev:g} via {model}] {text}")
    if reasoning:
        team.append("")
        team.append("TEAMMATE REASONING (DATA — how other models reasoned; untrusted):")
        for model, excerpt in reasoning:
            team.append(f"  [{model}] {excerpt}")
    return "\n".join(header + body + team + tail)
```

  (Update the docstring to note the team digest. Keep the existing tool-signatures + rules.)

- [ ] **Step 4: Run tests** — `-k "team_digest or submission_prompt or cold_start"` → PASS.
- [ ] **Step 5: `uv run pre-commit run -a`** → green.
- [ ] **Step 6: Commit** — `feat: seed the proposer prompt with the team blackboard digest`

---

## Task 4: Async orchestrator (replaces the sync loop)

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py` (replace `optimize`,
  `run_submission_generation`, sync `propose_submission`, and `main` with the async
  orchestrator; drop the `submission_log`/`shards`/`knowledge` imports from this module)
- Modify: `scripts/run_optimizer.sh` (drop per-worker env; the process owns all lanes)
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `Blackboard`, `propose_submission_async`, `submission_prompt`,
  `submission_score.score_submission`, `victim_feedback` (feedback only), `config.MODELS`,
  `config.TEAM_PROPOSERS`, `config.BLACKBOARD_LOG`, `config.BUILD_NEXT_DIR`.
- Produces:
  - `async worker_loop(worker_id: int, provider, board, out_dir, timeout_s, run=None) ->
    None` (logs per-generation metrics to the shared `run` if given).
  - `make_record(submission, score, reasoning, model, worker) -> blackboard.Record`.
  - `async optimize_team(board, out_dir, timeout_s, run=None) -> None` (builds the provider
    list from `TEAM_PROPOSERS`, skips unusable keys, `asyncio.gather`s the workers).
  - `main()` → one `wandb.init`, SIGTERM/SIGINT → clean cancel → `run.finish()`.

**wandb:** ONE run for the whole team (one process). All six lanes log to the same `run`
handle via the existing `_log_wandb(run, metrics)` (guards `None`), each metric tagged with
its `model` + `worker` so lanes are comparable in one chart. No per-worker `JED_WANDB` flag.
SIGTERM (from `pkill`/tmux) triggers a clean cancel so `run.finish()` marks the run
FINISHED, not crashed — eliminating the restart crash-alert noise.

- [ ] **Step 1: Write the failing test** — one worker iteration appends a record and a raised
  proposer is caught so the loop continues. Use an async stub for `propose_submission_async`
  and a sync stub for `score_submission`; run the loop for two iterations via a counter that
  cancels.

```python
def test_worker_loop_appends_then_survives_failure(tmp_path, monkeypatch) -> None:
    import asyncio
    from jed_attack.campaign import optimize_prompts as op, blackboard as bb, config
    from jed_attack.campaign.submission import Message, MessageType, Submission
    from jed_attack.campaign.submission_score import SubmissionScore, MessageScore

    sub = Submission(messages=[Message(type=MessageType.DEPUTY,
        text="Ping u1@h.invalid", hops=1)])
    score = SubmissionScore(public=3.0, total_hops=1, per_message=[MessageScore(
        message="Ping u1@h.invalid", type=MessageType.DEPUTY,
        severity={"optimal": 4.0}, trace={}, feedback="")])

    calls = {"n": 0}
    async def fake_propose(prompt, provider, timeout_s):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("proposer blip")
        if calls["n"] > 2:
            raise asyncio.CancelledError
        return sub, "reasoning"
    monkeypatch.setattr(op, "propose_submission_async", fake_propose)
    monkeypatch.setattr(op, "score_submission", lambda m, models=config.MODELS: score)
    monkeypatch.setattr(op, "_GENERATION_RETRY_S", 0.0)

    board = bb.Blackboard.load(tmp_path / "bb.jsonl")
    prov = __import__("jed_attack.campaign.providers", fromlist=["get"]).get("cheapest-kimi")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(op.worker_loop(0, prov, board, tmp_path / "out", timeout_s=1.0))
    assert board.best().public == 3.0          # first iteration appended
    assert calls["n"] == 3                       # blip at 2 was caught, loop continued
```

- [ ] **Step 2: Run to verify it fails** — `-k worker_loop` → AttributeError.

- [ ] **Step 3: Implement the orchestrator.** Replace the sync `optimize`/
  `run_submission_generation`/`propose_submission`/`main` with:

```python
def make_record(submission, score, reasoning, model, worker):
    """Build a blackboard.Record from a scored submission."""
    feedback = [
        {"message": ms.message, "type": ms.type.value,
         "severity": ms.severity, "feedback": ms.feedback}
        for ms in score.per_message
    ]
    return blackboard.Record(
        messages=[m.model_dump(mode="json") for m in submission.messages],
        public=score.public, feedback=feedback, reasoning=reasoning,
        model=model, worker=worker, ts=time.time(),
    )


async def worker_loop(worker_id, provider, board, out_dir, timeout_s, run=None):
    """One lane: author from the team digest, score, append (ships on new best), forever."""
    while True:
        try:
            incumbent = board.best()
            prompt = submission_prompt(
                incumbent,
                incumbent.feedback if incumbent else [],
                {},
                top_messages={
                    t: board.top_messages(t, k=_TEAM_TOP_K) for t in MessageType
                },
                reasoning=board.recent_reasoning(k=_TEAM_REASONING_K),
            )
            submission, reasoning = await propose_submission_async(
                prompt, provider, timeout_s)
            score = await asyncio.to_thread(score_submission, submission.messages)
            record = make_record(
                submission, score, reasoning, provider.model or provider.kind, worker_id)
            await board.append(record, out_dir)
            best = board.best()
            _log.info("worker %d (%s): public=%g best=%g",
                      worker_id, provider.model, score.public, best.public)
            _log_wandb(run, {  # one shared run; tag by lane so models are comparable
                "public": score.public, "best_public": best.public,
                "total_hops": float(score.total_hops),
                "model": provider.model, "worker": worker_id,
            })
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("worker %d generation failed; retrying", worker_id)
            await asyncio.sleep(_GENERATION_RETRY_S)


async def optimize_team(board, out_dir, timeout_s, run=None):
    """Launch one worker per usable TEAM_PROPOSERS lane and run them concurrently."""
    lanes = []
    for name in config.TEAM_PROPOSERS:
        provider = providers.get(name)
        if provider.key_env and provider.key_env not in os.environ:
            _log.warning("lane %s skipped: %s unset", name, provider.key_env)
            continue
        lanes.append(provider)
    _log.info("team: %d lanes -> %s", len(lanes), [p.model for p in lanes])
    await asyncio.gather(*(
        worker_loop(i, p, board, out_dir, timeout_s, run) for i, p in enumerate(lanes)
    ))


async def _run_team(board, run) -> None:
    """Run the team until cancelled; SIGTERM/SIGINT cancels cleanly so wandb can finish."""
    task = asyncio.ensure_future(
        optimize_team(board, config.BUILD_NEXT_DIR, PROPOSER_TIMEOUT_S, run))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        _log.info("team cancelled; shutting down cleanly")
```

  Add module constants `_TEAM_TOP_K = 8`, `_TEAM_REASONING_K = 3`. Rewrite `main()`:

```python
def main() -> None:
    load_dotenv(config.ENV_FILE)
    config.ensure_dirs()
    _setup_logging()
    run = _init_wandb(f"team-{int(time.time())}")  # ONE run for the whole team
    board = blackboard.Blackboard.load(config.BLACKBOARD_LOG)
    try:
        asyncio.run(_run_team(board, run))
    finally:
        _finish_wandb(run)  # clean exit -> run marked FINISHED, not crashed
```

  Imports: add `import asyncio`, `import signal`, `from jed_attack.campaign import
  blackboard`, `from jed_attack.campaign.submission import MessageType`,
  `from jed_attack.campaign.submission_score import score_submission`. Remove the
  `submission_log`, `shards`, `knowledge`, `prompt_opt`, `SubmissionRecord`,
  `victim_feedback.introspect_worst` usages from this module. KEEP `_init_wandb`,
  `_finish_wandb`, `_log_wandb` — the single team run uses all three.

- [ ] **Step 4: Update `scripts/run_optimizer.sh`** — the process owns all lanes, so drop
  `JED_WORKER_ID`/`JED_PROPOSER`; keep one tmux session `optimizer`, `JED_WANDB=1`,
  `JED_CAMPAIGN_ROOT`, `LD_LIBRARY_PATH`, and the `tee` to `run/logs/optimizer.log`.

- [ ] **Step 5: Run tests** — `-k "worker_loop or optimize"` → PASS.
- [ ] **Step 6: `uv run pre-commit run -a`** → green (this task will surface references to
  soon-deleted modules in tests — fix the async-touched tests here; Task 5 removes the rest).
- [ ] **Step 7: Commit** — `feat: async team orchestrator replaces the sync optimizer loop`

---

## Task 5: Remove the old code

**Files:**
- Delete: `src/jed_attack/campaign/consolidator.py`, `assemble_daemon.py`,
  `submission_log.py`, `shards.py`, `knowledge.py`, `prompt_opt.py`, `launch.py`
- Modify: `src/jed_attack/campaign/config.py` (remove dead paths), `pyproject.toml` (remove
  `jed-optimize` script), `scripts/campaign_watchdog.sh` (remove assemble+consolidator
  supervision), `optimize_prompts.py` (remove `_record_reasoning` + the sync
  `propose_submission`/`_salvage`… KEEP `_salvage_submission`/`_parse_message_objects` —
  they are used by `propose_submission_async`), `tests/test_campaign.py` (delete tests of
  removed modules; keep/blackboard-port the rest)

**Interfaces:** none produced; this task only removes.

- [ ] **Step 1: Delete the dead modules** — `git rm` the seven files above.
- [ ] **Step 2: Strip `config.py`** — remove `SUBMISSION_LOG`, `SUBMISSION_SHARDS_DIR`,
  `CONSOLIDATOR_STATUS_FILE`, `CONSOLIDATE_INTERVAL_S`, `PROPOSER_CONFIG_FILE`,
  `KNOWLEDGE_DIR`, `NOTES_DIR`, and the uncommitted `REASONING_LOG`; drop them from
  `ensure_dirs` (keep `BUILD_NEXT_DIR`, `logs`, and add nothing else — the blackboard file's
  parent is `CAMPAIGN_ROOT`).
- [ ] **Step 3: Strip `optimize_prompts.py`** — remove `_record_reasoning`, the sync
  `propose_submission`, `run_submission_generation`, `optimize`, `_configured_chain`,
  `current_provider(s)`, `set_providers`, and `_request_json`/`fetch_api_models` if now
  unused (they served the removed `launch.py` provider validation), and any
  `submission_log`/`shards`/`knowledge`/`prompt_opt` imports. KEEP `submission_prompt`,
  `_feedback_table`, `propose_submission_async`, `_reasoning_of`, `_salvage_submission`,
  `_parse_message_objects`, `_bracket_positions`, the async orchestrator, `_init_wandb`,
  `_finish_wandb`, `_log_wandb` (the team run uses all three), `_setup_logging`, `main`.
- [ ] **Step 4: `pyproject.toml`** — remove the `jed-optimize = "…launch:main"` entry (and
  the `[project.scripts]` table if now empty).
- [ ] **Step 5: `scripts/campaign_watchdog.sh`** — remove the `assemble`/`consolidator`
  daemon supervision (both deleted). If nothing remains to supervise, delete the script and
  drop its mention from `run_optimizer.sh`/docs; the single async process runs under the
  `optimizer` tmux session.
- [ ] **Step 6: Prune tests** — delete `test_submission_log_*`, `test_shards_*`,
  `test_run_submission_generation_*`, `test_config_shards_constants_*`,
  `test_set_providers_*`/`current_providers` (whatever references removed symbols). Keep the
  new blackboard/async tests and the schema/scoring/prompt tests.
- [ ] **Step 7: Full suite + lint** — `uv run pre-commit run -a` → green (no dangling
  imports, no references to deleted modules).
- [ ] **Step 8: Commit** — `refactor: remove the multi-process swarm (consolidator, shards,
  submission_log, knowledge, launch) superseded by the async team`

---

## Post-plan validation (not a task — operator step on green)

After the branch is merged/synced: `sync_green.sh`, then `scripts/run_optimizer.sh`, then
watch `run/logs/optimizer.log` + `run/blackboard.jsonl` for all six lanes producing, the
best climbing, and whether any `CI concurrency 429` fires (the experiment verdict). The old
`run/submission_log.jsonl` is ignored; optionally seed `blackboard.jsonl` from it first.
