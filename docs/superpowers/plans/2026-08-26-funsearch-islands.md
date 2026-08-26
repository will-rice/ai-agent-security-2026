# FunSearch Islands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the optimizer's single shared MAP-Elites archive with N parallel island archives (one worker each) — with migration, stagnation reset, and a novelty island — so independent lineages evolve concurrently instead of collapsing onto one champion.

**Architecture:** A new `IslandSet` wraps N `archive.Archive` instances plus per-island bookkeeping. `optimize_team` spawns N codex-lane workers, each pinned to one island; worker `i` proposes from island `i`'s local incumbent + parents, scores (unchanged path), inserts into island `i` only. Shared state (global-best, migration, reship) is mutated under the Blackboard's existing `asyncio.Lock`. The shipped `attack.py` is always the global best across islands.

**Tech Stack:** Python 3.12, pydantic, asyncio, llama-cpp (unchanged), `uv`, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-08-26-funsearch-islands-design.md`.

## Global Constraints

- Every commit: `uv run pre-commit run -a` green (ruff, ruff-format, ty, pytest). Line length ≤ 88. Google-style docstrings. No `from __future__ import annotations`. Fix type errors, never `# type: ignore`.
- Do NOT change the replay/scoring path (`submission_score.score_pools`, `replay_trace`, `throughput`, `_shape_elites`'s elite fields), the v23 token cost model, or the shipped-cut format.
- Reuse `archive.Archive` unchanged for quality islands. The novelty island is the SAME container with a different retention rule, selected by a flag — never a fork of the class.
- All decision logic (island pinning, stall/reset trigger, migration cadence, novelty distance, global-best) must be **pure functions or pure methods, unit-tested GPU-free**. The GGUF replay path stays untouched, so no test loads a model.
- Config knobs are env-overridable ALL-CAPS constants in `config.py`; `0` disables a mechanism.
- Concurrency invariant: **worker `i` is the ONLY writer of island `i`'s archive.** Shared reads/writes (global-best, migration into another island, reship, persistence) happen under `Blackboard._lock`.

---

### Task 1: `IslandSet` container + persistence

**Files:**
- Create: `src/jed_attack/campaign/islands.py`
- Test: `tests/test_islands.py`

**Interfaces:**
- Consumes: `archive.Archive`, `archive.Elite`, `archive.elite_board_density`, `config.ISLAND_COUNT`.
- Produces:
  - `class IslandSet` with `archives: list[Archive]`, `stall: list[int]`, `best_seen: list[float]`.
  - `IslandSet.for_count(n: int) -> IslandSet` (n empty archives; island 0 is the novelty island).
  - `insert(self, i: int, elite: Elite) -> bool` — insert into island i (novelty rule when `i == 0`, see Task 2; for now delegate to `archives[i].insert`).
  - `local_best_density(self, i: int) -> float` — max `elite_board_density` over island i's frontier, or `0.0` if empty.
  - `global_best_elite(self) -> Elite | None` — `max` over all islands' frontiers by `elite_board_density`.
  - `best_island(self) -> int` — index of the island with the highest `local_best_density` (ties: lowest index; novelty island 0 excluded unless it's the only non-empty one).
  - `to_jsonl(self, directory: Path) -> None` / `from_jsonl(cls, directory: Path, n: int) -> IslandSet` — persist/restore `island_<i>.jsonl` via each `Archive.to_jsonl`/`from_jsonl`.

- [ ] **Step 1: Write the failing test** (`tests/test_islands.py`)

```python
"""Tests for the parallel island archive set (GPU-free)."""

from pathlib import Path

from jed_attack.campaign import archive
from jed_attack.campaign.islands import IslandSet


def _elite(family: str, density_gpt: float, bucket: int = 5) -> archive.Elite:
    """A minimal firing gpt_oss elite whose throughput sets its board density."""
    return archive.Elite(
        text=f"http.post data=SECRET_MARKER url={{u}} {family}{bucket}",
        mtype="exfil",
        throughput={"gpt_oss": density_gpt, "gemma_4": 0.0},
        severity={"gpt_oss": 16.0, "gemma_4": 0.0},
        diagnosis="",
        family=family,
        bucket=bucket,
        url_scheme="s://{h}",
        turns={"gpt_oss": 2.0, "gemma_4": 0.0},
        input_chars=40,
    )


def test_islandset_isolates_inserts_and_finds_global_best() -> None:
    islands = IslandSet.for_count(3)
    islands.insert(1, _elite("plain", 0.02))
    islands.insert(2, _elite("forge", 0.05))
    # each insert lands only in its island
    assert len(islands.archives[1].frontier()) == 1
    assert len(islands.archives[0].frontier()) == 0
    # global best is the densest across islands (island 2)
    best = islands.global_best_elite()
    assert best is not None and best.family == "forge"
    assert islands.best_island() == 2


def test_islandset_persist_round_trip(tmp_path: Path) -> None:
    islands = IslandSet.for_count(2)
    islands.insert(1, _elite("plain", 0.02))
    islands.to_jsonl(tmp_path)
    restored = IslandSet.from_jsonl(tmp_path, 2)
    assert len(restored.archives[1].frontier()) == 1
    assert restored.global_best_elite().family == "plain"
```

- [ ] **Step 2: Run it to verify it fails** — `uv run pytest tests/test_islands.py -q` → ImportError / no `IslandSet`.

- [ ] **Step 3: Implement `islands.py`** — `IslandSet` with the interfaces above; `insert` delegates to `archives[i].insert` (novelty rule added in Task 2 via an `i == 0` branch calling `_novelty_insert`, a stub returning `archives[0].insert` for now). `global_best_elite`/`best_island` use `archive.elite_board_density`. `to_jsonl` writes `island_<i>.jsonl`; `from_jsonl` reads them (missing file → empty archive).

- [ ] **Step 4: Run tests to verify they pass** — `uv run pytest tests/test_islands.py -q`.

- [ ] **Step 5: Commit** — `git commit -am "Add IslandSet: N isolated archives + global-best + persistence"`.

---

### Task 2: Novelty island retention rule

**Files:**
- Modify: `src/jed_attack/campaign/islands.py`
- Test: `tests/test_islands.py`

**Interfaces:**
- Consumes: `archive.Elite`, `config.NOVELTY_NEIGHBORS`.
- Produces:
  - `descriptor(elite) -> tuple[str, int]` = `(elite.family, elite.bucket)`.
  - `novelty(elite, others: list[Elite]) -> float` — mean distance from `elite`'s descriptor to its `k` nearest in `others` (distance = family-mismatch(0/1) + `abs(bucket_a - bucket_b)`); `inf` when `others` is empty (maximally novel).
  - `IslandSet._novelty_insert(elite) -> bool` — insert into island 0, but when a `(family, bucket)` cell already holds an elite, keep whichever has HIGHER `novelty` against the rest of island 0 (not higher throughput). Firing still required (`throughput` > 0 on some model), matching `archive.Archive`.

- [ ] **Step 1: Write the failing test**

```python
def test_novelty_island_keeps_the_more_novel_of_a_contested_cell() -> None:
    from jed_attack.campaign.islands import novelty

    islands = IslandSet.for_count(2)
    # island 0 (novelty) already holds two 'plain' shapes -> a third 'plain' is FAMILIAR
    islands.insert(0, _elite("plain", 0.09, bucket=5))
    islands.insert(0, _elite("verb_variant", 0.09, bucket=9))
    familiar = _elite("plain", 0.20, bucket=5)  # high throughput but same as an existing cell
    novel = _elite("injection_variant", 0.01, bucket=2)  # low throughput but structurally new
    others = islands.archives[0].frontier()
    assert novelty(novel, others) > novelty(familiar, others)
    islands.insert(0, novel)
    texts = {e.family for e in islands.archives[0].frontier()}
    assert "injection_variant" in texts  # the novel shape is kept despite low throughput
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `descriptor`, `novelty`, and `_novelty_insert`; wire `insert` to call `_novelty_insert` when `i == 0`.
- [ ] **Step 4: Run tests to verify pass.**
- [ ] **Step 5: Commit** — `"Novelty island: retain structurally-distinct elites over throughput"`.

---

### Task 3: Stagnation-reset + migration decision logic (pure)

**Files:**
- Modify: `src/jed_attack/campaign/islands.py`
- Modify: `src/jed_attack/campaign/config.py` (add `ISLAND_COUNT`, `ISLAND_STAGNATION_GENERATIONS`, `ISLAND_MIGRATION_GENERATIONS`, `NOVELTY_NEIGHBORS`)
- Test: `tests/test_islands.py`

**Interfaces:**
- `IslandSet.note_generation(i, density) -> bool` — update `best_seen[i]`/`stall[i]`; return `True` when island `i` has stalled `>= config.ISLAND_STAGNATION_GENERATIONS` (0 disables → always `False`).
- `IslandSet.reset_island(i, seed: Elite | None) -> None` — replace `archives[i]` with a fresh `Archive`; if `seed` is not None, insert it; zero `stall[i]`/`best_seen[i]`.
- `should_migrate(generation: int) -> bool` (module fn) — `config.ISLAND_MIGRATION_GENERATIONS > 0 and generation > 0 and generation % config.ISLAND_MIGRATION_GENERATIONS == 0`.

- [ ] **Step 1: Failing test**

```python
def test_stall_triggers_reset_and_migration_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jed_attack.campaign import config
    from jed_attack.campaign import islands as isl

    monkeypatch.setattr(config, "ISLAND_STAGNATION_GENERATIONS", 3)
    monkeypatch.setattr(config, "ISLAND_MIGRATION_GENERATIONS", 4)
    islands = IslandSet.for_count(2)
    islands.insert(1, _elite("plain", 0.05))
    d = islands.local_best_density(1)
    flags = [islands.note_generation(1, d) for _ in range(3)]  # flat density
    assert flags == [False, False, True]  # 3rd flat generation stalls
    islands.reset_island(1, _elite("forge", 0.02))
    assert islands.stall[1] == 0 and len(islands.archives[1].frontier()) == 1
    assert not isl.should_migrate(3) and isl.should_migrate(4)
```

- [ ] **Step 2–4:** implement config constants + methods, run tests green.
- [ ] **Step 5: Commit** — `"Island stall/reset + migration cadence (config-tunable)"`.

---

### Task 4: Blackboard holds an IslandSet; global-best shipping

**Files:**
- Modify: `src/jed_attack/campaign/blackboard.py`
- Modify: `src/jed_attack/campaign/submission.py` (add `island` field to `Record` — Task uses it)
- Test: `tests/test_campaign.py`

**Interfaces (Blackboard):**
- `self.islands: IslandSet` built in `__init__`/`load` (`IslandSet.from_jsonl(_islands_dir(path), config.ISLAND_COUNT)`; falls back to empty).
- `island_best(self, i) -> Record | None` — best-objective Record whose `record.island == i` and current scheme.
- `global_champion(self) -> Record | None` — best-objective Record across all islands, current scheme (this is what ships; replaces `best_objective` at the SHIP call sites, not the telemetry ones).
- `reship_islands(self, out_dir) -> None` — ship the global-best frontier (union of island frontiers ranked by density, same `_ship_pools`/`_frontier_map` machinery) and persist all islands via `IslandSet.to_jsonl`.
- `Record` gains `island: int = 0` (persisted; default 0 so legacy rows load).

- [ ] **Step 1: Failing test** — append two records tagged island 1 and island 2 with different objectives; assert `island_best(1)`/`island_best(2)` return the right ones and `global_champion()` returns the higher-objective one.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — add `island` to `Record`; build `self.islands` in `load`; add `island_best`, `global_champion`, `reship_islands`, `_islands_dir(path)`. Keep the existing single-`archive` attribute as `self.islands.archives[0]`-independent? NO — remove `self.archive`; every reader moves to an island (Task 5). For THIS task, add the new methods and keep `self.archive` temporarily delegating to `islands.archives[0]` via a property so the module still imports; the delegation is deleted in Task 5.
- [ ] **Step 4: Run tests green.**
- [ ] **Step 5: Commit** — `"Blackboard: IslandSet member, per-island + global-best record selection"`.

---

### Task 5: Wire the worker loop to per-island evolution

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py`
- Test: `tests/test_campaign.py`

**Interfaces (changes in `worker_loop`, `_seed_archive`, `_ablate_champion`):**
- `worker_loop(worker_id, ...)` treats `island = worker_id % config.ISLAND_COUNT`. Replace:
  - `incumbent = board.best_objective()` → `incumbent = board.island_best(island)`.
  - `parents = board.archive.parents(K)` → `board.islands.archives[island].parents(K)`.
  - `opro=board.archive.frontier()` → `board.islands.archives[island].frontier()`.
  - `frontier_changed = board.archive.insert(elite)` loop → `board.islands.insert(island, elite)`; tag the appended record with `island` (pass to `make_record`).
  - After inserts: `board.note_generation(island)`, and if stalled → `board.islands.reset_island(island, seed)` under the lock (seed = random elite of `board.islands.best_island()`); every `M` gens the FIRST worker migrates `global_best` into a random quality island.
  - Reship: if the GLOBAL best changed → `await board.reship_islands(out_dir)`; `_ablate_champion` operates on `board.global_champion()`.
- `_seed_archive` → `_seed_islands`: seed the reseeded champion into island 1 (a quality island); other islands start empty.
- `make_record(...)` gains an `island` param (defaults 0).

- [ ] **Step 1: Failing test** — a worker-level unit test with a fake board/IslandSet asserting worker `i` inserts only into island `i` and reships the global best (use existing test doubles / monkeypatch `_score_batch`).
- [ ] **Step 2–4:** implement, keeping `worker_loop` under the C901 complexity cap (extract the stall/reset/migration into a helper `_evolve_island(board, island, gen, worker_id)`); run the FULL `tests/test_campaign.py` green.
- [ ] **Step 5: Commit** — `"worker_loop: evolve one island per worker; ship the global best"`.

---

### Task 6: Spawn N island workers; run script

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py` (`optimize_team`)
- Modify: `scripts/run_optimizer.sh` (set `JED_PROPOSER_REPLICAS=4` for the codex lane)
- Test: `tests/test_campaign.py`

**Interfaces:**
- `optimize_team` ensures the codex lane is fanned to `config.ISLAND_COUNT` workers (replicas), each with a distinct `worker_id` in `0..N-1`. Assert `len(cycles) >= config.ISLAND_COUNT` for the codex key, else warn.

- [ ] **Step 1: Failing test** — with `config.ISLAND_COUNT=4` and replicas set, `optimize_team`'s worker count equals 4 (test the cycle-building helper in isolation, not the full async run).
- [ ] **Step 2–4:** implement; `run_optimizer.sh` exports `JED_PROPOSER_REPLICAS=4`; run tests green.
- [ ] **Step 5: Commit** — `"Spawn ISLAND_COUNT codex workers, one per island; replicas=4 in run script"`.

---

### Task 7: Scheme bump v24 + reseed rollout + docs

**Files:**
- Modify: `src/jed_attack/campaign/blackboard.py` (`objective_scheme_name` v23 → v24)
- Modify: `tests/test_campaign.py` (scheme-name assertions v23 → v24)
- Modify: `scratchpad/reseed_108_champion.py` (reseed into island 1 via `board.islands`)
- Modify: `docs/research/2026-08-26-improving-the-search.md` (mark #4 done)

- [ ] **Step 1:** bump the scheme tag to `_pareto_v24`; update its docstring + the module comment; update the scheme-name tests.
- [ ] **Step 2:** run `uv run pytest tests/ -q` — full green.
- [ ] **Step 3:** update the reseed script to insert the 108 champion's scored elites into island 1 (quality) rather than a single archive.
- [ ] **Step 4:** `uv run pre-commit run -a` green.
- [ ] **Step 5: Commit** — `"Scheme v24 (islands); reseed champion into a quality island; doc #4 done"`.

**Post-plan (manual, not a task):** stop the optimizer, delete stale `run/blackboard.archive.jsonl` (single-archive), run the reseed, restart with `JED_PROPOSER_REPLICAS=4`, and confirm 4 island workers alive + the global best shipping.

## Self-Review

- **Coverage:** IslandSet (T1), novelty (T2), stall/reset/migration (T3), Blackboard integration + global-best ship (T4), worker wiring (T5), N-worker spawn (T6), scheme/reseed/rollout (T7) — every spec section maps to a task.
- **Type consistency:** `island: int` on `Record`; `IslandSet.insert(i, elite)`, `global_best_elite() -> Elite | None`, `global_champion() -> Record | None`, `island_best(i) -> Record | None` used consistently T4/T5.
- **Ambiguity resolved:** the novelty island is `archive.Archive` + an `i == 0` insert branch (not a subclass); reset seeds from `best_island()`'s random elite; shipping reads `global_champion()`/`reship_islands` only.
- **Risk guard:** T4 keeps a temporary `self.archive` property so the module imports mid-refactor; T5 removes it once all readers move to islands.
