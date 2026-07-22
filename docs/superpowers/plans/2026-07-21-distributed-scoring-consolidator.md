# Distributed scoring + consolidator — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Double scoring throughput by adding dylan as a second GPU host, and restructure
the archive write path into a lock-free map-reduce (per-worker shard files → one
consolidator).

**Architecture:** Approach B — dylan is serving-only; green's workers distribute replay
calls across all four `llama-server` endpoints (green×2 + dylan×2) with per-endpoint
failover. Workers write each scored candidate to a race-free per-entry shard file; a single
`consolidator` filters refusals, dedups by template, Pareto-merges the canonical archive,
and writes `total_score_est` to its own status file. The continuous `score_daemon` becomes
on-demand-before-ship.

**Tech Stack:** Python 3.13, `uv`, pytest, `aicomp_sdk`, llama.cpp `llama-server` over HTTP,
existing `jed_attack.campaign` package. Full design:
[docs/superpowers/specs/2026-07-21-distributed-scoring-consolidator-design.md](../specs/2026-07-21-distributed-scoring-consolidator-design.md).

## Global Constraints

- **Consolidator is the SOLE writer of `archive.jsonl`.** Workers never call
  `archive.insert`; they only write shard files. (`record_message` is the one change point.)
- **Shard files are race-free per-entry files**: one candidate per file, written
  temp-file + `os.replace`; the temp name must NOT match the consolidator's `*.json` glob.
- **`total_score_est` goes to `run/consolidator_status.json`, NEVER `score.json`** —
  `score.json` stays owned by the on-demand ground-truth `score.py` so estimate and real
  replay never clobber.
- **Notes are unchanged** — workers keep writing to the shared `"prompt_opt"` knowledge
  producer; do not touch `knowledge.py` or `_feedback_digest`.
- **Failover is idempotent**: replays are greedy-deterministic, so retrying a candidate on
  another endpoint is safe; all-endpoints-down degrades to severity 0 for that trial (never
  raises out of `score_prompt`).
- **Phase 1 must be complete + green-deployable before Phase 2.** dylan code paths must
  no-op when only the localhost endpoint is configured (`resolve_endpoints` falls back to
  the single default).
- **Never Kaggle-submit** anything as part of this work.
- Follow repo style: `uv run pre-commit run -a` green (ruff ≤88 cols, ty, pytest); Google
  docstrings; `logging` not `print`; no `from __future__ import annotations`.

---

## File Structure

- Create `src/jed_attack/campaign/shards.py` — per-entry shard write/claim.
- Create `src/jed_attack/campaign/consolidator.py` — the reduce process.
- Modify `src/jed_attack/campaign/config.py` — `SHARDS_DIR`, `CONSOLIDATOR_STATUS_FILE`,
  `CONSOLIDATE_INTERVAL_S`; `ensure_dirs` adds `SHARDS_DIR`.
- Modify `src/jed_attack/campaign/prompt_opt.py` — `record_message` writes a shard (not
  `archive.insert`); `_worker_id`; `score_prompt` distributes across endpoints via failover.
- Modify `src/jed_attack/harness/models.py` — add `resolve_endpoints`.
- Modify `src/jed_attack/campaign/score.py` — `finding_failover` helper; distribute the
  ground-truth replay across endpoints.
- Modify `scripts/campaign_watchdog.sh` — supervise `consolidator` instead of `score_daemon`.
- Modify `scripts/launch_optimizer_swarm.sh` — export `JED_WORKER_ID=$i` per worker.
- Create `scripts/serve_dylan.sh` — dylan llama-server launcher (Phase 2).
- Create `scripts/verify_score_identity.py` — cross-host severity check (Phase 2).
- Modify `tests/test_campaign.py` — tests for all Python units above.

---

# PHASE 1 — consolidation on green (single-host, fully testable)

### Task 1: config constants for shards + consolidator

**Files:**
- Modify: `src/jed_attack/campaign/config.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces: `config.SHARDS_DIR: Path`, `config.CONSOLIDATOR_STATUS_FILE: Path`,
  `config.CONSOLIDATE_INTERVAL_S: float`; `ensure_dirs()` creates `SHARDS_DIR`.

- [ ] **Step 1: Write the failing test**

```python
def test_config_shards_constants_and_ensure_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Shard/consolidator paths + interval exist, and ensure_dirs creates the shards dir."""
    from jed_attack.campaign import config

    assert config.SHARDS_DIR == config.CAMPAIGN_ROOT / "shards"
    assert (
        config.CONSOLIDATOR_STATUS_FILE
        == config.CAMPAIGN_ROOT / "consolidator_status.json"
    )
    assert config.CONSOLIDATE_INTERVAL_S > 0

    monkeypatch.setattr(config, "SHARDS_DIR", tmp_path / "shards")
    monkeypatch.setattr(config, "BUILD_NEXT_DIR", tmp_path / "bn")
    monkeypatch.setattr(config, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(config, "ARCHIVE_FILE", tmp_path / "arc.jsonl")
    monkeypatch.setattr(config, "CAMPAIGN_ROOT", tmp_path)
    config.ensure_dirs()
    assert (tmp_path / "shards").is_dir()
```

- [ ] **Step 2: Run it — expect fail** (`AttributeError: SHARDS_DIR`).
  Run: `uv run pytest tests/test_campaign.py::test_ensure_dirs_creates_shards_dir -v`

- [ ] **Step 3: Implement** — after `SCORE_CACHE` in `config.py`:

```python
SHARDS_DIR = CAMPAIGN_ROOT / "shards"  # per-worker scored-candidate shard files (map)
CONSOLIDATOR_STATUS_FILE = CAMPAIGN_ROOT / "consolidator_status.json"  # total_score_est
CONSOLIDATE_INTERVAL_S = float(os.getenv("JED_CONSOLIDATE_INTERVAL_S", "15"))
```

And add `SHARDS_DIR` to the `ensure_dirs()` tuple.

- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Commit** `config: shard dir + consolidator status/interval`.

---

### Task 2: `shards.py` — race-free per-entry write/claim

**Files:**
- Create: `src/jed_attack/campaign/shards.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces: `shards.write(entry: archive.Entry, shards_dir: Path, worker_id: str) -> Path`
  (atomic temp+replace, returns the final path); `shards.claim(shards_dir: Path) ->
  list[tuple[Path, archive.Entry]]` (glob `*.json`, parse, skip malformed/temp).

- [ ] **Step 1: Write the failing test**

```python
def test_shards_write_is_atomic_and_claim_reads_all(tmp_path: Path) -> None:
    """write produces a parseable *.json (no stray .tmp); claim returns every entry."""
    from jed_attack.campaign import archive, shards

    d = tmp_path / "shards"
    e1 = archive.Entry("A {urls}", 5, {"optimal": 80.0, "rules": 0.0, "hardened": 50.0}, 1.0)
    e2 = archive.Entry("B {addrs}", 8, {"optimal": 32.0, "rules": 32.0, "hardened": 32.0}, 2.0)
    p1 = shards.write(e1, d, "w1")
    p2 = shards.write(e2, d, "w2")
    assert p1.suffix == ".json" and p1.exists()
    assert not list(d.glob("*.tmp"))  # temp cleaned by the atomic replace
    claimed = shards.claim(d)
    templates = {entry.template for _, entry in claimed}
    assert templates == {"A {urls}", "B {addrs}"}
    assert {p for p, _ in claimed} == {p1, p2}
    # A half-written temp file is skipped, not parsed.
    (d / "w3-partial.json.tmp").write_text("{not json")
    assert len(shards.claim(d)) == 2
```

- [ ] **Step 2: Run — expect fail** (no module `shards`).
- [ ] **Step 3: Implement `src/jed_attack/campaign/shards.py`:**

```python
"""Per-entry shard files: the lock-free MAP half of the consolidator write path.

Each worker writes one scored :class:`archive.Entry` per file via temp-file + os.replace,
so a shard file is either fully present or absent — no partial reads, no lost writes, no
lock. The consolidator (:mod:`consolidator`) claims and deletes them.
"""

import json
import os
import uuid
from pathlib import Path

from jed_attack.campaign import archive


def write(entry: archive.Entry, shards_dir: Path, worker_id: str) -> Path:
    """Atomically write one scored entry to its own shard file.

    Args:
        entry: The scored candidate.
        shards_dir: The shards directory (created if missing).
        worker_id: Author id, only for readability of the filename.

    Returns:
        The final shard file path.
    """
    shards_dir.mkdir(parents=True, exist_ok=True)
    final = shards_dir / f"{worker_id}-{uuid.uuid4().hex}.json"
    tmp = final.with_name(final.name + ".tmp")  # .tmp never matches the *.json claim glob
    tmp.write_text(json.dumps(entry.to_json(), sort_keys=True), encoding="utf-8")
    os.replace(tmp, final)  # atomic
    return final


def claim(shards_dir: Path) -> list[tuple[Path, archive.Entry]]:
    """Read every complete shard file; skip malformed/half-written ones.

    Args:
        shards_dir: The shards directory.

    Returns:
        ``(path, entry)`` for each parseable shard, in sorted path order. The caller
        deletes the paths after merging.
    """
    if not shards_dir.exists():
        return []
    out: list[tuple[Path, archive.Entry]] = []
    for path in sorted(shards_dir.glob("*.json")):
        try:
            out.append((path, archive.Entry.from_json(json.loads(path.read_text()))))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out
```

- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Commit** `shards: race-free per-entry write/claim`.

---

### Task 3: `record_message` writes a shard, not the archive

**Files:**
- Modify: `src/jed_attack/campaign/prompt_opt.py` (`record_message`, add `_worker_id`,
  `import os`, `from jed_attack.campaign import ... shards`)
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `shards.write`, `config.SHARDS_DIR`.
- Produces: `record_message(template, hops, fitness) -> bool` (now always True; writes a
  shard + a knowledge note; does NOT touch `archive.jsonl`). `_worker_id() -> str`.

- [ ] **Step 1: Rewrite the existing `test_record_message_inserts_entry_and_notes`** (it
  currently asserts an archive insert) to assert the new shard behavior:

```python
def test_record_message_writes_shard_and_note_not_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """record_message writes a shard file + a note, and never touches the archive."""
    from jed_attack.campaign import archive, shards

    monkeypatch.setattr(prompt_opt.config, "SHARDS_DIR", tmp_path / "shards")
    monkeypatch.setattr(prompt_opt.config, "ARCHIVE_FILE", tmp_path / "arc.jsonl")
    monkeypatch.setattr(knowledge.config, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setenv("JED_WORKER_ID", "w7")
    fit = {"gates": {"optimal": 80.0, "rules": 0.0, "hardened": 53.0}, "cost_s": 1.2, "hops": 5}

    assert prompt_opt.record_message("SECRET_MARKER {urls}", 5, fit) is True

    claimed = shards.claim(tmp_path / "shards")
    assert [e.template for _, e in claimed] == ["SECRET_MARKER {urls}"]
    assert claimed[0][1].cost_s == 1.2
    assert not (tmp_path / "arc.jsonl").exists()  # archive untouched by the worker
    notes = knowledge.read_notes(tmp_path / "notes")
    assert any("cost_s=1.2" in n.text for n in notes)
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement.** In `prompt_opt.py` add `import os`, add `shards` to the
  campaign import, and replace the body of `record_message`:

```python
def _worker_id() -> str:
    """This worker's shard-file author id: ``JED_WORKER_ID`` env, else the pid."""
    return os.getenv("JED_WORKER_ID") or str(os.getpid())
```

```python
    gates = fitness["gates"]
    cost_s = fitness["cost_s"]
    entry = archive.Entry(template, hops, gates, cost_s)
    shards.write(entry, config.SHARDS_DIR, _worker_id())  # MAP: consolidator merges later
    knowledge.note(
        "prompt_opt",
        _NOTE_FORMAT.format(
            hops=hops,
            optimal=gates["optimal"],
            rules=gates["rules"],
            hardened=gates["hardened"],
            cost_s=cost_s,
            template=template,
        ),
    )
    return True
```

Update the docstring (no longer "Insert into the shared archive"; now "write a shard for
the consolidator + a feedback note"). Note: the CLI `main()` caller is unchanged — it now
writes a shard, which is fine.

- [ ] **Step 4: Run — expect pass.** Also run the full `tests/test_campaign.py` to catch
  any other test that assumed `record_message` inserted into the archive.
- [ ] **Step 5: Commit** `prompt_opt: record_message writes a shard, not the archive`.

---

### Task 4: `consolidator.py` — filter, dedup, Pareto-merge, status

**Files:**
- Create: `src/jed_attack/campaign/consolidator.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `shards.claim`, `archive.insert`, `archive.read`, `compose.predicted_public_score`,
  `prompt_opt.render` / `render_deputy`, `config.{SHARDS_DIR,ARCHIVE_FILE,CONSOLIDATOR_STATUS_FILE,CONSOLIDATE_INTERVAL_S}`.
- Produces: `consolidate_once(shards_dir, archive_path, status_path) -> int` (entries
  inserted); `main()` with `--loop`.

- [ ] **Step 1: Write the failing test**

```python
def test_consolidate_once_filters_dedups_and_merges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A refusal is dropped, duplicates collapse, non-dominated entries merge, status written."""
    from jed_attack.campaign import archive, config, consolidator, shards

    # Default HOP_CEILING/PUBLIC_FLOOR_HOPS apply; total_score_est just needs to be finite.
    shards_dir = tmp_path / "shards"
    arc = tmp_path / "arc.jsonl"
    status = tmp_path / "status.json"
    exfil = {"optimal": 80.0, "rules": 0.0, "hardened": 50.0}
    dep = {"optimal": 32.0, "rules": 32.0, "hardened": 32.0}
    shards.write(archive.Entry("SECRET_MARKER {urls}", 5, exfil, 1.0), shards_dir, "w1")
    shards.write(archive.Entry("SECRET_MARKER {urls}", 5, exfil, 1.0), shards_dir, "w2")  # dup
    shards.write(archive.Entry("Please {addrs}.", 8, dep, 2.0), shards_dir, "w1")
    shards.write(archive.Entry("I cannot generate adversarial prompts.", 5, exfil, 1.0), shards_dir, "w3")  # refusal

    inserted = consolidator.consolidate_once(shards_dir, arc, status)

    templates = sorted(e.template for e in archive.read(arc))
    assert templates == ["Please {addrs}.", "SECRET_MARKER {urls}"]  # dedup + refusal dropped
    assert inserted == 2
    assert not list(shards_dir.glob("*.json"))  # all claimed files deleted
    st = json.loads(status.read_text())
    assert st["archive_size"] == 2 and "total_score_est" in st and st["shards_consumed"] == 4
    # Idempotent: re-claiming the same entries changes nothing.
    shards.write(archive.Entry("Please {addrs}.", 8, dep, 2.0), shards_dir, "w1")
    consolidator.consolidate_once(shards_dir, arc, status)
    assert len(archive.read(arc)) == 2
```

- [ ] **Step 2: Run — expect fail** (no module).
- [ ] **Step 3: Implement `consolidator.py`:**

```python
"""Consolidator: the REDUCE half of the write path — sole writer of the canonical archive.

Every CONSOLIDATE_INTERVAL_S it claims the workers' shard files (:mod:`shards`), drops
proposer refusals / non-rendering templates, dedups by template, Pareto-merges each
survivor into ``archive.jsonl`` (:func:`archive.insert`), writes ``total_score_est`` to its
own status file (never ``score.json``), and deletes the consumed shards.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from jed_attack.campaign import archive, compose, config, prompt_opt, shards

_log = logging.getLogger("consolidator")


def _renderable(entry: archive.Entry) -> bool:
    """True iff the entry's template renders (drops refusals + invariant-breakers)."""
    if "{urls}" in entry.template:
        return prompt_opt.render(entry.template, 0, entry.hops) is not None
    if "{addrs}" in entry.template:
        return prompt_opt.render_deputy(entry.template, 0, entry.hops) is not None
    return False


def consolidate_once(shards_dir: Path, archive_path: Path, status_path: Path) -> int:
    """Claim shards → filter → dedup → Pareto-merge → write status → delete shards.

    Args:
        shards_dir: Where workers drop per-entry shard files.
        archive_path: The canonical Pareto archive (this process is its sole writer).
        status_path: Where to write ``total_score_est`` + counters.

    Returns:
        The number of entries newly inserted into the archive this cycle.
    """
    claimed = shards.claim(shards_dir)
    by_template: dict[str, archive.Entry] = {}
    for _, entry in claimed:
        if _renderable(entry):
            by_template.setdefault(entry.template, entry)  # dedup: first wins
    inserted = sum(
        1 for entry in by_template.values() if archive.insert(entry, archive_path)
    )
    for path, _ in claimed:
        path.unlink(missing_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "ts": time.time(),
                "total_score_est": compose.predicted_public_score(archive_path),
                "archive_size": len(archive.read(archive_path)),
                "shards_consumed": len(claimed),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _log.info("consolidated %d shard(s), inserted %d", len(claimed), inserted)
    return inserted


def main() -> None:
    """CLI: one consolidation pass, or ``--loop`` forever at CONSOLIDATE_INTERVAL_S."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true", help="run forever")
    args = parser.parse_args()
    config.ensure_dirs()
    while True:
        try:
            consolidate_once(
                config.SHARDS_DIR, config.ARCHIVE_FILE, config.CONSOLIDATOR_STATUS_FILE
            )
        except Exception:  # a bad cycle must never kill the loop
            _log.exception("consolidation cycle failed; retrying next tick")
        if not args.loop:
            break
        time.sleep(config.CONSOLIDATE_INTERVAL_S)


if __name__ == "__main__":
    main()
```

Note: `time.time()` is a real Python call (the Date.now ban is Workflow-script-only).

- [ ] **Step 4: Run — expect pass.** Run full `tests/test_campaign.py`.
- [ ] **Step 5: Commit** `consolidator: filter/dedup/Pareto-merge shards + status`.

---

### Task 5: watchdog supervises the consolidator; workers get a worker id

**Files:**
- Modify: `scripts/campaign_watchdog.sh` (swap `scoredaemon`→`consolidator`)
- Modify: `scripts/launch_optimizer_swarm.sh` (export `JED_WORKER_ID=$i`)

**Interfaces:** none (shell/ops). **This task has no unit test** — it is process wiring;
verify by the runtime smoke below, not pytest.

- [ ] **Step 1:** In `campaign_watchdog.sh`, replace the `start_daemon scoredaemon
  score_daemon` line with `start_daemon consolidator consolidator` (module
  `jed_attack.campaign.consolidator --loop`). Keep `start_daemon assemble assemble_daemon`.
  The score_daemon is no longer supervised (run on-demand before ship).
- [ ] **Step 2:** In `launch_optimizer_swarm.sh`, inside the `for i in $(seq 1 "$N")` loop,
  add `JED_WORKER_ID="$i"` to the worker's env (alongside `JED_PROPOSER_MODEL`/`JED_WANDB`).
- [ ] **Step 3: Runtime smoke (not pytest)** — documented for the deploy step, not run here:
  after deploy, `pgrep -f consolidator` shows one loop; `run/shards/` drains each cycle;
  `run/consolidator_status.json` updates; `run/archive.jsonl` grows; no `score_daemon` under
  the watchdog.
- [ ] **Step 4: Commit** `scripts: watchdog runs consolidator; workers get JED_WORKER_ID`.

---

**End of Phase 1.** At this point: `uv run pre-commit run -a` is green; the write path is
map-reduce on green alone (workers → shards → consolidator → archive), refusals/dupes are
filtered, and the continuous score_daemon is retired. This is independently deployable —
the endpoint list still resolves to localhost-only until Phase 2. **Deploy + restart the
green fleet here** (sync, wipe `run/shards/`, relaunch watchdog+swarm) and confirm the Phase
1 smoke before starting Phase 2.

---

# PHASE 2 — dylan scale-out (distributed endpoints)

### Task 6: `resolve_endpoints` (multi-endpoint per model)

**Files:**
- Modify: `src/jed_attack/harness/models.py`
- Test: `tests/test_campaign.py` (or `tests/test_models.py` if that is where model tests live)

**Interfaces:**
- Produces: `resolve_endpoints(model_key: str) -> list[str]` — `GPT_OSS_BASE_URLS` /
  `GEMMA_BASE_URLS` comma-split, else `[resolve_base_url(model_key)]`. `resolve_base_url`
  unchanged (still the single-endpoint default and the failover tail).

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_endpoints_splits_env_else_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_endpoints reads *_BASE_URLS (comma list), else the single default."""
    from jed_attack.harness import models

    monkeypatch.delenv("GPT_OSS_BASE_URLS", raising=False)
    assert models.resolve_endpoints("gpt_oss") == [models.resolve_base_url("gpt_oss")]
    monkeypatch.setenv(
        "GPT_OSS_BASE_URLS", "http://localhost:8080/v1, http://192.168.1.220:8080/v1"
    )
    assert models.resolve_endpoints("gpt_oss") == [
        "http://localhost:8080/v1",
        "http://192.168.1.220:8080/v1",
    ]
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement** in `models.py` after `resolve_base_url`:

```python
def resolve_endpoints(model_key: str) -> list[str]:
    """All served base URLs for a model: the ``*_BASE_URLS`` list, else the single default.

    green sets ``GPT_OSS_BASE_URLS`` / ``GEMMA_BASE_URLS`` to a comma-separated list
    (localhost first, then dylan) so scoring fans out across hosts; unset falls back to the
    one :func:`resolve_base_url` default, so single-host runs are unchanged.

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.

    Returns:
        Ordered, non-empty list of base URLs (localhost/default is the failover tail).
    """
    env_var = "GPT_OSS_BASE_URLS" if model_key == "gpt_oss" else "GEMMA_BASE_URLS"
    raw = os.environ.get(env_var, "")
    endpoints = [u.strip() for u in raw.split(",") if u.strip()]
    return endpoints or [resolve_base_url(model_key)]
```

- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Commit** `models: resolve_endpoints for multi-host scoring`.

---

### Task 7: `finding_failover` + distribute `score_prompt` across endpoints

**Files:**
- Modify: `src/jed_attack/campaign/score.py` (add `finding_failover`)
- Modify: `src/jed_attack/campaign/prompt_opt.py` (`score_prompt` builds per-(model,endpoint)
  factories and scores each cell via `finding_failover`)
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces: `score.finding_failover(chain, factories_ordered, guardrail_factory) ->
  finding | None` — tries each factory in order, returns the first non-raising result
  (including a legitimate `None` no-fire), returns `None` only if every factory raised.
- `score_prompt` distributes: `endpoints = {m: resolve_endpoints(m)}`,
  `factories = {(m, ep): factory(m, ep)}`, each cell picks a round-robin start endpoint and
  fails over through the model's list.

- [ ] **Step 1: Write the failing tests** (unit-test the helper directly — clean and
  live-model-free):

```python
def test_finding_failover_tries_next_on_connection_error() -> None:
    """finding_failover skips a raising factory and returns the next factory's finding."""
    from jed_attack.campaign import score

    sentinel = {"predicates": [{"severity": 5}]}

    def dead() -> object:
        raise ConnectionError("refused")

    def live() -> object:
        return object()

    calls: list[object] = []

    def fake_finding(chain, factory, guardrail):
        calls.append(factory)
        if factory is dead:
            raise ConnectionError("refused")
        return sentinel

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(score, "_finding", fake_finding)
        out = score.finding_failover(("m",), [dead, live], lambda: None)
    assert out is sentinel and calls == [dead, live]


def test_finding_failover_returns_none_when_all_endpoints_dead() -> None:
    """All factories raising -> None (scored 0 that trial), never propagates."""
    from jed_attack.campaign import score

    def boom(chain, factory, guardrail):
        raise ConnectionError("refused")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(score, "_finding", boom)
        assert score.finding_failover(("m",), [object(), object()], lambda: None) is None
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement.** In `score.py`:

```python
def finding_failover(
    chain: Sequence[str],
    factories_ordered: list[Callable[[], Any]],
    guardrail_factory: Callable[[], Any],
) -> dict[str, Any] | None:
    """Replay a chain, failing over across an ordered list of endpoint factories.

    Replays are greedy-deterministic, so retrying on another endpoint is safe. Returns the
    first factory's finding (a legitimate ``None`` no-fire counts as success); returns
    ``None`` only if EVERY factory raised (all endpoints down -> scored 0 this trial).

    Args:
        chain: The candidate message chain.
        factories_ordered: Agent factories to try in order (round-robin start + tail).
        guardrail_factory: The guardrail to replay under.

    Returns:
        The finding dict, or None.
    """
    for factory in factories_ordered:
        try:
            return _finding(chain, factory, guardrail_factory)
        except Exception:  # endpoint unreachable — fail over to the next
            _log.warning("scoring endpoint failed; failing over", exc_info=True)
            continue
    return None
```

In `prompt_opt.score_prompt`, replace the `factories`/`cell` block:

```python
    endpoints = {m: resolve_endpoints(m) for m in models}
    factories = {
        (m, ep): llama_server_agent_factory(m, ep) for m in models for ep in endpoints[m]
    }

    def cell(model_offset: int, model: str, trial: int, gname: str) -> tuple:
        message = fam.render(template, model_offset * trials + trial, k)
        if message is None:
            return (gname, model, 0.0, 0.0)
        eps = endpoints[model]
        start = (model_offset * trials + trial) % len(eps)
        ordered = [factories[(model, eps[(start + i) % len(eps)])] for i in range(len(eps))]
        started = time.monotonic()
        finding = score.finding_failover((message,), ordered, GATE_GUARDRAILS[gname])
        elapsed = time.monotonic() - started
        sev = gate._severity(finding["predicates"]) if finding else 0
        return (gname, model, float(sev), elapsed)
```

(Add `resolve_endpoints` to the `from jed_attack.harness.models import ...` line.)

- [ ] **Step 4: Run — expect pass.** Run full `tests/test_campaign.py`.
- [ ] **Step 5: Commit** `score/prompt_opt: distribute scoring across endpoints w/ failover`.

---

### Task 8: distribute the ground-truth `score.py` replay across endpoints

**Files:**
- Modify: `src/jed_attack/campaign/score.py` (`predict_public_score` builds
  per-(model,endpoint) factories; `_replay_public` calls `finding_failover`)
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `resolve_endpoints`, `finding_failover`. Behavior identical single-endpoint; a
  dead dylan fails over to green.

- [ ] **Step 1: Write the failing test** — assert `predict_public_score` builds one factory
  per (model, endpoint) when two endpoints are configured (monkeypatch `resolve_endpoints`
  to 2 URLs and `llama_server_agent_factory` to a recorder; assert both endpoints used):

```python
def test_score_replay_distributes_across_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """predict_public_score builds a factory per (model, endpoint) from resolve_endpoints."""
    from jed_attack.campaign import compose, config, score

    monkeypatch.setattr(config, "SCORE_CACHE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(compose, "compose_pool", lambda _p: ["SECRET_MARKER m"])
    built: list[tuple] = []
    monkeypatch.setattr(
        score, "resolve_endpoints", lambda m: [f"http://a/{m}", f"http://b/{m}"], raising=False
    )
    monkeypatch.setattr(
        score, "llama_server_agent_factory", lambda m, ep: built.append((m, ep)) or (lambda: None)
    )
    monkeypatch.setattr(score, "finding_failover", lambda *a, **k: None)  # no real replay
    score.predict_public_score(models=("gpt_oss",), max_new_replays=0)
    assert ("gpt_oss", "http://a/gpt_oss") in built and ("gpt_oss", "http://b/gpt_oss") in built
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement.** In `score.py` add `resolve_endpoints` to the models import,
  and in `predict_public_score` replace `factories = {m: ...}` with per-(model, endpoint)
  factories; in the replay call, build the model's ordered factory list and call
  `finding_failover`. Keep the cache key on `(chain, model)` (endpoint-agnostic — replays
  are deterministic across endpoints once the identity check passes).
- [ ] **Step 4: Run — expect pass.** Run full suite.
- [ ] **Step 5: Commit** `score: distribute ground-truth replay across endpoints`.

---

### Task 9: dylan serving + score-identity verification

**Files:**
- Create: `scripts/serve_dylan.sh`
- Create: `scripts/verify_score_identity.py`

**This task is infra + a runtime gate, not TDD.** The unit-testable pieces (endpoints,
failover) are Tasks 6–8; this task stands up dylan and *verifies* score parity.

- [ ] **Step 1:** Get llama.cpp + the two GGUFs onto dylan. Build llama.cpp with CUDA on
  dylan (its GPUs: 3090 sm_86, TITAN RTX sm_75). Transfer
  `gpt-oss-20b-Q4_K_M.gguf` + `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (rsync over the LAN; if
  green→dylan SSH is blocked, laptop-mediate). Put them under `~/models/` on dylan.
- [ ] **Step 2:** Write `scripts/serve_dylan.sh` mirroring green's serve command, bound to
  `0.0.0.0`, one model per card (gpt_oss→3090, gemma→TITAN RTX), ports 8080/8081, `--jinja
  -ngl 999`, `-np 4 -c 8192` (tune up if VRAM allows). Launch under tmux. Open dylan's
  firewall for green→8080/8081 (ufw/iptables) — confirm with `nc -z 192.168.1.220 8080`
  from green.
- [ ] **Step 3:** Write `scripts/verify_score_identity.py`: for a fixed set of ~5 known
  rendered candidates per family, replay each under `OptimalGuardrail` on green's endpoint
  and on dylan's endpoint for the same model, and assert equal severity + `score_cell`
  hash (or report the diff). Run it from green (which can reach both).
- [ ] **Step 4: GATE.** If severities match, proceed to Task 10 (enable dylan). If they
  diverge materially, STOP the scale-out: leave green-only endpoints (Phase 1 stands),
  record the divergence, and escalate — dylan cannot be trusted for scoring.
- [ ] **Step 5: Commit** `scripts: dylan serving + cross-host score-identity check`.

---

### Task 10: wire green's endpoint lists + deploy

**Files:**
- Modify: green `.env` (runtime, not committed) — or `scripts/launch_optimizer_swarm.sh` to
  export the endpoint lists.

**Infra/ops task — no unit test.**

- [ ] **Step 1:** Set on green (`.env` or the launcher):
  `GPT_OSS_BASE_URLS=http://localhost:8080/v1,http://192.168.1.220:8080/v1` and
  `GEMMA_BASE_URLS=http://localhost:8081/v1,http://192.168.1.220:8081/v1`.
- [ ] **Step 2:** Restart the green swarm so workers pick up the endpoint lists; the
  consolidator/composer are unaffected.
- [ ] **Step 3: Runtime verify:** worker throughput (candidates scored / minute) roughly
  doubles; `consolidator_status.json` cadence unchanged; kill dylan's servers briefly and
  confirm scoring continues (failover to green) without zero-score spikes.
- [ ] **Step 4: Commit** any launcher change `scripts: green endpoint lists (green+dylan)`.

---

## Post-plan

- Whole-branch review (the SDD final review).
- Deploy Phase 1 at its checkpoint above; deploy Phase 2 after the Task 9 identity gate
  passes. Both are fleet restarts (sync, wipe `run/shards/`, relaunch) — no Kaggle submit.
- Update the two campaign READMEs / serving docs if they describe the single-host layout.
