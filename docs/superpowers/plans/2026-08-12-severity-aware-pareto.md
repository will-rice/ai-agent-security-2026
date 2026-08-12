# Severity-aware 4-D Pareto + metric clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the QD-Pareto archive a severity-aware 4-D score vector (per-model throughput AND severity), ship by board-density, and rename the misleading wandb objective metrics.

**Architecture:** `Elite` carries `throughput` + `severity` dicts; `dominates` is Pareto over all `2*len(MODELS)` components; `ship_set`/champion rank by board-density `(severity+NOVELTY)/200/(gen_chars+FIXED)`. `_shape_elites` already reads per-model severity — it just stores it. The wandb block is renamed for clarity and gains `board_mean_models` + frontier gauges.

**Tech Stack:** Python 3, pydantic v2 (unaffected), `uv`, `pytest`, `ruff` (88 col), `ty`.

## Global Constraints

- `uv run pre-commit run -a` must stay green (ruff format, ruff 88-col E501, ty, pytest). Use `set -a && . ./.env && set +a` before `uv run`.
- Absolute imports; Google-style docstrings; no `from __future__ import annotations`.
- Never guess an interface — read the source. `config.MODELS`, `config.GATE_GUARDRAIL_NAME` (`"optimal"`), `config.NOVELTY_PER_CELL` (2.0), `config.FIXED_CHARS`, `config.ARCHIVE_FRONTIER_CAP` all exist.
- Severity source is the raw `_SEVERITY_W`-weighted value already stored in `severity_by_model[GATE_GUARDRAIL_NAME][model]` (weighted at `submission_score.py:528`) — do NOT re-weight.
- Preserve the blackboard MIN objective path (logging) and the frontier-authoritative shipping gate from the prior build. Leave unrelated `.claude/skills/submit-kernel/*` working-tree edits untouched.

---

### Task 1: board_density helper

**Files:**
- Modify: `src/jed_attack/campaign/submission_score.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces: `submission_score.board_density(severity: float, gen_chars: float, model: str) -> float` = `0.0` if `severity <= 0` else `(severity + config.NOVELTY_PER_CELL) / 200.0 / (gen_chars + config.FIXED_CHARS[model])`.

- [ ] **Step 1: Write the failing test**

```python
def test_board_density_rewards_severity_and_leanness() -> None:
    from jed_attack.campaign import config
    from jed_attack.campaign.submission_score import board_density
    lean_hi = board_density(16.0, 150.0, "gpt_oss")
    lean_lo = board_density(1.0, 150.0, "gpt_oss")
    fat_hi = board_density(16.0, 500.0, "gpt_oss")
    assert lean_hi > lean_lo          # higher severity -> higher density
    assert lean_hi > fat_hi           # leaner -> higher density
    assert board_density(0.0, 150.0, "gpt_oss") == 0.0  # non-firing -> 0
    exp = (16.0 + config.NOVELTY_PER_CELL) / 200.0 / (150.0 + config.FIXED_CHARS["gpt_oss"])
    assert lean_hi == exp
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_campaign.py::test_board_density_rewards_severity_and_leanness -v` -> FAIL.

- [ ] **Step 3: Implement** next to `throughput` in `submission_score.py`:

```python
def board_density(severity: float, gen_chars: float, model: str) -> float:
    """Per-model board value-per-char: the LB board a shape earns per char of budget.

    Mirrors _firing_templates' per-candidate board ((severity + NOVELTY_PER_CELL) / 200)
    over its per-candidate cost (gen_chars + FIXED_CHARS[model]). This is what fills a
    fixed char budget best, so ship_set ranks by it. 0 for non-firing (severity <= 0).
    """
    if severity <= 0.0:
        return 0.0
    return (severity + config.NOVELTY_PER_CELL) / 200.0 / (
        gen_chars + config.FIXED_CHARS[model]
    )
```

- [ ] **Step 4: Run to verify it passes**, then full suite.
- [ ] **Step 5: Commit** — `feat(score): board-density helper (per-model value-per-char)`.

---

### Task 2: Elite gains severity + 4-D dominance + ship-by-density

**Files:**
- Modify: `src/jed_attack/campaign/archive.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `submission_score.board_density`, `config.MODELS`, `config.ARCHIVE_FRONTIER_CAP`.
- Produces:
  - `Elite` gains `severity: dict[str, float]` (per model), AFTER `throughput` and BEFORE `diagnosis` in field order (asdict/persistence carries it).
  - `dominates(a, b)` — Pareto over BOTH throughput and severity across `config.MODELS` (all components `>=`, at least one `>`).
  - `Archive.ship_set()` — the frontier ranked by `sum(board_density(sev_m, gen_chars_m, m))` desc, top `ARCHIVE_FRONTIER_CAP`. Since the Elite stores throughput (not gen_chars), recover per-model gen_chars from throughput: `gen_chars_m = 1/throughput_m - FIXED_CHARS[m]` when `throughput_m > 0`, else non-firing (density 0).
  - `Archive.from_jsonl` tolerates a missing `severity` key (default all-zero).

- [ ] **Step 1: Write the failing tests**

```python
def test_elite_4d_dominance_uses_throughput_and_severity() -> None:
    from jed_attack.campaign import archive as ar
    def e(tg, tm, sg, sm):
        return ar.Elite("t", "exfil", {"gpt_oss": tg, "gemma_4": tm},
                        {"gpt_oss": sg, "gemma_4": sm}, "", "forge", 5)
    lean_weak = e(0.006, 0.007, 1.0, 1.0)
    lean_strong = e(0.006, 0.007, 16.0, 16.0)
    assert ar.dominates(lean_strong, lean_weak)      # equal throughput, higher severity
    assert not ar.dominates(lean_weak, lean_strong)
    tradeoff = e(0.009, 0.004, 1.0, 1.0)             # leaner but weaker vs lean_strong
    assert not ar.dominates(lean_strong, tradeoff)   # neither dominates -> both survive
    assert not ar.dominates(tradeoff, lean_strong)


def test_ship_set_ranks_by_board_density(monkeypatch) -> None:
    from jed_attack.campaign import archive as ar, config
    # two shapes on the frontier (a tradeoff pair): weak-lean vs strong-balanced
    weak = ar.Elite("WEAK", "exfil", {"gpt_oss": 0.009, "gemma_4": 0.009},
                    {"gpt_oss": 1.0, "gemma_4": 1.0}, "", "forge", 5)
    strong = ar.Elite("STRONG", "exfil", {"gpt_oss": 0.006, "gemma_4": 0.006},
                      {"gpt_oss": 16.0, "gemma_4": 16.0}, "", "forge", 6)
    arch = ar.Archive()
    arch.insert(weak); arch.insert(strong)
    ship = arch.ship_set()
    assert {e.text for e in ship} == {"WEAK", "STRONG"}      # both on frontier (tradeoff)
    assert ship[0].text == "STRONG"   # higher board-density ships first
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement.** Add `severity: dict[str, float]` to `Elite` (after `throughput`). Widen `dominates`:

```python
def dominates(a: Elite, b: Elite) -> bool:
    """True if a Pareto-dominates b over per-model throughput AND severity."""
    comps = [(a.throughput[m], b.throughput[m]) for m in config.MODELS]
    comps += [(a.severity[m], b.severity[m]) for m in config.MODELS]
    ge = all(av >= bv for av, bv in comps)
    gt = any(av > bv for av, bv in comps)
    return ge and gt
```

Rewrite `ship_set` to rank by board-density (import `board_density` from `submission_score`):

```python
def ship_set(self) -> list[Elite]:
    from jed_attack.campaign.submission_score import board_density

    def density(e: Elite) -> float:
        total = 0.0
        for m in config.MODELS:
            t = e.throughput[m]
            if t <= 0.0:
                continue
            gen_chars = 1.0 / t - config.FIXED_CHARS[m]
            total += board_density(e.severity[m], gen_chars, m)
        return total

    return sorted(self.frontier(), key=density, reverse=True)[
        : config.ARCHIVE_FRONTIER_CAP
    ]
```

In `from_jsonl`, tolerate a missing `severity` key:

```python
data = json.loads(line)
data.setdefault("severity", {m: 0.0 for m in config.MODELS})
arch.insert(Elite(**data))
```

- [ ] **Step 4: Run to verify pass**, then full suite (the Task-4-of-prior-build `test_archive_*` tests construct `Elite` positionally — update them to pass the new `severity` dict).
- [ ] **Step 5: Commit** — `feat(archive): severity axis (4-D Pareto) + ship by board-density`.

---

### Task 3: _shape_elites stores per-model severity

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- `_shape_elites` builds each `Elite` with `severity={m: severity_by_model[GATE_GUARDRAIL_NAME].get(m, 0.0) for m in config.MODELS}` from the SAME `per_message` entry it already reads for throughput. A non-firing model (severity 0) already yields `throughput` 0 there; severity 0 is consistent.

- [ ] **Step 1: Write the failing test** — model on the existing `_shape_elites` test; assert the produced Elite's `.severity` dict matches the fake `severity_by_model["optimal"]` per model, and that a zero-severity model gives `severity[m] == 0.0` AND `throughput[m] == 0.0`.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** — read the severity off the same `MessageScore` and pass it into `Elite(..., severity=...)`. Read the current `_shape_elites` to match its structure exactly; do not change the throughput/firing logic.
- [ ] **Step 4: Run to verify pass**, full suite.
- [ ] **Step 5: Commit** — `feat(loop): _shape_elites carries per-model severity`.

---

### Task 4: render helpers + prompts.toml show both axes

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py`, `src/jed_attack/campaign/prompts.toml`
- Test: `tests/test_campaign.py`

**Interfaces:**
- `_render_opro_table` / `_render_parents` render severity next to throughput per model (e.g. `gpt_oss(thru=0.0058, sev=976)`). No scalar objective leaks. Row sort may key on summed board-density (display only).
- `prompts.toml` OPRO/parents framing mentions BOTH axes (leaner AND higher-severity win, esp. on the binding model).

- [ ] **Step 1: Write the failing test** — extend the OPRO-table test: assert the rendered table contains a severity marker per model (`sev` and the numeric severity) alongside throughput, and still shows both model names, no scalar. Keep a guard that the new prose marker is absent from the schema-only dump (mirror the prior build's `test_prose_markers_absent_from_schema_only_dump`).
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** the render change + prompts.toml prose (match the existing `{{OPRO}}`/`{{PARENTS}}` voice). Read the current `_render_opro_table`/`_render_parents` first.
- [ ] **Step 4: Run to verify pass**, full suite; confirm existing prompt tests still bind.
- [ ] **Step 5: Commit** — `feat(opro): show per-model severity alongside throughput`.

---

### Task 5: wandb metric renames + board_mean_models + gauges + comment fix

**Files:**
- Modify: `src/jed_attack/campaign/optimize_prompts.py`
- Test: `tests/test_campaign.py`

**Interfaces:** in the `_log_wandb` dict (`optimize_prompts.py` ~538-581):
- Rename: `{m}_objective`→`board_{m}`; `best_objective`→`best_board_min_models`; `batch_mean_objective`→`batch_mean_board_min_models`; `best_gen_chars_bottleneck`→`champion_bottleneck_gen_chars`; `n_shapes`→`champion_n_shapes`; `refine_objective_gain`→`refine_board_gain`; `replay_s_{m}`→`replay_seconds_{m}`. In `_batch_score_metrics`, `batch_severity_{m}`→`batch_severity_raw_{m}`.
- Add `board_mean_models` = `mean(_project_boards(best_score).values())`.
- Add frontier gauges from `board.archive`: `frontier_size`, `frontier_families` (distinct `family`), `frontier_distinct_throughput` (distinct throughput-vector count), `frontier_distinct_severity` (distinct severity-vector count).
- Fix the stale comment at lines ~548-554: `best_objective` (now `best_board_min_models`) is the MIN over columns, not the mean.

- [ ] **Step 1: Write the failing test** — a `_log_wandb` capture test (monkeypatch the logger/`_log_wandb` sink, or call the metrics-builder if factored) asserting the new keys are present and the old ones absent, `board_mean_models == mean(board_gpt_oss, board_gemma_4)`, and `frontier_size`/`frontier_families` reflect a seeded archive. If the metrics dict is inline, factor the dict-building into a small pure helper `_generation_wandb_metrics(...)` so it is unit-testable, and call it from `_log_wandb`.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** the renames, additions, comment fix (and the optional extraction helper). Grep the repo + tests for the OLD key names and update any assertion that referenced them.
- [ ] **Step 4: Run to verify pass**, full suite.
- [ ] **Step 5: Commit** — `feat(metrics): clear wandb objective names + board_mean_models + frontier gauges`.

---

### Task 6: End-to-end + full green

**Files:**
- Test: `tests/test_campaign.py`

- [ ] **Step 1: Write an end-to-end test** — seed an archive from two shapes with DIFFERENT severity at the SAME throughput; run one generation with fakes; assert (a) the frontier keeps both (4-D non-domination), (b) `ship_set` orders the higher-severity shape first, (c) the shipped `attack.py` leads with the higher-severity shape's text, (d) `type_to_response_format_param(SubmissionBatch)` still builds. Make each assertion bind (revert-to-throughput-only would fail it).
- [ ] **Step 2: Run to verify it fails** (before the vector change would be present in the fake path) / passes on the built tree.
- [ ] **Step 3: Full gate** — `uv run pre-commit run -a`: ruff, ruff-format, ty, pytest all PASS.
- [ ] **Step 4: Commit** — `test(loop): end-to-end severity-aware Pareto + shipping`.

---

## Self-Review

**Spec coverage:** board_density (T1), 4-D dominance + severity field + ship-by-density + persistence tolerance (T2), `_shape_elites` severity (T3), render both axes (T4), metric renames + `board_mean_models` + gauges + comment fix (T5), end-to-end + green (T6). Migration (delete `run/blackboard.archive.jsonl` on deploy) is an operational step handled at restart, not a code task — noted here and in the deploy.

**Placeholder scan:** T3/T4/T5 steps point the implementer to read the current function bodies rather than repeating them, because they are single, existing functions the implementer must edit in place; every NEW function/carrying the interface has concrete code or a concrete assertion. No TBD/"handle edge cases".

**Type consistency:** `Elite.severity: dict[str, float]` used in T2/T3/T6; `board_density(severity, gen_chars, model) -> float` in T1/T2; `dominates` 4-D in T2/T6; metric keys in T5 match the spec table.

**Scope:** one subsystem (the score vector + its shipping + its telemetry); a single plan is correct.
