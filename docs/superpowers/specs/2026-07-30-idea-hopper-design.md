# Idea Hopper — Design

**Date:** 2026-07-30
**Status:** Draft for user review
**Component:** `src/jed_attack/campaign/idea_hopper.py`, `run/idea_hopper/`

## Goal

Automate the discovery and prioritization of next experiments for the JED Kaggle attack optimizer.

The first version is a **local-only, read-mostly idea hopper**: it reads our existing evidence, generates ranked experiment hypotheses, and writes a queue for a human/agent to choose from. It does **not** edit code, start scorer jobs, submit to Kaggle, or call network APIs.

## Motivation

The useful idea sources are now scattered:

- optimizer history in `run/blackboard.jsonl`
- artifact scores in `run/artifact_sweeps/**/metrics.json` and `run/artifact_eval_current/latest_metrics.json`
- public-kernel mining reports under `run/kaggle_research_cron/`
- Kaggle discussion caches under `run/kaggle_research_cron/latest_*_discussions.json`
- predicate and guardrail source in the SDK/repo
- strategy constraints in `docs/strategy.md`

Manually reading all of that works once, but it does not scale. The hopper turns those artifacts into a small, refreshed list of concrete experiments with evidence and gates. It should be a goblin with a clipboard, not a goblin with a chainsaw.

## Core decisions

1. **Local-only first.** The hopper consumes already-cached Kaggle/kernel/discussion data. Existing cron jobs can refresh those caches; the hopper itself does not require network.
2. **No automatic code changes.** The hopper only writes reports/queues under `run/idea_hopper/`.
3. **Evidence-backed ideas only.** Every generated idea must cite at least one source artifact and include the observation that produced it.
4. **Score-gated output.** Every idea must name an acceptance gate, usually a targeted test plus `artifact_lb_est_public` or raw/sec comparison.
5. **Rank by actionability, not novelty theater.** Prefer ideas that are small, testable, and likely to improve calibrated score or private-transfer robustness.

## Inputs

### Primary local inputs

- `docs/strategy.md`
- `src/jed_attack/campaign/assemble.py`
- `src/jed_attack/campaign/prompts.toml`
- `src/jed_attack/campaign/config.py`
- `run/blackboard.jsonl` when present
- `run/artifact_sweeps/**/metrics.json` when present
- `run/artifact_eval_current/latest_metrics.json` when present
- `run/kaggle_research_cron/latest_public_kernel_mining.md` when present
- `run/kaggle_research_cron/public_kernel_latest_mining_*/decoded_findings.md` when present
- `run/kaggle_research_cron/latest_*_discussions.json` when present

### Optional local inputs

- `run/judge_study*/rows.jsonl`
- `run/logs/*.log`
- `run/kaggle_research_cron/latest_kernel_scores.csv`
- built generated artifacts such as `run/build_next/attack.py`

Missing optional inputs are not errors; they reduce confidence on affected idea families.

## Outputs

The hopper writes only gitignored runtime artifacts:

- `run/idea_hopper/latest.md` — human-readable ranked queue
- `run/idea_hopper/latest.json` — structured version of the same queue
- `run/idea_hopper/queue.jsonl` — append-only history of generated ideas
- `run/idea_hopper/state.json` — lightweight de-duplication state

Each idea row has this shape:

```json
{
  "id": "20260730-001",
  "title": "Test REPLAY_SAFE_FRAC=0.97 in generated artifact",
  "hypothesis": "Top public latency-split kernels use 0.97; it may reduce timeout risk without dropping candidate count locally.",
  "source_evidence": [
    {
      "path": "run/kaggle_research_cron/public_kernel_latest_mining_20260730_000514/decoded_findings.md",
      "observation": "Latency-split family uses REPLAY_SAFE_FRAC=0.97."
    }
  ],
  "likely_files": ["src/jed_attack/campaign/assemble.py", "tests/test_campaign.py"],
  "category": "replay_sizing",
  "priority": "medium",
  "expected_upside": "Improve calibrated LB estimate by avoiding replay timeout/overpacking.",
  "risk": "Can underfill and lower public score if current 0.99 is safe.",
  "acceptance_gate": "artifact_lb_est_public beats current build_next by >= 0.25 and campaign tests pass.",
  "status": "proposed"
}
```

## Architecture

### 1. Collectors

Collectors normalize raw evidence into small records:

- `BlackboardCollector`: samples high-objective records, recent near-misses, and model asymmetries from `run/blackboard.jsonl`.
- `ArtifactMetricsCollector`: reads sweep/current artifact metrics and extracts score deltas, candidate counts, selected templates, replay cost, and calibrated LB estimates.
- `KernelMiningCollector`: reads decoded public-kernel findings and public kernel score snapshots.
- `DiscussionCollector`: reads cached Kaggle discussions and extracts new evaluator/scoring/private-transfer claims.
- `CodeSurfaceCollector`: reads current generated attack source/config to identify which public-kernel mechanics are already implemented.

Collectors should be conservative: if parsing fails, they emit a warning in `latest.md` and continue.

### 2. Idea generators

Generators consume normalized records and produce candidate ideas:

- `PublicKernelDeltaGenerator`: identifies mined public tactics absent from our generated attack or not recently score-gated.
- `NearMissGenerator`: identifies patterns that fire in one model, one template, or one guardrail proxy but not another.
- `ReplayEconomicsGenerator`: proposes sizing/cap/margin experiments from artifact metrics and timeout/candidate-count signals.
- `PredicateGapGenerator`: proposes minimal trace-shape experiments for non-exfil predicates using strategy/predicate notes.
- `DiscussionSignalGenerator`: turns fresh discussion/FAQ changes into validation or risk-check tasks.

The first implementation can use deterministic rules and keyword extraction. No LLM call is required. LLM summarization can be added later, but only after the deterministic queue is useful.

### 3. De-duplication

The hopper computes a stable fingerprint from:

- normalized title
- category
- likely files
- key constants/templates mentioned

It suppresses duplicate ideas already present in `state.json` unless new evidence changes priority or acceptance gates.

### 4. Ranking

Each idea receives numeric subscores:

- evidence strength: number and quality of source artifacts
- implementation size: fewer touched files and isolated changes rank higher
- scorer clarity: ideas with exact tests/metrics rank higher
- expected score impact: calibrated LB/raw-sec/candidate-count upside
- private-transfer value: diversity, encoding, non-public-overfit potential
- risk: timeout risk, dirty-worktree risk, long scorer cost

Initial ranking formula:

```text
priority_score =
  2.0 * evidence_strength
+ 1.5 * scorer_clarity
+ 1.5 * expected_score_impact
+ 1.0 * private_transfer_value
- 1.5 * risk
- 1.0 * implementation_size
```

The formula is intentionally simple and visible in `latest.md`; if a ranking looks wrong, we should change the weights rather than hide behind “AI vibes.”

## CLI

Primary commands:

```bash
uv run python -m jed_attack.campaign.idea_hopper
uv run python -m jed_attack.campaign.idea_hopper --limit 20
uv run python -m jed_attack.campaign.idea_hopper --watch --interval-min 30
```

Useful flags:

- `--out-dir run/idea_hopper`
- `--limit N`
- `--since-hours N` for discussion/log freshness
- `--include-low-confidence` to show weaker ideas instead of filtering them
- `--no-state` to ignore previous de-duplication state for one run

`--watch` is session-local and should be launched in tmux when used for long-running work. It should not use system cron by default.

## Report format

`latest.md` should start with a short operator summary:

```text
# Idea Hopper Report

Generated: 2026-07-30T...
Inputs read: blackboard, artifact metrics, kernel findings, discussion cache
Warnings: artifact_eval_current still running; no latest_metrics.json yet

## Top ideas
1. ...
```

Each idea section includes:

- title
- why now
- source evidence
- proposed patch target
- exact gate
- risk
- first implementation step

## Testing

Unit tests should not require GPU, Kaggle auth, network, or real W&B.

Required tests:

1. Parses artifact metrics and emits replay-sizing observations.
2. Parses decoded kernel findings and emits public-kernel delta observations.
3. Reads a small synthetic blackboard and emits a model-asymmetry/near-miss idea.
4. De-duplicates repeated ideas across two runs using `state.json`.
5. Writes `latest.md`, `latest.json`, and appends `queue.jsonl`.
6. Continues with warnings when optional inputs are missing or malformed.

Acceptance gate:

```bash
uv run pytest -q tests/test_campaign.py -k idea_hopper
uv run ruff check src/jed_attack/campaign/idea_hopper.py tests/test_campaign.py
uv run ty check src/jed_attack/campaign/idea_hopper.py tests/test_campaign.py
```

Full pre-commit remains required before committing implementation changes.

## Operational workflow

Manual one-shot:

```bash
uv run python -m jed_attack.campaign.idea_hopper
sed -n '1,160p' run/idea_hopper/latest.md
```

Session-local watch:

```bash
tmux new-session -d -s idea_hopper \
  'cd /home/will/projects/ai-agent-security-2026 && uv run python -m jed_attack.campaign.idea_hopper --watch --interval-min 30'
```

The hopper can run alongside the Kaggle research cron because it only reads the cron outputs. It should avoid running heavy artifact scorers or touching GPU state.

## Risks

- **Low-quality noisy ideas:** mitigated by requiring source evidence and scorer gates.
- **Overfitting to public kernels:** mitigated by explicitly scoring private-transfer value and labeling public-only throughput ideas.
- **Dirty worktree confusion:** mitigated by making the hopper non-mutating and writing only under `run/`.
- **Stale evidence:** mitigated by including input timestamps and warnings in `latest.md`.
- **LLM temptation creep:** first version is deterministic; LLM-assisted summarization is future work, not part of this spec.

## Out of scope

- Editing source files automatically.
- Starting artifact scorers or optimizers automatically.
- Kaggle submissions.
- Kaggle/network refreshes; existing cron jobs own that.
- Replacing the human/agent decision loop.

## Future extensions

- Add a `--from-latest-kernels` mode that invokes Kaggle research refresh first, with explicit network/auth approval.
- Add an “experiment pack” output that groups compatible ideas into a score-gated implementation plan.
- Add W&B summaries if local W&B run metadata is available.
- Add optional LLM summarization for discussion threads, constrained to cite exact cached sources.
