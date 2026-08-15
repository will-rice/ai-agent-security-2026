---
name: submit-kernel
description: Use when submitting a Kaggle code-competition notebook (kernel) — pushing, polling, and submitting a built kernel to a competition, checking the daily submission quota first, or when building the JED attack submission kernel for "AI Agent Security — Multi-Step Tool Attacks". Keywords: kaggle submit, kernels push, competition_submit_code, submission slot, quota, jed attack, submission.csv.
---

# submit-kernel

## Overview
Submit a Kaggle **code competition** entry: a notebook that Kaggle re-runs in the
scoring environment. Submitting spends a scarce **daily slot** — always check
quota first and verify the kernel builds before pushing.

## When to use
- Submitting the JED attack (`AI Agent Security — Multi-Step Tool Attacks`).
- Any Kaggle code-competition kernel push + submit + score-poll.
- Not for dataset uploads or plain-file (`competitions submit -f`) competitions.

## The tools (all Python, no flags — config lives in ALL-CAPS constants)
- `build_kernel.py` (this dir) — builds the kernel folder (notebook + `kernel-metadata.json`). Edit its constants (`ATTACK_PY`, `TITLE`, `MACHINE_SHAPE`, …) to change what's built; it compile-checks the embedded attack before writing.
- `submit.py` (this dir) — the one entry point: loads `.env`, builds, pushes, waits for COMPLETE, submits, polls the eval. Reuses `submit_kernel.py`'s functions.
- `submit_kernel.py` / `submission_quota.py` (nvidia-kaggle skill) — the underlying Kaggle push/submit/poll and daily-quota helpers.

Skill dir: `.claude/skills/submit-kernel` (in-repo) · nvidia-kaggle scripts: `/home/will/.agents/skills/nvidia-kaggle-skill/scripts`

## Workflow

Config is fixed in constants (no CLI flags), so a wrong accelerator or mismatched
slug can't be passed by accident — those were the two real submission failures.
To change what's submitted, edit the constants; then run from the repo root:

```
PY=.venv/bin/python
SK=.claude/skills/submit-kernel

# Build only (writes the kernel folder; compile-checks the embedded attack.py):
$PY $SK/build_kernel.py

# Build + push + submit + poll the eval (spends a daily slot — human-invoked):
$PY $SK/submit.py
```

`submit.py` is self-guarding: the competition submit only fires after the pushed
kernel run reaches COMPLETE, so a broken build fails before any slot is spent.
The eval poll then runs for hours; background it or let it run.

Constants worth knowing (in `build_kernel.py` unless noted):
- `ATTACK_PY` — the artifact embedded (default `run/build_next/attack.py`; point at a `run/submission_cuts/…` path for a frozen cut).
- `TITLE` — the kernel slug is DERIVED from it, so the metadata id always matches the pushed slug.
- `MACHINE_SHAPE = NvidiaTeslaT4` — this competition rejects P100 (`400 FAILED_PRECONDITION`).
- `MESSAGE`, `SUBMISSION_FILE = submission.csv` (in `submit.py`) — the competition scores `submission.csv`, produced by the rerun.

## JED specifics (this competition)
- Kernel cell 5 pattern: `if os.getenv('KAGGLE_IS_COMPETITION_RERUN'): JEDAttackInferenceServer().serve()` else write a zero-row `submission.csv`. On a normal push the notebook is fast (writes `attack.py` + placeholder); the **real ~5h eval runs only on Kaggle's competition rerun** after submit.
- Metadata: `competition_sources:[ai-agent-security-multi-step-tool-attacks]`, no model/dataset sources (the competition provides the models), `enable_gpu:true`, `machine_shape:NvidiaTeslaT4` (P100 is rejected at submit with `400 FAILED_PRECONDITION`), `enable_internet:false`.
- The metadata `id` slug MUST equal the slugified title, or Kaggle pushes under the title-derived slug and the status poll can't find the kernel. `build_kernel.py` derives the slug from `--title` to guarantee this.
- `attack.py` does NO grade-time self-sizing: it returns the whole embedded pool up to the SDK's `MAX_REPLAY_FINDINGS` (2000) and the gateway scores what completes before its replay deadline (partial credit since 2026-08-05, so oversizing is safe). The shipped count is just the pool size (≤2000). Cuts live in `run/submission_cuts/`; `build_kernel.py` embeds the `ATTACK_PY` constant (default `run/build_next/attack.py`).

## Common mistakes
- Skipping the quota check and burning the last slot on a bad build.
- `enable_internet:true` when the rerun env is offline — leave it false; the attack imports only `aicomp_sdk` + stdlib.
- Wrong `--file`: this competition scores `submission.csv`, produced by the rerun, not by the push.
- Editing the pushed notebook by hand — rebuild with `build_kernel.py` so the writefile cell and metadata stay consistent.
