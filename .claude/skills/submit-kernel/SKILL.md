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

## The three tools
- `build_kernel.py` (this dir) — builds the kernel folder (notebook + `kernel-metadata.json`) from an `attack.py`. Host-portable.
- `submission_quota.py` (nvidia-kaggle skill) — headroom BEFORE spending a slot.
- `submit_kernel.py` (nvidia-kaggle skill) — push → poll kernel → `competition_submit_code` → poll eval → report public score.

Skill dir: `.claude/skills/submit-kernel` (in-repo) · nvidia-kaggle scripts: `/home/will/.agents/skills/nvidia-kaggle-skill/scripts`

## Workflow

```
# run from the repo root
NK=/home/will/.agents/skills/nvidia-kaggle-skill/scripts
SK=.claude/skills/submit-kernel
PY=.venv/bin/python
COMP=ai-agent-security-multi-step-tool-attacks

# 1. Quota — STOP if 0 remaining (a spent slot cannot be reclaimed).
PYTHONPATH=$NK $PY $NK/submission_quota.py $COMP

# 2. Build the kernel folder from the chosen attack.py (default: the latest cut).
$PY $SK/build_kernel.py <ATTACK_PY> <OUT_DIR> --user <KAGGLE_USER> \
    --slug jed-attack-$(date -u +%Y%m%d-%H%M) --title "JED attack" --enable-gpu --no-internet

# 3. Verify locally before pushing (compile + inspect the writefile cell).
$PY -c "import json,py_compile,pathlib,tempfile,sys; nb=json.load(open(sys.argv[1])); \
src=[''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('%%writefile')][0].split('\n',1)[1]; \
f=tempfile.NamedTemporaryFile('w',suffix='.py',delete=False); f.write(src); f.close(); py_compile.compile(f.name,doraise=True); print('attack cell compiles')" <OUT_DIR>/<slug>.ipynb

# 4. Push + submit + poll (the submission file the notebook produces is submission.csv).
# NOTE: the Kaggle SDK USES $KAGGLE_API_TOKEN as the bearer token — pass the REAL
# value (from the repo .env), never a dummy like `1`, or the push 401s.
set -a; . /home/will/projects/ai-agent-security-2026/.env; set +a
PYTHONPATH=$NK $PY $NK/submit_kernel.py <OUT_DIR> --file submission.csv --message "<msg>"
```

## JED specifics (this competition)
- Kernel cell 5 pattern: `if os.getenv('KAGGLE_IS_COMPETITION_RERUN'): JEDAttackInferenceServer().serve()` else write a zero-row `submission.csv`. On a normal push the notebook is fast (writes `attack.py` + placeholder); the **real ~5h eval runs only on Kaggle's competition rerun** after submit.
- Metadata: `competition_sources:[ai-agent-security-multi-step-tool-attacks]`, no model/dataset sources (the competition provides the models), `enable_gpu:true`, `enable_internet:false`.
- `attack.py` self-sizes at grade time (`_HARD_N_CAP=2000`, `_REPLAY_SAFE_FRAC=0.97`). Cuts live in `run/submission_cuts/`; `build_kernel.py` embeds whichever `attack.py` you pass.

## Common mistakes
- Skipping the quota check and burning the last slot on a bad build.
- `enable_internet:true` when the rerun env is offline — leave it false; the attack imports only `aicomp_sdk` + stdlib.
- Wrong `--file`: this competition scores `submission.csv`, produced by the rerun, not by the push.
- Editing the pushed notebook by hand — rebuild with `build_kernel.py` so the writefile cell and metadata stay consistent.
