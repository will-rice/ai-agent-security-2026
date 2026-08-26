#!/usr/bin/env bash
# One pass of the Kaggle competition research poll: runs the nvidia-kaggle
# skill's scripts to ingest fresh discussions + kernels and refresh the
# latest_*.json / CSV artifacts the optimizer and idea-hopper consume from
# run/kaggle_research_cron/.
#
# The loop lives in cron (a crontab entry wraps this in `flock -n`, every 30
# min). Each Kaggle call is bounded by STEP_TIMEOUT_S so a hung request can
# never wedge the cron slot.
set -u

ROOT="/home/will/projects/ai-agent-security-2026"
SKILL="/home/will/.agents/skills/nvidia-kaggle-skill"
PY="$ROOT/.venv/bin/python"
COMPETITION="ai-agent-security-multi-step-tool-attacks"
OUT="$ROOT/run/kaggle_research_cron"
LOG="$ROOT/run/logs/kaggle_research_cron.log"
STEP_TIMEOUT_S="${KAGGLE_RESEARCH_STEP_TIMEOUT_S:-300}"

# Cron starts with a minimal environment: pin HOME so the Kaggle client finds
# ~/.kaggle/kaggle.json, and expose the skill's scripts on PYTHONPATH.
export HOME="${HOME:-/home/will}"
export PYTHONPATH="$SKILL/scripts"
mkdir -p "$OUT" "$ROOT/run/logs"

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== $ts kaggle research tick ===" >>"$LOG"
FAILS=0

step() {
  # step LABEL cmd... — run a bounded step, logging timeout/failure but never
  # aborting the pass (a transient API failure shouldn't skip later steps).
  # Counts failures so the pass only stamps last_success_utc on a clean run.
  local label="$1"
  shift
  timeout "$STEP_TIMEOUT_S" "$@"
  local status=$?
  if [[ "$status" -eq 124 ]]; then
    echo "TIMEOUT: $label exceeded ${STEP_TIMEOUT_S}s" >>"$LOG"
    FAILS=$((FAILS + 1))
  elif [[ "$status" -ne 0 ]]; then
    echo "FAILED: $label exited $status" >>"$LOG"
    FAILS=$((FAILS + 1))
  fi
  return 0
}

step discussion_ingest \
  "$PY" "$SKILL/scripts/discussion_ingest.py" "$COMPETITION" \
  --max-pages 2 --sort-by updated --page-size 20 --nofetch-comments >>"$LOG" 2>&1

step discussion_query_recent \
  "$PY" "$SKILL/scripts/discussion_query.py" "$COMPETITION" --limit 25 --as-json \
  >"$OUT/latest_discussions.json" 2>>"$LOG"
step discussion_query_private \
  "$PY" "$SKILL/scripts/discussion_query.py" "$COMPETITION" --search private --limit 15 --as-json \
  >"$OUT/latest_private_discussions.json" 2>>"$LOG"
step discussion_query_score \
  "$PY" "$SKILL/scripts/discussion_query.py" "$COMPETITION" --search score --limit 15 --as-json \
  >"$OUT/latest_score_discussions.json" 2>>"$LOG"
step discussion_query_multi \
  "$PY" "$SKILL/scripts/discussion_query.py" "$COMPETITION" --search multi --limit 15 --as-json \
  >"$OUT/latest_multi_discussions.json" 2>>"$LOG"

step kernel_ingest_daterun \
  "$PY" "$SKILL/scripts/kernel_ingest.py" "$COMPETITION" \
  --max-pages 1 --sort-by dateRun --page-size 20 >>"$LOG" 2>&1
step kernel_ingest_votecount \
  "$PY" "$SKILL/scripts/kernel_ingest.py" "$COMPETITION" \
  --max-pages 1 --sort-by voteCount --page-size 20 >>"$LOG" 2>&1

step fetch_top_kernel_scores \
  "$PY" "$SKILL/scripts/fetch_top_kernel_scores.py" "$COMPETITION" \
  --sort descending --max-pages 2 --page-size 20 \
  >"$OUT/latest_kernel_scores.csv" 2>>"$LOG"

now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "$now" >"$OUT/last_run_utc"
if [[ "$FAILS" -eq 0 ]]; then
  echo "$now" >"$OUT/last_success_utc"
fi
echo "=== $ts kaggle research tick complete ($FAILS step failure(s)) ===" >>"$LOG"
