#!/usr/bin/env bash
# Launch the DEPUTY-hedge optimizer in a detached tmux session.
#
# Same engine as run_optimizer.sh, but points the proposer at the deputy mission
# (JED_MISSION_PATH=missions/deputy.md -> author ONLY type=deputy shapes) and a
# SEPARATE cold-start board (JED_CAMPAIGN_ROOT=run_deputy) so deputy shapes are not
# compared against (and always beaten by) the sev-16 exfil champion. The objective is
# predicate-agnostic (minimizes cost per FIRING candidate), so it credits the sev-4
# CONFUSED_DEPUTY fire and hunts the leanest deputy shape for the private-board hedge.
#
# Idempotent: tears down ANY running optimizer (exfil or deputy) + both tmux sessions
# first, then starts fresh. GPU1 (Ada), both victims, like the exfil optimizer.
set -euo pipefail

SESSION=deputy
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
MISSION="$REPO/src/jed_attack/campaign/missions/deputy.md"

STOP_TIMEOUT_S="${JED_OPTIMIZER_STOP_TIMEOUT_S:-330}"
OPTIMIZER_PYTHON_PATTERN='^(.*/)?python([0-9]+([.][0-9]+)?)? -m jed_attack[.]campaign[.]optimize_prompts($| )'
pkill -TERM -f "$OPTIMIZER_PYTHON_PATTERN" 2>/dev/null || true
for _ in $(seq 1 "$STOP_TIMEOUT_S"); do
  pgrep -f "$OPTIMIZER_PYTHON_PATTERN" >/dev/null || break
  sleep 1
done
pkill -KILL -f "$OPTIMIZER_PYTHON_PATTERN" 2>/dev/null || true
tmux kill-session -t optimizer 2>/dev/null || true
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

tmux new-session -d -s "$SESSION" -c "$REPO" \
  "exec bash -lc 'mkdir -p run_deputy/logs; \
    export JED_CAMPAIGN_ROOT=\"$REPO/run_deputy\" \
      JED_MISSION_PATH=\"$MISSION\" JED_WANDB=1 \
      JED_PROPOSER_REPLICAS=2 JED_ISLANDS=2 \
      JED_GPU_GPT=1 JED_GPU_GEMMA=1 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      LD_LIBRARY_PATH=\"/usr/local/cuda-12.8/lib64:\${LD_LIBRARY_PATH:-}\"; \
    while true; do \
      uv run python -m jed_attack.campaign.optimize_prompts 2>&1 | tee -a run_deputy/logs/optimizer.log; \
      echo \"[wrapper] deputy optimizer exited \$? at \$(date -u +%H:%M) -- resuming in 15s\" | tee -a run_deputy/logs/optimizer.log; \
      sleep 15; \
    done'"

echo "DEPUTY optimizer launched in tmux session '$SESSION' (mission=deputy.md, board=run_deputy)"
echo "  watch:  tmux attach -t $SESSION    log: run_deputy/logs/optimizer.log"
