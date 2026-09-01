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
# Idempotent for DEPUTY ONLY: tears down just the 'deputy' tmux session (whose SIGHUP
# kills its own python) so it can coexist with the exfil optimizer on the OTHER GPU.
# Does NOT pkill-by-pattern. Runs BOTH victims on GPU1 (Ada 48GB) -- the ONLY GPU that
# fits gemma-4-26B (16GB file + its huge tool-schema KV context tops the 3090's 24GB), so
# two full 2-model optimizers cannot coexist; this takes GPU1 with exfil stopped.
set -euo pipefail

SESSION=deputy
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
MISSION="$REPO/src/jed_attack/campaign/missions/deputy.md"

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 2

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
