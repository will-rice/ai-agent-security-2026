#!/bin/bash
# DEPUTY-HEDGE optimizer, meant to run ON dylan. Same agentic machinery as the public
# exfil optimizer (src/jed_attack/scripts/run_optimizer.sh), but: authors ONLY type=deputy
# shapes and scores CONFUSED_DEPUTY under the optimal guardrail (the objective is
# predicate-agnostic), on a SEPARATE board (run_deputy) so it cold-starts deputy instead of
# inheriting the exfil champion. gpt on GPU0 (3090), gemma on GPU1 (TITAN).
#
# The ONLY thing that makes this the deputy lane instead of exfil is JED_MISSION_PATH ->
# the committed deputy mission. No file-swap: the exfil launcher leaves it unset (defaults
# to missions/exfil.md), this one points it at missions/deputy.md.
set -u
SESSION=deputy
REPO="$HOME/projects/ai-agent-security-2026"
cd "$REPO"
LDLIB=$(echo "$REPO"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':')
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "exec bash -lc 'mkdir -p run_deputy/logs; \
    export JED_CAMPAIGN_ROOT=\"$REPO/run_deputy\" JED_WANDB=1 \
      JED_TEAM_PROPOSERS=\"codex-agentic\" \
      JED_MISSION_PATH=\"$REPO/src/jed_attack/campaign/missions/deputy.md\" \
      JED_PROPOSER_REPLICAS=2 JED_ISLANDS=2 JED_NUM_GPUS=2 \
      JED_GPU_GPT=0 JED_GPU_GEMMA=1 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      JED_MODELS_DIR=\"$REPO/models\" \
      LD_LIBRARY_PATH=\"$LDLIB\"; \
    while true; do \
      uv run python -m jed_attack.campaign.optimize_prompts 2>&1 | tee -a run_deputy/logs/optimizer.log; \
      echo \"[wrapper] optimizer exited \$? at \$(date -u +%H:%M) -- resuming from run_deputy board in 15s\" | tee -a run_deputy/logs/optimizer.log; \
      sleep 15; \
    done'"
echo "deputy optimizer launched in tmux session '$SESSION' on dylan"
echo "  watch:  ssh dylan tmux attach -t $SESSION"
