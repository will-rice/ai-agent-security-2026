#!/bin/bash
# Launch the prompt-optimizer swarm: N workers running jed_attack.campaign.optimize_prompts.
# Each worker asks the local served model (JED_PROPOSER=local) for multi-post template
# variants, scores them on both models, and promotes winners to the shared, fcntl-locked
# best_prompt.json. Run ON green (or `ssh green bash scripts/launch_optimizer_swarm.sh`).
# Sanity-checks one generation first and refuses to launch a parametric-only loop (i.e. if
# the proposer model can't produce parseable JSON). Host config via env (green defaults):
#   CAMPAIGN_REPO=$HOME/ai-agent-security-2026   JED_CAMPAIGN_ROOT=$HOME/campaign-run
#   JED_PROPOSER_MODEL=gpt_oss   N_WORKERS=3
set -u
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
REPO="${CAMPAIGN_REPO:-$HOME/ai-agent-security-2026}"
export JED_CAMPAIGN_ROOT="${JED_CAMPAIGN_ROOT:-$HOME/campaign-run}"
N="${N_WORKERS:-3}"
LOGD="$JED_CAMPAIGN_ROOT/logs"
mkdir -p "$LOGD"
cd "$REPO" || { echo "!! repo not found: $REPO"; exit 1; }

if pgrep -f 'campaign.optimize_prompts' >/dev/null; then
  echo "!! optimizer workers already running:"
  pgrep -af 'campaign.optimize_prompts' | grep -v pgrep
  echo "   kill them first (pkill -f campaign.optimize_prompts) for a fresh set. Aborting."
  exit 1
fi

# One generation with proposer model $1; succeed (return 0) only if the model actually
# produced parseable proposals (no "returned no valid proposals" parametric fallback).
sanity() {
  echo "--- sanity generation: proposer model = $1 ---"
  JED_PROPOSER_MODEL="$1" timeout 400 \
    uv run python -m jed_attack.campaign.optimize_prompts --generations 1 --proposals 3 \
    >"$LOGD/optimizer-sanity.log" 2>&1
  grep -vE '^wandb:' "$LOGD/optimizer-sanity.log" | tail -12
  ! grep -q 'returned no valid proposals' "$LOGD/optimizer-sanity.log"
}

MODEL="${JED_PROPOSER_MODEL:-gpt_oss}"
if ! sanity "$MODEL"; then
  echo "    $MODEL produced no parseable JSON; retrying with gemma_4"
  if sanity gemma_4; then
    MODEL=gemma_4
  else
    echo "!! no model produced parseable proposals; loop would be parametric-only. NOT launching."
    echo "   see $LOGD/optimizer-sanity.log"
    exit 2
  fi
fi

echo "==> launching $N worker(s), proposer model = $MODEL"
for i in $(seq 1 "$N"); do
  wb=0
  [ "$i" -eq 1 ] && wb=1  # worker 1 owns the single wandb run; the rest run wandb-off
  (cd "$REPO" && JED_PROPOSER_MODEL="$MODEL" JED_WANDB="$wb" setsid nohup \
    uv run python -m jed_attack.campaign.optimize_prompts >"$LOGD/optimizer-$i.log" 2>&1 &)
  echo "    worker $i launched (wandb=$wb) -> $LOGD/optimizer-$i.log"
done
sleep 3
echo "--- live optimizer processes ---"
pgrep -af optimize_prompts | grep -v pgrep || echo "!! none found — check $LOGD/optimizer-*.log"
echo "==> done. logs: $LOGD/optimizer-*.log | wandb: will-rice/jed-prompt-opt"
