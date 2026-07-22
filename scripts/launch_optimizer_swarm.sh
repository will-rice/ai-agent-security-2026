#!/bin/bash
# Launch the prompt-optimizer swarm: N workers running jed_attack.campaign.optimize_prompts.
# Each worker asks the configured proposer to author a whole submission, scores it on both
# served models, and writes the scored SubmissionRecord as a shard for the consolidator to
# append to the submission log. Run ON green (or `ssh green bash scripts/launch_optimizer_swarm.sh`).
# Sanity-checks one generation first and refuses to launch if no proposer completes a
# generation. Host config via env (green defaults):
#   CAMPAIGN_REPO=$HOME/projects/ai-agent-security-2026   JED_CAMPAIGN_ROOT=$REPO/run
#   JED_PROPOSER=gpt_oss   N_WORKERS=3
set -u
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
REPO="${CAMPAIGN_REPO:-$HOME/projects/ai-agent-security-2026}"
export JED_CAMPAIGN_ROOT="${JED_CAMPAIGN_ROOT:-$REPO/run}"
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

# One generation with proposer $1; succeed (return 0) only if a generation actually
# completed (a "public=" summary line means the proposer answered, scored, and recorded).
sanity() {
  echo "--- sanity generation: proposer = $1 ---"
  JED_PROPOSER="$1" timeout 400 \
    uv run python -m jed_attack.campaign.optimize_prompts --generations 1 \
    >"$LOGD/optimizer-sanity.log" 2>&1
  grep -vE '^wandb:' "$LOGD/optimizer-sanity.log" | tail -12
  grep -q 'public=' "$LOGD/optimizer-sanity.log"
}

PROPOSER="${JED_PROPOSER:-gpt_oss}"
if ! sanity "$PROPOSER"; then
  echo "    $PROPOSER completed no generation; retrying with gemma_4"
  if sanity gemma_4; then
    PROPOSER=gemma_4
  else
    echo "!! no proposer completed a generation. NOT launching."
    echo "   see $LOGD/optimizer-sanity.log"
    exit 2
  fi
fi

echo "==> launching $N worker(s), proposer = $PROPOSER"
for i in $(seq 1 "$N"); do
  wb=0
  [ "$i" -eq 1 ] && wb=1  # worker 1 owns the single wandb run; the rest run wandb-off
  (cd "$REPO" && JED_PROPOSER="$PROPOSER" JED_WANDB="$wb" JED_WORKER_ID="$i" setsid nohup \
    uv run python -m jed_attack.campaign.optimize_prompts >"$LOGD/optimizer-$i.log" 2>&1 &)
  echo "    worker $i launched (wandb=$wb) -> $LOGD/optimizer-$i.log"
done
sleep 3
echo "--- live optimizer processes ---"
pgrep -af optimize_prompts | grep -v pgrep || echo "!! none found — check $LOGD/optimizer-*.log"
echo "==> done. logs: $LOGD/optimizer-*.log | wandb: will-rice/jed-prompt-opt"
