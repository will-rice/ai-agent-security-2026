#!/usr/bin/env bash
# Launch the whole-submission optimizer in a detached tmux session.
#
# One process owns every proposer lane (async team orchestrator), so there is
# no per-worker env — a single tmux session runs the whole team.
#
# Run this on the serving host (green) from the repo root. It is the clean
# replacement for `setsid nohup ... &`, which left the launching ssh channel
# half-open. tmux decouples the worker from ssh entirely and lets you watch it
# live (`tmux attach -t optimizer`).
#
# Idempotent: tears down any existing worker (tmux session + stray process)
# first, then starts fresh. Restarting the worker after a prompt/code change is
# just: sync the tree, then re-run this script.
#
# The worker itself is CPU/network only (it calls the llama-servers over HTTP);
# GPU serving lives in the sibling `gemma`/`gptoss` tmux sessions.
set -euo pipefail

SESSION=optimizer
REPO="$(cd "$(dirname "$0")/.." && pwd)"

tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f jed_attack.campaign.optimize_prompts 2>/dev/null || true
sleep 1

# Env is set inside the pane's login shell: a new-session inherits the tmux
# SERVER's stale environment (the server is already up for gemma/gptoss), not
# this script's, so exporting here would not reach the worker.
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "exec bash -lc 'mkdir -p run/logs; \
    export JED_CAMPAIGN_ROOT=\"$REPO/run\" JED_WANDB=1 \
      LD_LIBRARY_PATH=\"/usr/local/cuda-12.8/lib64:\${LD_LIBRARY_PATH:-}\"; \
    uv run python -m jed_attack.campaign.optimize_prompts 2>&1 \
      | tee -a run/logs/optimizer.log'"

echo "optimizer worker launched in tmux session '$SESSION'"
echo "  watch:   tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  restart: scripts/run_optimizer.sh"
