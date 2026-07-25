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
# The worker scores IN-PROCESS: it loads the gpt_oss + gemma GGUFs itself via
# llama-cpp-python (deterministic, matches the T4 gateway), so the old `gemma`/`gptoss`
# llama-server tmux sessions are NO LONGER needed. llama-cpp-python is now a uv-managed
# CUDA wheel (pinned `llama-cuda` index in pyproject), so plain `uv run` below syncs the
# env normally -- no more manual build, no more --no-sync. CUDA_DEVICE_ORDER=PCI_BUS_ID
# maps device 0 = 3090, 1 = Ada (config.MODEL_GPU).
set -euo pipefail

SESSION=optimizer
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Graceful stop, THEN kill the session. The worker catches SIGTERM/SIGINT to cancel
# cleanly and wandb.finish() its run; `tmux kill-session` sends SIGHUP, which it does
# NOT catch, so killing the session first strands the wandb run as a zombie "running".
# So: SIGTERM the worker, wait for it to exit (letting wandb finalize), SIGKILL any
# straggler, then drop the session. One in-flight score_submission runs in a thread
# that asyncio cancellation cannot stop and may use the full 300-second replay budget,
# so the default grace period must exceed that budget.
STOP_TIMEOUT_S="${JED_OPTIMIZER_STOP_TIMEOUT_S:-330}"
OPTIMIZER_PYTHON_PATTERN='^(.*/)?python([0-9]+([.][0-9]+)?)? -m jed_attack[.]campaign[.]optimize_prompts($| )'
pkill -TERM -f "$OPTIMIZER_PYTHON_PATTERN" 2>/dev/null || true
for _ in $(seq 1 "$STOP_TIMEOUT_S"); do
  pgrep -f "$OPTIMIZER_PYTHON_PATTERN" >/dev/null || break
  sleep 1
done
pkill -KILL -f "$OPTIMIZER_PYTHON_PATTERN" 2>/dev/null || true
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

# Env is set inside the pane's login shell: a new-session inherits the tmux
# SERVER's stale environment (the server is already up for gemma/gptoss), not
# this script's, so exporting here would not reach the worker.
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "exec bash -lc 'mkdir -p run/logs; \
    export JED_CAMPAIGN_ROOT=\"$REPO/run\" JED_WANDB=1 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      DYLAN_JUDGE_URL=http://192.168.1.220:8100 \
      LD_LIBRARY_PATH=\"/usr/local/cuda-12.8/lib64:\${LD_LIBRARY_PATH:-}\"; \
    uv run python -m jed_attack.campaign.optimize_prompts 2>&1 \
      | tee -a run/logs/optimizer.log'"

echo "optimizer worker launched in tmux session '$SESSION'"
echo "  watch:   tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  restart: scripts/run_optimizer.sh"
