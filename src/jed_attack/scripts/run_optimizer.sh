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
REPO="$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"

# Graceful stop, THEN kill the session. The worker catches SIGTERM/SIGINT to cancel
# cleanly and wandb.finish() its run; `tmux kill-session` sends SIGHUP, which it does
# NOT catch, so killing the session first strands the wandb run as a zombie "running".
# So: SIGTERM the worker, wait for it to exit (letting wandb finalize), SIGKILL any
# straggler, then drop the session. One in-flight score_pools runs in a thread
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
# Sole proposer lane: codex-gpt55 (gpt-5.5 via the codex ChatGPT-account Responses
# backend, authed by the OAuth token in ~/.codex/auth.json, no env key). Measured
# 2026-08-12 as the steadiest author (obj 33-47, zero drops) while the CheapestInference
# lanes had grown a ~30% ship-invariant drop rate and a wide low-side spread under the
# schema-as-source-of-truth prompt, so mimo/minimax are dropped. CAVEAT: gpt-5.5 spends
# the PERSONAL ChatGPT Pro quota per generation and the token expires (~days, then needs
# `codex login`); as the SOLE lane it burns quota faster and a lapsed token stalls the
# whole roster -- watch for 401s / quota 429s.
# WORKERS=ISLANDS=2 (not 4): scoring is the bottleneck and SERIALIZES on the per-model
# llama.cpp lock (submission_score._model_locks) -- score_pools already replays both
# victims concurrently across the two GPUs for ONE candidate, so a single worker
# saturates both GPUs and the rest just BLOCK on the locks. Only 1 scorer + 1 worker
# proposing ahead (hiding codex latency) does real work; more workers are pure lock
# contention + wasted codex quota. So 2 replicas / 2 islands (JED_ISLANDS=2), both
# seeded from the champion (island_best -> global_champion fallback).
#
# The public-board objective is scored in-process from the gpt_oss/gemma GGUFs alone --
# no judge fleet involved.
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "exec bash -lc 'mkdir -p run/logs; \
    export JED_CAMPAIGN_ROOT=\"$REPO/run\" JED_WANDB=1 \
      JED_TEAM_PROPOSERS=\"codex-agentic\" \
      JED_PROPOSER_REPLICAS=2 JED_ISLANDS=2 \
      JED_GPU_GPT=1 JED_GPU_GEMMA=1 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      LD_LIBRARY_PATH=\"/usr/local/cuda-12.8/lib64:\${LD_LIBRARY_PATH:-}\"; \
    uv run python -m jed_attack.campaign.optimize_prompts 2>&1 \
      | tee -a run/logs/optimizer.log'"

echo "optimizer worker launched in tmux session '$SESSION'"
echo "  watch:   tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  restart: src/jed_attack/scripts/run_optimizer.sh"
