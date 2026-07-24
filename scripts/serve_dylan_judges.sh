#!/usr/bin/env bash
# Dylan judge service: vLLM (AWQ Qwen3-32B on the RTX 3090) + the FastAPI judge app.
#
# vLLM runs as a STANDALONE server in its own venv (~/vllm-venv) -- it is NOT a repo
# dependency; our code only HTTP-calls its OpenAI endpoint. The FastAPI judge service
# runs from the synced repo via `uv run` and calls that vLLM endpoint with guided-JSON.
#
# The TITAN RTX (Turing) is left idle: vLLM's AWQ/Marlin kernels need Ampere+ (the 3090).
# Forward-compat: with a 2nd matched 3090, add `--data-parallel-size 2` to the vllm line
# -> 2 replicas, one per GPU, vLLM routes across them (one endpoint, ~2x throughput).
#
# Idempotent: re-running tears down the tmux sessions and restarts.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="$HOME/vllm-venv"
MODEL="${VLLM_MODEL:-bullerwins/Qwen3-32B-awq}"

# One-time: create the vLLM venv + install vLLM (skipped if already present).
if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
  python3 -m venv "$VLLM_VENV"
  "$VLLM_VENV/bin/pip" install -q --upgrade pip
  "$VLLM_VENV/bin/pip" install -q vllm
fi

# vLLM OpenAI server on the 3090 (device 0), guided decoding on, :8000.
tmux kill-session -t vllm 2>/dev/null || true
sleep 1
# --served-model-name pins the served id to $MODEL so the FastAPI service (which reads
# config.VLLM_MODEL, same default) always requests a name vLLM actually serves. Override
# both by exporting VLLM_MODEL before running (config.VLLM_MODEL reads the same env var).
tmux new-session -d -s vllm \
  "env CUDA_VISIBLE_DEVICES=0 \"$VLLM_VENV/bin/vllm\" serve \"$MODEL\" \
     --served-model-name \"$MODEL\" \
     --quantization awq_marlin --gpu-memory-utilization 0.92 \
     --max-model-len 8192 --port 8000"

# FastAPI judge service from the synced repo (uv run auto-syncs), :8100.
tmux kill-session -t judgesvc 2>/dev/null || true
sleep 1
tmux new-session -d -s judgesvc -c "$REPO" \
  "exec bash -lc 'uv run uvicorn jed_attack.campaign.judge_service:app \
     --host 0.0.0.0 --port 8100'"

echo "vLLM (:8000, model=$MODEL) + judge service (:8100) launched on dylan"
echo "  vLLM logs:  tmux attach -t vllm"
echo "  svc logs:   tmux attach -t judgesvc"
