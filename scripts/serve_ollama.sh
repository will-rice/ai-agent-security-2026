#!/usr/bin/env bash
# User-space ollama on green (no sudo): install to ~/ollama, serve Qwen3-32B on the
# Ada GPU (device 1 under PCI_BUS_ID) in a detached tmux session. The judge is offline
# and independent of the T4-faithful scorer, so ollama (not the in-process llama-cpp
# path) is fine here. Idempotent: re-running tears down and restarts the server.
#
# NOTE: current ollama releases ship a `.tar.zst` asset (the old `.tgz` URL 404s), and
# the tarball unpacks to bin/ + lib/ (the CUDA runner libs live under lib/ollama, so
# LD_LIBRARY_PATH must point there). We resolve the latest amd64 asset via the GitHub
# releases API rather than hardcoding a version.
set -euo pipefail

SESSION=ollama
OLLAMA_DIR="$HOME/ollama"

if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
  mkdir -p "$OLLAMA_DIR"
  url=$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
    | grep browser_download_url \
    | grep -oE 'https://[^"]*ollama-linux-amd64.tar.zst' | head -1)
  curl -fsSL "$url" -o /tmp/ollama.tar.zst
  tar --zstd -xf /tmp/ollama.tar.zst -C "$OLLAMA_DIR"
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1
tmux new-session -d -s "$SESSION" \
  "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
     OLLAMA_HOST=127.0.0.1:11434 \
     LD_LIBRARY_PATH=\"$OLLAMA_DIR/lib/ollama:\${LD_LIBRARY_PATH:-}\" \
   \"$OLLAMA_DIR/bin/ollama\" serve"

sleep 5
"$OLLAMA_DIR/bin/ollama" pull qwen3:32b
echo "ollama serving qwen3:32b on GPU1 (tmux session '$SESSION')"
