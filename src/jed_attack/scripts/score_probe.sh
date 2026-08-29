#!/usr/bin/env bash
# Grade a candidate through the REAL optimizer objective (not raw gen tokens).
#   bash src/jed_attack/scripts/score_probe.sh <candidate.json>
# <candidate.json> is one message (or a list) in ship shape:
#   {"type":"exfil","text":"... {u} ... SECRET_MARKER ...","url_scheme":"a://{h}","hops":1}
# Prints objective_mean (MINIMIZE) + per-model input/gen/severity/turns. Inherits the
# caller's GPU env (CUDA_VISIBLE_DEVICES, JED_GPU_*, LD_LIBRARY_PATH).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec env CUDA_DEVICE_ORDER=PCI_BUS_ID JED_MODELS_DIR="$PWD/models" \
  .venv/bin/python -m jed_attack.scripts.score_probe "$1"
