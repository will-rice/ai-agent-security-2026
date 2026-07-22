#!/bin/bash
# Campaign watchdog: supervise the composer daemons (assemble + scoredaemon); restart any
# that die (60s loop). Host config via env (green defaults shown):
#   CAMPAIGN_REPO=$HOME/projects/ai-agent-security-2026   JED_CAMPAIGN_ROOT=$REPO/run
# Deliberately dumb + self-contained. The optimizer swarm that feeds the archive is
# launched separately (scripts/launch_optimizer_swarm.sh / `uv run jed-optimize`).
set -u
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
REPO="${CAMPAIGN_REPO:-$HOME/projects/ai-agent-security-2026}"
export JED_CAMPAIGN_ROOT="${JED_CAMPAIGN_ROOT:-$REPO/run}"
LOGD="$JED_CAMPAIGN_ROOT/logs"
mkdir -p "$LOGD"

start_daemon() { # name module
  local name="$1" mod="$2"
  pgrep -f "jed_attack.campaign.$mod --loop" >/dev/null && return
  echo "$(date -u '+%F %T') $name dead -> restart" >>"$LOGD/watchdog.log"
  (cd "$REPO" && setsid nohup uv run python -m "jed_attack.campaign.$mod" --loop \
    >>"$LOGD/$name.log" 2>&1 &)
}

while true; do
  start_daemon assemble assemble_daemon
  start_daemon consolidator consolidator
  sleep 60
done
