#!/bin/bash
# Campaign watchdog: supervise the 4 campaign daemons + N producer codex agents; restart any
# that die (60s loop). Host config via env (green defaults shown):
#   CAMPAIGN_REPO=$HOME/ai-agent-security-2026   JED_CAMPAIGN_ROOT=$HOME/campaign-run
#   N_PRODUCERS=5   WORKTREE_BASE=$HOME/exp
# Deliberately dumb + self-contained; producers run codex in per-agent worktrees.
set -u
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
REPO="${CAMPAIGN_REPO:-$HOME/ai-agent-security-2026}"
export JED_CAMPAIGN_ROOT="${JED_CAMPAIGN_ROOT:-$HOME/campaign-run}"
N="${N_PRODUCERS:-5}"
WTB="${WORKTREE_BASE:-$HOME/exp}"
LOGD="$JED_CAMPAIGN_ROOT/logs"
mkdir -p "$LOGD"

start_daemon() { # name module
  local name="$1" mod="$2"
  pgrep -f "jed_attack.campaign.$mod --loop" >/dev/null && return
  echo "$(date -u '+%F %T') $name dead -> restart" >>"$LOGD/watchdog.log"
  (cd "$REPO" && setsid nohup uv run python -m "jed_attack.campaign.$mod" --loop \
    >>"$LOGD/$name.log" 2>&1 &)
}

start_producer() { # n
  local n="$1" wt="$WTB/agent-$n"
  tmux has-session -t "codex-$n" 2>/dev/null && return
  [ -f "$wt/task.md" ] || return
  echo "$(date -u '+%F %T') producer $n dead -> restart" >>"$LOGD/watchdog.log"
  tmux new-session -d -s "codex-$n" \
    "cd $wt && export PATH=$HOME/.local/bin:\$PATH JED_CAMPAIGN_ROOT=$JED_CAMPAIGN_ROOT && \
     codex exec -s danger-full-access \"\$(cat task.md)\" 2>&1 | tee $LOGD/producer-$n.log"
}

while true; do
  start_daemon harvest harvest
  start_daemon gate gate_daemon
  start_daemon assemble assemble_daemon
  start_daemon scorewatch score_watch
  start_daemon scoredaemon score_daemon
  for n in $(seq 1 "$N"); do start_producer "$n"; done
  sleep 60
done
