#!/usr/bin/env bash
# Run from your workstation (the machine with the working tree + ssh access to green).
# Lints + commits the prompt-optimizer change, deploys it to green by fast-forward, then
# runs the committed green-side launcher (scripts/launch_optimizer_swarm.sh) to sanity-check
# one generation and start the swarm.
#
# Safe to re-run: only commits if the tree is dirty, only deploys if green is behind, and
# the green launcher refuses to start a parametric-only loop or double-launch workers.
# Host override via env:  GREEN_HOST=green   GREEN_REPO='~/ai-agent-security-2026'
#
# Run:  bash scripts/deploy_and_launch_optimizer.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GREEN_HOST="${GREEN_HOST:-green}"
GREEN_REPO="${GREEN_REPO:-\$HOME/ai-agent-security-2026}"  # remote-expanded, keep the \$
cd "$REPO" || { echo "!! repo not found at $REPO"; exit 1; }

echo "==> [1/4] lint (pre-commit; auto-fixes formatting, retries once)"
uv run pre-commit run --files \
    src/jed_attack/campaign/optimize_prompts.py tests/test_campaign.py \
  || uv run pre-commit run --files \
    src/jed_attack/campaign/optimize_prompts.py tests/test_campaign.py \
  || { echo "!! pre-commit still failing after one retry — fix and re-run"; exit 1; }

echo "==> [2/4] commit (only if there are changes)"
if git diff --quiet && git diff --cached --quiet; then
  echo "    nothing to commit (already committed)"
else
  git add -A
  git commit -m "prompt-opt: local served-model proposer (codex is provider-blocked)

codex exec returns 'Request blocked' from its provider safety on these red-team
prompts even with truthful competition context. Default the proposer to a
locally-served target model (gpt_oss/gemma_4) via llama_server_chat_client --
the same open-weight models the competition scores, no external provider.
JED_PROPOSER selects the backend (local|codex); JED_PROPOSER_MODEL the model.
Add scripts to launch the optimizer swarm on green."
fi

LOCAL_HEAD="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "==> [3/4] deploy to $GREEN_HOST (fast-forward only)"
GREEN_HEAD="$(ssh "$GREEN_HOST" "cd $GREEN_REPO && git rev-parse HEAD")" \
  || { echo "!! could not reach $GREEN_HOST"; exit 1; }
if [ "$GREEN_HEAD" = "$LOCAL_HEAD" ]; then
  echo "    $GREEN_HOST already at $(git rev-parse --short HEAD) — skipping deploy"
else
  if ! git merge-base --is-ancestor "$GREEN_HEAD" "$LOCAL_HEAD"; then
    echo "!! remote head $GREEN_HEAD is not an ancestor of local — diverged; resolve manually"
    exit 1
  fi
  BUNDLE=/tmp/jed-optimizer.bundle
  git bundle create "$BUNDLE" "${GREEN_HEAD}..${BRANCH}"
  scp "$BUNDLE" "$GREEN_HOST":/tmp/jed-optimizer.bundle
  ssh "$GREEN_HOST" "cd $GREEN_REPO && git fetch /tmp/jed-optimizer.bundle $BRANCH && git merge --ff-only FETCH_HEAD && echo \"    $GREEN_HOST now at \$(git rev-parse --short HEAD)\""
fi

echo "==> [4/4] sanity + launch (green: scripts/launch_optimizer_swarm.sh)"
ssh "$GREEN_HOST" "bash $GREEN_REPO/scripts/launch_optimizer_swarm.sh"
RC=$?
if [ "$RC" -eq 2 ]; then
  echo "==> launch aborted: proposer produced no valid JSON on either model (see above)."
elif [ "$RC" -ne 0 ]; then
  echo "==> remote step failed (rc=$RC)."
else
  echo "==> ALL DONE — swarm is running on $GREEN_HOST."
fi
exit "$RC"
