"""Campaign paths, models, and knobs — env-overridable.

Runtime state lives under ``run/`` (git-ignored). All daemons and the producers read
these so paths stay consistent across the fleet.
"""

import os
from pathlib import Path

# Repo/runtime roots. CAMPAIGN_ROOT points at the run/ tree; override per host.
CAMPAIGN_ROOT = Path(os.getenv("JED_CAMPAIGN_ROOT", "run")).resolve()

CANDIDATES_DIR = (
    CAMPAIGN_ROOT / "candidates"
)  # run/candidates/<producer>.jsonl (producers append)
HARVEST_FILE = CAMPAIGN_ROOT / "harvest" / "candidates.jsonl"  # deduped collection
GATE_LEDGER = CAMPAIGN_ROOT / "gate_ledger.jsonl"  # one Verdict per chain (newest wins)
BUILD_NEXT_DIR = (
    CAMPAIGN_ROOT / "build_next"
)  # assembled attack_src.py + attack.py + status
LEADERBOARD = CAMPAIGN_ROOT / "leaderboard.jsonl"
FLOOR_DIR = CAMPAIGN_ROOT / "floor"  # promoted best submission

# Shared cross-agent knowledge log (see knowledge.py). Each writer appends its own
# <producer>.jsonl so the fleet learns from every agent's tries without locks.
KNOWLEDGE_DIR = CAMPAIGN_ROOT / "knowledge"
ATTEMPTS_DIR = KNOWLEDGE_DIR / "attempts"  # every probed chain, fired or not
NOTES_DIR = KNOWLEDGE_DIR / "notes"  # free-form insights + gate lessons

# The two target models and their served base URLs (llama-server on green).
MODELS: tuple[str, ...] = ("gpt_oss", "gemma_4")

# Max tool hops per interaction. MUST match the scored gateway's replay depth
# (aicomp_sdk DEFAULT_MAX_TOOL_HOPS = AttackRunConfig.max_tool_hops = 8), or
# multi-post candidates (one message → up to 8 SECRET_MARKER posts) are
# under-counted locally and wrongly gated out. Single-post scores are unchanged.
EVAL_HOPS = 8

# Concurrent replay workers for the GPU consumers (gate, score, produce). Sized to
# the gemma llama-server's batch slots (-np 8) so concurrent requests fill the
# continuous-batching slots instead of queuing one-at-a-time. gemma is the binding
# server; gpt (fewer slots) just queues the excess, which is harmless.
REPLAY_WORKERS = int(os.getenv("JED_REPLAY_WORKERS", "8"))

# Returned-candidate ceiling for the assembled submission (see docs/strategy.md).
MAX_CANDIDATES = int(os.getenv("JED_MAX_CANDIDATES", "300"))

# Calibrated-score daemon: the single-message ship pool it scores + its result/cache.
SCORE_CAP = int(os.getenv("JED_SCORE_CAP", "300"))
SCORE_FILE = CAMPAIGN_ROOT / "score.json"  # latest calibrated public-LB prediction
SCORE_CACHE = CAMPAIGN_ROOT / "score_cache.jsonl"  # per-(message,model) replay cache


def ensure_dirs() -> None:
    """Create the runtime directories if missing."""
    for path in (
        CANDIDATES_DIR,
        HARVEST_FILE.parent,
        BUILD_NEXT_DIR,
        FLOOR_DIR,
        ATTEMPTS_DIR,
        NOTES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
