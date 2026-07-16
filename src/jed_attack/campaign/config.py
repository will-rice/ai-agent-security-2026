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

# The two target models and their served base URLs (llama-server on green).
MODELS: tuple[str, ...] = ("gpt_oss", "gemma_4")

# Returned-candidate ceiling for the assembled submission (see docs/strategy.md).
MAX_CANDIDATES = int(os.getenv("JED_MAX_CANDIDATES", "300"))


def ensure_dirs() -> None:
    """Create the runtime directories if missing."""
    for path in (CANDIDATES_DIR, HARVEST_FILE.parent, BUILD_NEXT_DIR, FLOOR_DIR):
        path.mkdir(parents=True, exist_ok=True)
