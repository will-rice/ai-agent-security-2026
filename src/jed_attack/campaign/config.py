"""Campaign paths, models, and knobs — env-overridable.

Runtime state lives under ``run/`` (git-ignored). All daemons and the producers read
these so paths stay consistent across the fleet.
"""

import os
from pathlib import Path

# Repo/runtime roots. Everything the campaign writes lives under the repo's git-ignored
# ``run/`` tree, so state is self-contained and cwd-independent (config.py is at
# <repo>/src/jed_attack/campaign/config.py, so parents[3] is the repo root). Override
# with JED_CAMPAIGN_ROOT only for an out-of-tree runtime.
_REPO_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_ROOT = Path(os.getenv("JED_CAMPAIGN_ROOT", str(_REPO_ROOT / "run"))).resolve()

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

# Ship ARTIFACT: the shape of build_next/attack.py. "adaptive" (default) ships the
# self-sizing adaptive.build_adaptive run() that probes the live env and fills only as
# many candidates as fit the 9000s/cell budget on the actual hardware — it CANNOT time
# out the way a fixed pool does (a static 150× multi-post pool overran the T4 and scored
# a zero). "static" ships the fixed assemble() pool (legacy; only safe when the
# per-candidate replay cost is known to fit the budget — e.g. single-post).
SHIP_ARTIFACT = os.getenv("JED_SHIP_ARTIFACT", "adaptive")

# Ship-pool source (the STATIC path / score-daemon proxy). "multipost" uses the
# deterministic multi-post template directly (produce.multipost_chain — the winning
# attack); "ledger" uses the gate-adopted pool.
SHIP_POOL = os.getenv("JED_SHIP_POOL", "multipost")

# Ship-pool ranking (only used when SHIP_POOL="ledger"). "public" ranks candidates by
# their OptimalGuardrail (public LB)
# score and fills every slot with the highest-firing multi-post — maximizes the visible
# LB, bets the private guardrail is lenient. "robust" ranks by strict-guardrail survival
# and reserves SHIP_HEDGE_SLOTS for diverse chains — anti-overfit (bets private strict).
SHIP_RANK = os.getenv("JED_SHIP_RANK", "public")

# Ship-pool slots reserved for diverse NON-SECRET_MARKER chains (Family-1
# read-then-post exfil + confused-deputy) in "robust" ranking only. The SECRET_MARKER
# multi-post template drives the public score and bets the private LB is taint-based;
# these slots hedge the other private guardrail family (a payload scanner).
SHIP_HEDGE_SLOTS = int(os.getenv("JED_SHIP_HEDGE_SLOTS", "20"))

# Prompt-optimization incumbents (one per attack FAMILY — the portfolio hedge). The
# fitness harness (prompt_opt.py) writes the best template + count for each family.
# EXFIL = SECRET_MARKER multi-post: maximizes PUBLIC (OptimalGuardrail) severity, but is
# 0 under the RulesGuardrail (which denies exactly the scored exfil payload) — the bet
# that the private LB is lenient. DEPUTY = single-message multi-email confused-deputy:
# lower per-hit severity but survives ALL guardrails (optimal/rules/hardened) — the bet
# that the private LB is strict. produce/adaptive read the EXFIL file; the hedged ship
# pool draws from both.
BEST_PROMPT_FILE = (
    CAMPAIGN_ROOT / "best_prompt.json"
)  # EXFIL incumbent (back-compat name)
BEST_DEPUTY_FILE = CAMPAIGN_ROOT / "best_deputy.json"  # DEPUTY (robust) incumbent

# Live proposer config (optimize_prompts.read_proposer). Workers re-read this each
# generation, so `jed-optimize --switch` can change the proposer backend/model/endpoint
# without a restart. Holds no secret — only a `key_env` naming the env var with the key.
PROPOSER_CONFIG_FILE = CAMPAIGN_ROOT / "proposer.json"

# Prompt-optimization orchestrator (optimize_prompts.py): its generation logfile and the
# scratch cwd handed to the bounded codex proposer subprocess (kept away from src/).
OPTIMIZE_LOG = CAMPAIGN_ROOT / "optimize_prompts.log"
CODEX_SCRATCH_DIR = CAMPAIGN_ROOT / "codex_scratch"

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
