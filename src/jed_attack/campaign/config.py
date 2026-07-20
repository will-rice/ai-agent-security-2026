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

# API tokens (ZAI_API_KEY, CHEAPEST_API_KEY) load from a gitignored repo-root .env.
# Pass this explicit path to load_dotenv: bare load_dotenv()/find_dotenv() cannot locate
# .env under ``python -m`` (no reliable calling frame), so the swarm ran keyless
# and dropped every api proposer. An explicit path is deterministic.
ENV_FILE = _REPO_ROOT / ".env"

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

# Ship ARTIFACT: the shape of build_next/attack.py. "composed" (default) packs the
# Pareto archive (archive.py) via the submission composer (compose.py): reserves a
# public floor of the pinned exfil template, then greedily fills the rest of the
# green-seconds budget with the archive entry with the best surviving robust weight per
# green-second (min(gates["rules"], gates["hardened"]) / cost_s) — replaces the
# exfil-only ship path with a maximin hedge over whichever guardrail is private, while
# keeping a public-LB floor. "static" ships the fixed assemble() pool at SCORE_CAP
# candidates — locked to the PROVEN config: 80 × K=5, the only submission that scored
# (34.315). "adaptive" ships the self-sizing adaptive.build_adaptive run(); it
# OVER-fills and timed out on the T4 (v7 blanked).
SHIP_ARTIFACT = os.getenv("JED_SHIP_ARTIFACT", "composed")

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
ARCHIVE_FILE = CAMPAIGN_ROOT / "archive.jsonl"  # Pareto archive of scored messages

# Live proposer config (optimize_prompts.read_proposer). Workers re-read this each
# generation, so `jed-optimize --switch` can change the proposer backend/model/endpoint
# without a restart. Holds no secret — only a `key_env` naming the env var with the key.
PROPOSER_CONFIG_FILE = CAMPAIGN_ROOT / "proposer.json"

# Prompt-optimization orchestrator (optimize_prompts.py): its generation logfile and the
# scratch cwd handed to the bounded codex proposer subprocess (kept away from src/).
OPTIMIZE_LOG = CAMPAIGN_ROOT / "optimize_prompts.log"
CODEX_SCRATCH_DIR = CAMPAIGN_ROOT / "codex_scratch"

# Ship + calibrated-score pool size. 80 is the PROVEN T4 ceiling for K=5 multi-post: the
# 80-candidate pool scored 34.315, everything larger (105/120/230) timed out to ~0. Both
# the shipped static pool (assemble cap) and the local score daemon use this, so local
# calibration reflects exactly what ships.
SCORE_CAP = int(os.getenv("JED_SCORE_CAP", "80"))
SCORE_FILE = CAMPAIGN_ROOT / "score.json"  # latest calibrated public-LB prediction
SCORE_CACHE = CAMPAIGN_ROOT / "score_cache.jsonl"  # per-(message,model) replay cache

# Green-seconds ceiling for the composer's budget check (submission composer, later
# task). Calibrated from the real T4 pass/fail boundary: the 80-candidate K=5 pool
# PASSED and scored 34.315, the 105-candidate pool TIMED OUT. So CEILING ≈ 80 × the
# mean green replay-seconds of one K=5 candidate. PROVISIONAL: this default (2.5s/
# candidate, i.e. 80 * 2.5 = 200.0) is a placeholder guess, not yet a green
# measurement — Task 8 measures the real mean on green and overwrites this constant.
GREEN_SECONDS_CEILING = float(os.getenv("JED_GREEN_CEILING_S", "200.0"))

# Pack to this fraction of the ceiling, leaving margin for T4-vs-green variance (the
# ceiling is measured on green; the actual submission runs on the stricter T4).
BUDGET_FILL_FRACTION = 0.85

# Green-seconds of exfil reserved for the public floor (the promoted best submission,
# see config.FLOOR_DIR). Modest default; tune once real green per-candidate cost_s is
# measured (Task 8).
PUBLIC_FLOOR_COST_S = float(os.getenv("JED_PUBLIC_FLOOR_COST_S", "20.0"))


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
