"""Campaign paths, models, and knobs — env-overridable.

Runtime state lives under ``run/`` (git-ignored). The optimizer swarm, the composer, and
the score/assemble daemons all read these so paths stay consistent across the fleet.
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

BUILD_NEXT_DIR = (
    CAMPAIGN_ROOT / "build_next"
)  # composed attack.py + build_next_status.json

# Shared cross-agent knowledge log (see knowledge.py). Each writer appends its own
# <producer>.jsonl so the fleet learns from every agent's tries without locks.
KNOWLEDGE_DIR = CAMPAIGN_ROOT / "knowledge"
NOTES_DIR = KNOWLEDGE_DIR / "notes"  # free-form insights

# The two target models and their served base URLs (llama-server on green).
MODELS: tuple[str, ...] = ("gpt_oss", "gemma_4")

# Max tool hops per interaction. MUST match the scored gateway's replay depth
# (aicomp_sdk DEFAULT_MAX_TOOL_HOPS = AttackRunConfig.max_tool_hops = 8), or
# multi-post candidates (one message → up to 8 SECRET_MARKER posts) are
# under-counted locally and wrongly gated out. Single-post scores are unchanged.
EVAL_HOPS = 8

# Concurrent replay workers for the GPU consumers (score, prompt-opt). Sized to
# the gemma llama-server's batch slots (-np 8) so concurrent requests fill the
# continuous-batching slots instead of queuing one-at-a-time. gemma is the binding
# server; gpt (fewer slots) just queues the excess, which is harmless.
REPLAY_WORKERS = int(os.getenv("JED_REPLAY_WORKERS", "8"))

ARCHIVE_FILE = CAMPAIGN_ROOT / "archive.jsonl"  # Pareto archive of scored messages

# Live proposer config (optimize_prompts.read_proposer). Workers re-read this each
# generation, so `jed-optimize --switch` can change the proposer backend/model/endpoint
# without a restart. Holds no secret — only a `key_env` naming the env var with the key.
PROPOSER_CONFIG_FILE = CAMPAIGN_ROOT / "proposer.json"

# Prompt-optimization orchestrator (optimize_prompts.py): its generation logfile and the
# scratch cwd handed to the bounded codex proposer subprocess (kept away from src/).
OPTIMIZE_LOG = CAMPAIGN_ROOT / "optimize_prompts.log"
CODEX_SCRATCH_DIR = CAMPAIGN_ROOT / "codex_scratch"

# Calibrated-score pool size. 80 is the PROVEN T4 ceiling for K=5 multi-post: the
# 80-candidate pool scored 34.315, everything larger (105/120/230) timed out to ~0.
SCORE_CAP = int(os.getenv("JED_SCORE_CAP", "80"))
SCORE_FILE = CAMPAIGN_ROOT / "score.json"  # latest calibrated public-LB prediction
SCORE_CACHE = CAMPAIGN_ROOT / "score_cache.jsonl"  # per-(message,model) replay cache

# Green-seconds ceiling for the composer's budget check (submission composer). Calibrated
# from the real T4 pass/fail boundary: the 80-candidate K=5 pool PASSED and scored
# 34.315, the 105-candidate pool TIMED OUT. So CEILING ≈ 80 × the mean green replay-
# seconds of one K=5 candidate. PROVISIONAL: this default (2.5s/candidate, i.e.
# 80 * 2.5 = 200.0) is a placeholder guess, not yet a green measurement.
GREEN_SECONDS_CEILING = float(os.getenv("JED_GREEN_CEILING_S", "200.0"))

# Pack to this fraction of the ceiling, leaving margin for T4-vs-green variance (the
# ceiling is measured on green; the actual submission runs on the stricter T4).
BUDGET_FILL_FRACTION = 0.85

# Green-seconds of the pinned exfil reserved for the public-LB floor of the composed
# pool. Modest default; tune once real green per-candidate cost_s is measured.
PUBLIC_FLOOR_COST_S = float(os.getenv("JED_PUBLIC_FLOOR_COST_S", "20.0"))


def ensure_dirs() -> None:
    """Create the runtime directories the composer pipeline writes to."""
    for path in (
        BUILD_NEXT_DIR,
        NOTES_DIR,
        ARCHIVE_FILE.parent,  # == CAMPAIGN_ROOT; also holds score.json + score cache
        CAMPAIGN_ROOT / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)
