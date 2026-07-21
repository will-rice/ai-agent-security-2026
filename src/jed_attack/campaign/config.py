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

SCORE_FILE = CAMPAIGN_ROOT / "score.json"  # latest calibrated public-LB prediction
SCORE_CACHE = CAMPAIGN_ROOT / "score_cache.jsonl"  # per-(message,model) replay cache

# Green-seconds ceiling for the composer's budget check. The real T4 boundary is the
# 80-candidate K=5 pool that scored 34.315 (105 TIMED OUT). We reproduce that proven
# pool: the floor below reserves 80 exfil copies (96 green-seconds), and this ceiling
# must leave budget (ceiling * BUDGET_FILL_FRACTION) >= 96 for them to fit, so
# 120 * 0.85 = 102 >= 96. Deputy (~21s each) is then squeezed out, keeping the ship pool
# exfil-heavy like the proven submission; the swarm's cheaper deputy entries pack into
# any slack as they appear.
GREEN_SECONDS_CEILING = float(os.getenv("JED_GREEN_CEILING_S", "120.0"))

# Pack to this fraction of the ceiling, leaving margin for T4-vs-green variance (the
# ceiling is measured on green; the actual submission runs on the stricter T4).
BUDGET_FILL_FRACTION = 0.85

# Green-seconds of the pinned exfil reserved for the public-LB floor of the composed
# pool. 96 = 80 copies * the pinned entry's 1.2s cost_s, so the floor reproduces the
# proven 80-candidate exfil pool (the public-LB scorer). Lower this to trade exfil floor
# for more deputy (the private-LB hedge).
PUBLIC_FLOOR_COST_S = float(os.getenv("JED_PUBLIC_FLOOR_COST_S", "96.0"))


def ensure_dirs() -> None:
    """Create the runtime directories the composer pipeline writes to."""
    for path in (
        BUILD_NEXT_DIR,
        NOTES_DIR,
        ARCHIVE_FILE.parent,  # == CAMPAIGN_ROOT; also holds score.json + score cache
        CAMPAIGN_ROOT / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)
