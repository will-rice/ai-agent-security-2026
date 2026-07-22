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

SHARDS_DIR = CAMPAIGN_ROOT / "shards"  # per-worker scored-candidate shard files (map)
CONSOLIDATOR_STATUS_FILE = CAMPAIGN_ROOT / "consolidator_status.json"  # total_score_est
CONSOLIDATE_INTERVAL_S = float(os.getenv("JED_CONSOLIDATE_INTERVAL_S", "15"))

# Tool-hop ceiling for the composer's budget check — the binding T4 resource (replay
# work ∝ tool hops), DETERMINISTIC and identical on green and T4 (unlike measured
# wall-time, which fleet contention inflates). Calibrated to the empirical boundary:
# 80×K5 = 400 hops PASSED (scored 34.315); 105×K5 = 525 and 230×K2 = 460 BLANKED. 460 is
# the observed failure point; BUDGET_FILL_FRACTION keeps us safely under it.
HOP_CEILING = int(os.getenv("JED_HOP_CEILING", "460"))

# Pack to this fraction of the ceiling, leaving margin below the observed 460-hop fail
# point (460 * 0.85 = 391 hops ≈ the proven 400-hop pool). Hops are green==T4, so this
# margin is pure pass/fail safety, not a green-vs-T4 correction.
BUDGET_FILL_FRACTION = 0.85

# Tool hops reserved for the public-LB floor of the composed pool: copies of the search
# best public-value (exfil) entry until their summed hops cover this. 240 ≈ 48×K5 exfil,
# leaving ~151 hops (391 - 240) for the private deputy fill — a public/private hedge.
# Lower this to trade public floor for more deputy (the private-LB bet).
PUBLIC_FLOOR_HOPS = int(os.getenv("JED_PUBLIC_FLOOR_HOPS", "240"))

# The competition's per-cell novelty bonus (see score.py: raw += 2 * unique_cells). Each
# rendered copy carries unique .invalid domains/addresses -> a distinct score cell, so a
# firing copy is worth (surviving_severity + NOVELTY_PER_CELL). Single source for the
# marginal-value objective (compose + prompt_opt) and the in-loop score estimate.
NOVELTY_PER_CELL = 2.0


def ensure_dirs() -> None:
    """Create the runtime directories the composer pipeline writes to."""
    for path in (
        BUILD_NEXT_DIR,
        NOTES_DIR,
        ARCHIVE_FILE.parent,  # == CAMPAIGN_ROOT; also holds score.json + score cache
        CAMPAIGN_ROOT / "logs",
        SHARDS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
