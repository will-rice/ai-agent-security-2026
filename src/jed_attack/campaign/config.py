"""Campaign paths, models, and knobs — env-overridable.

Runtime state lives under ``run/`` (git-ignored). The async team and the blackboard
both read these so paths stay consistent.
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
)  # assembled attack.py + build_next_status.json

# The two target models and their served base URLs (llama-server on green).
MODELS: tuple[str, ...] = ("gpt_oss", "gemma_4")

# The proposer models the async team rotates. optimize_team GROUPS these by API key into
# one lane per key (one worker per lane), and the lane's worker rotates through its
# models one generation at a time — so only one request per key is ever in flight (the
# cheapestinference concurrency cap is per-KEY, confirmed). Two lanes result:
# CHEAPEST_API_KEY (all 6 CI models — the new token covers every one) and ZAI_API_KEY
# (glm-5-turbo — structured-capable + fast). A model whose key_env is unset is skipped.
TEAM_PROPOSERS: tuple[str, ...] = (
    "cheapest-kimi",
    "cheapest-kimi2.6",
    "cheapest-deepseek",
    "cheapest-glm5.2",
    "cheapest-minimax",
    "cheapest-mimo",
    "zai-glm5-turbo",
)

# Max tool hops per interaction. MUST match the scored gateway's replay depth
# (aicomp_sdk DEFAULT_MAX_TOOL_HOPS = AttackRunConfig.max_tool_hops = 8), or
# multi-post candidates (one message → up to 8 SECRET_MARKER posts) are
# under-counted locally and wrongly gated out. Single-post scores are unchanged.
EVAL_HOPS = 8

# Prompt-optimization orchestrator (optimize_prompts.py): its generation logfile.
OPTIMIZE_LOG = CAMPAIGN_ROOT / "optimize_prompts.log"

# The team's shared memory (blackboard.py): an append-only JSONL record of every scored
# submission (messages, score, feedback, the proposer's reasoning, model, worker). The
# in-memory blackboard is rebuilt from it on start (warm restart).
BLACKBOARD_LOG = CAMPAIGN_ROOT / "blackboard.jsonl"

# Tool-hop ceiling for the submission budget check — the binding T4 resource (replay
# work ∝ tool hops), DETERMINISTIC and identical on green and T4 (unlike measured
# wall-time, which fleet contention inflates). Calibrated to the empirical boundary:
# 80×K5 = 400 hops PASSED (scored 34.315); 105×K5 = 525 and 230×K2 = 460 BLANKED. 460 is
# the observed failure point; BUDGET_FILL_FRACTION keeps us safely under it.
HOP_CEILING = int(os.getenv("JED_HOP_CEILING", "460"))

# Pack to this fraction of the ceiling, leaving margin below the observed 460-hop fail
# point (460 * 0.85 = 391 hops ≈ the proven 400-hop pool). Hops are green==T4, so this
# margin is pure pass/fail safety, not a green-vs-T4 correction.
BUDGET_FILL_FRACTION = 0.85

# Hard cap on the number of messages in a submission — enforced structurally on the
# model's `Submission` pydantic (see submission.py: Field(max_length=...)), so the model
# can never author more than this. The empirical T4 boundary reads as a COUNT limit as
# much as a hop one: 80 messages PASSED; 105 and 230 BLANKED. Each replayed candidate
# pays a fixed per-candidate cost (fresh SandboxEnv reset + agent first-inference)
# INDEPENDENT of its hops, so message count is a distinct binding resource from total
# hops. This cap keeps us at the proven-safe 80-count regime; the hop half is T4-fit.
MAX_SHIP_MESSAGES = int(os.getenv("JED_MAX_SHIP_MESSAGES", "80"))

# The competition's per-cell novelty bonus (raw += 2 * unique_cells). Each rendered copy
# carries unique .invalid domains/addresses -> a distinct score cell, so a firing copy
# is worth (surviving_severity + NOVELTY_PER_CELL). Single source for the scorer's cell
# bonus (submission_score.py).
NOVELTY_PER_CELL = 2.0

# Tier-2 victim introspection (victim_feedback.introspect_worst): number of the
# lowest-severity messages in a scored submission to probe with an extra "what
# change would flip your decision" turn against the victim itself. Bounds the
# extra per-submission inference cost to a fixed count, not O(submission size).
INTROSPECT_WORST_N = int(os.getenv("JED_INTROSPECT_WORST_N", "3"))


def ensure_dirs() -> None:
    """Create the runtime directories the submission pipeline writes to."""
    for path in (BUILD_NEXT_DIR, CAMPAIGN_ROOT / "logs"):
        path.mkdir(parents=True, exist_ok=True)
