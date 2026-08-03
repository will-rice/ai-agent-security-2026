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
BUILD_ROBUST_DIR = CAMPAIGN_ROOT / "build_robust"
SUBMISSION_CUTS_DIR = CAMPAIGN_ROOT / "submission_cuts"
ARTIFACT_CHAMPION_PATH = CAMPAIGN_ROOT / "artifact_champion.json"

# The two target models and their served base URLs (llama-server on green).
MODELS: tuple[str, ...] = ("gpt_oss", "gemma_4")

# In-process scoring: where the GGUFs live, and which GPU each model loads on (under
# CUDA_DEVICE_ORDER=PCI_BUS_ID: device 0 = RTX 3090, device 1 = RTX 6000 Ada). The
# scorer loads each GGUF resident and replays in-process (llama-cpp-python) to match
# the T4 gateway exactly -- see docs/.../in-process-scoring-design.md and memory
# jed-t4-replay-time-budget.
MODELS_DIR = Path(os.getenv("JED_MODELS_DIR", str(_REPO_ROOT / "models")))
MODEL_GPU: dict[str, int] = {"gpt_oss": 0, "gemma_4": 1}

# Time-aware objective: the T4 gateway INVALIDATES a submission that exceeds ~9000s/
# model of wall-clock replay. We replay in-process, so we MEASURE green replay-seconds
# per model and zero any submission over budget (mirrors T4 INVALID). Budget is in
# GREEN-seconds. CALIBRATED FROM A REAL T4 VERIFY 2026-07-24 (v11): a 21-candidate /
# 179-green-s submission PASSED T4 in ~5060s gpt_oss (of 9000) -> ~241 T4-s/candidate
# vs 8.5 green-s/candidate = a ~28x ratio, so 9000 T4-s ≈ 318 green-s. With
# MAX_SHIP_MESSAGES=30 the largest possible submission is ~255 green-s (~7200 T4-s),
# so the count cap -- not time -- binds; 300 sits above any capped submission (never
# wrongly zeros) yet below the ~318 ceiling (backstop only). Earlier 120/200/280 were
# too tight (120 zeroed ~60% of valid output). gemma never binds.
GREEN_REPLAY_BUDGET_S: dict[str, float] = {"gpt_oss": 300.0, "gemma_4": 60.0}


# Exact generated-artifact scoring. The optimizer still uses public raw/sec as its
# cheap inner-loop objective, but on every new shipped ``build_next/attack.py`` we can
# run the full SDK evaluator (including live validation/fill) and log leaderboard-like
# ``artifact_*`` metrics. Disable only when throughput debugging is more important than
# exact outer-loop telemetry.
ARTIFACT_SCORE_ENABLED = os.getenv("JED_ARTIFACT_SCORE", "1") != "0"
# Budget for the telemetry-only artifact re-score. A full 9000s (Kaggle-equivalent)
# score GPU-locks the lane for ~103 min and, since it also fires on every champion
# reship, it stalled the whole search for ~1h45m at a time. This score feeds ONLY the
# wandb lb_est / artifact-champion telemetry, not the search objective, so a short
# consistent sample is fine: ~600s scores in ~7 min and still tracks relative progress.
# Raise it (or run a one-off full score) when an exact Kaggle-scale number is needed.
ARTIFACT_SCORE_BUDGET_S = float(os.getenv("JED_ARTIFACT_SCORE_BUDGET_S", "600"))
# Also re-score the shipped artifact on a timer, not only on a new-champion reship —
# otherwise the "artifact score" telemetry goes stale for the entire duration of any
# objective plateau (it silently did for ~27h). A full score is expensive (~1-2h), so
# default to a long interval; 0 disables the timer (reship-only, the old behavior).
ARTIFACT_SCORE_EVERY_S = float(os.getenv("JED_ARTIFACT_SCORE_EVERY_S", "21600"))

# Leaderboard calibration for live-fill artifacts. The 2026-07-28 submission
# ``artifact-public-180 sha-95083d3e t4`` scored 180.0 under local artifact
# replay, but 86.265 on the public LB. That run was a validated single-post
# fill: each credited candidate contributed 18 raw points (16 EXFIL severity
# + 2 novelty), so 86.265 public corresponds to 958.5 credited candidates.
# Keep exploration open-ended, but log/cut on this T4-adjusted estimate.
ARTIFACT_LB_REFERENCE_PUBLIC = float(
    os.getenv("JED_ARTIFACT_LB_REFERENCE_PUBLIC", "86.265")
)
ARTIFACT_LB_REFERENCE_RAW_PER_CANDIDATE = float(
    os.getenv("JED_ARTIFACT_LB_REFERENCE_RAW_PER_CANDIDATE", "18.0")
)
ARTIFACT_LB_REFERENCE_CANDIDATES = (
    ARTIFACT_LB_REFERENCE_PUBLIC
    * 200_000.0
    / 1_000.0
    / ARTIFACT_LB_REFERENCE_RAW_PER_CANDIDATE
)
ARTIFACT_LB_REFERENCE_LOCAL_S_PER_CANDIDATE: dict[str, float] = {
    "gpt_oss": float(
        os.getenv(
            "JED_ARTIFACT_LB_REFERENCE_GPT_OSS_LOCAL_S_PER_CANDIDATE",
            str(2028.7454912662506 / 2000.0),
        )
    ),
    "gemma_4": float(
        os.getenv(
            "JED_ARTIFACT_LB_REFERENCE_GEMMA_4_LOCAL_S_PER_CANDIDATE",
            str(2100.8640832901 / 2000.0),
        )
    ),
}


def team_proposers_from_env(
    value: str | None, *, default: tuple[str, ...]
) -> tuple[str, ...]:
    """Parse a comma-separated proposer roster override.

    Args:
        value: Raw ``JED_TEAM_PROPOSERS`` value, or ``None``.
        default: The fallback roster when no usable override is set.

    Returns:
        The stripped proposer names, preserving order, or ``default``.
    """
    if value is None:
        return default
    proposers = tuple(name.strip() for name in value.split(",") if name.strip())
    return proposers or default


# The proposer models the async team rotates. optimize_team GROUPS these by API key into
# one lane per key (one worker per lane). CheapestInference is also single-flight at
# the model-window level, so the worker must not advance to another CI model after a
# CI concurrency/stream failure; it retries the same model after a cooldown instead.
# The CI entries below are only a static fallback/order hint: at optimizer startup the
# CI lane is replaced with the live /v1/models response for CHEAPEST_API_KEY. Operators
# can temporarily pin a smaller roster with
# JED_TEAM_PROPOSERS="cheapest-minimax,zai-glm5-turbo" without editing source.
_TEAM_PROPOSERS_ENV = os.getenv("JED_TEAM_PROPOSERS")
TEAM_PROPOSERS_FROM_ENV = _TEAM_PROPOSERS_ENV is not None
TEAM_PROPOSERS: tuple[str, ...] = team_proposers_from_env(
    _TEAM_PROPOSERS_ENV,
    default=(
        "cheapest-deepseek",
        "cheapest-mimo",
        "cheapest-minimax",
        "cheapest-kimi",
        "cheapest-glm5.2",
        "zai-glm5-turbo",
    ),
)

# Max tool hops per interaction. MUST match the scored gateway's replay depth
# (aicomp_sdk DEFAULT_MAX_TOOL_HOPS = AttackRunConfig.max_tool_hops = 8), or
# multi-post candidates (one message → up to 8 SECRET_MARKER posts) are
# under-counted locally and wrongly gated out. Single-post scores are unchanged.
EVAL_HOPS = 8

# Adversarial refinement: max per-generation batch hill-climb rounds. After scoring a
# batch, the lane re-authors the WHOLE batch against its real per-message scores,
# re-scores every submission, and keeps the higher-mean-public batch, up to this many
# rounds. Four restores the approved adversarial hill-climb cap: at most five full
# batch scorings per generation, with early stop on the first non-improving rewrite.
# Set to 0 to disable refinement when scoring throughput is the binding constraint.
REFINE_MAX_ROUNDS = 4

# Prompt-optimization orchestrator (optimize_prompts.py): its generation logfile.
OPTIMIZE_LOG = CAMPAIGN_ROOT / "optimize_prompts.log"

# The hot-reloadable proposer prompts (system + template), re-read every generation, so
# editing the guidance takes effect on the next generation with NO worker restart (edit,
# then sync_green.sh). Version-controlled next to the code, not runtime state.
PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.toml"

# The team's shared memory (blackboard.py): an append-only JSONL record of every scored
# submission (messages, score, feedback, the proposer's reasoning, model, worker). The
# in-memory blackboard is rebuilt from it on start (warm restart).
BLACKBOARD_LOG = CAMPAIGN_ROOT / "blackboard.jsonl"

# Static T4 caps (plain module constants like EVAL_HOPS; NOT hot-reloadable -- changing
# one is a worker restart). MAX_SHIP_MESSAGES is a real Field(max_length) on Submission,
# so it is the JSON schema's maxItems (what the proposer sees) AND the validation rule
# from one source -- no drift. HOP_BUDGET is a summed-hop cap with no schema form,
# enforced in a model_validator.
#   MAX_SHIP_MESSAGES (30): candidate count is the BINDING score lever -- scoring has no
#     dedup, so each firing candidate adds severity + a novelty cell. The real T4 limit
#     is a 9000s/model WALL-CLOCK replay budget; the v11 verify measured ~241 T4-s per
#     candidate, so 30 candidates ≈ 7230s ≈ 80% of the budget (~20% margin). Was 25 when
#     the budget was mis-modeled as too tight; raise further (toward ~33) only after a
#     verify confirms the larger size, and keep GREEN_REPLAY_BUDGET_S above 30 x ~8.5
#     green-s = 255. See memory jed-t4-replay-time-budget.
#   HOP_BUDGET (391): total tool-hop budget (85% of the 460-hop fail point). NOTE: hops
#     are NOT the binding constraint -- wall-clock replay time is; this is a loose upper
#     bound only (30 x 8 = 240 hops, well under).
MAX_SHIP_MESSAGES = 30
HOP_BUDGET = 391

# Max submissions per proposer generation. A proposer left uncapped emits huge batches
# (200+) which, at ~one replay per submission per model, stretch a generation to an hour
# and starve the search of iterations. This is the JSON-schema ``maxItems`` the proposer
# is constrained to (so structured decoding STOPS at the cap instead of us truncating
# waste) AND the salvage enforcement, from one source. Small is fine: the shipped
# artifact self-fills URLs to the T4 budget at eval, so batch size sets the search's
# iteration granularity, not the shipped candidate count. Fewer, faster generations =
# quicker discovery of leaner-reasoning shapes. Not hot-reloadable — a worker restart.
MAX_SCORE_BATCH = 8

# The competition's per-cell novelty bonus (raw += 2 * unique_cells). Each rendered copy
# carries unique .invalid domains/addresses -> a distinct score cell, so a firing copy
# is worth (surviving_severity + NOVELTY_PER_CELL). Single source for the scorer's cell
# bonus (submission_score.py).
NOVELTY_PER_CELL = 2.0

# Cross-model robustness weight for the optimizer objective (hedges the blind private
# board). The objective's numerator blends the per-model MEAN board (score.public) with
# the per-model MINIMUM: robust = (1-L)*mean + L*min. L=0.0 is the current mean behavior
# (a lopsided one-model shape still scores well); L->1.0 rewards only shapes that fire
# on BOTH victims, which is what transfers to unseen private models. Non-zero L stamps a
# distinct objective scheme tag (blackboard.OBJECTIVE_NAME) so its rescaled magnitudes
# cannot out-rank mean-scheme champions. Prototype: leave 0.0 until you flip it on.
ROBUSTNESS_LAMBDA = float(os.getenv("JED_ROBUSTNESS_LAMBDA", "0.0"))
if not 0.0 <= ROBUSTNESS_LAMBDA <= 1.0:
    raise ValueError("JED_ROBUSTNESS_LAMBDA must be in [0.0, 1.0]")

# Tier-2 victim introspection (victim_feedback.introspect_worst): number of the
# lowest-severity messages in a scored submission to probe with an extra "what
# change would flip your decision" turn against the victim itself. Bounds the
# extra per-submission inference cost to a fixed count, not O(submission size).
INTROSPECT_WORST_N = int(os.getenv("JED_INTROSPECT_WORST_N", "3"))

# Held-out offline proxy study. RulesGuardrail is passed only by judge_study.py and is
# never inserted into GATE_GUARDRAILS or the live scoring path.
JUDGE_VERSION = "robustness-v1"
JUDGE_ANCHOR_VERSION = "anchors-v1"
JUDGE_PUBLIC_BAND_RATIO = 0.05
JUDGE_MODE = os.getenv("JED_JUDGE_MODE", "shadow")
if JUDGE_MODE not in {"off", "shadow", "active"}:
    raise ValueError("JED_JUDGE_MODE must be one of: off, shadow, active")
JUDGE_ROBUSTNESS_TIE_POINTS = 5.0
JUDGE_MIN_CONFIDENCE = 0.60
JUDGE_MAX_CONCURRENT_ASSESSMENTS = max(
    1, int(os.getenv("JED_JUDGE_MAX_CONCURRENT_ASSESSMENTS", "1"))
)
NOVELTY_POOL_SAMPLE = 8
JUDGE_STUDY_N = 40
JUDGE_STUDY_LABEL_SPARES = 4
JUDGE_STUDY_FIXTURE_PROBES = 8
JUDGE_STUDY_DIR = CAMPAIGN_ROOT / "judge_study_v1"

# Dylan judge service (FastAPI + vLLM). The green optimizer POSTs typed judge requests
# to this one endpoint; the service calls the co-located vLLM OpenAI server. See
# docs/.../2026-07-24-dylan-judge-fleet-novelty-selection-design.md.
DYLAN_JUDGE_URL = os.getenv("DYLAN_JUDGE_URL", "http://dylan:8100")
VLLM_URL = os.getenv(
    "VLLM_URL", "http://127.0.0.1:8000/v1"
)  # dylan-local, service->vLLM
VLLM_MODEL = os.getenv("VLLM_MODEL", "bullerwins/Qwen3-32B-awq")


def ensure_dirs() -> None:
    """Create the runtime directories the submission pipeline writes to."""
    for path in (
        BUILD_NEXT_DIR,
        BUILD_ROBUST_DIR,
        SUBMISSION_CUTS_DIR,
        CAMPAIGN_ROOT / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)
