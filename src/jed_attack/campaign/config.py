"""Campaign paths, models, and knobs — env-overridable.

Runtime state lives under ``run/`` (git-ignored). The async team and the blackboard
both read these so paths stay consistent.
"""

import os
import tomllib
from pathlib import Path
from typing import Any

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

# The hot-reloadable proposer prompts (system + template), re-read every generation, so
# editing the guidance takes effect on the next generation with NO worker restart (edit,
# then sync_green.sh). Version-controlled next to the code, not runtime state.
PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.toml"

# The team's shared memory (blackboard.py): an append-only JSONL record of every scored
# submission (messages, score, feedback, the proposer's reasoning, model, worker). The
# in-memory blackboard is rebuilt from it on start (warm restart).
BLACKBOARD_LOG = CAMPAIGN_ROOT / "blackboard.jsonl"

# HOP_CEILING, BUDGET_FILL_FRACTION, MAX_SHIP_MESSAGES are HOT-RELOADABLE numeric caps:
# they live in tunables.toml, reached via the module __getattr__ below, so they can be
# tuned with NO worker restart (edit tunables.toml + sync_green.sh). Calibration behind
# the defaults (binding T4 resources; deterministic, green==T4):
#   HOP_CEILING (460): tool-hop ceiling; replay work scales with hops. 80x5 = 400 hops
#     PASSED (34.315); 105x5=525, 230x2=460 BLANKED -> 460 = the observed fail point.
#   BUDGET_FILL_FRACTION (0.85): pack to this fraction (460 * 0.85 = 391 ~ proven 400).
#   MAX_SHIP_MESSAGES (80): message-count cap -- a distinct T4 resource from hops (each
#     replayed candidate pays a fixed per-candidate cost). 80 PASSED; 105/230 BLANKED.
TUNABLES_FILE = Path(__file__).resolve().parent / "tunables.toml"

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


_TUNABLE_DEFAULTS: dict[str, float] = {
    "max_ship_messages": 80,
    "hop_ceiling": 460,
    "budget_fill_fraction": 0.85,
}
_TUNABLE_ATTRS: dict[str, tuple[str, type]] = {
    "MAX_SHIP_MESSAGES": ("max_ship_messages", int),
    "HOP_CEILING": ("hop_ceiling", int),
    "BUDGET_FILL_FRACTION": ("budget_fill_fraction", float),
}
_tunables_cache: dict[str, Any] = {}
_tunables_mtime: float = -1.0


def _tunables() -> dict[str, Any]:
    """The hot-reloadable caps (tunables.toml), re-parsed only when the file changes.

    Cached by mtime so a per-Message-validation attribute read costs one ``stat()``, not
    a TOML parse; an edit is picked up on its next stat (so no worker restart).
    """
    global _tunables_cache, _tunables_mtime
    try:
        mtime = TUNABLES_FILE.stat().st_mtime
    except OSError:
        return _tunables_cache
    if mtime != _tunables_mtime:
        _tunables_mtime = mtime
        try:
            _tunables_cache = tomllib.loads(TUNABLES_FILE.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            pass  # keep the last-good values through a mid-edit parse error
    return _tunables_cache


def __getattr__(name: str) -> Any:  # noqa: ANN401 - PEP 562 hook: heterogeneous int|float
    """Route MAX_SHIP_MESSAGES / HOP_CEILING / BUDGET_FILL_FRACTION to tunables.toml.

    A module ``__getattr__`` (PEP 562) fires only for names not bound as normal globals,
    so NOT defining these three as constants sends every ``config.MAX_SHIP_MESSAGES``
    access here — hot with no caller change. The Submission schema enforces the
    caps in its model_validator (not a Field), so a change takes effect live.
    """
    if name in _TUNABLE_ATTRS:
        key, cast = _TUNABLE_ATTRS[name]
        return cast(_tunables().get(key, _TUNABLE_DEFAULTS[key]))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
