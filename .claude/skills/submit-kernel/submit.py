#!/usr/bin/env python3
"""Build and submit the JED kernel to Kaggle in one Python step.

No flags: every setting is an ALL-CAPS CONSTANT (here and in build_kernel.py).
Run with no arguments to build the kernel, push it, wait for the kernel run to
COMPLETE, submit it to the competition, and poll the evaluation. Submitting
spends a scarce daily slot, so this is a deliberate, human-invoked action
(AGENTS.md: hold submission for human direction).

It is self-guarding: the competition submit only happens after the pushed
kernel run reaches COMPLETE, so a broken build fails before any slot is spent.
Reuses the nvidia-kaggle skill's submit_kernel functions rather than
reimplementing the Kaggle API dance.
"""

import os
import sys
from pathlib import Path

import build_kernel

# --- Fixed submission config (constants, not flags) ---
NVIDIA_KAGGLE_SCRIPTS = Path("/home/will/.agents/skills/nvidia-kaggle-skill/scripts")
ENV_FILE = build_kernel.REPO_ROOT / ".env"  # supplies KAGGLE_API_TOKEN
SUBMISSION_FILE = "submission.csv"  # the output the competition rerun scores
MESSAGE = "Leaner 2000/pool cut: gpt ~22-tok commentary-forge + gemma ~24-tok unquoted-toolcall, tiled across all single-token hosts to ship the full 2000-replay budget of lean fires (targeting the ~1633 completions the LB top implies)."
POLL_INTERVAL_S = 30
POLL_TIMEOUT_S = 6 * 60 * 60  # 6h covers the push run plus the rerun eval
# ALWAYS ship the full pool -- the SDK replays up to MAX_REPLAY_FINDINGS = 2000 per model,
# so 2000 firing candidates per pool is the target EVERY submission. The completable count
# is NOT a fixed ~1010 (that was the champion's slow replay rate wrongly generalized): it
# SCALES with leanness -- leaner candidates replay faster, so more of them complete before
# the deadline. The public LB top (~147) implies ~1633 completed per column (147 / 0.09),
# far above 1010, so under-shipping caps the score by starving the grader of candidates to
# replay. Floor = 2000: never ship a pool that cannot fill the 2000-replay budget. Ordering
# is leanest-first (firing_only), so if the gate yields >2000 the leanest 2000 are the ones
# that replay.
MIN_SHIP_CANDIDATES = 2000
# ------------------------------------------------------


def _load_env(path: Path) -> None:
    """Load KEY=VALUE lines from .env into the environment (for KAGGLE_API_TOKEN).

    The Kaggle SDK uses ``KAGGLE_API_TOKEN`` as the bearer token, so it must hold
    the real value — never a placeholder, or the push 401s.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main() -> None:
    """Build → push → wait for COMPLETE → submit → poll eval → report score."""
    _load_env(ENV_FILE)
    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit(f"KAGGLE_API_TOKEN not set (expected in {ENV_FILE})")

    # FIRING GATE: a cut cannot be submitted unless every embedded candidate is verified
    # firing (a hash-bound verified.json from `python -m jed_attack.scripts.verify_cut`).
    # This fails BEFORE the push, so a dud pool can never reach a submission.
    from jed_attack.campaign.verify import is_verified

    attack_path = build_kernel.REPO_ROOT / build_kernel.ATTACK_PY
    ok, reason = is_verified(attack_path)
    if not ok:
        sys.exit(f"FIRING GATE: refusing to submit -- {reason}")
    print(f"firing gate: {reason} ({build_kernel.ATTACK_PY})")

    # SIZE GATE: ship the FULL 2000-replay budget every time. A pool below 2000 starves
    # the grader of candidates and caps the score (the completable count scales with
    # leanness -- the LB top completes ~1633/column, far above the old ~1010 belief).
    # Refuse before the push; the fix is more hosts / more lean shapes to fill 2000.
    from jed_attack.campaign.verify import _extract_pool

    _src = attack_path.read_text(encoding="utf-8")
    for _var, _model in (("_FORGE", "gpt_oss"), ("_PLAIN", "gemma_4")):
        _n = len(_extract_pool(_src, _var))
        if _n < MIN_SHIP_CANDIDATES:
            sys.exit(
                f"SIZE GATE: refusing to submit -- {_model} pool has {_n} firing "
                f"candidates < {MIN_SHIP_CANDIDATES} (under-fills the budget; build "
                f"more hosts / a higher fire-rate shape)"
            )
        print(f"size gate: {_model} {_n} firing candidates (>= {MIN_SHIP_CANDIDATES})")

    sys.path.insert(0, str(NVIDIA_KAGGLE_SCRIPTS))
    import submit_kernel as sk

    folder = build_kernel.build()
    slug = sk.read_kernel_metadata(str(folder))["id"]
    print(f"built {folder} (kernel {slug})")

    api = sk.get_api()
    version = sk.push_kernel(api, str(folder))
    status, _ = sk.poll_kernel(api, slug, POLL_INTERVAL_S, POLL_TIMEOUT_S)
    if status != "complete":
        sys.exit(f"kernel run did not complete (status={status}); not submitting")

    # Make the description unique per pushed version — poll_submission matches by it,
    # so a constant message could latch onto a prior submission's stale score.
    message = f"{MESSAGE} (v{version})"
    if not sk.submit_to_competition(
        api, slug, build_kernel.COMPETITION, SUBMISSION_FILE, version, message
    ):
        sys.exit("competition submission rejected")

    eval_status, score, _ = sk.poll_submission(
        api, build_kernel.COMPETITION, message, POLL_INTERVAL_S, POLL_TIMEOUT_S
    )
    print(f"eval: {eval_status} | public score: {score}")
    if eval_status != "complete":
        sys.exit(1)


if __name__ == "__main__":
    main()
