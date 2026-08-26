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
MESSAGE = "url-after-end 108 champion: gpt forge-before-host (host truly last) + plain gemma, 1826 firing/pool (verified all_fire). Re-submitting the best-known real-board shape as the clean floor after the objective was recalibrated to total input+output tokens."
POLL_INTERVAL_S = 30
POLL_TIMEOUT_S = 6 * 60 * 60  # 6h covers the push run plus the rerun eval
# Never ship an under-filled pool. A pool below the budget-completable count leaves grader
# time idle before its deadline: champion_gated shipped ~994 firing and scored 89.5, vs the
# full 2000-candidate champion's 104.4. The real floor is the completable count -- the grader
# only replays ~985-1010 candidates before the deadline (memory completion-ceiling-aug15), so
# any pool comfortably above that (e.g. the 1826-host url-after-end 108 champion) never runs
# the grader dry. Floor set at 1200 = the completable ceiling plus margin, NOT the leaner-pool
# host count; a pool above it is fine even if it does not reach 1900 single-token hosts.
MIN_SHIP_CANDIDATES = 1200
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
    # firing (a hash-bound verified.json from scripts/verify_cut.py). This fails BEFORE
    # the push, so a dud pool can never reach a submission.
    from jed_attack.campaign.verify import is_verified

    attack_path = build_kernel.REPO_ROOT / build_kernel.ATTACK_PY
    ok, reason = is_verified(attack_path)
    if not ok:
        sys.exit(f"FIRING GATE: refusing to submit -- {reason}")
    print(f"firing gate: {reason} ({build_kernel.ATTACK_PY})")

    # SIZE GATE: never ship a pool below the budget-completable count -- it leaves grader
    # time idle and scores less than a bigger pool (996 gpt / champion_gated 89.5 both
    # under-filled). Refuse before the push; the fix is more hosts / higher fire-rate.
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
