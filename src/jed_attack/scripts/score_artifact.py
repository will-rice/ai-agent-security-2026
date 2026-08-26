"""Pre-submit check: score a built ``attack.py`` at the full Kaggle-equivalent budget.

In-loop artifact scoring is disabled during the search (it GPU-locks the lane and
stalls the optimizer — see ``config.ARTIFACT_SCORE_ENABLED``). Run this deliberately
before submitting to get the leaderboard-estimate metrics for the shipped artifact.

Loads its own resident GGUF backends, so stop the optimizer first (or expect GPU
contention). Usage::

    uv run python -m jed_attack.scripts.score_artifact [attack_path] [--budget-s 9000]
"""

import argparse
import logging
from pathlib import Path

from jed_attack.campaign import config
from jed_attack.campaign.artifact_score import score_artifact_metrics

_HEADLINE_KEYS = (
    "artifact_public",
    "artifact_lb_est_public",
    "artifact_lb_est_score_raw",
    "artifact_lb_est_candidate_count_min",
    "artifact_sha256",
)


def main() -> None:
    """Score the artifact and log its metrics, headline numbers first."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "attack_path",
        nargs="?",
        default=str(config.CAMPAIGN_ROOT / "build_next" / "attack.py"),
        help="Built attack.py to score (default: the shipped build_next/attack.py).",
    )
    parser.add_argument(
        "--budget-s",
        type=float,
        default=config.ARTIFACT_SCORE_BUDGET_S,
        help="Per-model SDK generation/replay budget (Kaggle uses 9000).",
    )
    args = parser.parse_args()

    path = Path(args.attack_path).resolve()
    if not path.exists():
        raise SystemExit(f"artifact not found: {path}")

    logging.info(
        "scoring %s against %s at %.0fs/model (this takes a while)...",
        path,
        ", ".join(config.MODELS),
        args.budget_s,
    )
    metrics = score_artifact_metrics(path, budget_s=args.budget_s)

    logging.info("=== headline ===")
    for key in _HEADLINE_KEYS:
        logging.info("%s = %s", key, metrics.get(key))
    logging.info("=== all metrics ===")
    for key in sorted(metrics):
        logging.info("%s = %s", key, metrics[key])


if __name__ == "__main__":
    main()
