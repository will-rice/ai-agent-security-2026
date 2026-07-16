"""Fast deterministic-agent evaluation for quick iteration."""

from __future__ import annotations

import argparse
import logging

from jed_attack.harness.runner import run_attack
from jed_attack.submission.attack import AttackAlgorithm

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("jed_attack.smoke")


def main() -> None:
    """Run the submission against the deterministic agent and log its score."""
    parser = argparse.ArgumentParser(description="JED attack smoke test")
    parser.add_argument("--budget-s", type=float, default=30.0)
    args = parser.parse_args()

    result = run_attack(AttackAlgorithm, agent="deterministic", budget_s=args.budget_s)
    _log.info(
        "score=%.3f raw=%.1f findings=%d unique_cells=%d time=%.1fs guardrail=%s",
        result.score,
        result.score_raw,
        result.findings_count,
        result.unique_cells,
        result.time_taken,
        result.guardrail_id,
    )


if __name__ == "__main__":
    main()
