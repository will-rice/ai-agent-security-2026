"""Run the held-out RulesGuardrail proxy study (requires the local victim GPUs)."""

import argparse
from pathlib import Path

from jed_attack.campaign import config
from jed_attack.campaign.judge_study import run_study


def main() -> None:
    """Parse bounded operational arguments and write the study artifacts."""
    parser = argparse.ArgumentParser(
        description="Held-out robustness judge proxy study"
    )
    parser.add_argument("--blackboard", type=Path, default=config.BLACKBOARD_LOG)
    parser.add_argument("--output-dir", type=Path, default=config.JUDGE_STUDY_DIR)
    parser.add_argument("--n", type=int, default=config.JUDGE_STUDY_N)
    parser.add_argument("--heldout-fraction", type=float, default=0.3)
    args = parser.parse_args()
    result = run_study(
        blackboard_path=args.blackboard,
        output_dir=args.output_dir,
        n=args.n,
        heldout_fraction=args.heldout_fraction,
    )
    print(f"ready={result.report.ready} pairs={result.report.heldout_pairs}")


if __name__ == "__main__":
    main()
