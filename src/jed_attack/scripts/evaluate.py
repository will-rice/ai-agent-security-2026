"""Evaluate the submission against a real GGUF model on a GPU host."""

import argparse
import logging
from pathlib import Path

from jed_attack.harness.models import download_gguf, gguf_agent_factory
from jed_attack.harness.runner import run_attack
from jed_attack.submission.attack import AttackAlgorithm

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("jed_attack.evaluate")


def main() -> None:
    """Download the requested GGUF and evaluate the submission against it."""
    parser = argparse.ArgumentParser(description="JED attack real-model evaluation")
    parser.add_argument("model", choices=["gpt_oss", "gemma_4"])
    parser.add_argument("--budget-s", type=float, default=1800.0)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    model_path = download_gguf(args.model, args.models_dir)
    _log.info("model ready: %s", model_path)
    factory = gguf_agent_factory(args.model, model_path)

    result = run_attack(
        AttackAlgorithm,
        agent_factory=factory,
        agent_label=args.model,
        budget_s=args.budget_s,
    )
    _log.info(
        "[%s] score=%.3f raw=%.1f findings=%d unique_cells=%d time=%.1fs",
        args.model,
        result.score,
        result.score_raw,
        result.findings_count,
        result.unique_cells,
        result.time_taken,
    )


if __name__ == "__main__":
    main()
