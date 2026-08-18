"""Verify a submission cut fires 100%, then stamp the gate manifest.

Replays every embedded candidate on its own victim, drops any that don't fire
EXFILTRATION, rewrites the cut to the firing-only pools, and writes the hash-bound
``verified.json`` manifest the submit gate requires. Needs the resident GGUFs (GPU);
replay is deterministic/greedy, so one pass is definitive.

Usage:
    uv run python scripts/verify_cut.py run/submission_cuts/<name>/attack.py
"""

import argparse
import logging
from pathlib import Path

from jed_attack.campaign.verify import verify_and_filter_cut


def main() -> None:
    """Verify + filter a cut and report the per-pool firing/dropped counts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cut", help="path to the cut's attack.py")
    args = parser.parse_args()

    manifest = verify_and_filter_cut(Path(args.cut))
    for var, pool in manifest["pools"].items():
        logging.info(
            "%s -> %-8s %d firing, %d dropped (of %d) = %.1f%%",
            var,
            pool["model"],
            pool["firing"],
            pool["dropped"],
            pool["input"],
            100.0 * pool["firing"] / max(1, pool["input"]),
        )
    logging.info(
        "all_fire=%s  sha256=%s  -> %s",
        manifest["all_fire"],
        manifest["attack_sha256"][:12],
        "cut is submission-ready (gate will pass)",
    )


if __name__ == "__main__":
    main()
