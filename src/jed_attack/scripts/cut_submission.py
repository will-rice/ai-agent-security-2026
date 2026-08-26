"""Cut a competition submission from the current blackboard champion.

Regenerates ``attack.py`` from ``best_objective`` and injects it into the shipped
notebook's ``%%writefile /kaggle/working/attack.py`` cell -- the deterministic,
error-prone build step. This is pure fill (no model / no GPU), so it is safe to run
while the optimizer holds both GGUFs resident.

Optional ``--push`` runs ``kaggle kernels push`` (needs Kaggle creds in the env). The
push is gated behind the flag because it is an externally visible action that can spend
a competition slot on rerun.

Usage:
    uv run python -m jed_attack.scripts.cut_submission          # build + inject only
    uv run python -m jed_attack.scripts.cut_submission --push   # ...and push the kernel
"""

import argparse
import json
import logging
import py_compile
import subprocess
import sys

from jed_attack.campaign import config
from jed_attack.campaign.blackboard import (
    Blackboard,
    Record,
    _champion_map,
    _ship_pools,
)

KERNEL_DIR = config.CAMPAIGN_ROOT / "submission_kernel" / "jed-attack-submission"
NOTEBOOK = KERNEL_DIR / "jed-attack-submission.ipynb"
WRITEFILE_MARKER = "%%writefile /kaggle/working/attack.py"


def main() -> None:
    """Build the champion notebook and, with ``--push``, push the kernel."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("cut_submission")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--push", action="store_true", help="kaggle kernels push after building"
    )
    parser.add_argument(
        "--diversity-band",
        type=float,
        default=0.1,
        help=(
            "Ship the champion with the MOST distinct shapes whose objective is within "
            "this fractional band of the best (e.g. 0.1 = within 10%%). Default 0.1: "
            "maximize distinct shapes for robustness -- diversity never hurts the "
            "public board (throughput is shape-count-insensitive) and hedges the "
            "stricter private re-scoring. Pass 0.0 to ship the strict-objective "
            "champion instead."
        ),
    )
    args = parser.parse_args()

    champion = build_and_inject(log, args.diversity_band)
    write_cut_record(champion)

    if args.push:
        log.info("pushing kernel: %s", KERNEL_DIR)
        subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)], check=True)
        log.info("pushed -- poll with: kaggle kernels status <slug>")


def build_and_inject(log: logging.Logger, diversity_band: float = 0.0) -> Record:
    """Dump the champion to attack.py and inject it into the notebook cell.

    Args:
        log: Logger for the build summary.
        diversity_band: Fractional objective band for shape-diversity selection (see
            :meth:`Blackboard.best_diverse`); 0.0 ships the strict-objective champion.
    """
    blackboard = Blackboard.load(config.BLACKBOARD_LOG)
    champion = (
        blackboard.best_diverse(diversity_band)
        if diversity_band > 0.0
        else blackboard.best_objective()
    )
    if champion is None:
        sys.exit("no champion in blackboard -- nothing to ship")

    # Route through the SAME per-model router the live Blackboard reship path uses
    # (_champion_map + _ship_pools -> assemble.build_permodel), so this script's
    # --push cut can never disagree with what the optimizer would ship: each pool
    # goes only to its own victim, not the flat both-pools-to-both-victims cut.
    pools = _champion_map(champion)
    n_candidates = sum(len(chains) for chains in pools.values())
    attack_path = _ship_pools(pools, config.BUILD_NEXT_DIR)
    py_compile.compile(str(attack_path), doraise=True)
    attack_src = attack_path.read_text(encoding="utf-8")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = _writefile_cell(notebook)
    cell["source"] = f"{WRITEFILE_MARKER}\n{attack_src}"
    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    log.info(
        "champion objective=%.3f fires=%s shapes=%d -> %d candidates",
        champion.objective,
        champion.fires,
        len(champion.messages),
        n_candidates,
    )
    log.info("attack.py: %d bytes | notebook updated: %s", len(attack_src), NOTEBOOK)
    return champion


def _writefile_cell(notebook: dict) -> dict:
    """Return the code cell that writes attack.py, raising if it is missing."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(WRITEFILE_MARKER):
            return cell
    sys.exit(f"no `{WRITEFILE_MARKER}` cell in {NOTEBOOK}")


def write_cut_record(champion: Record) -> None:
    """Persist a small provenance record of what was shipped."""
    config.SUBMISSION_CUTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "objective": champion.objective,
        "public": champion.public,
        "fires": champion.fires,
        "n_shapes": len(champion.messages),
        "shapes": [str(m["text"]) for m in champion.messages],
    }
    (config.SUBMISSION_CUTS_DIR / "last_cut.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
