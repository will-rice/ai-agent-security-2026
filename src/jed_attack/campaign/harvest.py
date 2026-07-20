"""Harvest daemon: collect deduped producer candidates into the harvest file."""

import json
import logging

from jed_attack.campaign import config, store
from jed_attack.campaign.daemon import run_daemon

_log = logging.getLogger("harvest")


def harvest_once() -> int:
    """Read all producer candidate files, dedupe, and write the harvest file.

    Returns:
        The number of unique candidates collected.
    """
    candidates = store.read_all()
    config.HARVEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.HARVEST_FILE.open("w", encoding="utf-8") as handle:
        for cand in candidates:
            handle.write(
                json.dumps(
                    {
                        "chain": list(cand.chain),
                        "producer": cand.producer,
                        "chain_id": cand.chain_id,
                    }
                )
                + "\n"
            )
    _log.info("collected %d candidates", len(candidates))
    return len(candidates)


def main() -> None:
    """Entry point for the harvest daemon."""
    run_daemon("harvest", harvest_once, 60.0)


if __name__ == "__main__":
    main()
