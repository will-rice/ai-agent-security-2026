"""Score-watch daemon: aggregate producer counts + adopted pool to the leaderboard."""

import json
import logging
import time

from jed_attack.campaign import config, gate, store
from jed_attack.campaign.daemon import run_daemon

_log = logging.getLogger("scorewatch")


def watch_once() -> None:
    """Append a leaderboard snapshot: per-producer candidate counts + adopted totals."""
    candidates = store.read_all()
    adopted = gate.read_adopted()
    by_producer: dict[str, int] = {}
    for cand in candidates:
        by_producer[cand.producer] = by_producer.get(cand.producer, 0) + 1
    snapshot = {
        "t": time.time(),
        "candidates": len(candidates),
        "adopted": len(adopted),
        "total_robust_severity": sum(v.robust_severity for v in adopted),
        "by_producer": by_producer,
    }
    config.LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    with config.LEADERBOARD.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot) + "\n")
    _log.info(
        "candidates=%d adopted=%d total_robust_sev=%d by_producer=%s",
        snapshot["candidates"],
        snapshot["adopted"],
        snapshot["total_robust_severity"],
        by_producer,
    )


def main() -> None:
    """Entry point for the score-watch daemon."""
    run_daemon("scorewatch", watch_once, 120.0)


if __name__ == "__main__":
    main()
