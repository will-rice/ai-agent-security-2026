"""Assemble daemon: rebuild build_next/attack.py from the gate ledger's adopted pool.

Sole owner of ``build_next``. Builds the SAME single-message pool the score daemon
scores (``single_message=True, cap=config.SCORE_CAP``), so the shipped artifact never
drifts from ``score.json`` — the stale multi-message pool that this used to emit (bare
``assemble()`` defaults) was the drift that shipped a wrong pool before.
"""

import logging

from jed_attack.campaign import assemble, config
from jed_attack.campaign.daemon import run_daemon

_log = logging.getLogger("assemble")


def assemble_once() -> None:
    """Rebuild the shippable attack.py (single-message pool) from adopted candidates."""
    path = assemble.assemble(single_message=True, cap=config.SCORE_CAP)
    _log.info("assembled single-message pool (cap=%d) -> %s", config.SCORE_CAP, path)


def main() -> None:
    """Entry point for the assemble daemon."""
    run_daemon("assemble", assemble_once, 120.0)


if __name__ == "__main__":
    main()
