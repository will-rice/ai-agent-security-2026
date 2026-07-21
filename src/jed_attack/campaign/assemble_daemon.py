"""Assemble daemon: rebuild build_next/attack.py from the composer's ship pool.

Sole owner of ``build_next``. Runs ``compose.build``, which reserves a public floor of
the pinned exfil template and fills the rest of the green-seconds budget with the best
robust-surviving archive entries (see ``compose.py``).
"""

import logging

from jed_attack.campaign import compose, config
from jed_attack.campaign.daemon import run_daemon

_log = logging.getLogger("assemble")


def assemble_once() -> None:
    """Rebuild the shippable attack.py from the composer's archive-packed pool."""
    path = compose.build(config.BUILD_NEXT_DIR)
    _log.info("assembled composed submission -> %s", path)


def main() -> None:
    """Entry point for the assemble daemon."""
    run_daemon("assemble", assemble_once, 120.0)


if __name__ == "__main__":
    main()
