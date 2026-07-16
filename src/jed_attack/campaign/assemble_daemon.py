"""Assemble daemon: rebuild build_next/attack.py from the gate ledger's adopted pool."""

import logging

from jed_attack.campaign import assemble
from jed_attack.campaign.daemon import run_daemon

_log = logging.getLogger("assemble")


def assemble_once() -> None:
    """Rebuild the submission attack.py from the adopted candidates."""
    path = assemble.assemble()
    _log.info("assembled -> %s", path)


def main() -> None:
    """Entry point for the assemble daemon."""
    run_daemon("assemble", assemble_once, 120.0)


if __name__ == "__main__":
    main()
