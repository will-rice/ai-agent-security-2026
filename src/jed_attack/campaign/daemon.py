"""Shared loop scaffolding for campaign daemons."""

import argparse
import logging
import time
from collections.abc import Callable


def run_daemon(name: str, once: Callable[[], object], default_interval: float) -> None:
    """Run a daemon's ``once`` step, once or on a ``--loop`` interval.

    Args:
        name: Daemon name (for logging + arg parsing).
        once: The per-iteration callable (exceptions are logged, not fatal).
        default_interval: Default seconds between iterations in loop mode.
    """
    parser = argparse.ArgumentParser(description=f"{name} daemon")
    parser.add_argument("--loop", action="store_true", help="run continuously")
    parser.add_argument("--interval", type=float, default=default_interval)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {name} %(message)s")
    log = logging.getLogger(name)
    while True:
        try:
            once()
        except Exception:
            log.exception("iteration failed")
        if not args.loop:
            break
        time.sleep(args.interval)
