"""Assemble daemon: rebuild build_next/attack.py from the configured ship artifact.

Sole owner of ``build_next``. Per ``config.SHIP_ARTIFACT`` it writes the composer's
archive-packed submission (default — ``compose.build``, which reserves a public floor
of the pinned exfil template and fills the rest of the green-seconds budget with the
best robust-surviving archive entry; see ``compose.py``), the self-sizing adaptive
submission (``adaptive.build_adaptive``, which probes the live T4 env at run time and
fills only what fits the 9000s/cell budget), or the legacy static single-message pool
the score daemon scores.
"""

import json
import logging

from jed_attack.campaign import adaptive, assemble, compose, config
from jed_attack.campaign.daemon import run_daemon

_log = logging.getLogger("assemble")


def assemble_once() -> None:
    """Rebuild the shippable attack.py per ``config.SHIP_ARTIFACT``.

    ``composed`` (default) writes the archive-packed submission; ``adaptive`` writes
    the self-sizing submission; ``static`` writes the fixed single-message pool
    (``single_message=True, cap=config.SCORE_CAP``).
    """
    if config.SHIP_ARTIFACT == "composed":
        path = compose.build(config.BUILD_NEXT_DIR)
        _log.info("assembled composed submission -> %s", path)
    elif config.SHIP_ARTIFACT == "adaptive":
        path = adaptive.build_adaptive(config.BUILD_NEXT_DIR)
        (config.BUILD_NEXT_DIR / "build_next_status.json").write_text(
            json.dumps({"source": "adaptive", "self_sizing": True}, indent=2),
            encoding="utf-8",
        )
        _log.info("assembled adaptive submission -> %s", path)
    else:
        path = assemble.assemble(single_message=True, cap=config.SCORE_CAP)
        _log.info("assembled static pool (cap=%d) -> %s", config.SCORE_CAP, path)


def main() -> None:
    """Entry point for the assemble daemon."""
    run_daemon("assemble", assemble_once, 120.0)


if __name__ == "__main__":
    main()
