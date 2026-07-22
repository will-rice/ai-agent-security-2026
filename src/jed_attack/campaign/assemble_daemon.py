"""Assemble daemon: rebuild build_next/attack.py from the global-best submission.

Sole owner of ``build_next``. Runs ``assemble.build`` on the messages of
``submission_log.best`` — the highest-scoring whole-submission the incumbent loop has
evaluated so far (see ``submission_log.py``).
"""

import logging

from jed_attack.campaign import assemble, config, submission_log
from jed_attack.campaign.daemon import run_daemon

_log = logging.getLogger("assemble")


def assemble_once() -> None:
    """Rebuild the shippable attack.py from the global-best logged submission."""
    rec = submission_log.best(config.SUBMISSION_LOG)
    if rec is None:
        _log.info("submission log is empty; nothing to assemble yet")
        return
    texts = [message["text"] for message in rec.messages]
    path = assemble.build(texts, config.BUILD_NEXT_DIR)
    _log.info("assembled authored submission -> %s", path)


def main() -> None:
    """Entry point for the assemble daemon."""
    run_daemon("assemble", assemble_once, 120.0)


if __name__ == "__main__":
    main()
