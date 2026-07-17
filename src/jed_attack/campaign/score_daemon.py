"""Score daemon: keep ``run/score.json`` fresh with the calibrated public-LB prediction.

Replays the single-message ship pool under the public guardrail and caches each
``(message, model)`` result (deterministic under greedy decoding), so after a warmup
only newly-adopted messages replay each cycle. The update loop reads ``score.json``
for the latest number with no on-demand replay. See ``score.predict_public_score``.
"""

import json
import logging

from jed_attack.campaign import config, score
from jed_attack.campaign.daemon import run_daemon

_log = logging.getLogger("scoredaemon")


def score_once() -> int:
    """Recompute the calibrated public-LB prediction and write score.json.

    Returns:
        The number of messages replayed this cycle (0 once the cache is warm).
    """
    result = score.predict_public_score()
    config.SCORE_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log.info(
        "public_lb=%s candidates=%d new_replays=%d",
        result["public_lb"],
        result["candidates"],
        result["new_replays"],
    )
    return int(result["new_replays"])


def main() -> None:
    """Entry point for the score daemon."""
    run_daemon("scoredaemon", score_once, 300.0)


if __name__ == "__main__":
    main()
