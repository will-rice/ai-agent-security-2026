"""Score daemon: keep ``run/score.json`` fresh with the calibrated public-LB prediction.

Replays the single-message ship pool under the public guardrail and caches each
``(message, model)`` result (deterministic under greedy decoding), so after a warmup
only newly-adopted messages replay each cycle. The update loop reads ``score.json``
for the latest number with no on-demand replay. See ``score.predict_public_score``.
"""

import json
import logging

from jed_attack.campaign import assemble, config, score
from jed_attack.campaign.daemon import run_daemon

_log = logging.getLogger("scoredaemon")


def score_once() -> int:
    """Recompute the calibrated public-LB prediction and regenerate the ship pool.

    Scores the single-message ship pool, writes score.json, then re-assembles
    ``build_next/attack.py`` from the SAME ``config.SCORE_CAP`` pool it just
    scored — so the shippable artifact can never drift from score.json's number
    (both come from ``assemble.ship_pool_chains(config.SCORE_CAP)``).

    Returns:
        The number of messages replayed this cycle (0 once the cache is warm).
    """
    result = score.predict_public_score()
    config.SCORE_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Keep the shippable pool in lockstep with the scored pool (same cap, same source).
    assemble.assemble(single_message=True, cap=config.SCORE_CAP)
    _log.info(
        "public_lb=%s candidates=%d new_replays=%d uncached=%d",
        result["public_lb"],
        result["candidates"],
        result["new_replays"],
        result.get("uncached", 0),
    )
    return int(result["new_replays"])


def main() -> None:
    """Entry point for the score daemon."""
    # Short interval: each cycle bounds its own replay budget, so it writes a fresh,
    # climbing score.json every few minutes rather than blocking on a full warmup.
    run_daemon("scoredaemon", score_once, 60.0)


if __name__ == "__main__":
    main()
