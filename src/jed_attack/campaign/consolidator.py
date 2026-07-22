"""Consolidator: the REDUCE half of the submission write path.

Every CONSOLIDATE_INTERVAL_S it claims the workers' ``SubmissionRecord`` shard files
(:mod:`shards`) and appends every one of them to the submission log
(:mod:`submission_log`) — no filtering, no dedup, no Pareto merge: decision C keeps
every scored submission. It writes the best public/private totals to its status file
and deletes the consumed shards.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from jed_attack.campaign import config, shards, submission_log

_log = logging.getLogger("consolidator")


def consolidate_submissions_once(
    shards_dir: Path, log_path: Path, status_path: Path
) -> int:
    """Claim SubmissionRecord shards, append every one to the log, report the best.

    No filter, no template dedup, no Pareto merge: decision C keeps every scored
    submission in the log.

    Args:
        shards_dir: Where workers drop per-submission shard files.
        log_path: The append-only submission log (:mod:`submission_log`).
        status_path: Where to write ``best_public``/``best_private`` + counters.

    Returns:
        The number of records newly appended to the log this cycle.
    """
    claimed = shards.claim(
        shards_dir, from_json=submission_log.SubmissionRecord.from_json
    )
    for _, record in claimed:
        submission_log.append(record, log_path)
    for path, _ in claimed:
        path.unlink(missing_ok=True)

    incumbent = submission_log.best(log_path)
    status_path.write_text(
        json.dumps(
            {
                "ts": time.time(),
                "best_public": incumbent.public if incumbent else None,
                "best_private": incumbent.private if incumbent else None,
                "log_size": len(submission_log.read(log_path)),
                "shards_consumed": len(claimed),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _log.info("consolidated %d submission shard(s)", len(claimed))
    return len(claimed)


def main() -> None:
    """CLI: one consolidation pass, or ``--loop`` forever at CONSOLIDATE_INTERVAL_S."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true", help="run forever")
    args = parser.parse_args()
    config.ensure_dirs()
    while True:
        try:
            consolidate_submissions_once(
                config.SUBMISSION_SHARDS_DIR,
                config.SUBMISSION_LOG,
                config.CONSOLIDATOR_STATUS_FILE,
            )
        except Exception:  # a bad cycle must never kill the loop
            _log.exception("consolidation cycle failed; retrying next tick")
        if not args.loop:
            break
        time.sleep(config.CONSOLIDATE_INTERVAL_S)


if __name__ == "__main__":
    main()
