"""Consolidator: the REDUCE half of the write path, sole writer of the archive.

Every CONSOLIDATE_INTERVAL_S it claims the workers' shard files (:mod:`shards`), drops
proposer refusals / non-rendering templates, dedups by template, Pareto-merges each
survivor into ``archive.jsonl`` (:func:`archive.insert`), writes ``total_score_est`` to
its own status file (never ``score.json``), and deletes the consumed shards.

Alongside that Entry->Pareto-archive path (unchanged, kept for the running pipeline),
:func:`consolidate_submissions_once` claims ``SubmissionRecord`` shards and appends
every one of them to the submission log (:mod:`submission_log`) — no filtering, no
dedup, no Pareto merge: decision C keeps every scored submission.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from jed_attack.campaign import (
    archive,
    compose,
    config,
    prompt_opt,
    shards,
    submission_log,
)

_log = logging.getLogger("consolidator")


def _renderable(entry: archive.Entry) -> bool:
    """True iff the entry's template renders (drops refusals + invariant-breakers)."""
    if "{urls}" in entry.template:
        return prompt_opt.render(entry.template, 0, entry.hops) is not None
    if "{addrs}" in entry.template:
        return prompt_opt.render_deputy(entry.template, 0, entry.hops) is not None
    return False


def consolidate_once(shards_dir: Path, archive_path: Path, status_path: Path) -> int:
    """Claim shards → filter → dedup → Pareto-merge → write status → delete shards.

    Args:
        shards_dir: Where workers drop per-entry shard files.
        archive_path: The canonical Pareto archive (this process is its sole writer).
        status_path: Where to write ``total_score_est`` + counters.

    Returns:
        The number of entries newly inserted into the archive this cycle.
    """
    claimed = shards.claim(shards_dir)
    existing_entries = archive.read(archive_path)
    existing_keys = {
        (e.template, e.hops, tuple(sorted(e.gates.items()))) for e in existing_entries
    }

    by_template: dict[str, archive.Entry] = {}
    for _, entry in claimed:
        if _renderable(entry):
            by_template.setdefault(entry.template, entry)  # dedup: first wins

    inserted = 0
    for entry in by_template.values():
        entry_key = (entry.template, entry.hops, tuple(sorted(entry.gates.items())))
        if entry_key not in existing_keys:
            if archive.insert(entry, archive_path):
                inserted += 1
                existing_keys.add(entry_key)

    for path, _ in claimed:
        path.unlink(missing_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "ts": time.time(),
                "total_score_est": compose.predicted_public_score(archive_path),
                "archive_size": len(archive.read(archive_path)),
                "shards_consumed": len(claimed),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _log.info("consolidated %d shard(s), inserted %d", len(claimed), inserted)
    return inserted


def consolidate_submissions_once(
    shards_dir: Path, log_path: Path, status_path: Path
) -> int:
    """Claim SubmissionRecord shards, append every one to the log, report the best.

    No ``_renderable`` filter, no template dedup, no Pareto merge: decision C keeps
    every scored submission in the log.

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
            consolidate_once(
                config.SHARDS_DIR, config.ARCHIVE_FILE, config.CONSOLIDATOR_STATUS_FILE
            )
        except Exception:  # a bad cycle must never kill the loop
            _log.exception("consolidation cycle failed; retrying next tick")
        if not args.loop:
            break
        time.sleep(config.CONSOLIDATE_INTERVAL_S)


if __name__ == "__main__":
    main()
