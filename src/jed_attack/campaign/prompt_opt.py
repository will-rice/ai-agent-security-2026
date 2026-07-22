"""Per-worker identity for the submission shard write path (:mod:`shards`)."""

import os


def _worker_id() -> str:
    """This worker's shard-file author id: ``JED_WORKER_ID`` env, else the pid."""
    return os.getenv("JED_WORKER_ID") or str(os.getpid())
