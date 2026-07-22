"""Per-entry shard files: the lock-free MAP half of the consolidator write path.

Each worker writes one scored record (:class:`archive.Entry` or
:class:`submission_log.SubmissionRecord`) per file via temp-file + ``os.replace``, so a
shard file is either fully present or absent — no partial reads, no lost writes, no
lock. The consolidator (:mod:`consolidator`) claims and deletes them.
"""

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from jed_attack.campaign import archive


class _JsonRecord(Protocol):
    """Anything shard-writable: it can serialize itself to a JSON-able dict."""

    def to_json(self) -> dict[str, Any]: ...


def write(entry: _JsonRecord, shards_dir: Path, worker_id: str) -> Path:
    """Atomically write one scored record to its own shard file.

    Args:
        entry: The scored candidate (an ``archive.Entry`` or a ``SubmissionRecord``).
        shards_dir: The shards directory (created if missing).
        worker_id: Author id, only for readability of the filename.

    Returns:
        The final shard file path.
    """
    shards_dir.mkdir(parents=True, exist_ok=True)
    final = shards_dir / f"{worker_id}-{uuid.uuid4().hex}.json"
    tmp = final.with_name(
        final.name + ".tmp"
    )  # .tmp never matches the *.json claim glob
    tmp.write_text(json.dumps(entry.to_json(), sort_keys=True), encoding="utf-8")
    tmp.replace(final)  # atomic (os.replace under the hood)
    return final


def claim(
    shards_dir: Path,
    from_json: Callable[[dict[str, Any]], Any] = archive.Entry.from_json,
) -> list[tuple[Path, Any]]:
    """Read every complete shard file; skip malformed/half-written ones.

    Args:
        shards_dir: The shards directory.
        from_json: Reconstructs a record from its parsed JSON dict. Defaults to
            ``archive.Entry.from_json``; pass
            ``submission_log.SubmissionRecord.from_json`` to claim submission shards.

    Returns:
        ``(path, record)`` for each parseable shard, in sorted path order. The caller
        deletes the paths after merging.
    """
    if not shards_dir.exists():
        return []
    out: list[tuple[Path, Any]] = []
    for path in sorted(shards_dir.glob("*.json")):
        try:
            out.append((path, from_json(json.loads(path.read_text()))))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out
