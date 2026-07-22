"""Per-entry shard files: the lock-free MAP half of the consolidator write path.

Each worker writes one scored :class:`archive.Entry` per file via temp-file +
``os.replace``, so a shard file is either fully present or absent — no partial reads, no
lost writes, no lock. The consolidator (:mod:`consolidator`) claims and deletes them.
"""

import json
import uuid
from pathlib import Path

from jed_attack.campaign import archive


def write(entry: archive.Entry, shards_dir: Path, worker_id: str) -> Path:
    """Atomically write one scored entry to its own shard file.

    Args:
        entry: The scored candidate.
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


def claim(shards_dir: Path) -> list[tuple[Path, archive.Entry]]:
    """Read every complete shard file; skip malformed/half-written ones.

    Args:
        shards_dir: The shards directory.

    Returns:
        ``(path, entry)`` for each parseable shard, in sorted path order. The caller
        deletes the paths after merging.
    """
    if not shards_dir.exists():
        return []
    out: list[tuple[Path, archive.Entry]] = []
    for path in sorted(shards_dir.glob("*.json")):
        try:
            out.append((path, archive.Entry.from_json(json.loads(path.read_text()))))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out
