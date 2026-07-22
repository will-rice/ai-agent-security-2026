"""Append-only log of every evaluated whole-submission with its score and feedback.

Kept memory (decision C): nothing is ever pruned. The global best is simply the
max-scoring record over the log, so the loop never needs to re-derive it from shard
state.

Writes are concurrency-safe: an ``fcntl`` exclusive lock on a sibling ``.lock`` file
guards the read-modify-write, and the rewrite itself is a temp-file + ``os.replace``.
"""

import fcntl
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubmissionRecord:
    """One evaluated whole-submission: its messages, score, and per-message feedback.

    Attributes:
        messages: The submitted message sequence.
        public: Public-LB score for this submission.
        private: Private-LB score for this submission.
        feedback: Per-message feedback dicts from the scoring loop (opaque here).
        ts: Caller-supplied timestamp (epoch seconds) this record was logged at.
    """

    messages: list[str]
    public: float
    private: float
    feedback: list[dict]
    ts: float

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for this record."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SubmissionRecord":
        """Build a SubmissionRecord from a parsed JSON dict.

        Args:
            data: A dict as produced by :meth:`to_json`.

        Returns:
            The SubmissionRecord.
        """
        return cls(
            messages=list(data["messages"]),
            public=float(data["public"]),
            private=float(data["private"]),
            feedback=list(data["feedback"]),
            ts=float(data["ts"]),
        )


def read(path: Path) -> list[SubmissionRecord]:
    """Read every record from the submission log, skipping malformed lines.

    Args:
        path: The submission log jsonl file.

    Returns:
        The parsed records, in file order. Empty if the file does not exist.
    """
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(SubmissionRecord.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return records


def append(record: SubmissionRecord, path: Path) -> None:
    """Append ``record`` to the submission log. Nothing is ever pruned (decision C).

    Locks the log (a sibling ``<path>.lock`` file, ``fcntl`` exclusive), reads the
    current records, appends ``record``, and rewrites atomically (temp file +
    ``os.replace``).

    Args:
        record: The submission record to append.
        path: The submission log jsonl file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / (path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # released on close (end of the with-block)
        records = read(path)
        records.append(record)
        _write_all(records, path)


def best(path: Path) -> SubmissionRecord | None:
    """Return the record with the highest ``public`` score (the shipped board).

    Args:
        path: The submission log jsonl file.

    Returns:
        The best record, ties broken by higher ``private``, or None if the log is
        empty or absent.
    """
    records = read(path)
    if not records:
        return None
    return max(records, key=lambda r: (r.public, r.private))


def _write_all(records: list[SubmissionRecord], path: Path) -> None:
    """Atomically rewrite the submission log with ``records``.

    Args:
        records: The full record list to persist.
        path: The submission log jsonl file.
    """
    tmp = path.parent / (path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
    tmp.replace(path)  # atomic (os.replace under the hood)
