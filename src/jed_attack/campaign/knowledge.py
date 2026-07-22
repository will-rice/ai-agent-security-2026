"""Shared cross-agent knowledge log — free-form notes the fleet writes for itself.

Agents (and the submission optimizer) append free-form insights and dead ends here so
each learns from the others instead of re-deriving them.

Concurrency is lock-free: each writer appends to its own ``<producer>.jsonl`` (no torn
writes); readers glob and merge. All state lives under ``run/knowledge/``.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from jed_attack.campaign import config

INSIGHT = "insight"  # default note kind


@dataclass(frozen=True)
class Note:
    """A free-form knowledge note (agent insight)."""

    producer: str
    kind: str
    text: str
    ts: float = 0.0


def note(
    producer: str,
    text: str,
    *,
    kind: str = INSIGHT,
    notes_dir: Path | None = None,
) -> None:
    """Append a free-form knowledge note for the rest of the fleet.

    Args:
        producer: Author id (e.g. ``agent-2`` or ``prompt_opt``).
        text: The lesson, in one or two sentences.
        kind: The note kind (defaults to ``INSIGHT``).
        notes_dir: Override for the notes dir (defaults to config).
    """
    entry = Note(producer=producer, kind=kind, text=text, ts=time.time())
    _append(asdict(entry), (notes_dir or config.NOTES_DIR) / f"{producer}.jsonl")


def _append(record: dict, path: Path) -> None:
    """Append one json record as a line to a per-writer file.

    Args:
        record: The json-serializable record.
        path: The ``<producer>.jsonl`` file to append to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
