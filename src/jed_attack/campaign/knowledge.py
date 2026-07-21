"""Shared cross-agent knowledge log — free-form notes the fleet writes for itself.

Agents (and the prompt optimizer) append free-form insights and dead ends here so each
learns from the others instead of re-deriving them. The proposer feedback digest
(``optimize_prompts._feedback_digest``) reads the ``prompt_opt`` notes back into the
next generation's prompt.

Concurrency is lock-free: each writer appends to its own ``<producer>.jsonl`` (no torn
writes); readers glob and merge. All state lives under ``run/knowledge/``.
"""

import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jed_attack.campaign import config
from jed_attack.campaign.archive import chain_id

_log = logging.getLogger("knowledge")

INSIGHT = "insight"  # default note kind


@dataclass(frozen=True)
class Note:
    """A free-form knowledge note (agent insight)."""

    producer: str
    kind: str
    text: str
    chain_id: str = ""
    ts: float = 0.0


def note(
    producer: str,
    text: str,
    *,
    kind: str = INSIGHT,
    chain: Sequence[str] = (),
    notes_dir: Path | None = None,
) -> None:
    """Append a free-form knowledge note for the rest of the fleet.

    Args:
        producer: Author id (e.g. ``agent-2`` or ``prompt_opt``).
        text: The lesson, in one or two sentences.
        kind: The note kind (defaults to ``INSIGHT``).
        chain: Optional related chain (records its id for reference).
        notes_dir: Override for the notes dir (defaults to config).
    """
    entry = Note(
        producer=producer,
        kind=kind,
        text=text,
        chain_id=chain_id(chain) if chain else "",
        ts=time.time(),
    )
    _append(asdict(entry), (notes_dir or config.NOTES_DIR) / f"{producer}.jsonl")


def read_notes(notes_dir: Path | None = None) -> list[Note]:
    """Read every note, newest first.

    Args:
        notes_dir: Override for the notes dir (defaults to config).

    Returns:
        All notes sorted by timestamp, newest first.
    """
    notes = [
        Note(
            producer=str(line.get("producer", "")),
            kind=str(line.get("kind", INSIGHT)),
            text=str(line.get("text", "")),
            chain_id=str(line.get("chain_id", "")),
            ts=float(line.get("ts", 0.0)),
        )
        for line in _read_dir(notes_dir or config.NOTES_DIR)
    ]
    notes.sort(key=lambda n: n.ts, reverse=True)
    return notes


def _append(record: dict, path: Path) -> None:
    """Append one json record as a line to a per-writer file.

    Args:
        record: The json-serializable record.
        path: The ``<producer>.jsonl`` file to append to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_dir(directory: Path) -> Iterable[dict]:
    """Yield parsed json objects from every ``*.jsonl`` in a dir, skipping bad lines.

    Args:
        directory: The notes dir.

    Yields:
        Parsed dicts, oldest file first.
    """
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
