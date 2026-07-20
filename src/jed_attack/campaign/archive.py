"""Pareto archive of scored attack messages over the guardrail gate-vector.

A persistent, deduplication-free jsonl of :class:`Entry` records, kept non-dominated
under the ``{optimal, rules, hardened}`` gate vector (guardrails.GATE_GUARDRAILS).
The composer (a later task) draws its submission pool from this archive instead of a
single family incumbent, so it can hedge across whichever guardrail turns out to be
private. A message marked ``pinned`` (the proven exfil template) is never evicted, even
if a later entry dominates it — it is the one candidate with a real scored LB result.

Mirrors the locking/atomic-write pattern in ``prompt_opt._write_best`` /
``record_prompt``: an ``fcntl`` exclusive lock on a sibling ``.lock`` file guards the
read-modify-write, and the rewrite itself is a temp-file + ``os.replace``.
"""

import fcntl
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Entry:
    """One scored attack message and its per-gate severity.

    Attributes:
        template: The message template.
        hops: Tool-call hops the template was scored at.
        gates: Mean severity per guardrail gate, e.g. ``{"optimal": .., "rules": ..,
            "hardened": ..}``.
        cost_s: Wall-clock seconds the scoring took (for cost-aware composing).
        pinned: If True, this entry is never evicted by :func:`insert`, regardless of
            domination.
    """

    template: str
    hops: int
    gates: dict[str, float] = field(default_factory=dict)
    cost_s: float = 0.0
    pinned: bool = False

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for this entry."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Entry":
        """Build an Entry from a parsed JSON dict.

        Args:
            data: A dict as produced by :meth:`to_json`.

        Returns:
            The Entry.
        """
        return cls(
            template=str(data["template"]),
            hops=int(data["hops"]),
            gates=dict(data.get("gates", {})),
            cost_s=float(data.get("cost_s", 0.0)),
            pinned=bool(data.get("pinned", False)),
        )


def dominates(a: Entry, b: Entry) -> bool:
    """Return True iff ``a`` Pareto-dominates ``b`` on the gate vector.

    Args:
        a: Candidate dominator.
        b: Candidate dominated.

    Returns:
        True iff ``a.gates[g] >= b.gates[g]`` for every gate in ``b.gates`` AND
        ``a.gates[g] > b.gates[g]`` for at least one.
    """
    at_least_as_good = all(a.gates.get(g, 0.0) >= v for g, v in b.gates.items())
    strictly_better = any(a.gates.get(g, 0.0) > v for g, v in b.gates.items())
    return at_least_as_good and strictly_better


def read(path: Path) -> list[Entry]:
    """Read every entry from the archive, skipping malformed lines.

    Args:
        path: The archive jsonl file.

    Returns:
        The parsed entries, in file order. Empty if the file does not exist.
    """
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(Entry.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return entries


def insert(entry: Entry, path: Path) -> bool:
    """Insert ``entry`` into the archive if it is not Pareto-dominated.

    Locks the archive (a sibling ``<path>.lock`` file, ``fcntl`` exclusive), reads the
    current entries, and either rejects ``entry`` (an existing entry dominates it) or
    keeps it, dropping every existing NON-pinned entry that ``entry`` dominates. Pinned
    entries are never evicted, even if ``entry`` dominates them. The rewrite is atomic
    (temp file + ``os.replace``).

    Args:
        entry: The candidate entry to add.
        path: The archive jsonl file.

    Returns:
        True if ``entry`` was added, False if it was rejected as dominated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / (path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # released on close (end of the with-block)
        existing = read(path)
        if any(dominates(other, entry) for other in existing):
            return False
        kept = [
            other for other in existing if other.pinned or not dominates(entry, other)
        ]
        kept.append(entry)
        _write_all(kept, path)
    return True


def _write_all(entries: list[Entry], path: Path) -> None:
    """Atomically rewrite the archive with ``entries``.

    Args:
        entries: The full entry list to persist.
        path: The archive jsonl file.
    """
    tmp = path.parent / (path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_json(), sort_keys=True) + "\n")
    tmp.replace(path)  # atomic (os.replace under the hood)
