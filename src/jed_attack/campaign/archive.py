"""Pareto archive of scored attack messages over the guardrail gate-vector.

A persistent, deduplication-free jsonl of :class:`Entry` records, kept non-dominated
over the ``{optimal, rules, hardened}`` gates (maximized) AND ``hops`` (minimized),
so the hop-efficient templates the composer needs to maximize the total submission
survive the frontier. (``cost_s`` is stored as a contention-polluted monitoring signal
only — the hop budget, not wall-time, is what the composer packs against.)
The composer draws its submission pool from this archive instead of a single family
incumbent, so it can hedge across whichever guardrail turns out to be private. Every
entry is discovered by the seedless search; the archive keeps only non-dominated ones.

Writes are concurrency-safe: an ``fcntl`` exclusive lock on a sibling ``.lock`` file
guards the read-modify-write, and the rewrite itself is a temp-file + ``os.replace``.
"""

import fcntl
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def chain_id(chain: Sequence[str]) -> str:
    """Return a stable dedup id for a message chain.

    Args:
        chain: The user-message sequence.

    Returns:
        First 16 hex chars of the sha256 of the joined chain.
    """
    return hashlib.sha256(" ".join(chain).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Entry:
    """One scored attack message and its per-gate severity.

    Attributes:
        template: The message template.
        hops: Tool-call hops the template was scored at.
        gates: Mean severity per guardrail gate, e.g. ``{"optimal": .., "rules": ..,
            "hardened": ..}``.
        cost_s: Wall-clock seconds the scoring took (for cost-aware composing).
    """

    template: str
    hops: int
    gates: dict[str, float] = field(default_factory=dict)
    cost_s: float = 0.0

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
        )


def dominates(a: Entry, b: Entry) -> bool:
    """Return True iff ``a`` Pareto-dominates ``b`` over the gate vector AND hops.

    ``hops`` is a MINIMIZED objective alongside the (maximized) gates: the composer
    maximizes the total submission by packing marginal value per tool-hop
    (``(gate + novelty) / hops``), so a lower-hop template with equal gates has strictly
    higher value and MUST survive the frontier. Dropping hops from it would let a
    higher-hop, marginally-higher-gate entry evict the hop-efficient one that actually
    maximizes the packed total. (Hops is deterministic and green==T4, unlike ``cost_s``,
    which is a contention-polluted monitoring signal only.)

    Args:
        a: Candidate dominator.
        b: Candidate dominated.

    Returns:
        True iff ``a.gates[g] >= b.gates[g]`` for every gate in ``b.gates`` AND
        ``a.hops <= b.hops``, with at least one of those inequalities strict.
    """
    gates_ge = all(a.gates.get(g, 0.0) >= v for g, v in b.gates.items())
    hops_le = a.hops <= b.hops
    if not (gates_ge and hops_le):
        return False
    gates_gt = any(a.gates.get(g, 0.0) > v for g, v in b.gates.items())
    return gates_gt or a.hops < b.hops


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
    """Insert ``entry`` into the archive unless it is Pareto-dominated.

    Locks the archive (a sibling ``<path>.lock`` file, ``fcntl`` exclusive), reads the
    current entries, and either rejects ``entry`` (an existing entry dominates it) or
    keeps it, dropping every existing entry that ``entry`` dominates. The rewrite is
    atomic (temp file + ``os.replace``).

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
        kept = [other for other in existing if not dominates(entry, other)]
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
