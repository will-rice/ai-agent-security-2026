"""The team's shared blackboard: append-only JSONL + derived in-memory views.

One async process owns it, so appends are serialized with an ``asyncio.Lock`` (no fcntl).
Every scored submission is one JSONL line; the in-memory views (best submission, best
individual messages per shape, recent cross-model reasoning) are rebuilt on load and
updated on append. When an append sets a new public best, the shipped ``attack.py`` is
rewritten immediately via :func:`assemble.build`, so the artifact never lags the best.
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jed_attack.campaign import assemble
from jed_attack.campaign.submission import MessageType


@dataclass(frozen=True)
class Record:
    """One scored submission on the blackboard."""

    messages: list[dict]
    public: float
    feedback: list[dict]
    reasoning: str
    model: str
    worker: int
    ts: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Record":
        return cls(
            messages=list(data["messages"]),
            public=float(data["public"]),
            feedback=list(data["feedback"]),
            reasoning=str(data.get("reasoning", "")),
            model=str(data.get("model", "")),
            worker=int(data.get("worker", 0)),
            ts=float(data["ts"]),
        )


def _severity_sum(entry: dict) -> float:
    """Total severity of one feedback entry across guardrails."""
    return float(sum(entry.get("severity", {}).values()))


class Blackboard:
    """In-memory team memory backed by an append-only JSONL file."""

    def __init__(self, path: Path, records: list[Record]) -> None:
        self._path = path
        self._records = records
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, path: Path) -> "Blackboard":
        """Warm-start: replay the JSONL into memory (skips malformed lines)."""
        records: list[Record] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(Record.from_json(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return cls(path, records)

    def best(self) -> Record | None:
        """The highest-``public`` record, or ``None`` if empty."""
        if not self._records:
            return None
        return max(self._records, key=lambda r: r.public)

    def top_messages(self, mtype: MessageType, k: int) -> list[tuple[str, str, float]]:
        """Best-scoring individual messages of a shape: ``(text, model, severity)``.

        Ranked by severity-sum, deduped by text, best first. This is the cross-model
        material a worker on one model learns from another's wins.
        """
        best_by_text: dict[str, tuple[str, float]] = {}
        for record in self._records:
            for entry in record.feedback:
                if entry.get("type") != mtype.value:
                    continue
                sev = _severity_sum(entry)
                text = entry.get("message", "")
                if sev <= 0 or not text:
                    continue
                if text not in best_by_text or sev > best_by_text[text][1]:
                    best_by_text[text] = (record.model, sev)
        ranked = sorted(
            ((t, m, s) for t, (m, s) in best_by_text.items()),
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:k]

    def recent_reasoning(self, k: int, chars: int = 800) -> list[tuple[str, str]]:
        """The most recent non-empty reasoning blobs: ``(model, excerpt)`` (bounded)."""
        out: list[tuple[str, str]] = []
        for record in reversed(self._records):
            if record.reasoning:
                out.append((record.model, record.reasoning[:chars]))
            if len(out) >= k:
                break
        return out

    async def append(self, record: Record, out_dir: Path) -> None:
        """Append a record, persist it, and reship ``attack.py`` if it is a new best."""
        async with self._lock:
            prior_best = self.best()
            self._records.append(record)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
            if prior_best is None or record.public > prior_best.public:
                assemble.build([m["text"] for m in record.messages], out_dir)
