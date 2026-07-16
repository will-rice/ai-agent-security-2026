"""Candidate store — producers emit firing attack chains; harvest reads them.

A Candidate is one attack chain (user-message sequence) plus provenance.
Producers append their firing chains to ``run/candidates/<producer>.jsonl``;
the harvester reads all of them deduped by ``chain_id``. The gate then
validates each unique chain (see gate.py).
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jed_attack.campaign import config


def chain_id(chain: Sequence[str]) -> str:
    """Return a stable dedup id for a message chain.

    Args:
        chain: The user-message sequence.

    Returns:
        First 16 hex chars of the sha256 of the joined chain.
    """
    return hashlib.sha256(" ".join(chain).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Candidate:
    """A discovered attack chain with provenance."""

    chain: tuple[str, ...]
    producer: str
    predicates: tuple[str, ...] = ()
    self_scores: dict[str, float] = field(default_factory=dict)
    chain_id: str = ""

    @classmethod
    def make(
        cls,
        chain: Sequence[str],
        producer: str,
        *,
        predicates: Sequence[str] = (),
        self_scores: Mapping[str, float] | None = None,
    ) -> "Candidate":
        """Build a Candidate with a computed chain_id.

        Args:
            chain: The user-message sequence.
            producer: Producer id (e.g. ``agent-2``).
            predicates: Predicates the chain fired at discovery.
            self_scores: Producer-measured scores per model.

        Returns:
            The Candidate.
        """
        return cls(
            chain=tuple(chain),
            producer=producer,
            predicates=tuple(predicates),
            self_scores=dict(self_scores or {}),
            chain_id=chain_id(chain),
        )


def emit(candidate: Candidate, root: Path | None = None) -> Path:
    """Append a candidate to its producer's jsonl file.

    Args:
        candidate: The candidate to record.
        root: Override for the candidates dir (defaults to config).

    Returns:
        The producer file path written.
    """
    candidates_dir = root or config.CANDIDATES_DIR
    candidates_dir.mkdir(parents=True, exist_ok=True)
    path = candidates_dir / f"{candidate.producer}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(candidate), sort_keys=True) + "\n")
    return path


def read_all(root: Path | None = None) -> list[Candidate]:
    """Read every producer file, deduped by chain_id (first occurrence kept).

    Args:
        root: Override for the candidates dir (defaults to config).

    Returns:
        The deduped list of candidates.
    """
    candidates_dir = root or config.CANDIDATES_DIR
    seen: dict[str, Candidate] = {}
    if not candidates_dir.exists():
        return []
    for path in sorted(candidates_dir.glob("*.jsonl")):
        for line in _read_lines(path):
            cand = Candidate(
                chain=tuple(line.get("chain", ())),
                producer=str(line.get("producer", path.stem)),
                predicates=tuple(line.get("predicates", ())),
                self_scores=dict(line.get("self_scores", {})),
                chain_id=str(line.get("chain_id") or chain_id(line.get("chain", ()))),
            )
            if cand.chain and cand.chain_id not in seen:
                seen[cand.chain_id] = cand
    return list(seen.values())


def _read_lines(path: Path) -> Iterable[dict]:
    """Yield parsed JSON objects from a jsonl file, skipping bad lines.

    Args:
        path: The jsonl file.

    Yields:
        Parsed dicts.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
