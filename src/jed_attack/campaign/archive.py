"""MAP-Elites + Pareto archive of attack shapes.

The unit is a scored shape with a 4-D vector: per-model throughput AND per-model
severity (one of each per victim model). Cells are keyed by (family, gen_char_bucket);
each cell keeps its Pareto-non-dominated elites, and the globally non-dominated set
(frontier) is the elite pool that ships, ranked by summed per-model board-density.
Selection is Pareto over the raw per-model throughput/severity, never a scalar.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from jed_attack.campaign import config
from jed_attack.campaign.submission_score import board_density


@dataclass(frozen=True)
class Elite:
    """A single scored shape held in the archive.

    Attributes:
        text: The shape's literal message text.
        mtype: The message type ("exfil" or "deputy").
        throughput: Per-model throughput, `{model: 1/(gen_chars+FIXED)}`.
        severity: Per-model predicate severity, `{model: severity}`.
        diagnosis: Free-text scorer/judge diagnosis for this shape.
        family: The MAP-Elites structural family (behavioral descriptor axis 1).
        bucket: The gen_char_bucket cost bin (behavioral descriptor axis 2).
    """

    text: str
    mtype: str
    throughput: dict[str, float]
    severity: dict[str, float]
    diagnosis: str
    family: str
    bucket: int


def dominates(a: Elite, b: Elite) -> bool:
    """True if a Pareto-dominates b over per-model throughput AND severity."""
    comps = [(a.throughput[m], b.throughput[m]) for m in config.MODELS]
    comps += [(a.severity[m], b.severity[m]) for m in config.MODELS]
    ge = all(av >= bv for av, bv in comps)
    gt = any(av > bv for av, bv in comps)
    return ge and gt


class Archive:
    """A behavioral grid of Pareto-non-dominated shape elites."""

    def __init__(self) -> None:
        self._cells: dict[tuple[str, int], list[Elite]] = {}

    def insert(self, elite: Elite) -> bool:
        """Insert a scored elite into its (family, bucket) cell.

        Args:
            elite: The scored shape to insert.

        Returns:
            Whether the elite entered the global frontier.
        """
        cell = self._cells.setdefault((elite.family, elite.bucket), [])
        if any(dominates(x, elite) for x in cell):
            return False
        cell[:] = [x for x in cell if not dominates(elite, x)]
        cell.append(elite)
        return elite in self.frontier()

    def frontier(self) -> list[Elite]:
        """Globally non-dominated elites, across all cells."""
        allx = [x for cell in self._cells.values() for x in cell]
        return [x for x in allx if not any(dominates(y, x) for y in allx if y is not x)]

    def ship_set(self) -> list[Elite]:
        """The frontier ranked by summed per-model board-density, capped at the cap."""

        def density(e: Elite) -> float:
            total = 0.0
            for m in config.MODELS:
                t = e.throughput[m]
                if t <= 0.0:
                    continue
                gen_chars = 1.0 / t - config.FIXED_CHARS[m]
                total += board_density(e.severity[m], gen_chars, m)
            return total

        return sorted(self.frontier(), key=density, reverse=True)[
            : config.ARCHIVE_FRONTIER_CAP
        ]

    def parents(self, k: int) -> list[Elite]:
        """Sample k parents biased to the frontier + under-filled cells.

        Args:
            k: Number of parents to return.

        Returns:
            Up to k elites, preferring the frontier and falling back to elites
            from the least-filled cells.
        """
        front = self.frontier()
        # bias to the frontier; deterministic (no RNG in-loop) — rotate by cell count.
        under = sorted(self._cells.items(), key=lambda kv: len(kv[1]))
        picks = front[:k] or [x for _, cell in under for x in cell][:k]
        return picks

    def to_jsonl(self, path: Path) -> None:
        """Persist the frontier as one JSON object per line."""
        path.write_text(
            "\n".join(json.dumps(asdict(x)) for x in self.frontier()), encoding="utf-8"
        )

    @classmethod
    def from_jsonl(cls, path: Path) -> "Archive":
        """Rebuild an archive by re-inserting every elite from a persisted jsonl."""
        arch = cls()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    data.setdefault("severity", dict.fromkeys(config.MODELS, 0.0))
                    arch.insert(Elite(**data))
        return arch
