"""Parallel island archive set for FunSearch-style islands search.

Wraps N independent :class:`archive.Archive` instances so N proposer workers
can explore in isolation (avoiding premature convergence on a single shared
frontier), while still exposing a global best across all islands for
reporting/shipping. Island 0 is reserved as the "novelty" island (its
insert rule is extended in a later task); for now every island delegates
plainly to its own :meth:`archive.Archive.insert`.
"""

from dataclasses import dataclass
from pathlib import Path

from jed_attack.campaign import archive
from jed_attack.campaign.archive import Archive, Elite, elite_board_density


@dataclass
class IslandSet:
    """N isolated :class:`Archive` instances plus stagnation bookkeeping.

    Attributes:
        archives: One archive per island.
        stall: Per-island count of consecutive inserts that did not improve
            :attr:`best_seen` (bookkeeping for a later migration/reset task).
        best_seen: Per-island best :func:`archive.elite_board_density` ever
            observed (bookkeeping for a later migration/reset task).
    """

    archives: list[Archive]
    stall: list[int]
    best_seen: list[float]

    @classmethod
    def for_count(cls, n: int) -> "IslandSet":
        """Build an ``IslandSet`` of ``n`` empty archives.

        Args:
            n: Number of islands. Island 0 is the novelty island.

        Returns:
            A fresh ``IslandSet`` with n empty archives and zeroed bookkeeping.
        """
        return cls(
            archives=[Archive() for _ in range(n)],
            stall=[0] * n,
            best_seen=[0.0] * n,
        )

    def insert(self, i: int, elite: Elite) -> bool:
        """Insert ``elite`` into island ``i``.

        Args:
            i: Island index.
            elite: The scored shape to insert.

        Returns:
            Whether the elite entered island ``i``'s frontier.
        """
        if i == 0:
            # novelty rule added in Task 2
            return self.archives[0].insert(elite)
        return self.archives[i].insert(elite)

    def local_best_density(self, i: int) -> float:
        """Max :func:`archive.elite_board_density` over island ``i``'s frontier.

        Args:
            i: Island index.

        Returns:
            The best board-density on island ``i``, or 0.0 if its frontier is
            empty.
        """
        frontier = self.archives[i].frontier()
        if not frontier:
            return 0.0
        return max(elite_board_density(e) for e in frontier)

    def global_best_elite(self) -> Elite | None:
        """The single best elite across all islands' frontiers.

        Returns:
            The elite with the highest :func:`archive.elite_board_density`
            across every island's frontier, or ``None`` if all are empty.
        """
        all_elites = [e for a in self.archives for e in a.frontier()]
        if not all_elites:
            return None
        return max(all_elites, key=elite_board_density)

    def best_island(self) -> int:
        """Index of the island with the highest :meth:`local_best_density`.

        Ties break toward the lowest index. Island 0 (novelty) is excluded
        unless it is the only non-empty island.

        Returns:
            The best island's index.
        """
        if len(self.archives) == 1:
            return 0
        non_empty = [i for i, a in enumerate(self.archives) if a.frontier()]
        candidates = (
            range(len(self.archives))
            if non_empty == [0]
            else range(1, len(self.archives))
        )
        return max(candidates, key=self.local_best_density)

    def to_jsonl(self, directory: Path) -> None:
        """Persist each island's frontier to ``directory/island_<i>.jsonl``.

        Args:
            directory: Directory to write ``island_<i>.jsonl`` files into.
        """
        for i, a in enumerate(self.archives):
            a.to_jsonl(directory / f"island_{i}.jsonl")

    @classmethod
    def from_jsonl(cls, directory: Path, n: int) -> "IslandSet":
        """Restore an ``IslandSet`` of ``n`` islands from ``directory``.

        A missing ``island_<i>.jsonl`` file yields an empty archive for that
        island (see :meth:`archive.Archive.from_jsonl`).

        Args:
            directory: Directory containing ``island_<i>.jsonl`` files.
            n: Number of islands to restore.

        Returns:
            The restored ``IslandSet``.
        """
        archives = [
            archive.Archive.from_jsonl(directory / f"island_{i}.jsonl")
            for i in range(n)
        ]
        return cls(archives=archives, stall=[0] * n, best_seen=[0.0] * n)
