"""Parallel island archive set for FunSearch-style islands search.

Wraps N independent :class:`archive.Archive` instances so N proposer workers
can explore in isolation (avoiding premature convergence on a single shared
frontier), while still exposing a global best across all islands for
reporting/shipping. Island 0 is the "novelty" island: it keeps
structurally-DISTINCT elites over higher-throughput ones (see
:meth:`IslandSet._novelty_insert`); every other island delegates plainly to
its own :meth:`archive.Archive.insert`.
"""

from dataclasses import dataclass
from pathlib import Path

from jed_attack.campaign import archive, config
from jed_attack.campaign.archive import Archive, Elite, elite_board_density


def descriptor(elite: Elite) -> tuple[str, int]:
    """The behavioral descriptor ``(family, bucket)`` used for novelty distance.

    Args:
        elite: The elite to describe.

    Returns:
        Its ``(family, bucket)`` MAP-Elites cell key.
    """
    return (elite.family, elite.bucket)


def novelty(elite: Elite, others: list[Elite]) -> float:
    """Mean distance from ``elite``'s descriptor to its k nearest in ``others``.

    k is ``config.NOVELTY_NEIGHBORS``. Distance between two descriptors is
    family-mismatch (0 same / 1 different) plus
    ``abs(bucket_a - bucket_b)``. An ``elite`` with no neighbors (``others`` empty) is
    maximally novel.

    Args:
        elite: The elite to score.
        others: Candidate neighbor pool (typically an island's frontier).

    Returns:
        The mean distance to the k nearest neighbors, or ``float("inf")`` when
        ``others`` is empty.
    """
    if not others:
        return float("inf")
    family, bucket = descriptor(elite)
    distances = sorted(
        (family != other.family) + abs(bucket - other.bucket) for other in others
    )
    nearest = distances[: config.NOVELTY_NEIGHBORS]
    return sum(nearest) / len(nearest)


def _fires(elite: Elite) -> bool:
    """Whether ``elite`` fires on >=1 model (matches ``archive.model_density``)."""
    return any(
        elite.throughput.get(m, 0.0) > 0.0 and elite.severity.get(m, 0.0) > 0.0
        for m in config.MODELS
    )


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
            return self._novelty_insert(elite)
        return self.archives[i].insert(elite)

    def _novelty_insert(self, elite: Elite) -> bool:
        """Insert ``elite`` into island 0, the novelty island.

        Matches :meth:`archive.Archive.insert`'s firing requirement, but a contested
        elite -- one that shares ``elite``'s ``(family, bucket)`` cell, or that
        Pareto-dominates ``elite`` (so it would otherwise keep ``elite`` off the
        island's :meth:`archive.Archive.frontier` forever) -- is resolved by
        :func:`novelty` against the rest of island 0, not by throughput/severity
        dominance. When ``elite`` is the more novel of the two, every blocker is
        evicted from the underlying archive and ``elite`` is inserted; otherwise
        island 0 is left unchanged.

        Args:
            elite: The scored shape to insert.

        Returns:
            Whether ``elite`` entered island 0's frontier.
        """
        if not _fires(elite):
            return False
        archive0 = self.archives[0]
        frontier = archive0.frontier()
        cell = descriptor(elite)
        blockers = [
            other
            for other in frontier
            if descriptor(other) == cell or archive.dominates(other, elite)
        ]
        if blockers:
            elite_novelty = novelty(elite, frontier)
            blocker_novelty = max(
                novelty(blocker, [o for o in frontier if o is not blocker])
                for blocker in blockers
            )
            if elite_novelty <= blocker_novelty:
                return False
            # ``elite`` wins its own cell outright: purge it wholesale rather than
            # only removing the FRONTIER blockers found above. ``Archive.insert``
            # only prunes a cell of members its newcomer dominates -- it never
            # retroactively prunes a member some OTHER cell's elite later came to
            # dominate, so a stale off-frontier straggler can otherwise survive in
            # ``cell`` and veto the re-insert below even though it already lost the
            # novelty contest (it was never even a candidate; it just squats the cell).
            archive0._cells[cell] = []
            for blocker in blockers:
                if descriptor(blocker) == cell:
                    continue  # already cleared above
                other_cell = archive0._cells.get(descriptor(blocker))
                if other_cell is not None and blocker in other_cell:
                    other_cell.remove(blocker)
        return archive0.insert(elite)

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
