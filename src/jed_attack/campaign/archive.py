"""MAP-Elites + Pareto archive of attack shapes.

The unit is a scored shape with a 4-D vector: per-model throughput AND per-model
severity (one of each per victim model). Cells are keyed by (family, gen_char_bucket);
each cell keeps its Pareto-non-dominated elites, and the globally non-dominated set
(frontier) is the elite pool that ships, ranked by summed per-model board-density.
Selection is Pareto over the raw per-model throughput/severity, never a scalar.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path

from jed_attack.campaign import config


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
        url_scheme: EXFIL URL template ('{h}' host slot) the shape's shipped candidates
            post to; default reproduces the historical ``http://<host>.co``.
        turns: Per-model agent turn count for this shape's replay, ``{model: turns}``.
            Diagnostic only (NOT in the objective): persisted so we can measure whether
            firing turns are a constant 2 or vary (and thus whether turns belongs in the
            cost model). Empty for elites persisted before this field existed.
        input_chars: The shape's input message length. A hedge on the real grader's
            prefill cost (local replay is input-length-insensitive, but the grader's
            model server may charge for it). A ship-set tiebreaker, not a board-density
            term. 0 for elites persisted before this field existed.
    """

    text: str
    mtype: str
    throughput: dict[str, float]
    severity: dict[str, float]
    diagnosis: str
    family: str
    bucket: int
    url_scheme: str = "http://{h}.co"
    turns: dict[str, float] = field(default_factory=dict)
    input_chars: int = 0


def dominates(a: Elite, b: Elite) -> bool:
    """True if a Pareto-dominates b over per-model throughput AND severity."""
    comps = [(a.throughput[m], b.throughput[m]) for m in config.MODELS]
    comps += [(a.severity[m], b.severity[m]) for m in config.MODELS]
    ge = all(av >= bv for av, bv in comps)
    gt = any(av > bv for av, bv in comps)
    return ge and gt


def elite_board_density(elite: Elite) -> float:
    """Summed per-model board-density for ``elite`` -- the shared ship/rank key.

    ``board_density = (sev + NOVELTY_PER_CELL)/200 / (input_cost + gen_tokens + FIXED)``
    and ``throughput = 1/(input_cost + gen_tokens + FIXED)`` share the same denominator,
    so ``board_density = (sev + NOVELTY_PER_CELL)/200 * throughput`` EXACTLY -- applied
    directly on the elite's cached (already input-inclusive) per-model throughput, never
    recovered back to ``gen_tokens`` (a recovery would double-count the input cost baked
    into ``throughput``, since inverting ``1/t`` yields ``input_cost + gen_tokens``, not
    ``gen_tokens`` alone). Non-firing models (``throughput <= 0``) contribute 0. The
    single source of truth for "how good is this elite to ship" -- used by
    :meth:`Archive.ship_set`, the OPRO trajectory table, and the reported champion, so
    all three rank elites identically.
    """
    total = 0.0
    for m in config.MODELS:
        t = elite.throughput[m]
        if t <= 0.0:
            continue
        total += (elite.severity[m] + config.NOVELTY_PER_CELL) / 200.0 * t
    return total


def model_density(elite: Elite, model: str) -> float:
    """One model's board-density for ``elite`` (0.0 if it doesn't fire on ``model``).

    Applies the ``board_density = (sev + NOVELTY_PER_CELL)/200 * throughput`` identity
    directly to the cached per-model throughput (see :func:`elite_board_density` -- no
    ``gen_tokens`` recovery), so a pool/rank/select site can judge a shape by its OWN
    victim's density instead of the summed cross-model total (which lets the
    uniformly-denser model's shapes crowd the other out of every rank/select site -- the
    OPRO table, archive parents, and the shipped frontier all rank per-model for this
    reason).
    """
    t = elite.throughput.get(model, 0.0)
    if t <= 0.0:
        return 0.0
    return (elite.severity.get(model, 0.0) + config.NOVELTY_PER_CELL) / 200.0 * t


def rank_by_model_density(elites: Iterable[Elite]) -> list[Elite]:
    """Interleave ``elites`` per-model (round-robin) so neither model crowds out.

    Each model in :data:`config.MODELS` ranks its OWN firing elites
    (``throughput[model] > 0``) by :func:`model_density`, best first; the per-model
    rankings are then interleaved round-robin (model 0's best, model 1's best, model
    0's 2nd-best, ...), deduplicated by identity so a shape that fires on BOTH models is
    emitted once, at its first (best) slot. A plain sort by *summed* board-density lets
    a uniformly-denser model (historically gemma-plain vs gpt-forge) monopolize every
    top-N slot; this is the shared fix used by the OPRO table and archive parents.

    Args:
        elites: The candidate elites (typically an archive frontier).

    Returns:
        The elites in interleaved best-first order; every input elite that fires on at
        least one model appears exactly once.
    """
    per_model = [
        sorted(
            (e for e in elites if e.throughput.get(m, 0.0) > 0.0),
            key=lambda e, m=m: model_density(e, m),
            reverse=True,
        )
        for m in config.MODELS
    ]
    seen: set[int] = set()
    out: list[Elite] = []
    for group in zip_longest(*per_model):
        for e in group:
            if e is None or id(e) in seen:
                continue
            seen.add(id(e))
            out.append(e)
    return out


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
        """The frontier ranked by board-density, ties broken toward shorter input.

        Primary key is summed per-model board-density; the secondary key prefers a
        shorter input message (``-input_chars``) -- a free hedge on the grader's prefill
        cost that never overrides a denser shape.
        """
        return sorted(
            self.frontier(),
            key=lambda e: (elite_board_density(e), -e.input_chars),
            reverse=True,
        )[: config.ARCHIVE_FRONTIER_CAP]

    def parents(self, k: int) -> list[Elite]:
        """Up to k parents: per-model-balanced frontier specialists + exploration fill.

        Interleaves each model's frontier specialists by their OWN board-density
        (:func:`rank_by_model_density`), so a uniformly-denser model's shapes cannot
        crowd the other model's specialists out of a small ``k`` (the bug a fixed
        prefix ``front[:k]`` had -- gemma-heavy frontiers shipped zero gpt parents).
        If that interleaved list is still short of ``k``, tops up from the
        least-filled archive cells -- MAP-Elites' diversity/exploration material,
        wired in here (it was dead code on the normal frontier-full path before),
        skipping any shape already picked.

        Args:
            k: Number of parents to return.

        Returns:
            Up to k elites: per-model-balanced frontier specialists first, then
            under-filled-cell exploration material.
        """
        picks = rank_by_model_density(self.frontier())[:k]
        if len(picks) >= k:
            return picks
        seen = {id(e) for e in picks}
        under = sorted(self._cells.items(), key=lambda kv: len(kv[1]))
        for _, cell in under:
            for e in cell:
                if id(e) in seen:
                    continue
                picks.append(e)
                seen.add(id(e))
                if len(picks) >= k:
                    return picks
        return picks

    def to_jsonl(self, path: Path) -> None:
        """Persist the frontier as one JSON object per line."""
        path.write_text(
            "\n".join(json.dumps(asdict(x)) for x in self.frontier()), encoding="utf-8"
        )

    @classmethod
    def from_jsonl(cls, path: Path) -> "Archive":
        """Rebuild an archive by re-inserting every elite from a persisted jsonl.

        A line with no ``severity`` key is a pre-4-D throughput-only record (the
        schema before the severity axis). A zero-severity stand-in for it would
        win the throughput axis and never be Pareto-dominated, silently polluting
        the frontier forever. Structurally impossible instead: the whole file is
        treated as stale-schema and discarded, returning an empty archive so the
        caller's cold-start path cleanly re-seeds 4-D elites.
        """
        arch = cls()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    if "severity" not in data:
                        return cls()
                    arch.insert(Elite(**data))
        return arch
