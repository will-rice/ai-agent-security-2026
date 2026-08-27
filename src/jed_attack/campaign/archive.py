"""MAP-Elites + Pareto archive of attack shapes.

The unit is a scored shape with a 4-D vector: per-model TOKEN COST (to MINIMIZE) and
per-model severity (one of each per victim model). Cells are keyed by
``(family, input_char_bucket)``; each cell keeps its Pareto-non-dominated elites, and
the globally non-dominated set (frontier) is the elite pool that ships, ranked by each
elite's BEST (fewest-token) firing column -- FEWEST wins.
Selection is Pareto over the raw per-model tokens/severity, never a scalar.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path

from jed_attack.campaign import config


@dataclass(frozen=True)
class Elite:
    """A single scored shape held in the archive.

    Attributes:
        text: The shape's literal message text.
        mtype: The message type ("exfil" or "deputy").
        tokens: Per-model total token cost, ``{model: input_tokens + gen_tokens +
            FIXED_TOKENS[model]}``. MINIMIZE -- a model the shape does NOT fire on
            carries ``float("inf")`` (see
            :func:`jed_attack.campaign.submission_score.message_token_cost`).
        severity: Per-model predicate severity, `{model: severity}`.
        diagnosis: Free-text scorer/judge diagnosis for this shape.
        family: The MAP-Elites structural family (behavioral descriptor axis 1).
        bucket: The input_char_bucket length bin (behavioral descriptor axis 2).
        url_scheme: EXFIL URL template ('{h}' host slot) the shape's shipped candidates
            post to; default reproduces the historical ``http://<host>.co``.
        input_chars: The shape's input message length. A hedge on the real grader's
            prefill cost (local replay is input-length-insensitive, but the grader's
            model server may charge for it). A ship-set tiebreaker, not a token-cost
            term. 0 for elites persisted before this field existed.
    """

    text: str
    mtype: str
    tokens: dict[str, float]
    severity: dict[str, float]
    diagnosis: str
    family: str
    bucket: int
    url_scheme: str = "http://{h}.co"
    input_chars: int = 0


def dominates(a: Elite, b: Elite) -> bool:
    """True if a Pareto-dominates b: fewer-or-equal tokens AND no-lower severity.

    Tokens are MINIMIZED (fewer wins), severity is MAXIMIZED (more wins) -- ``a``
    dominates ``b`` when it is at least as good as ``b`` on every per-model axis and
    strictly better on at least one.
    """
    token_comps = [(a.tokens[m], b.tokens[m]) for m in config.MODELS]
    severity_comps = [(a.severity[m], b.severity[m]) for m in config.MODELS]
    le = all(av <= bv for av, bv in token_comps)
    ge = all(av >= bv for av, bv in severity_comps)
    strict = any(av < bv for av, bv in token_comps) or any(
        av > bv for av, bv in severity_comps
    )
    return le and ge and strict


def model_density(elite: Elite, model: str) -> float:
    """One model's token cost for ``elite`` (``+inf`` if it doesn't fire on ``model``).

    MINIMIZE. Reads the elite's OWN per-model token cost directly (already ``+inf`` for
    a non-firing model, see :attr:`Elite.tokens`), so a pool/rank/select site can judge
    a shape by its OWN victim's cost instead of a cross-model aggregate (which lets the
    uniformly-leaner model's shapes crowd the other out of every rank/select site -- the
    OPRO table, archive parents, and the shipped frontier all rank per-model for this
    reason).
    """
    severity = elite.severity.get(model, 0.0)
    if severity <= 0.0:  # not firing -> never ranks/ships
        return float("inf")
    return elite.tokens.get(model, float("inf"))


def elite_board_density(elite: Elite) -> float:
    """``elite``'s BEST per-model token cost -- the shared ship/rank key. MINIMIZE.

    An elite is authored for, and scored against, exactly ONE model's pool (see
    :func:`jed_attack.campaign.optimize_prompts._shape_elites`) -- it is structurally a
    single-model specialist, never firing on both columns at once. So the elite's
    aggregate cost is the MIN of :func:`model_density` over :data:`config.MODELS`: its
    one real (finite) firing column, or ``+inf`` if it fires on neither (never worth
    shipping). This differs from the SUBMISSION-level objective
    (:func:`jed_attack.campaign.optimize_prompts._score_total_tokens`, which means
    across models and requires BOTH pools to fire) because a submission's
    ``per_message`` spans both pools while one elite never does. The single source of
    truth for "how good is this elite to ship" -- used by :meth:`Archive.ship_set`, the
    OPRO trajectory table, and the reported champion, so all three rank elites
    identically.
    """
    return min(model_density(elite, m) for m in config.MODELS)


def rank_by_model_density(elites: Iterable[Elite]) -> list[Elite]:
    """Interleave ``elites`` per-model (round-robin) so neither model crowds out.

    Each model in :data:`config.MODELS` ranks its OWN firing elites
    (``model_density < inf``) by :func:`model_density`, LEANEST (fewest tokens) first;
    the per-model rankings are then interleaved round-robin (model 0's leanest, model
    1's leanest, model 0's 2nd-leanest, ...), deduplicated by identity so a shape that
    fires on BOTH models is emitted once, at its first (best) slot. A plain sort by the
    *mean* token cost lets a uniformly-leaner model (historically gemma-plain vs
    gpt-forge) monopolize every top-N slot; this is the shared fix used by the OPRO
    table and archive parents.

    Args:
        elites: The candidate elites (typically an archive frontier).

    Returns:
        The elites in interleaved best-first order; every input elite that fires on at
        least one model appears exactly once.
    """
    per_model = [
        sorted(
            (e for e in elites if model_density(e, m) < float("inf")),
            key=lambda e, m=m: model_density(e, m),
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
        """The frontier ranked by token cost (fewest first), ties toward shorter input.

        Primary key is the mean per-model token cost (:func:`elite_board_density`,
        ascending); the secondary key prefers a shorter input message
        (``e.input_chars``, ascending) -- a free hedge on the grader's prefill cost that
        never overrides a leaner shape.
        """
        return sorted(
            self.frontier(),
            key=lambda e: (elite_board_density(e), e.input_chars),
        )[: config.ARCHIVE_FRONTIER_CAP]

    def parents(self, k: int) -> list[Elite]:
        """Up to k parents: per-model-balanced frontier specialists + exploration fill.

        Interleaves each model's frontier specialists by their OWN token cost
        (:func:`rank_by_model_density`), so a uniformly-leaner model's shapes cannot
        crowd the other model's specialists out of a small ``k`` (the bug a fixed
        prefix ``front[:k]`` had -- gemma-heavy frontiers shipped zero gpt parents).
        If that interleaved list is still short of ``k``, tops up from the
        least-filled archive cells -- MAP-Elites' diversity/exploration material --
        skipping any shape already picked OR any non-firing shape (firing is intrinsic
        to selection, so a generate-but-never-fire shape is never a parent).

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
                if id(e) in seen or not any(
                    model_density(e, m) < float("inf") for m in config.MODELS
                ):
                    continue  # firing is intrinsic: never parent a non-firing shape
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

        A line with no ``severity`` OR ``tokens`` key is a stale-schema record (either
        the pre-4-D throughput-only schema, or a pre-v26 ``throughput`` field). A
        zero-severity/zero-cost stand-in for either would never be Pareto-dominated,
        silently polluting the frontier forever. Structurally impossible instead: the
        whole file is treated as stale-schema and discarded, returning an empty archive
        so the caller's cold-start path cleanly re-seeds fresh elites.
        """
        arch = cls()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    if "severity" not in data or "tokens" not in data:
                        return cls()
                    # Filter to the current Elite fields so a schema change that DROPS
                    # a field (e.g. a removed diagnostic) never crashes board load on an
                    # older-scheme island row -- unknown keys are ignored, matching
                    # Record.from_json's tolerance on the flat blackboard.
                    arch.insert(
                        Elite(
                            **{
                                k: v
                                for k, v in data.items()
                                if k in Elite.__dataclass_fields__
                            }
                        )
                    )
        return arch
