"""The team's shared blackboard: append-only JSONL + derived in-memory views.

One async process owns it, so appends are serialized with an ``asyncio.Lock`` (no
fcntl). Every scored submission is one JSONL line; the in-memory views (best submission,
best individual messages per shape, recent cross-model reasoning) are rebuilt on load
and updated on append. When an append sets a new optimizer-objective best, the
shipped ``attack.py`` is rewritten immediately via the per-model router
(:func:`assemble.build_permodel`), so the artifact never lags the campaign champion.
"""

import asyncio
import functools
import json
import logging
from itertools import zip_longest
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jed_attack.campaign import archive, assemble, config, fill
from jed_attack.campaign.archive import Elite
from jed_attack.campaign.islands import IslandSet
from jed_attack.campaign.judge_policy import (
    CandidateObjective,
    JudgeAssessment,
    compare_candidates,
)
from jed_attack.campaign.submission import MessageType, Submission

_log = logging.getLogger(__name__)


# Tag stamped on every record scored under the CURRENT optimizer objective. Bump this
# whenever the shipping/objective scheme changes: champion selection prefers rows
# carrying the current tag so a prior scheme's incomparable magnitudes cannot freeze
# the logged champion (see _objective_key). The current tag is ``v24`` (the islands
# scheme: the shipped pool is now the GLOBAL best across the island archives --
# per-island lineages + migration + stagnation reset + a novelty island -- same v22/v23
# cost model, see docstring). NOTE: ROBUSTNESS_LAMBDA is currently UN-WIRED -- the score
# never reads it, so a non-zero value only changes the tag/pool below, not the score.
def objective_scheme_name(
    robustness_lambda: float,
    portfolio_lambda: float = 0.0,
    gate: str = config.GATE_GUARDRAIL_NAME,
) -> str:
    """Scheme tag for the current objective weights.

    ``v24`` moves shipping from a single Pareto archive to an island model
    (:mod:`jed_attack.campaign.islands`): N parallel per-island archives with their own
    lineages, periodic migration between islands, stagnation reset for islands that stop
    improving, and a dedicated novelty island. The shipped pool is the GLOBAL best
    elite across all island archives, not one archive's frontier. The cost model and
    diversity axes are UNCHANGED from ``v23``/``v22`` (throughput = value /
    (input_tokens + gen_tokens); diversity keyed on input-length + reply-suppression
    shape_family) -- only the archive topology changed, so this is a pure re-key: v23
    and earlier rows are stale (discarded on rank). ``v23`` re-keyed the archive's
    diversity axis from generated length to INPUT length (``input_char_bucket``) and
    added a reply-suppression flag to shape_family: generated output is floored (28/29
    on every firing shape), so the old gen-char axis illuminated nothing and the search
    collapsed to one cell; input length is the axis that varies and that the objective
    minimises. ``v20`` was the MAP-Elites + Pareto
    shipping scheme: the shipped pool is the archive's Pareto frontier over the two raw
    per-model throughput columns (see :mod:`jed_attack.campaign.archive`), NOT a single
    MIN-scalar champion. The MIN
    objective helpers (:func:`_objective_key`, :meth:`Blackboard.best_objective`) are
    RETAINED for logging/telemetry only -- they no longer decide what ships. The scheme
    encodes the gate guardrail (``config.GATE_GUARDRAIL_NAME``) so switching guardrails
    auto-starts a clean pool; a non-zero portfolio weight tags diversity-on, its own
    pool. NOTE: ``robustness_lambda`` is currently UN-WIRED (only affects the tag/pool
    below, never the score itself); threaded through only so a future robustness-aware
    objective has its own pool ready.
    """
    base = f"{gate}_pareto_v25"
    if robustness_lambda != 0.0:
        base = f"robust{robustness_lambda:g}_{gate}_pareto_v25"
    if portfolio_lambda != 0.0:
        base = f"portfolio{portfolio_lambda:g}_{base}"
    return base


OBJECTIVE_NAME = objective_scheme_name(
    config.ROBUSTNESS_LAMBDA, config.PORTFOLIO_LAMBDA
)


class Record(BaseModel):
    """One scored submission on the blackboard.

    Carries the authored two-pool ``Submission`` -- the pools live in exactly one
    place, ``record.submission.gpt_oss``/``gemma_4`` (matching ``config.MODELS``); a
    ``Record`` never re-declares them. The champion reship fills each pool via
    :meth:`Submission.candidate_chains` and ships a per-model map through the router
    (see :func:`_champion_map`, :func:`assemble.build_permodel`).
    """

    model_config = ConfigDict(frozen=True)

    submission: Submission
    public: float
    feedback: list[dict]
    reasoning: str = ""
    model: str = ""
    worker: int = 0
    ts: float
    valid: bool = True
    invalid_reason: str | None = None
    fires: bool = False
    assessment: dict[str, Any] | None = None
    objective: float = 0.0
    objective_sum: float = (
        0.0  # lexicographic 2nd key: total board (headroom above min)
    )
    objective_tiebreaker: float = 0.0
    objective_name: str = "legacy_public"
    gen_chars: float = (
        0.0  # bottleneck-model generated chars to fire (the objective's cost)
    )
    # Per-model public board {model: board}; the mean of these is ``public``. Persisted
    # so the robustness objective and the incumbent block can see per-victim spread.
    public_by_model: dict[str, float] = Field(default_factory=dict)
    # Which FunSearch island (islands.IslandSet index) authored/scored this record.
    # Persisted so island_best/global_champion can re-derive per-island rankings from
    # the flat JSONL alone. Defaults to 0 so a pre-islands row (single shared archive,
    # island 0's role under the Task-4 bridge) loads unchanged.
    island: int = 0

    @model_validator(mode="before")
    @classmethod
    def _tolerate_legacy_rows(cls, data: object) -> object:
        """Rewrite an older-shape row into the current schema before validation.

        A pre-two-pool row has a flat ``messages`` field (list of message dicts) and no
        ``submission`` -- load it into BOTH named pools so an old harvest board still
        warm-starts. Both pools, not just ``gpt_oss``, because ``Submission`` requires
        every pool non-empty (``Field(min_length=config.MIN_SHIP_MESSAGES)``, always
        >=1 -- see ``config.MIN_SHIP_MESSAGES``'s "the mean objective needs both
        per-model columns populated") -- a genuinely empty pool cannot validate at all.
        This also reproduces the ORIGINAL pre-two-pool behavior byte-for-byte: that flat
        pool was shipped identically to both victims (one shared list, "scored on BOTH
        victims"), so duplicating it into both named pools is the faithful
        reconstruction, not an arbitrary filler. A row persisted before ``fires``
        existed infers it from per-message severity (see :func:`_feedback_fires`)
        instead of silently defaulting to non-firing.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "submission" not in data and "messages" in data:
            legacy_messages = data.pop("messages")
            data["submission"] = {
                "gpt_oss": legacy_messages,
                "gemma_4": list(legacy_messages),
            }
        if "fires" not in data:
            data["fires"] = _feedback_fires(list(data.get("feedback", [])))
        return data

    @property
    def messages(self) -> list[dict]:
        """Both pools concatenated in ``config.MODELS`` order -- the flat message view.

        The record stores its pools on ``submission``; consumers that only need the
        authored shapes (diversity count, judge-study candidate source, incumbent
        prompt block) read this flat concatenation.
        """
        return [m.model_dump(mode="json") for _, m in self.submission.all_messages()]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for this record."""
        return self.model_dump(mode="json")

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Record":
        """Build a Record from a parsed JSON dict (tolerant of the legacy flat shape).

        See :meth:`_tolerate_legacy_rows` for the pre-validation rewrite.
        """
        return cls.model_validate(data)


def _fill_chains(specs: list[tuple[str, str, str]], cap: int) -> list[list[str]]:
    """Fill ``(text, type, url_scheme)`` specs into typed candidate chains.

    Thin wrapper around :func:`fill.ordered_chains` that returns plain ``list``s (not
    tuples) -- the shape :func:`assemble.build_permodel` expects, since it now takes
    typed python pools and serializes them internally (no JSON string in between).
    """
    return [list(chain) for chain in fill.ordered_chains(specs, cap)]


def _champion_map(record: "Record") -> dict[str, list[list[str]]]:
    """Per-model filled candidate chains for a champion record's authored Submission.

    Delegates straight to :meth:`Submission.candidate_chains` -- the record's pools
    live in exactly one place (``record.submission``), so there is no separate
    spec-building path to keep in sync with it. The result feeds straight into
    :func:`assemble.build_permodel` (see :func:`_ship_pools`), which routes each pool
    to its own victim at grade time.
    """
    return {
        model: [
            list(chain)
            for chain in record.submission.candidate_chains(
                model, config.SHIP_CANDIDATE_CAP
            )
        ]
        for model in config.MODELS
    }


def _ship_pools(pools: dict[str, list[list[str]]], out_dir: Path) -> Path:
    """Route the ``gpt_oss``/``gemma_4`` filled pools through the per-model router.

    ``assemble.build_permodel``'s two positional pools are model-specific
    (forge=``gpt_oss``, plain=``gemma_4``), not generic first/second -- this pins that
    mapping in one place so every ship call site agrees. An empty pool is valid to
    ``build_permodel`` (it just ships that victim nothing, scoring 0) -- this can
    happen on the frontier path when every surviving elite fires on only one model
    (the champion path always fills both pools, see ``Submission``'s per-pool
    ``min_length``). Not an error, but silent, so it's logged.
    """
    for model, chains in pools.items():
        if not chains:
            _log.warning(
                "shipping %s with an EMPTY pool -- that model's column will score 0",
                model,
            )
    return assemble.build_permodel(pools["gpt_oss"], pools["gemma_4"], out_dir)


def _islands_dir(board_path: Path) -> Path:
    """The directory of per-island archive JSONLs persisted alongside the blackboard.

    Mirrors the ``blackboard.jsonl`` idiom: each island's frontier
    lives in its own file (``island_<i>.jsonl``, see :meth:`islands.IslandSet.to_jsonl`)
    under a sibling directory (``<stem>.islands/``) so a warm start reloads every
    island from the same directory.
    """
    return board_path.with_name(board_path.stem + ".islands")


def _frontier_map(arch: archive.Archive) -> dict[str, list[list[str]]]:
    """Per-model filled candidate chains for the archive's Pareto ship-set.

    Ranks and caps EACH pool by its OWN victim's board-density
    (:func:`archive.model_density`): a global summed-density truncation (the old
    :meth:`Archive.ship_set`) let the denser model monopolize every
    ``ARCHIVE_FRONTIER_CAP`` slot -- gemma-plain shapes are uniformly denser than
    gpt-forge, so the top-N were 100% gemma and gpt shipped an EMPTY pool (gpt column
    scored 0). Filtering per model here (``throughput[model] > 0``) keeps a gpt-only
    elite out of gemma's pool and vice versa, and the per-model cap guarantees BOTH
    victims get their densest firing shapes. Ties on density break toward the SHORTER
    input (``-input_chars``): input prefill is a real but small replay cost (measured
    ~1/45 of a generated token, folded into throughput at INPUT_PREFILL_WEIGHT), so it
    never overrides density but does decide equal-density shapes toward less prefill.
    Feeds :func:`assemble.build_permodel` via :func:`_ship_pools`.
    """
    frontier = arch.frontier()
    out: dict[str, list[list[str]]] = {}
    for model in config.MODELS:
        firing = sorted(
            (e for e in frontier if archive.model_density(e, model) > 0.0),
            key=lambda e: (archive.model_density(e, model), -e.input_chars),
            reverse=True,
        )[: config.ARCHIVE_FRONTIER_CAP]
        out[model] = _fill_chains(
            [(e.text, e.mtype, e.url_scheme) for e in firing],
            config.SHIP_CANDIDATE_CAP,
        )
    return out


def _severity_sum(entry: dict) -> float:
    """Total severity of one feedback entry across guardrails."""
    return float(sum(entry.get("severity", {}).values()))


def _feedback_fires(feedback: list[dict]) -> bool:
    """Infer old-row firing status from persisted per-message severity."""
    return any(_severity_sum(entry) > 0.0 for entry in feedback)


def _assessment_objective(record: Record) -> CandidateObjective:
    """Convert a record's persisted assessment into policy comparison input."""
    return CandidateObjective(
        valid=record.valid,
        firing=record.fires,
        public=record.public,
        gen_tokens=0.0,
        assessment=(
            JudgeAssessment.model_validate(record.assessment)
            if record.assessment is not None
            else None
        ),
    )


def _more_robust_record(left: Record, right: Record) -> Record:
    """Return the record preferred by the judge-aware policy."""
    decision = compare_candidates(
        _assessment_objective(left), _assessment_objective(right)
    )
    if decision.winner == "b":
        return right
    return left


def _objective_key(
    record: Record,
) -> tuple[bool, bool, float, float, float, float]:
    """Ranking key for campaign objective champions.

    Firing dominates: a valid firing record always outranks a non-firing one. Among
    firing records, CURRENT-scheme rows (``objective_name == OBJECTIVE_NAME``) outrank
    older-scheme rows regardless of magnitude, because a change of objective denominator
    changes the scale — the persisted ``objective`` of a stale-scheme row is not
    comparable to a current one (e.g. a per-hop objective of 18 dwarfs an equivalent
    per-generated-char objective, which would otherwise freeze the champion forever).
    Only within one scheme is the numeric ``objective`` a valid comparison.

    The comparison is LEXICOGRAPHIC: the MEAN ``objective`` (average of the two
    per-model columns — the actual LB metric) is primary; ties break on
    ``objective_sum`` (total board across models — numerically redundant with the mean
    primary for two models, but kept as a defensive tiebreak for legacy/partial rows),
    then ``objective_tiebreaker`` (distinct both-model shapes, a private hedge), then
    ``public``. Records predating a field default it to 0.0, so a new row with the same
    mean but a real sum outranks an old sum-less row without a bump. Legacy rows have no
    objective and only win by public score when no scored records exist.
    """
    return (
        record.valid and record.fires,
        record.objective_name == OBJECTIVE_NAME,
        record.objective,
        record.objective_sum,
        record.objective_tiebreaker,
        record.public,
    )


class Blackboard:
    """In-memory team memory backed by an append-only JSONL file."""

    def __init__(
        self,
        path: Path,
        records: list[Record],
        initial_islands: IslandSet | None = None,
    ) -> None:
        self._path = path
        self._records = records
        # The N-island MAP-Elites + Pareto archive set: what actually ships
        # (:meth:`reship_islands`). Direct constructions (tests, cold start) get a
        # fresh empty set; :meth:`load` warm-starts it from the sibling islands dir.
        self.islands = (
            initial_islands
            if initial_islands is not None
            else IslandSet.for_count(config.ISLAND_COUNT)
        )
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, path: Path) -> "Blackboard":
        """Warm-start: replay the JSONL into memory (skips malformed lines)."""
        records: list[Record] = []
        malformed_lines: list[int] = []
        if path.exists():
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(Record.from_json(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    malformed_lines.append(line_no)
                    continue
        if malformed_lines:
            # A few stragglers (a crash mid-append) or a schema migration that
            # legitimately invalidates most/all old rows are both just "drop and
            # warm-start from whatever remains" -- an empty board is a cold start,
            # which is fine. Warn (loudly, with the dropped-row count) instead of
            # crashing the optimizer: refusing to warm-start would be strictly worse
            # than shipping fewer candidates from a stale/near-empty board.
            _log.warning(
                "skipped %d malformed blackboard row(s) while loading %s; "
                "first lines=%s",
                len(malformed_lines),
                path,
                malformed_lines[:5],
            )
        # Warm-starts every island's frontier from the per-island archives under
        # _islands_dir -- the SAME directory :meth:`reship_islands` persists to -- so
        # load and the live write path agree (a restart reloads exactly what last
        # shipped; there is no longer a separate archive to fall out of sync).
        return cls(
            path, records, IslandSet.from_jsonl(_islands_dir(path), config.ISLAND_COUNT)
        )

    def best_public(self) -> Record | None:
        """The highest-``public`` record, or ``None`` if empty."""
        if not self._records:
            return None
        return max(self._records, key=lambda r: r.public)

    def best_objective(self) -> Record | None:
        """The highest optimizer-objective record, or ``None`` if empty."""
        if not self._records:
            return None
        return max(self._records, key=_objective_key)

    def island_best(self, i: int) -> Record | None:
        """The best current-scheme record on island ``i``, or ``None``.

        Mirrors :meth:`best_objective` (the same :func:`_objective_key` ranking),
        restricted to records tagged ``island == i`` under the current objective
        scheme (``objective_name == OBJECTIVE_NAME``); a stale-scheme row's magnitude
        is not comparable to a current one (see :func:`_objective_key`), so it is
        excluded outright here rather than merely outranked.

        Args:
            i: Island index.

        Returns:
            Island ``i``'s best-objective record, or ``None`` if it has none under the
            current scheme.
        """
        candidates = [
            record
            for record in self._records
            if record.island == i and record.objective_name == OBJECTIVE_NAME
        ]
        if not candidates:
            return None
        return max(candidates, key=_objective_key)

    def global_champion(self) -> Record | None:
        """The best current-scheme record across every island, or ``None``.

        The ship/telemetry champion for the islands search: the max of
        :meth:`island_best` over every island in :attr:`islands`. It is the reship
        *trigger* (:func:`worker_loop` reships when it advances) and the ablate target,
        not itself a ship call site -- :meth:`reship_islands` ships the union of every
        island's frontier. Telemetry call sites that report the flat-board champion
        regardless of scheme keep using :meth:`best_objective`.

        Returns:
            The best-objective record across all islands, or ``None`` if none exist
            under the current scheme.
        """
        bests = [
            record
            for i in range(len(self.islands.archives))
            if (record := self.island_best(i)) is not None
        ]
        if not bests:
            return None
        return max(bests, key=_objective_key)

    def best_diverse(self, band: float) -> Record | None:
        """Champion with the MOST distinct shapes within ``band`` of the best objective.

        For a private-board hedge: shape-TEXT diversity is what survives a stricter
        private guardrail (a blocked phrasing kills all URL-variants of that one shape
        at once), but the public-throughput objective is nearly flat across shape count
        (``fill`` expands any shape count to the same candidate cap), so a strict
        :meth:`best_objective` selects a lean low-shape build. This keeps the objective
        within ``band`` (a fractional drop, e.g. 0.1 = 10%) of the best while maximizing
        distinct shapes. ``band=0.0`` reduces to the strict champion. Falls back to
        :meth:`best_objective` when no current-scheme firing record exists.

        Args:
            band: Max fractional objective drop from the best to still be eligible.

        Returns:
            The most-diverse eligible record, or ``None`` if the board is empty.
        """
        firing = [
            record
            for record in self._records
            if record.valid and record.fires and record.objective_name == OBJECTIVE_NAME
        ]
        if not firing:
            return self.best_objective()
        floor = max(record.objective for record in firing) * (1.0 - band)
        within = [record for record in firing if record.objective >= floor]
        return max(within, key=lambda record: (len(record.messages), record.objective))

    def best(self) -> Record | None:
        """Compatibility alias for the exact-public champion."""
        return self.best_public()

    def best_robust(self) -> Record | None:
        """The judge-robust champion within the public-regression band."""
        public_best = self.best_public()
        if public_best is None:
            return None
        floor = public_best.public * (1.0 - config.JUDGE_PUBLIC_BAND_RATIO)
        candidates = [
            record
            for record in self._records
            if record.valid
            and record.fires
            and record.public >= floor
            and record.assessment is not None
            and record.assessment.get("status") == "available"
        ]
        if not candidates:
            return public_best
        return functools.reduce(_more_robust_record, candidates)

    def mechanism_references(self, k: int) -> list[str]:
        """Distinct high-public mechanism labels for archive-relative judge context."""
        labels: list[str] = []
        for record in sorted(self._records, key=lambda item: item.public, reverse=True):
            if record.assessment is None:
                continue
            assessment = JudgeAssessment.model_validate(record.assessment)
            if assessment.status != "available" or assessment.mechanism is None:
                continue
            for label in assessment.mechanism.mechanism_labels:
                if label not in labels:
                    labels.append(label)
                if len(labels) == k:
                    return labels
        return labels

    def top_messages(self, mtype: MessageType, k: int) -> list[tuple[str, str, float]]:
        """Best-scoring individual messages of a shape, balanced across victim models.

        Each feedback entry's VICTIM model (its own ``"model"`` tag -- see
        :func:`optimize_prompts.make_record`) buckets it into that victim's own
        top-by-severity ranking, deduped by text within the bucket; the two known
        victims' (:data:`~jed_attack.campaign.config.MODELS`) rankings are then
        interleaved round-robin so neither victim's messages crowd the other out of a
        small ``k`` -- a single global top-k by severity was victim-blind once one
        victim's shapes were uniformly more severe or more numerous (and, before the
        per-message tag existed, the entry was mislabeled with the PROPOSER lane
        instead of the victim it actually fired on). PRESENT vs ABSENT matters: a
        legacy row persisted before the ``"model"`` tag existed has no ``"model"`` key
        at all and falls back to the record's proposer lane; an entry that carries an
        EXPLICIT ``""`` (:func:`optimize_prompts.make_record`'s deliberate "no reliable
        victim" marker for a mis-aligned score) is used VERBATIM, never silently
        reattributed to ``record.model`` -- proposer lanes can be literally named
        "gpt_oss"/"gemma_4" (see ``providers.py``), so falling an explicit "" through to
        ``record.model`` could misattribute an unattributed entry into a real victim's
        bucket. Either way, a victim outside ``config.MODELS`` (including "") lands in
        one shared catch-all bucket appended after the known victims -- graceful, never
        a crash.
        """
        other = "_other"
        buckets: dict[str, dict[str, tuple[str, float]]] = {
            m: {} for m in (*config.MODELS, other)
        }
        for record in self._records:
            for entry in record.feedback:
                if entry.get("type") != mtype.value:
                    continue
                sev = _severity_sum(entry)
                text = entry.get("message", "")
                if sev <= 0 or not text:
                    continue
                # Present-but-"" (make_record's explicit "unattributed" marker) must
                # NOT fall through to record.model -- only an ABSENT key (a genuine
                # legacy row) does.
                victim = entry["model"] if "model" in entry else record.model
                bucket = buckets[victim] if victim in buckets else buckets[other]
                if text not in bucket or sev > bucket[text][1]:
                    bucket[text] = (victim, sev)
        ranked = [
            sorted(buckets[m].items(), key=lambda kv: kv[1][1], reverse=True)
            for m in (*config.MODELS, other)
        ]
        out: list[tuple[str, str, float]] = []
        for group in zip_longest(*ranked):
            for item in group:
                if item is None:
                    continue
                text, (victim, sev) = item
                out.append((text, victim, sev))
                if len(out) >= k:
                    return out
        return out

    def recent_reasoning(self, k: int, chars: int = 800) -> list[tuple[str, str]]:
        """The most recent non-empty reasoning blobs: ``(model, excerpt)`` (bounded)."""
        out: list[tuple[str, str]] = []
        for record in reversed(self._records):
            if record.reasoning:
                out.append((record.model, record.reasoning[:chars]))
            if len(out) >= k:
                break
        return out

    async def append(self, record: Record, out_dir: Path, reship: bool = True) -> bool:
        """Append a record, persist it, and (optionally) reship on a new best.

        Args:
            record: The scored record to append and persist.
            out_dir: Where a new-best reship writes ``attack.py``.
            reship: Whether a new optimizer-objective best reships ``attack.py`` via
                the per-model router (:func:`assemble.build_permodel`). The default
                preserves the legacy best-submission ship path; the curated-pool loop
                passes ``False`` so append only records candidates and curation is the
                sole ship path (no double ship).

        Returns:
            ``True`` when this append rewrote ``attack.py``; otherwise ``False``.
        """
        async with self._lock:
            prior_best = self.best_objective()
            self._records.append(record)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
                handle.flush()
            if reship and self.best_objective() is record and record is not prior_best:
                _ship_pools(_champion_map(record), out_dir)
                return True
            return False

    def reship_best(self, out_dir: Path) -> None:
        """Rewrite ``attack.py`` from the current objective champion (no-op if empty).

        Used on warm start so the shipped artifact reflects the flat file's best
        immediately, rather than lagging until a future generation beats it.

        Args:
            out_dir: Where :func:`assemble.build_permodel` writes ``attack.py``.
        """
        best = self.best_objective()
        if best is not None:
            _ship_pools(_champion_map(best), out_dir)

    async def reship_islands(self, out_dir: Path) -> None:
        """Ship the union of every island's frontier, and persist all islands.

        Merges each island's frontier (:meth:`archive.Archive.frontier`) into a fresh
        :class:`archive.Archive` via :meth:`archive.Archive.insert` -- re-establishing
        Pareto dominance ACROSS islands (an elite from one island can dominate one from
        another) -- so :func:`_frontier_map`/:func:`_ship_pools` rank and ship the
        resulting cross-island union the same way a single archive's frontier ships.
        Persists every island to its own sibling JSONL via
        :meth:`islands.IslandSet.to_jsonl` under the same lock so the shipped artifact
        and the on-disk islands never disagree.

        Args:
            out_dir: Where :func:`assemble.build_permodel` writes ``attack.py``.
        """
        async with self._lock:
            union = archive.Archive()
            for island_archive in self.islands.archives:
                for elite in island_archive.frontier():
                    union.insert(elite)
            _ship_pools(_frontier_map(union), out_dir)
            islands_dir = _islands_dir(self._path)
            islands_dir.mkdir(parents=True, exist_ok=True)
            self.islands.to_jsonl(islands_dir)

    def champion_by_board_density(self) -> Elite | None:
        """Board-density leader across every island's frontier (logging only).

        The whole cross-island union ships (:meth:`reship_islands`); this is only the
        single representative surfaced in logs/telemetry.
        :meth:`islands.IslandSet.global_best_elite` returns exactly the
        :func:`archive.elite_board_density` leader over all island frontiers -- the same
        shape the shipped union leads with -- so the reported champion matches what
        actually ships first, never a selection gate.

        Returns:
            The board-density-leading elite across all islands, or ``None`` if every
            island is empty.
        """
        return self.islands.global_best_elite()

    def reship_champions(self, public_out_dir: Path, robust_out_dir: Path) -> None:
        """Rewrite exact-public and robust champion artifacts independently."""
        public = self.best_public()
        robust = self.best_robust()
        if public is not None:
            _ship_pools(_champion_map(public), public_out_dir)
        if robust is not None:
            _ship_pools(_champion_map(robust), robust_out_dir)
