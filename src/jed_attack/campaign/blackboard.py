"""The team's shared blackboard: append-only JSONL + derived in-memory views.

One async process owns it, so appends are serialized with an ``asyncio.Lock`` (no
fcntl). Every scored submission is one JSONL line; the in-memory views (best submission,
best individual messages per shape, recent cross-model reasoning) are rebuilt on load
and updated on append. When an append sets a new optimizer-objective best, the
shipped ``attack.py`` is rewritten immediately via :func:`assemble.build`, so the
artifact never lags the campaign champion.
"""

import asyncio
import functools
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jed_attack.campaign import assemble, config, fill
from jed_attack.campaign.judge_policy import (
    CandidateObjective,
    JudgeAssessment,
    compare_candidates,
)
from jed_attack.campaign.submission import MessageType

_log = logging.getLogger(__name__)


# Tag stamped on every record scored under the CURRENT optimizer objective (the MEAN
# over victims of each model's public board per its OWN generated char). Bump this
# whenever the objective changes scale: champion selection prefers rows carrying the
# current tag so a prior scheme's incomparable magnitudes cannot freeze the champion.
# See _objective_key. A non-zero robustness weight blends the mean rate toward the
# worst model's rate, which rescales the objective, so it earns its own tag and pool.
def objective_scheme_name(
    robustness_lambda: float, portfolio_lambda: float = 0.0
) -> str:
    """Scheme tag for the current objective weights.

    The objective is the projected filled+trimmed public board (LB points), with a
    per-candidate cost ``agent_turns + CHAR_TIEBREAK*gen_chars`` (turns-first), plus a
    PORTFOLIO_LAMBDA distinct-shape diversity bonus. The ``v9`` bump retires the earlier
    pools; a non-zero robustness or portfolio weight rescales it, so each earns its own
    tag and champion pool.
    """
    base = "public_raw_per_gen_char_v9"
    if robustness_lambda != 0.0:
        base = f"robust{robustness_lambda:g}_raw_per_gen_char_v9"
    if portfolio_lambda != 0.0:
        base = f"portfolio{portfolio_lambda:g}_{base}"
    return base


OBJECTIVE_NAME = objective_scheme_name(
    config.ROBUSTNESS_LAMBDA, config.PORTFOLIO_LAMBDA
)


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
    valid: bool = True
    invalid_reason: str | None = None
    fires: bool = False
    assessment: dict[str, Any] | None = None
    objective: float = 0.0
    objective_tiebreaker: float = 0.0
    objective_name: str = "legacy_public"
    gen_chars: float = (
        0.0  # bottleneck-model generated chars to fire (the objective's cost)
    )
    # Per-model public board {model: board}; the mean of these is ``public``. Persisted
    # so the robustness objective and the incumbent block can see per-victim spread.
    public_by_model: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for this record."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Record":
        """Build a Record from a parsed JSON dict (tolerant of older/missing fields)."""
        return cls(
            messages=list(data["messages"]),
            public=float(data["public"]),
            feedback=list(data["feedback"]),
            reasoning=str(data.get("reasoning", "")),
            model=str(data.get("model", "")),
            worker=int(data.get("worker", 0)),
            ts=float(data["ts"]),
            valid=bool(data.get("valid", True)),
            invalid_reason=(
                str(data["invalid_reason"])
                if data.get("invalid_reason") is not None
                else None
            ),
            fires=bool(data.get("fires", _feedback_fires(list(data["feedback"])))),
            assessment=(
                dict(data["assessment"])
                if isinstance(data.get("assessment"), dict)
                else None
            ),
            objective=float(data.get("objective", 0.0)),
            objective_tiebreaker=float(data.get("objective_tiebreaker", 0.0)),
            objective_name=str(data.get("objective_name", "legacy_public")),
            gen_chars=float(data.get("gen_chars", 0.0)),
            public_by_model={
                str(m): float(v)
                for m, v in dict(data.get("public_by_model", {})).items()
            },
        )


def _champion_json(record: "Record") -> str:
    """Serialize a champion record's messages as the filled candidate-list JSON.

    Fills the record's message shapes into the shipped candidate list via the shared
    :mod:`fill` round-robin -- the same one the scorer projects -- so what ships matches
    what was scored. Fills directly (no ``Submission`` re-validation) so a persisted
    record always ships.
    """
    templates = [
        fill.templatize(str(m["text"])) or str(m["text"]) for m in record.messages
    ]
    chains = fill.ordered_chains(templates, config.SHIP_CANDIDATE_CAP)
    return json.dumps([list(chain) for chain in chains], separators=(",", ":"))


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
        replay_seconds=0.0,
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


def _objective_key(record: Record) -> tuple[bool, bool, float, float, float]:
    """Ranking key for campaign objective champions.

    Firing dominates: a valid firing record always outranks a non-firing one. Among
    firing records, CURRENT-scheme rows (``objective_name == OBJECTIVE_NAME``) outrank
    older-scheme rows regardless of magnitude, because a change of objective denominator
    changes the scale — the persisted ``objective`` of a stale-scheme row is not
    comparable to a current one (e.g. a per-hop objective of 18 dwarfs an equivalent
    per-generated-char objective, which would otherwise freeze the champion forever).
    Only within one scheme is the numeric ``objective`` a valid comparison. Legacy rows
    have no objective and only win by public score when no scored records exist.
    """
    return (
        record.valid and record.fires,
        record.objective_name == OBJECTIVE_NAME,
        record.objective,
        record.objective_tiebreaker,
        record.public,
    )


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
            total = len(records) + len(malformed_lines)
            # A few stragglers (a crash mid-append) are fine to skip, but losing most
            # or all rows means a systematic schema/parse regression — proceeding would
            # silently ship a stale/empty attack.py, so fail loudly instead.
            if not records or len(malformed_lines) > 0.5 * total:
                raise RuntimeError(
                    f"blackboard load dropped {len(malformed_lines)}/{total} rows as "
                    f"malformed (first lines={malformed_lines[:5]}) from {path}; "
                    "refusing to warm-start from a degraded board"
                )
            _log.warning(
                "skipped %d malformed blackboard row(s) while loading %s; "
                "first lines=%s",
                len(malformed_lines),
                path,
                malformed_lines[:5],
            )
        return cls(path, records)

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

    async def append(self, record: Record, out_dir: Path, reship: bool = True) -> bool:
        """Append a record, persist it, and (optionally) reship on a new best.

        Args:
            record: The scored record to append and persist.
            out_dir: Where a new-best reship writes ``attack.py``.
            reship: Whether a new optimizer-objective best reships ``attack.py`` via
                :func:`assemble.build`. The default preserves the legacy
                best-submission ship path; the curated-pool loop passes ``False`` so
                append only records candidates and curation is the sole ship path (no
                double ship).

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
                assemble.build(_champion_json(record), out_dir)
                return True
            return False

    def reship_best(self, out_dir: Path) -> None:
        """Rewrite ``attack.py`` from the current objective champion (no-op if empty).

        Used on warm start so the shipped artifact reflects the flat file's best
        immediately, rather than lagging until a future generation beats it.

        Args:
            out_dir: Where :func:`assemble.build` writes ``attack.py``.
        """
        best = self.best_objective()
        if best is not None:
            assemble.build(_champion_json(best), out_dir)

    def reship_champions(self, public_out_dir: Path, robust_out_dir: Path) -> None:
        """Rewrite exact-public and robust champion artifacts independently."""
        public = self.best_public()
        robust = self.best_robust()
        if public is not None:
            assemble.build(_champion_json(public), public_out_dir)
        if robust is not None:
            assemble.build(_champion_json(robust), robust_out_dir)
