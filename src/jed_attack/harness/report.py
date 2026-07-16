"""Post-run score attribution and a JSONL run archive for ablations."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aicomp_sdk.scoring import SEVERITY_W, get_score_cell_hash

from jed_attack.harness.runner import RunResult


@dataclass(frozen=True, slots=True)
class Breakdown:
    """Attribution of raw score into severity and diversity components."""

    severity_total: int
    diversity_total: int
    unique_cells: int
    per_predicate: dict[str, int] = field(default_factory=dict)


def breakdown(findings: Sequence[Mapping[str, Any]]) -> Breakdown:
    """Split raw score into severity and diversity, per predicate.

    Args:
        findings: Validated findings (with predicates + score cell signatures).

    Returns:
        A ``Breakdown`` summarizing the score composition.
    """
    per_predicate: dict[str, int] = {}
    severity_total = 0
    cells: set[str] = set()
    for finding in findings:
        cell_hash = get_score_cell_hash(finding)
        if cell_hash is not None:
            cells.add(cell_hash)
        for predicate in finding.get("predicates", []):
            weight = SEVERITY_W.get(int(predicate.get("severity", 1)), 1)
            severity_total += weight
            name = str(predicate.get("predicate", "UNKNOWN"))
            per_predicate[name] = per_predicate.get(name, 0) + weight
    return Breakdown(
        severity_total=severity_total,
        diversity_total=2 * len(cells),
        unique_cells=len(cells),
        per_predicate=per_predicate,
    )


def save_run(
    result: RunResult,
    report: Breakdown,
    *,
    archive_dir: Path,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Append a run record to the JSONL archive for reproducible ablations.

    Args:
        result: The ``RunResult`` for the run.
        report: The score breakdown.
        archive_dir: Directory holding ``runs.jsonl``.
        extra: Optional extra fields (e.g. attack config) to record.

    Returns:
        The archive file path.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / "runs.jsonl"
    record = {
        "score": result.score,
        "score_raw": result.score_raw,
        "findings_count": result.findings_count,
        "unique_cells": result.unique_cells,
        "agent_label": result.agent_label,
        "guardrail_id": result.guardrail_id,
        "time_taken": result.time_taken,
        "breakdown": asdict(report),
        "extra": dict(extra or {}),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path
