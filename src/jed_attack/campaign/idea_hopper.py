"""Local evidence-to-experiment queue for the JED campaign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceEvidence:
    """One cited source observation behind an idea."""

    path: str
    observation: str


@dataclass(frozen=True)
class Idea:
    """One proposed experiment with evidence and a score gate."""

    id: str
    title: str
    hypothesis: str
    source_evidence: tuple[SourceEvidence, ...]
    likely_files: tuple[str, ...]
    category: str
    priority: str
    priority_score: float
    expected_upside: str
    risk: str
    acceptance_gate: str
    status: str = "proposed"
    fingerprint: str = ""


@dataclass(frozen=True)
class HopperReport:
    """The result of one idea-hopper pass."""

    generated_utc: str
    inputs_read: tuple[str, ...]
    warnings: tuple[str, ...]
    ideas: tuple[Idea, ...]


def default_repo_root() -> Path:
    """Return the repository root from this module path."""
    return Path(__file__).resolve().parents[3]


def run_once(
    repo_root: Path | None = None,
    out_dir: Path | None = None,
    limit: int = 20,
    include_low_confidence: bool = False,
    use_state: bool = True,
) -> HopperReport:
    """Collect local evidence, generate ideas, rank them, and write reports."""
    root = (repo_root or default_repo_root()).resolve()
    output_dir = out_dir or (root / "run" / "idea_hopper")
    context = _collect_context(root)
    ideas: list[Idea] = []
    if not include_low_confidence:
        ideas = [idea for idea in ideas if idea.priority != "low"]
    ideas = sorted(ideas, key=lambda item: item.priority_score, reverse=True)[:limit]
    report = HopperReport(
        generated_utc=datetime.now(UTC).isoformat(),
        inputs_read=tuple(context["inputs_read"]),
        warnings=tuple(context["warnings"]),
        ideas=tuple(ideas),
    )
    _write_outputs(report, output_dir, use_state=use_state)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate a ranked JED idea queue.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--include-low-confidence", action="store_true")
    parser.add_argument("--no-state", action="store_true")
    args = parser.parse_args(argv)
    report = run_once(
        out_dir=args.out_dir,
        limit=args.limit,
        include_low_confidence=args.include_low_confidence,
        use_state=not args.no_state,
    )
    print(f"wrote {len(report.ideas)} idea(s)")
    return 0


def _collect_context(root: Path) -> dict[str, list[str]]:
    """Collect only required-path existence/warning data for the skeleton task."""
    required = (
        "docs/strategy.md",
        "src/jed_attack/campaign/assemble.py",
        "src/jed_attack/campaign/prompts.toml",
        "src/jed_attack/campaign/config.py",
    )
    inputs_read: list[str] = []
    warnings: list[str] = []
    for relative in required:
        path = root / relative
        if path.exists():
            inputs_read.append(relative)
        else:
            warnings.append(f"missing primary input: {relative}")
    optional = (
        "run/blackboard.jsonl",
        "run/artifact_eval_current/latest_metrics.json",
        "run/kaggle_research_cron/latest_public_kernel_mining.md",
    )
    for relative in optional:
        if not (root / relative).exists():
            warnings.append(f"missing optional input: {relative}")
    return {"inputs_read": inputs_read, "warnings": warnings}


def _write_outputs(
    report: HopperReport,
    out_dir: Path,
    *,
    use_state: bool,
) -> None:
    """Write markdown, JSON, queue history, and state files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _report_payload(report)
    (out_dir / "latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "latest.md").write_text(_render_markdown(report), encoding="utf-8")
    with (out_dir / "queue.jsonl").open("a", encoding="utf-8") as handle:
        for idea in report.ideas:
            handle.write(json.dumps(_idea_payload(idea), sort_keys=True) + "\n")
    if use_state:
        fingerprints = [idea.fingerprint for idea in report.ideas if idea.fingerprint]
        (out_dir / "state.json").write_text(
            json.dumps({"fingerprints": fingerprints}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    elif not (out_dir / "state.json").exists():
        (out_dir / "state.json").write_text(
            json.dumps({"fingerprints": []}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _report_payload(report: HopperReport) -> dict[str, Any]:
    return {
        "generated_utc": report.generated_utc,
        "inputs_read": list(report.inputs_read),
        "warnings": list(report.warnings),
        "ideas": [_idea_payload(idea) for idea in report.ideas],
    }


def _idea_payload(idea: Idea) -> dict[str, Any]:
    payload = asdict(idea)
    payload["source_evidence"] = [asdict(item) for item in idea.source_evidence]
    return payload


def _render_markdown(report: HopperReport) -> str:
    lines = [
        "# Idea Hopper Report",
        "",
        f"Generated: {report.generated_utc}",
        f"Inputs read: {', '.join(report.inputs_read) or '-'}",
        f"Warnings: {len(report.warnings)}",
        "",
        "## Top ideas",
        "",
    ]
    if not report.ideas:
        lines.append("No ideas generated from the available local evidence.")
    for index, idea in enumerate(report.ideas, start=1):
        lines.extend(
            [
                f"{index}. **{idea.title}**",
                f"   - Category: `{idea.category}`",
                f"   - Priority: `{idea.priority}` ({idea.priority_score:.2f})",
                f"   - Gate: {idea.acceptance_gate}",
            ]
        )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
