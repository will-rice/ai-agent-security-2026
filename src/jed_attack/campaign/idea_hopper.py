"""Local evidence-to-experiment queue for the JED campaign."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


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


@dataclass(frozen=True)
class ArtifactMetricRecord:
    """One artifact sweep's public-score and candidate-count metrics."""

    path: str
    template: str
    lb_est_public: float
    candidate_count: float


@dataclass(frozen=True)
class CodeSurface:
    """The generated-artifact knobs currently visible in campaign source."""

    replay_safe_frac: float | None
    template_names: tuple[str, ...]
    source_path: str


@dataclass(frozen=True)
class BlackboardSignal:
    """One historical model result from the local campaign blackboard."""

    model: str
    public: float
    objective: float
    fires: bool
    text: str
    path: str


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
    ideas = _generate_ideas(context)
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


def _collect_context(root: Path) -> dict[str, Any]:
    """Collect local code, artifact, and public-kernel evidence."""
    inputs_read: list[str] = []
    warnings: list[str] = []
    _read_primary_paths(root, inputs_read, warnings)
    return {
        "inputs_read": inputs_read,
        "warnings": warnings,
        "code_surface": _read_code_surface(root, inputs_read, warnings),
        "artifact_metrics": _read_artifact_metrics(root, inputs_read, warnings),
        "kernel_findings": _read_kernel_findings(root, inputs_read, warnings),
        "blackboard_signals": _read_blackboard(root, inputs_read, warnings),
        "discussion_signals": _read_discussions(root, inputs_read, warnings),
    }


def _read_primary_paths(
    root: Path,
    inputs_read: list[str],
    warnings: list[str],
) -> None:
    for relative in (
        "docs/strategy.md",
        "src/jed_attack/campaign/prompts.toml",
        "src/jed_attack/campaign/config.py",
    ):
        path = root / relative
        if path.exists():
            inputs_read.append(relative)
        else:
            warnings.append(f"missing primary input: {relative}")


def _read_text(
    path: Path,
    relative: str,
    inputs_read: list[str],
    warnings: list[str],
) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        warnings.append(f"missing optional input: {relative}")
        return ""
    inputs_read.append(relative)
    return text


def _read_code_surface(
    root: Path,
    inputs_read: list[str],
    warnings: list[str],
) -> CodeSurface:
    relative = "src/jed_attack/campaign/assemble.py"
    text = _read_text(root / relative, relative, inputs_read, warnings)
    replay_match = re.search(r"_REPLAY_SAFE_FRAC\s*=\s*([0-9.]+)", text)
    replay_safe_frac = float(replay_match.group(1)) if replay_match else None
    template_names = tuple(re.findall(r'\("([^"]+)",\s*"[^"]*"\)', text))
    return CodeSurface(
        replay_safe_frac=replay_safe_frac,
        template_names=template_names,
        source_path=relative,
    )


def _read_artifact_metrics(
    root: Path,
    inputs_read: list[str],
    warnings: list[str],
) -> list[ArtifactMetricRecord]:
    records: list[ArtifactMetricRecord] = []
    for path in sorted((root / "run" / "artifact_sweeps").glob("**/metrics.json")):
        relative = path.relative_to(root).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warnings.append(f"malformed artifact metrics: {relative}")
            continue
        inputs_read.append(relative)
        if not isinstance(data, Mapping):
            warnings.append(f"malformed artifact metrics: {relative}")
            continue
        try:
            lb_est_public = float(data.get("artifact_lb_est_public") or 0.0)
            candidate_count = float(data.get("artifact_candidate_count_mean") or 0.0)
        except (AttributeError, TypeError, ValueError):
            warnings.append(f"malformed artifact metrics: {relative}")
            continue
        records.append(
            ArtifactMetricRecord(
                path=relative,
                template=path.parent.name,
                lb_est_public=lb_est_public,
                candidate_count=candidate_count,
            )
        )
    return records


def _read_kernel_findings(
    root: Path,
    inputs_read: list[str],
    warnings: list[str],
) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    base = root / "run" / "kaggle_research_cron"
    for path in sorted(base.glob("public_kernel_latest_mining_*/decoded_findings.md")):
        relative = path.relative_to(root).as_posix()
        text = _read_text(path, relative, inputs_read, warnings)
        if text:
            findings.append((relative, text))
    if not findings:
        warnings.append(
            "missing optional input: "
            "run/kaggle_research_cron/public_kernel_latest_mining_*/decoded_findings.md"
        )
    return findings


def _read_blackboard(
    root: Path,
    inputs_read: list[str],
    warnings: list[str],
) -> list[BlackboardSignal]:
    """Read local blackboard rows into the fields relevant to idea generation."""
    relative = "run/blackboard.jsonl"
    path = root / relative
    signals: list[BlackboardSignal] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        warnings.append(f"missing optional input: {relative}")
        return signals
    inputs_read.append(relative)
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"malformed blackboard row: {relative}:{line_no}")
            continue
        if not isinstance(row, Mapping):
            warnings.append(f"malformed blackboard row: {relative}:{line_no}")
            continue
        messages = row.get("messages") or []
        first_text = ""
        if messages and isinstance(messages, list) and isinstance(messages[0], Mapping):
            first_text = str(messages[0].get("text") or "")
        try:
            public = float(row.get("public") or 0.0)
            objective = float(row.get("objective") or 0.0)
        except (TypeError, ValueError):
            warnings.append(f"malformed blackboard row: {relative}:{line_no}")
            continue
        signals.append(
            BlackboardSignal(
                model=str(row.get("model") or ""),
                public=public,
                objective=objective,
                fires=bool(row.get("fires")),
                text=first_text,
                path=f"{relative}:{line_no}",
            )
        )
    return signals


def _read_discussions(
    root: Path,
    inputs_read: list[str],
    warnings: list[str],
) -> list[tuple[str, str]]:
    """Read cached Kaggle discussions without fetching anything remotely."""
    base = root / "run" / "kaggle_research_cron"
    signals: list[tuple[str, str]] = []
    for path in sorted(base.glob("latest_*_discussions.json")):
        relative = path.relative_to(root).as_posix()
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warnings.append(f"malformed discussion cache: {relative}")
            continue
        inputs_read.append(relative)
        if isinstance(rows, list):
            for row in rows[:20]:
                if isinstance(row, Mapping):
                    text = f"{row.get('title', '')}\n{row.get('body_markdown', '')}"
                    signals.append((relative, text))
    if not signals:
        warnings.append(
            "missing optional input: "
            "run/kaggle_research_cron/latest_*_discussions.json"
        )
    return signals


def _generate_ideas(context: dict[str, Any]) -> list[Idea]:
    ideas = _public_kernel_delta_ideas(context)
    ideas.extend(_blackboard_ideas(context))
    ideas.extend(_discussion_ideas(context))
    return [_with_fingerprint(idea) for idea in ideas]


def _public_kernel_delta_ideas(context: dict[str, Any]) -> list[Idea]:
    code = cast(CodeSurface, context["code_surface"])
    findings = cast(list[tuple[str, str]], context["kernel_findings"])
    ideas: list[Idea] = []
    for path, text in findings:
        if "REPLAY_SAFE_FRAC=0.97" in text and code.replay_safe_frac != 0.97:
            ideas.append(
                Idea(
                    id="replay-safe-frac-097",
                    title="Test REPLAY_SAFE_FRAC=0.97 in generated artifact",
                    hypothesis=(
                        "Top public latency-split kernels use 0.97; the change may "
                        "reduce timeout risk versus the current generated setting."
                    ),
                    source_evidence=(
                        SourceEvidence(
                            path,
                            "Decoded public kernels report REPLAY_SAFE_FRAC=0.97.",
                        ),
                    ),
                    likely_files=(
                        "src/jed_attack/campaign/assemble.py",
                        "tests/test_campaign.py",
                    ),
                    category="replay_sizing",
                    priority="medium",
                    priority_score=5.5,
                    expected_upside=(
                        "Improve calibrated LB estimate by avoiding replay "
                        "timeout/overpacking."
                    ),
                    risk=(
                        "Can underfill and lower public score if current 0.99 is safe."
                    ),
                    acceptance_gate=(
                        "artifact_lb_est_public beats current build_next by >= 0.25 "
                        "and campaign tests pass."
                    ),
                )
            )
        if "measured latency split" in text.lower():
            ideas.append(
                Idea(
                    id="measured-latency-split",
                    title="Test measured latency split without model hints",
                    hypothesis=(
                        "Public leaders classify fast/slow behavior from observed "
                        "latency "
                        "rather than relying on exposed model hints."
                    ),
                    source_evidence=(
                        SourceEvidence(
                            path,
                            "Decoded findings recommend testing measured latency "
                            "split.",
                        ),
                    ),
                    likely_files=(
                        "src/jed_attack/campaign/assemble.py",
                        "tests/test_campaign.py",
                    ),
                    category="template_selection",
                    priority="medium",
                    priority_score=5.0,
                    expected_upside=(
                        "May improve generalization if model hints are absent or "
                        "stale in "
                        "the evaluator."
                    ),
                    risk=(
                        "Can choose the wrong template if warmup/probe latency "
                        "is noisy."
                    ),
                    acceptance_gate=(
                        "artifact_lb_est_public beats current build_next by >= 0.25 "
                        "and "
                        "selected-template telemetry remains stable on both models."
                    ),
                )
            )
    return ideas


def _blackboard_ideas(context: dict[str, Any]) -> list[Idea]:
    """Identify message families that behave differently across target models."""
    signals = cast(list[BlackboardSignal], context["blackboard_signals"])
    by_text: dict[str, list[BlackboardSignal]] = {}
    for signal in signals:
        if signal.text:
            by_text.setdefault(signal.text, []).append(signal)
    for rows in by_text.values():
        fired = {row.model for row in rows if row.fires and row.public > 0.0}
        failed = {row.model for row in rows if not row.fires or row.public <= 0.0}
        if fired and failed:
            evidence = tuple(
                SourceEvidence(
                    row.path,
                    f"{row.model}: public={row.public}, fires={row.fires}",
                )
                for row in rows[:4]
            )
            return [
                Idea(
                    id="blackboard-model-asymmetry",
                    title="Investigate model-asymmetric firing in blackboard history",
                    hypothesis=(
                        "A message family fires on one target model but not another; "
                        "a template split or portfolio hedge may improve both-model "
                        "score."
                    ),
                    source_evidence=evidence,
                    likely_files=(
                        "src/jed_attack/campaign/prompts.toml",
                        "src/jed_attack/campaign/assemble.py",
                    ),
                    category="near_miss",
                    priority="medium",
                    priority_score=5.25,
                    expected_upside=(
                        "Recover score lost to one-model failures without expanding "
                        "replay cost much."
                    ),
                    risk=(
                        "May overfit noisy historical rows if the asymmetry came from "
                        "transient latency."
                    ),
                    acceptance_gate=(
                        "New candidate family improves both-model public raw/sec or "
                        "artifact_lb_est_public by >= 0.25."
                    ),
                )
            ]
    return []


def _discussion_ideas(context: dict[str, Any]) -> list[Idea]:
    """Flag cached evaluator/scoring updates that require local validation."""
    discussions = cast(list[tuple[str, str]], context["discussion_signals"])
    for path, text in discussions:
        lowered = text.lower()
        if (
            "runtime enforcement" in lowered
            or "static replay" in lowered
            or "secret-exfiltration scoring" in lowered
        ):
            return [
                Idea(
                    id="discussion-evaluator-risk-review",
                    title="Review cached evaluator/scoring discussion changes",
                    hypothesis=(
                        "Cached Kaggle discussion mentions evaluator/runtime/scoring "
                        "changes; our constants and scorer calibration may need "
                        "validation."
                    ),
                    source_evidence=(
                        SourceEvidence(
                            path,
                            "Discussion cache mentions runtime, static replay, or "
                            "secret-exfiltration scoring changes.",
                        ),
                    ),
                    likely_files=(
                        "docs/strategy.md",
                        "src/jed_attack/campaign/config.py",
                    ),
                    category="evaluator_risk",
                    priority="medium",
                    priority_score=4.75,
                    expected_upside=(
                        "Avoid optimizing against stale evaluator assumptions."
                    ),
                    risk=(
                        "May produce only documentation or calibration work, not a "
                        "direct score increase."
                    ),
                    acceptance_gate=(
                        "Confirm docs/config still match cached FAQ; if not, update "
                        "strategy and rerun artifact calibration."
                    ),
                )
            ]
    return []


def _with_fingerprint(idea: Idea) -> Idea:
    fingerprint = _fingerprint(idea)
    return Idea(
        id=idea.id,
        title=idea.title,
        hypothesis=idea.hypothesis,
        source_evidence=idea.source_evidence,
        likely_files=idea.likely_files,
        category=idea.category,
        priority=idea.priority,
        priority_score=idea.priority_score,
        expected_upside=idea.expected_upside,
        risk=idea.risk,
        acceptance_gate=idea.acceptance_gate,
        status=idea.status,
        fingerprint=fingerprint,
    )


def _fingerprint(idea: Idea) -> str:
    text = "|".join(
        [
            idea.title.lower().strip(),
            idea.category,
            ",".join(sorted(idea.likely_files)),
        ]
    )
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


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
