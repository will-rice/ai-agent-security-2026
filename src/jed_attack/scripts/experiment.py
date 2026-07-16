"""Run the attack against the served models and record scores — Codex's inner loop.

Scores the submission against gpt_oss and/or gemma via running ``llama-server``
endpoints (see ``docs/serving.md``) under the public guardrail (the leaderboard
scorer), writes a ``submission.csv``-style summary, and appends each run to the
JSONL archive for comparison across experiments.
"""

import argparse
import logging
from pathlib import Path

from jed_attack.harness.models import llama_server_agent_factory, resolve_base_url
from jed_attack.harness.report import breakdown, save_run
from jed_attack.harness.runner import RunResult, run_attack
from jed_attack.submission.attack import AttackAlgorithm

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("jed_attack.experiment")

# Submission Id prefix per model key (Kaggle uses "gemma", not "gemma_4").
_ID_PREFIX = {"gpt_oss": "gpt_oss", "gemma_4": "gemma"}


def _run_one(
    model_key: str,
    *,
    base_url: str | None,
    budget_s: float,
    max_candidates: int | None,
    archive_dir: Path,
) -> RunResult:
    """Score the attack against one served model under the public guardrail.

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.
        base_url: Explicit server base URL, or None to resolve from env.
        budget_s: Wall-clock budget for generation and (separately) replay.
        max_candidates: Optional override of the returned-candidate ceiling.
        archive_dir: Directory holding ``runs.jsonl``.

    Returns:
        The ``RunResult`` for this model.
    """
    url = resolve_base_url(model_key, base_url)
    _log.info("[%s] using server %s", model_key, url)
    factory = llama_server_agent_factory(model_key, url)
    attack_config = {} if max_candidates is None else {"max_candidates": max_candidates}

    result = run_attack(
        AttackAlgorithm,
        agent_factory=factory,
        agent_label=model_key,
        budget_s=budget_s,
        attack_config=attack_config,
    )
    report = breakdown(result.findings)
    save_run(
        result,
        report,
        archive_dir=archive_dir,
        extra={"budget_s": budget_s, **attack_config},
    )
    _log.info(
        "[%s] score=%.3f findings=%d cells=%d sev=%d div=%d %s",
        model_key,
        result.score,
        result.findings_count,
        result.unique_cells,
        report.severity_total,
        report.diversity_total,
        report.per_predicate,
    )
    return result


def main() -> None:
    """Run experiments against the served models and write a submission summary."""
    parser = argparse.ArgumentParser(description="JED attack experiment runner")
    parser.add_argument(
        "--model", choices=["both", "gpt_oss", "gemma_4"], default="both"
    )
    parser.add_argument("--budget-s", type=float, default=900.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--gpt-oss-url", default=None)
    parser.add_argument("--gemma-url", default=None)
    parser.add_argument("--archive-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/submission.csv"))
    args = parser.parse_args()

    keys = ["gpt_oss", "gemma_4"] if args.model == "both" else [args.model]
    urls = {"gpt_oss": args.gpt_oss_url, "gemma_4": args.gemma_url}
    rows: list[tuple[str, float]] = []
    for model_key in keys:
        result = _run_one(
            model_key,
            base_url=urls[model_key],
            budget_s=args.budget_s,
            max_candidates=args.max_candidates,
            archive_dir=args.archive_dir,
        )
        rows.append((f"{_ID_PREFIX[model_key]}_public", result.score))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Id,Score", *(f"{name},{score:.4f}" for name, score in rows)]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log.info("wrote %s (public scores only; private guardrail is hidden)", args.out)


if __name__ == "__main__":
    main()
