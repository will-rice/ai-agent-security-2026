"""Offline study: does the Qwen judge correlate with the faithful public score?

Reads the live team blackboard's already-faithfully-scored submissions (post KV-fix,
so each record's ``public`` is a trustworthy label and its per-message feedback is the
observed effect the judge needs), judges each with the surrogate-guardrail judge, and
reports Spearman rank correlation of judge score vs faithful public. Writes report.json
+ pairs.csv (+ scatter.png if matplotlib is present) under config.JUDGE_STUDY_DIR.

NO re-scoring: the study consumes the labels the live optimizer already produced, so it
runs ALONGSIDE the optimizer with no GPU contention -- the judge runs on ollama/Qwen
(Ada), the optimizer keeps its GGUF scorer. Per-model correlation is not reported: the
blackboard persists only the aggregate ``public``, not per-model boards.
"""

import argparse
import csv
import json
import logging

from scipy.stats import spearmanr

from jed_attack.campaign import config
from jed_attack.campaign.judge import judge_submission
from jed_attack.campaign.submission import Message

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("judge_correlation")


def main() -> None:
    """Judge each blackboard submission and report judge-vs-public correlation."""
    parser = argparse.ArgumentParser(description="Qwen judge correlation study")
    parser.add_argument("--blackboard", default=str(config.BLACKBOARD_LOG))
    parser.add_argument("--n", type=int, default=config.JUDGE_STUDY_N)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.blackboard)]
    sample = rows if len(rows) <= args.n else stratified_sample(rows, args.n)
    _log.info("judging %d of %d blackboard submissions", len(sample), len(rows))

    results = []
    for i, row in enumerate(sample):
        messages = [Message(**m) for m in row["messages"]]
        feedback = [entry["feedback"] for entry in row["feedback"]]
        verdict = judge_submission(messages, feedback)
        public = float(row["public"])
        results.append(
            {
                "faithful_public": public,
                "judge_score": verdict.score,
                "judge_rationale": verdict.rationale,
                "n_messages": len(messages),
            }
        )
        _log.info("[%d/%d] public=%.3f judge=%d", i + 1, len(sample), public, verdict.score)

    judge_scores: list[int] = [int(r["judge_score"]) for r in results]
    public_scores: list[float] = [float(r["faithful_public"]) for r in results]
    rho, pvalue = spearmanr(judge_scores, public_scores)

    config.JUDGE_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "n": len(results),
        "spearman_rho_public": float(rho),
        "spearman_pvalue": float(pvalue),
    }
    (config.JUDGE_STUDY_DIR / "report.json").write_text(json.dumps(report, indent=2))
    with (config.JUDGE_STUDY_DIR / "pairs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    _write_scatter(judge_scores, public_scores, float(rho))
    _log.info("REPORT %s", json.dumps(report, indent=2))


def stratified_sample(rows: list[dict], n: int) -> list[dict]:
    """Sample ``n`` submissions spread across the faithful-public range.

    Used only when the blackboard has more than ``n`` records; spreads the sample so the
    correlation is not dominated by whichever score band happens to be most populated.
    Includes up to 5 zero-public rows as a low anchor; the rest are drawn evenly across
    the sorted non-zero range (the maximum is always retained).

    Args:
        rows: Parsed blackboard records (each has ``public``, ``messages``, ``feedback``).
        n: Target sample size.

    Returns:
        Up to ``n`` records (fewer only if the blackboard is smaller than ``n``).
    """
    zeros = [r for r in rows if r["public"] == 0.0][:5]
    nonzero = sorted((r for r in rows if r["public"] > 0.0), key=lambda r: r["public"])
    want = max(1, n - len(zeros))
    step = max(1, len(nonzero) // want)
    picked = nonzero[::step][:want]
    # Ensure the maximum value is included if the stride/truncation dropped it.
    if picked and nonzero and picked[-1]["public"] < nonzero[-1]["public"]:
        picked = picked[:-1] + [nonzero[-1]]
    return picked + zeros


def _write_scatter(judge: list[int], public: list[float], rho: float) -> None:
    """Write a judge-vs-public scatter PNG if matplotlib is available (optional)."""
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")  # headless backend BEFORE importing pyplot
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        _log.info("matplotlib absent; skipping scatter (pairs.csv has the data)")
        return
    plt.figure()
    plt.scatter(public, judge)
    plt.xlabel("faithful public score")
    plt.ylabel("judge score")
    plt.title(f"judge vs public (Spearman rho={rho:.2f})")
    plt.savefig(config.JUDGE_STUDY_DIR / "scatter.png", dpi=120)


if __name__ == "__main__":
    main()
