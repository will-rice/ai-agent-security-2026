"""Offline study: does the Qwen judge correlate with the faithful public score?

Samples archived submissions, RE-SCORES them with the fixed in-process scorer (the
archived publics are KV-polluted -- used ONLY to stratify, never as labels), judges
each with the surrogate-guardrail judge, and reports Spearman rank correlation of
judge score vs faithful public (and vs each per-model board). Writes a JSON report +
CSV (and a scatter PNG if matplotlib is present) under config.JUDGE_STUDY_DIR.

Green-only (loads the resident GGUF scorer): run with `uv run --no-sync` and with the
live optimizer STOPPED (a second gpt_oss load OOMs the 3090). See the plan's Task 3.
"""

import argparse
import csv
import json
import logging

from scipy.stats import spearmanr

from jed_attack.campaign import config
from jed_attack.campaign.judge import judge_submission
from jed_attack.campaign.submission import Message
from jed_attack.campaign.submission_score import score_submission

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("judge_correlation")


def main() -> None:
    """Run the correlation study and write its report."""
    parser = argparse.ArgumentParser(description="Qwen judge correlation study")
    parser.add_argument(
        "--archive", default=str(config.BLACKBOARD_LOG) + ".pre-kvfix.bak"
    )
    parser.add_argument("--n", type=int, default=config.JUDGE_STUDY_N)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.archive)]
    sample = stratified_sample(rows, args.n)
    _log.info("sampled %d of %d archived submissions", len(sample), len(rows))

    results = []
    for i, row in enumerate(sample):
        messages = [Message(**m) for m in row["messages"]]
        scored = score_submission(messages)  # faithful labels
        verdict = judge_submission(
            messages, [ms.feedback for ms in scored.per_message]
        )
        results.append(
            {
                "faithful_public": scored.public,
                **{f"public_{m}": v for m, v in scored.public_by_model.items()},
                "judge_score": verdict.score,
                "judge_rationale": verdict.rationale,
                "n_messages": len(messages),
            }
        )
        _log.info(
            "[%d/%d] public=%.3f judge=%d",
            i + 1,
            len(sample),
            scored.public,
            verdict.score,
        )

    judge_scores: list[int] = [int(r["judge_score"]) for r in results]
    public_scores: list[float] = [float(r["faithful_public"]) for r in results]
    rho, pvalue = spearmanr(judge_scores, public_scores)

    config.JUDGE_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    model_rhos = {
        m: float(spearmanr(judge_scores, [float(r[f"public_{m}"]) for r in results])[0])
        for m in config.MODELS
    }
    report = {
        "n": len(results),
        "spearman_rho_public": float(rho),
        "spearman_pvalue": float(pvalue),
        "spearman_rho_by_model": model_rhos,
    }
    (config.JUDGE_STUDY_DIR / "report.json").write_text(json.dumps(report, indent=2))
    with (config.JUDGE_STUDY_DIR / "pairs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    _write_scatter(judge_scores, public_scores, float(rho))
    _log.info("REPORT %s", json.dumps(report, indent=2))


def stratified_sample(rows: list[dict], n: int) -> list[dict]:
    """Sample ``n`` archived submissions spread across the archived-public range.

    The archived public is KV-polluted, so it is used ONLY to spread the sample (get
    weak/medium/strong attacks), never as a label. Includes up to 5 zero-public rows as
    a low anchor; the rest are drawn evenly across the sorted non-zero range.

    Args:
        rows: Parsed archived blackboard records (each has ``public``, ``messages``).
        n: Target sample size.

    Returns:
        Up to ``n`` records (fewer only if the archive is smaller than ``n``).
    """
    zeros = [r for r in rows if r["public"] == 0.0][:5]
    nonzero = sorted((r for r in rows if r["public"] > 0.0), key=lambda r: r["public"])
    want = max(1, n - len(zeros))
    step = max(1, len(nonzero) // want)
    picked = nonzero[::step][:want]
    # Ensure the maximum value is included if not already present
    if picked and nonzero and picked[-1]["public"] < nonzero[-1]["public"]:
        picked = picked[:-1] + [nonzero[-1]]
    return picked + zeros


def _write_scatter(judge: list[int], public: list[float], rho: float) -> None:
    """Write a judge-vs-public scatter PNG if matplotlib is available (optional)."""
    try:
        import matplotlib  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        matplotlib.use("Agg")
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
