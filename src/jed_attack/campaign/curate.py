"""Novelty-aware pool curation: build the shipped attack pool from scored candidates.

Selection = three objectives: the FAITHFUL replay score is the firing floor (only
candidates that fire are eligible), the SEVERITY judge ranks eligible candidates (a
private-LB proxy), and the NOVELTY judge gates admission (a candidate joins only if it
adds diversity vs the pool-so-far) -- so the shipped pool is high-quality AND diverse,
not 30 near-identical exfils. ``select_pool`` takes a passed-in candidate collection so
the same shape works when the proposer later returns ``list[Submission]``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jed_attack.campaign import assemble, config
from jed_attack.campaign.blackboard import Blackboard
from jed_attack.campaign.judge import (
    NoveltyScore,
    SeverityScore,
    judge_novelty,
    judge_severity,
)
from jed_attack.campaign.submission import Message


@dataclass
class Candidate:
    """One selection unit: its message(s), a display text, and whether it fires."""

    messages: list[Message]
    text: str
    fires: bool
    feedback: list[str] = field(default_factory=list)


def select_pool(
    candidates: list[Candidate],
    severity_fn: Callable[[Candidate], SeverityScore],
    novelty_fn: Callable[[Candidate, list[str]], NoveltyScore],
    threshold: float,
    cap: int,
) -> list[Candidate]:
    """Curate a diverse, high-quality pool from ``candidates``.

    Eligible = fires (replay floor). Rank eligible by ``severity_fn`` (private-LB proxy),
    then greedily admit in that order, skipping any whose ``novelty_fn`` vs the
    pool-so-far is below ``threshold``, until ``cap`` are admitted.

    Args:
        candidates: The candidate pool to select from.
        severity_fn: Scores a candidate's elicited severity (private-LB proxy).
        novelty_fn: Scores a candidate's novelty against the pool-so-far's texts.
        threshold: Minimum novelty score required to admit a candidate.
        cap: Maximum number of candidates to admit.

    Returns:
        The admitted candidates, in admission order.
    """
    eligible = [c for c in candidates if c.fires]
    ranked = sorted(eligible, key=lambda c: severity_fn(c).score, reverse=True)
    pool: list[Candidate] = []
    for cand in ranked:
        if len(pool) >= cap:
            break
        if novelty_fn(cand, [p.text for p in pool]).score >= threshold:
            pool.append(cand)
    return pool


def curate_from_blackboard(
    board: Blackboard, out_dir: Path, run: Any = None
) -> list[Candidate]:
    """Build + ship a curated pool from the blackboard's firing candidates.

    Extracts one Candidate per unique message across the blackboard's records (fires =
    its replay severity > 0), runs :func:`select_pool` with the real judges, ships the
    pool via :func:`assemble.build`, and logs pool metrics to wandb.

    Args:
        board: The team's shared blackboard to curate from.
        out_dir: Output directory for the assembled submission.
        run: An optional wandb run to log pool metrics to.

    Returns:
        The curated pool of candidates.
    """
    candidates = _blackboard_candidates(board)
    pool = select_pool(
        candidates,
        judge_severity_of,
        judge_novelty_of,
        config.NOVELTY_ADMIT_THRESHOLD,
        config.MAX_SHIP_MESSAGES,
    )
    assemble.build([c.text for c in pool], out_dir)
    if run is not None:
        _log_pool(run, pool)
    return pool


def judge_severity_of(cand: Candidate) -> SeverityScore:
    """Adapt a Candidate to the severity judge (its messages + feedback).

    Args:
        cand: The candidate to score.

    Returns:
        The severity judge's verdict.
    """
    return judge_severity(cand.messages, cand.feedback)


def judge_novelty_of(cand: Candidate, pool_sample: list[str]) -> NoveltyScore:
    """Adapt a Candidate to the novelty judge (its messages vs the pool sample).

    Args:
        cand: The candidate to score.
        pool_sample: The current pool's message texts to compare against.

    Returns:
        The novelty judge's verdict.
    """
    return judge_novelty(cand.messages, pool_sample)


def _blackboard_candidates(board: Blackboard) -> list[Candidate]:
    """Build one Candidate per unique message across records.

    A candidate fires if its recorded replay severity (summed across guardrails) is
    greater than zero.

    Args:
        board: The blackboard to extract candidates from.

    Returns:
        One Candidate per unique message text, in first-seen order.
    """
    seen: set[str] = set()
    out: list[Candidate] = []
    for record in board._records:
        for msg, fb in zip(record.messages, record.feedback, strict=True):
            text = msg["text"]
            if text in seen:
                continue
            seen.add(text)
            severity = fb.get("severity", {})
            fires = bool(severity) and max(severity.values(), default=0.0) > 0.0
            out.append(
                Candidate(
                    messages=[Message.model_validate(msg)],
                    text=text,
                    fires=fires,
                    feedback=[fb.get("feedback", "")],
                )
            )
    return out


def _log_pool(run: Any, pool: list[Candidate]) -> None:
    """Log pool size to wandb (per-candidate judge scores are logged during selection).

    Args:
        run: The wandb run to log to.
        pool: The curated pool.
    """
    run.log({"pool_size": len(pool)})
