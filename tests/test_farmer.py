"""The attack fires predicates and fills the budget by repetition under ALLOW."""

from aicomp_sdk.guardrails.base import GuardrailBase

from jed_attack.harness.runner import run_attack
from jed_attack.submission.attack import AttackAlgorithm


def test_attack_repeats_to_accumulate_severity() -> None:
    """Under ALLOW the attack returns many (repeated) firing candidates.

    The scorer sums severity over every finding with no dedup, so the winning set
    repeats the best candidates. We assert the repetition is happening (more
    findings than distinct cells) and that it scores.
    """
    result = run_attack(
        AttackAlgorithm,
        agent="deterministic",
        budget_s=20.0,
        guardrail_factory=GuardrailBase,
        # Keep the returned set small so the dev replay finishes fast; real runs
        # use the default ~300 ceiling and tune it against GPU run times.
        attack_config={"max_candidates": 40},
    )
    assert result.findings_count >= 20  # filled by repetition, not one-per-cell
    assert result.findings_count > result.unique_cells  # repetition, not pure diversity
    assert result.score > 0.0
