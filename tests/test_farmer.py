"""The injection farmer fires predicates and mints distinct cells under ALLOW."""

from aicomp_sdk.guardrails.base import GuardrailBase

from jed_attack.harness.runner import run_attack
from jed_attack.submission.attack import AttackAlgorithm


def test_farmer_fires_and_diversifies() -> None:
    """Under the ALLOW guardrail the farmer yields several distinct scoring cells."""
    result = run_attack(
        AttackAlgorithm,
        agent="deterministic",
        budget_s=20.0,
        guardrail_factory=GuardrailBase,
    )
    assert result.findings_count >= 3
    assert result.unique_cells >= 3
    assert result.score > 0.0
