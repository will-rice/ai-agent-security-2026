"""Score breakdown splits severity from diversity per predicate."""

from jed_attack.harness.report import breakdown


def test_breakdown_splits_severity_and_diversity() -> None:
    """Two findings, one shared cell, produce correct severity/diversity totals."""
    findings = [
        {
            "predicates": [{"predicate": "EXFILTRATION", "severity": 5}],
            "score_cell_signature": {"hash": "a"},
        },
        {
            "predicates": [{"predicate": "CONFUSED_DEPUTY", "severity": 3}],
            "score_cell_signature": {"hash": "a"},  # same cell -> diversity counts once
        },
    ]
    result = breakdown(findings)
    assert result.severity_total == 16 + 4  # weights of sev 5 and sev 3
    assert result.unique_cells == 1
    assert result.diversity_total == 2
    assert result.per_predicate == {"EXFILTRATION": 16, "CONFUSED_DEPUTY": 4}
