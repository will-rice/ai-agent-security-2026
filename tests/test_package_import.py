"""Package import smoke tests."""


def test_jed_attack_imports() -> None:
    """The package and its live harness subpackage import cleanly."""
    import jed_attack
    import jed_attack.harness

    assert jed_attack.__version__


def test_aicomp_sdk_available() -> None:
    """The vendored competition SDK is importable with its key entry points."""
    from aicomp_sdk.attacks.contracts import AttackAlgorithmBase, AttackCandidate
    from aicomp_sdk.evaluation.runner import evaluate_redteam

    assert callable(evaluate_redteam)
    assert issubclass(AttackAlgorithmBase, object)
    assert AttackCandidate.from_messages(["hi"]).user_messages == ("hi",)
