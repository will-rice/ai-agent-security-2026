"""Guards that the submission depends only on aicomp_sdk + stdlib."""

import ast
import sys
from pathlib import Path

SUBMISSION = Path(__file__).resolve().parents[1] / "src/jed_attack/submission/attack.py"


def test_attack_defines_algorithm() -> None:
    """attack.py defines an AttackAlgorithm subclassing the SDK base."""
    from aicomp_sdk.attacks.contracts import AttackAlgorithmBase

    from jed_attack.submission.attack import AttackAlgorithm

    assert issubclass(AttackAlgorithm, AttackAlgorithmBase)


def test_attack_imports_only_allowed_roots() -> None:
    """Static check: attack.py imports only stdlib or aicomp_sdk, never jed_attack."""
    tree = ast.parse(SUBMISSION.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert "jed_attack" not in roots, "submission must not import the harness package"
    stdlib = set(sys.stdlib_module_names)
    for root in roots:
        assert root == "aicomp_sdk" or root in stdlib, f"disallowed import root: {root}"
