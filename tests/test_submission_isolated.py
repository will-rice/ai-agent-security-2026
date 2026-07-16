"""Guards that the submission depends only on aicomp_sdk + stdlib."""

import ast
import importlib
import sys

import pytest


def test_attack_defines_algorithm() -> None:
    """attack.py defines an AttackAlgorithm subclassing the SDK base."""
    from aicomp_sdk.attacks.contracts import AttackAlgorithmBase

    from jed_attack.submission.attack import AttackAlgorithm

    assert issubclass(AttackAlgorithm, AttackAlgorithmBase)


def test_attack_imports_only_allowed_roots() -> None:
    """Static check: the BUILT submission imports only stdlib or aicomp_sdk.

    The dev-time ``attack.py`` imports the ``jed_attack.submission.templates``
    sibling, which the builder (Task 10) inlines into a single self-contained
    file. Until the builder exists, this check is skipped.
    """
    pytest.importorskip("jed_attack.scripts.build_submission")
    builder = importlib.import_module("jed_attack.scripts.build_submission")

    source = builder.build_attack_source()
    tree = ast.parse(source)
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
