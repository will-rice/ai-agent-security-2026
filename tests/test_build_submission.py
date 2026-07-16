"""The built submission is a single file importing only aicomp_sdk + stdlib."""

import ast
import sys
from typing import cast

from jed_attack.scripts.build_submission import build_attack_source


def test_built_source_has_no_harness_imports() -> None:
    """No jed_attack.* imports survive inlining; only aicomp_sdk/stdlib remain."""
    source = build_attack_source()
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert "jed_attack" not in roots
    stdlib = set(sys.stdlib_module_names)
    assert all(r == "aicomp_sdk" or r in stdlib for r in roots)


def test_built_source_defines_algorithm() -> None:
    """The built module exec's and exposes an AttackAlgorithm subclass."""
    from aicomp_sdk.attacks.contracts import AttackAlgorithmBase

    namespace: dict[str, object] = {}
    exec(compile(build_attack_source(), "attack.py", "exec"), namespace)  # noqa: S102
    attack_cls = cast("type", namespace["AttackAlgorithm"])
    assert issubclass(attack_cls, AttackAlgorithmBase)


def test_built_source_runs_end_to_end() -> None:
    """The inlined AttackAlgorithm actually runs — catches broken flattened refs.

    Definition-time exec cannot catch a broken ``templates.x`` / ``selector.x``
    reference because those live inside methods; only running ``run`` exercises
    them. This runs the built class directly against a SandboxEnv under the
    permissive guardrail and asserts it produces firing candidates.
    """
    from aicomp_sdk.agents import build_agent
    from aicomp_sdk.attacks.contracts import AttackAlgorithmBase, AttackRunConfig
    from aicomp_sdk.core.env.sandbox import SandboxEnv
    from aicomp_sdk.core.predicates import eval_predicates
    from aicomp_sdk.evaluation.ops import resolve_fixtures_dir
    from aicomp_sdk.guardrails.base import GuardrailBase

    namespace: dict[str, object] = {}
    exec(compile(build_attack_source(), "attack.py", "exec"), namespace)  # noqa: S102
    attack_cls = cast("type[AttackAlgorithmBase]", namespace["AttackAlgorithm"])

    env = SandboxEnv(
        seed=123,
        fixtures_dir=resolve_fixtures_dir(),
        agent=build_agent("deterministic"),
        guardrail=GuardrailBase(),
        max_tool_hops=4,
    )
    candidates = attack_cls().run(
        env, AttackRunConfig(time_budget_s=15.0, max_tool_hops=4)
    )
    assert candidates, "built attack produced no candidates"
    # At least one candidate replays to a firing trace under the permissive guardrail.
    fresh = SandboxEnv(
        seed=123,
        fixtures_dir=resolve_fixtures_dir(),
        agent=build_agent("deterministic"),
        guardrail=GuardrailBase(),
        max_tool_hops=4,
    )
    fresh.reset()
    for message in candidates[0].user_messages:
        fresh.interact(message, max_tool_hops=4)
    assert eval_predicates(fresh.export_trace_dict())
