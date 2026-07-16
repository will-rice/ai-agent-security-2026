"""Thin wrapper over the SDK's scorer-equivalent red-team evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aicomp_sdk.agents import AgentSelection
from aicomp_sdk.evaluation.ops import AttackGuardrailSpec
from aicomp_sdk.evaluation.runner import evaluate_redteam


@dataclass(frozen=True, slots=True)
class RunResult:
    """Normalized outcome of one red-team evaluation."""

    score: float
    score_raw: float
    findings_count: int
    unique_cells: int
    time_taken: float
    guardrail_id: str
    agent_label: str


def run_attack(
    attack_cls: type,
    *,
    agent: str = "deterministic",
    budget_s: float = 30.0,
    agent_factory: Callable[[], Any] | None = None,
    agent_label: str | None = None,
    guardrail_factory: Callable[[], Any] | None = None,
    attack_config: Mapping[str, Any] | None = None,
    fixtures_dir: Path | None = None,
) -> RunResult:
    """Evaluate an attack class and return its normalized score.

    Uses the SDK's ``evaluate_redteam`` so the score matches the public
    leaderboard scale. Pass ``agent`` for a built-in backend (``deterministic``,
    ``gpt_oss``, ``gemma_4``) or ``agent_factory`` for a custom (e.g. GGUF) agent.

    Args:
        attack_cls: The AttackAlgorithm subclass to evaluate.
        agent: Built-in agent selection; ignored when ``agent_factory`` is given.
        budget_s: Wall-clock budget for generation (and, separately, replay).
        agent_factory: Optional zero-arg factory returning a custom agent.
        agent_label: Human label when using a custom factory.
        guardrail_factory: Optional zero-arg guardrail factory; ``None`` uses the
            SDK default (OPTIMAL public). Pass ``GuardrailBase`` for a permissive
            ALLOW guardrail (deterministic-agent dev tests need this).
        attack_config: Optional config mapping passed to the attack constructor.
        fixtures_dir: Optional explicit fixtures directory.

    Returns:
        A populated ``RunResult``.
    """
    kwargs: dict[str, Any] = {
        "budget_s": budget_s,
        "attack_config": dict(attack_config or {}),
        "fixtures_dir": fixtures_dir,
    }
    if agent_factory is not None:
        kwargs["agent_factory"] = agent_factory
        kwargs["agent_label"] = agent_label or "custom"
    else:
        kwargs["agent_selection"] = AgentSelection(agent)
    if guardrail_factory is not None:
        kwargs["attack_guardrail_spec"] = AttackGuardrailSpec(
            id="allow_dev",
            version="1",
            guardrail_factory=guardrail_factory,
        )

    execution = evaluate_redteam(attack_cls, **kwargs)
    attack = execution.attack
    assert attack is not None  # REDTEAM track always populates attack
    return RunResult(
        score=attack.score,
        score_raw=attack.score_raw,
        findings_count=attack.findings_count,
        unique_cells=attack.unique_cells,
        time_taken=attack.time_taken,
        guardrail_id=attack.guardrail_id,
        agent_label=execution.agent.label,
    )
