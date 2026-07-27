"""Exact generated-artifact scoring for optimizer outer-loop telemetry."""

import hashlib
import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from types import ModuleType
from typing import cast

from aicomp_sdk.agents.protocol import AgentProtocol
from aicomp_sdk.agents.types import AgentDecision, AgentStateSnapshot, AgentToolSpec
from aicomp_sdk.attacks.contracts import (
    AttackAlgorithmBase,
    AttackCandidate,
    AttackRunConfig,
)
from aicomp_sdk.core.env.api import AttackEnvProtocol
from aicomp_sdk.core.runtime_history import RuntimeHistory

from jed_attack.campaign import config
from jed_attack.campaign import submission_score as resident_score
from jed_attack.harness.models import ResidentAgentFactory
from jed_attack.harness.report import breakdown
from jed_attack.harness.runner import RunResult, run_attack

ArtifactMetric = float | int | str
AgentFactory = Callable[[], AgentProtocol]


def load_attack_class(attack_path: Path) -> type[AttackAlgorithmBase]:
    """Load ``AttackAlgorithm`` from a generated ``attack.py`` artifact.

    Args:
        attack_path: The generated artifact to import.

    Returns:
        The loaded SDK attack class.

    Raises:
        ImportError: If Python cannot build an import spec for the artifact.
        TypeError: If the artifact does not expose a valid ``AttackAlgorithm``.
    """
    resolved = attack_path.resolve()
    source_hash = _sha256(resolved)[:16]
    module_name = f"_jed_attack_artifact_{source_hash}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import attack artifact: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return _attack_class_from_module(module, resolved)


def score_artifact_metrics(
    attack_path: Path,
    *,
    budget_s: float = config.ARTIFACT_SCORE_BUDGET_S,
    models: Sequence[str] = config.MODELS,
) -> dict[str, ArtifactMetric]:
    """Score a built artifact exactly and return W&B-ready ``artifact_*`` metrics.

    This path runs the SDK evaluator against the generated ``attack.py`` itself, so
    live validation/fill inside the artifact is included before replay scoring. It
    reuses the campaign's resident inline GGUF backends and per-model locks so it
    does not load duplicate model contexts or interleave with inner-loop scoring.

    Args:
        attack_path: Generated ``attack.py`` to score.
        budget_s: Per-model SDK generation/replay budget.
        models: Model keys to evaluate.

    Returns:
        Flat metrics suitable for direct W&B logging.
    """
    resolved = attack_path.resolve()
    attack_cls = load_attack_class(resolved)
    per_model: dict[str, tuple[RunResult, int]] = {
        model: _score_one_model(attack_cls, model, budget_s) for model in models
    }
    scores = [result.score for result, _ in per_model.values()]
    raws = [result.score_raw for result, _ in per_model.values()]
    findings = [result.findings_count for result, _ in per_model.values()]
    cells = [result.unique_cells for result, _ in per_model.values()]
    replay_s = [result.time_taken for result, _ in per_model.values()]
    candidate_counts = [count for _, count in per_model.values()]

    metrics: dict[str, ArtifactMetric] = {
        "artifact_valid": 1.0,
        "artifact_budget_s": budget_s,
        "artifact_sha256": _sha256(resolved),
        "artifact_path": str(resolved),
        "artifact_public": mean(scores) if scores else 0.0,
        "artifact_score_raw": mean(raws) if raws else 0.0,
        "artifact_findings": sum(findings),
        "artifact_unique_cells": sum(cells),
        "artifact_time_taken_s": sum(replay_s),
        "artifact_candidate_count_mean": mean(candidate_counts)
        if candidate_counts
        else 0.0,
        "artifact_candidate_count_min": min(candidate_counts)
        if candidate_counts
        else 0,
        "artifact_candidate_count_max": max(candidate_counts)
        if candidate_counts
        else 0,
    }
    for model, (result, candidate_count) in per_model.items():
        report = breakdown(result.findings)
        metrics |= {
            f"artifact_{model}_public": result.score,
            f"artifact_{model}_raw": result.score_raw,
            f"artifact_{model}_findings": result.findings_count,
            f"artifact_{model}_unique_cells": result.unique_cells,
            f"artifact_{model}_time_taken_s": result.time_taken,
            f"artifact_{model}_candidate_count": candidate_count,
            f"artifact_{model}_severity": report.severity_total,
            f"artifact_{model}_diversity": report.diversity_total,
        }
    return metrics


def _attack_class_from_module(
    module: ModuleType, attack_path: Path
) -> type[AttackAlgorithmBase]:
    attack_cls = getattr(module, "AttackAlgorithm", None)
    if not isinstance(attack_cls, type):
        raise TypeError(f"{attack_path} does not define class AttackAlgorithm")
    if not issubclass(attack_cls, AttackAlgorithmBase):
        raise TypeError(
            f"{attack_path} AttackAlgorithm must inherit from AttackAlgorithmBase"
        )
    return cast("type[AttackAlgorithmBase]", attack_cls)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_one_model(
    attack_cls: type[AttackAlgorithmBase],
    model_key: str,
    budget_s: float,
) -> tuple[RunResult, int]:
    """Score ``attack_cls`` against one resident model and count returned candidates."""
    counts: list[int] = []
    counting_cls = _counting_attack_class(attack_cls, counts)
    lock = resident_score.resident_backend_lock(model_key)
    with lock:
        result = run_attack(
            counting_cls,
            agent_factory=_resident_factory(model_key),
            agent_label=model_key,
            budget_s=budget_s,
        )
    return result, counts[-1] if counts else 0


def _counting_attack_class(
    attack_cls: type[AttackAlgorithmBase], counts: list[int]
) -> type[AttackAlgorithmBase]:
    """Wrap an attack class so artifact telemetry records its returned count."""

    class CountingAttack(AttackAlgorithmBase):
        """Delegate attack that records the generated candidate count."""

        def __init__(self, config: Mapping[str, object] | None = None) -> None:
            super().__init__(config)
            self._delegate = attack_cls(config=config)

        def run(
            self, env: AttackEnvProtocol, config: AttackRunConfig
        ) -> list[AttackCandidate]:
            """Run the wrapped artifact and record how many candidates it returns."""
            candidates = self._delegate.run(env, config)
            counts.append(len(candidates))
            return candidates

    return CountingAttack


def _resident_factory(model_key: str) -> AgentFactory:
    """Return a zero-arg SDK factory backed by the campaign's resident model."""
    resident = resident_score.resident_backend(model_key)

    def factory() -> AgentProtocol:
        return cast("AgentProtocol", _ResettingAgent(resident(), resident))

    return factory


class _ResettingAgent:
    """Proxy that clears a resident llama.cpp KV cache on env reset/restore."""

    def __init__(
        self, agent: AgentProtocol, resident: ResidentAgentFactory
    ) -> None:
        self._agent = agent
        self._resident = resident

    def next_action(
        self,
        *,
        history: RuntimeHistory,
        tools: Sequence[AgentToolSpec],
    ) -> AgentDecision:
        """Delegate model action generation to the wrapped SDK agent."""
        return self._agent.next_action(history=history, tools=tools)

    def reset_state(self) -> None:
        """Reset both llama.cpp backend cache and SDK agent state."""
        self._resident.reset()
        self._agent.reset_state()

    def snapshot_state(self) -> AgentStateSnapshot:
        """Return the wrapped SDK agent's snapshot."""
        return self._agent.snapshot_state()

    def restore_state(self, snapshot: AgentStateSnapshot) -> None:
        """Restore after clearing llama.cpp backend cache."""
        self._resident.reset()
        self._agent.restore_state(snapshot)
