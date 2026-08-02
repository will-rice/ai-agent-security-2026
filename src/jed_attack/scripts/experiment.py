"""Run the attack against the real models and record scores — Codex's inner loop.

Scores the submission against gpt_oss and/or gemma under the public guardrail
(the leaderboard scorer), writes a ``submission.csv``-style summary, and appends
each run to the JSONL archive for comparison across experiments. The default
backend is the in-process GGUF path used by the campaign scorer; the old
``llama-server`` HTTP path remains available for comparison.
"""

import argparse
import hashlib
import importlib.util
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

from aicomp_sdk.agents.protocol import AgentProtocol
from aicomp_sdk.agents.types import AgentDecision, AgentStateSnapshot, AgentToolSpec
from aicomp_sdk.attacks.contracts import AttackAlgorithmBase
from aicomp_sdk.core.runtime_history import RuntimeHistory

from jed_attack.campaign import config as campaign_config
from jed_attack.harness.models import (
    ResidentAgentFactory,
    gguf_agent_factory,
    gguf_target_path,
    llama_server_agent_factory,
    resolve_base_url,
)
from jed_attack.harness.report import breakdown, save_run
from jed_attack.harness.runner import RunResult, run_attack

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("jed_attack.experiment")

# Submission Id prefix per model key (Kaggle uses "gemma", not "gemma_4").
_ID_PREFIX = {"gpt_oss": "gpt_oss", "gemma_4": "gemma"}
Backend = Literal["inline", "server"]
AgentFactory = Callable[[], AgentProtocol]


def load_attack_class(attack_path: Path | None = None) -> type[AttackAlgorithmBase]:
    """Load the ``AttackAlgorithm`` class to score.

    Args:
        attack_path: Optional generated ``attack.py`` artifact. ``None`` keeps the
            package-source attack for legacy/source-development experiments.

    Returns:
        The loaded SDK attack class.

    Raises:
        FileNotFoundError: If ``attack_path`` is given but does not exist.
        TypeError: If the artifact does not expose a valid SDK ``AttackAlgorithm``.
        ImportError: If Python cannot build a module spec for the artifact.
    """
    if attack_path is None:
        from jed_attack.submission.attack import AttackAlgorithm

        return AttackAlgorithm

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


def attack_artifact_metadata(attack_path: Path | None) -> dict[str, str]:
    """Return archive metadata identifying the scored attack artifact."""
    if attack_path is None:
        return {"attack_source": "package", "attack_path": "", "attack_sha256": ""}
    resolved = attack_path.resolve()
    return {
        "attack_source": "artifact",
        "attack_path": str(resolved),
        "attack_sha256": _sha256(resolved),
    }


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ResettingAgent:
    """Proxy that clears a resident llama.cpp KV cache on env reset/restore."""

    def __init__(self, agent: AgentProtocol, resident: ResidentAgentFactory) -> None:
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


def agent_factory_for_model(
    model_key: str,
    *,
    backend: Backend,
    base_url: str | None,
) -> tuple[AgentFactory, dict[str, str]]:
    """Build the requested model backend for a local reproduction run.

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.
        backend: ``"inline"`` for resident GGUF scoring, ``"server"`` for
            the legacy OpenAI-compatible ``llama-server`` endpoint.
        base_url: Optional server URL override; only used for ``backend="server"``.

    Returns:
        A zero-arg SDK agent factory plus archive metadata identifying the backend.
    """
    if backend == "server":
        url = resolve_base_url(model_key, base_url)
        return llama_server_agent_factory(model_key, url), {
            "model_backend": "server",
            "model_base_url": url,
        }

    model_path = gguf_target_path(model_key, campaign_config.MODELS_DIR)
    resident = gguf_agent_factory(
        model_key,
        model_path,
        main_gpu=campaign_config.MODEL_GPUS[model_key][0],
    )

    def factory() -> AgentProtocol:
        return cast("AgentProtocol", _ResettingAgent(resident(), resident))

    return factory, {
        "model_backend": "inline",
        "model_path": str(model_path),
    }


def _run_one(
    model_key: str,
    *,
    attack_cls: type[AttackAlgorithmBase],
    attack_metadata: dict[str, str],
    backend: Backend,
    base_url: str | None,
    budget_s: float,
    max_candidates: int | None,
    archive_dir: Path,
) -> RunResult:
    """Score the attack against one served model under the public guardrail.

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.
        attack_cls: The loaded attack class, either package source or artifact.
        attack_metadata: Archive fields identifying the attack source/hash.
        backend: Model backend to use for this reproduction run.
        base_url: Explicit server base URL, or None to resolve from env.
        budget_s: Wall-clock budget for generation and (separately) replay.
        max_candidates: Optional override of the returned-candidate ceiling.
        archive_dir: Directory holding ``runs.jsonl``.

    Returns:
        The ``RunResult`` for this model.
    """
    factory, model_metadata = agent_factory_for_model(
        model_key, backend=backend, base_url=base_url
    )
    if backend == "server":
        _log.info("[%s] using server %s", model_key, model_metadata["model_base_url"])
    else:
        _log.info("[%s] using inline GGUF %s", model_key, model_metadata["model_path"])
    attack_config = {} if max_candidates is None else {"max_candidates": max_candidates}

    result = run_attack(
        attack_cls,
        agent_factory=factory,
        agent_label=model_key,
        budget_s=budget_s,
        attack_config=attack_config,
    )
    report = breakdown(result.findings)
    save_run(
        result,
        report,
        archive_dir=archive_dir,
        extra={
            "budget_s": budget_s,
            **attack_config,
            **attack_metadata,
            **model_metadata,
        },
    )
    _log.info(
        "[%s] score=%.3f findings=%d cells=%d sev=%d div=%d %s",
        model_key,
        result.score,
        result.findings_count,
        result.unique_cells,
        report.severity_total,
        report.diversity_total,
        report.per_predicate,
    )
    return result


def main() -> None:
    """Run experiments against the served models and write a submission summary."""
    parser = argparse.ArgumentParser(description="JED attack experiment runner")
    parser.add_argument(
        "--model", choices=["both", "gpt_oss", "gemma_4"], default="both"
    )
    parser.add_argument("--budget-s", type=float, default=900.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--attack-path",
        type=Path,
        default=None,
        help=(
            "Generated attack.py artifact to score, e.g. run/build_next/attack.py. "
            "Unset keeps the package source attack."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["inline", "server"],
        default="inline",
        help="Model backend: inline GGUF scoring by default, or legacy llama-server.",
    )
    parser.add_argument("--gpt-oss-url", default=None)
    parser.add_argument("--gemma-url", default=None)
    parser.add_argument("--archive-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/submission.csv"))
    args = parser.parse_args()

    attack_cls = load_attack_class(args.attack_path)
    attack_metadata = attack_artifact_metadata(args.attack_path)
    if args.attack_path is not None:
        _log.info(
            "scoring attack artifact %s sha256=%s",
            attack_metadata["attack_path"],
            attack_metadata["attack_sha256"],
        )

    keys = ["gpt_oss", "gemma_4"] if args.model == "both" else [args.model]
    urls = {"gpt_oss": args.gpt_oss_url, "gemma_4": args.gemma_url}
    rows: list[tuple[str, float]] = []
    for model_key in keys:
        result = _run_one(
            model_key,
            attack_cls=attack_cls,
            attack_metadata=attack_metadata,
            backend=args.backend,
            base_url=urls[model_key],
            budget_s=args.budget_s,
            max_candidates=args.max_candidates,
            archive_dir=args.archive_dir,
        )
        rows.append((f"{_ID_PREFIX[model_key]}_public", result.score))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Id,Score", *(f"{name},{score:.4f}" for name, score in rows)]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log.info("wrote %s (public scores only; private guardrail is hidden)", args.out)


if __name__ == "__main__":
    main()
