"""Experiment script helpers load exact attack artifacts."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from aicomp_sdk.agents.protocol import AgentProtocol
from aicomp_sdk.agents.types import AgentDecision, AgentStateSnapshot, AgentToolSpec
from aicomp_sdk.attacks.contracts import AttackAlgorithmBase
from aicomp_sdk.core.runtime_history import RuntimeHistory
from pytest import MonkeyPatch

from jed_attack.scripts import experiment


def test_load_attack_class_from_path_loads_exact_artifact(tmp_path: Path) -> None:
    """An explicit attack path is imported instead of the package source attack."""
    attack_path = tmp_path / "attack.py"
    attack_path.write_text(
        "\n".join(
            [
                "from aicomp_sdk.attacks.contracts import (",
                "    AttackAlgorithmBase,",
                "    AttackCandidate,",
                ")",
                "",
                "class AttackAlgorithm(AttackAlgorithmBase):",
                "    artifact_marker = 'generated'",
                "",
                "    def run(self, env, config):",
                "        return [AttackCandidate.from_messages(['artifact message'])]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    attack_cls = experiment.load_attack_class(attack_path)

    assert issubclass(attack_cls, AttackAlgorithmBase)
    assert cast("Any", attack_cls).artifact_marker == "generated"


def test_attack_artifact_metadata_records_path_and_hash(tmp_path: Path) -> None:
    """Run archives include enough identity to compare local runs to Kaggle builds."""
    attack_path = tmp_path / "attack.py"
    attack_path.write_text("print('artifact')\n", encoding="utf-8")

    metadata = experiment.attack_artifact_metadata(attack_path)

    assert metadata["attack_path"] == str(attack_path)
    assert metadata["attack_sha256"]


def test_inline_backend_uses_resident_gguf_and_gpu_placement(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The default exact-repro backend is in-process GGUF, not llama-server."""
    calls: list[tuple[str, Path, int | None]] = []
    resets: list[str] = []
    fake_agent = _FakeAgent()

    def fake_gguf_agent_factory(
        model_key: str, model_path: Path, *, main_gpu: int | None = None
    ) -> experiment.ResidentAgentFactory:
        calls.append((model_key, model_path, main_gpu))

        class Resident:
            def reset(self) -> None:
                resets.append("reset")

            def __call__(self) -> AgentProtocol:
                return fake_agent

        return cast("experiment.ResidentAgentFactory", Resident())

    monkeypatch.setattr(experiment.campaign_config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(experiment.campaign_config, "MODEL_GPU", {"gpt_oss": 7})
    monkeypatch.setattr(experiment, "gguf_agent_factory", fake_gguf_agent_factory)

    factory, metadata = experiment.agent_factory_for_model(
        "gpt_oss", backend="inline", base_url=None
    )
    agent = factory()
    agent.reset_state()

    assert calls == [
        ("gpt_oss", tmp_path / "gpt-oss-20b-Q4_K_M.gguf", 7),
    ]
    assert metadata == {
        "model_backend": "inline",
        "model_path": str(tmp_path / "gpt-oss-20b-Q4_K_M.gguf"),
    }
    assert resets == ["reset"]
    assert fake_agent.reset_count == 1


def test_server_backend_uses_llama_server_only_when_requested(
    monkeypatch: MonkeyPatch,
) -> None:
    """The stale llama-server path is still available as an explicit comparison mode."""
    calls: list[tuple[str, str]] = []

    def fake_llama_server_agent_factory(
        model_key: str, base_url: str
    ) -> experiment.AgentFactory:
        calls.append((model_key, base_url))
        return lambda: _FakeAgent()

    monkeypatch.setattr(
        experiment, "llama_server_agent_factory", fake_llama_server_agent_factory
    )

    factory, metadata = experiment.agent_factory_for_model(
        "gemma_4", backend="server", base_url="http://example.test/v1"
    )

    assert factory is not None
    assert calls == [("gemma_4", "http://example.test/v1")]
    assert metadata == {
        "model_backend": "server",
        "model_base_url": "http://example.test/v1",
    }


class _FakeAgent:
    def __init__(self) -> None:
        self.reset_count = 0

    def next_action(
        self,
        *,
        history: RuntimeHistory,
        tools: Sequence[AgentToolSpec],
    ) -> AgentDecision:
        raise AssertionError("not used")

    def reset_state(self) -> None:
        self.reset_count += 1

    def snapshot_state(self) -> AgentStateSnapshot:
        return {"version": 1, "backend": "fake", "data": {}}

    def restore_state(self, snapshot: AgentStateSnapshot) -> None:
        self.reset_state()
