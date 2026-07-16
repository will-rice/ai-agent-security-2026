"""Shared GGUF model-server shell for Kaggle-hosted llama.cpp agents."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aicomp_sdk.agents.hf_chat_template.backends.llama_cpp import (
    LlamaCppChatTemplateBackend,
)
from aicomp_sdk.agents.hf_chat_template.types import (
    HFBackendConfig,
    HFGenerationBackendProtocol,
)
from aicomp_sdk.agents.protocol import AgentProtocol
from aicomp_sdk.agents.types import AGENT_STATE_VERSION, AgentStateSnapshot
from kaggle_evaluation.jed_attack_134815 import remote_agent as ra


@dataclass(frozen=True)
class GgufModelSpec:
    model_label: str
    backend_label: str
    model_id: str
    repo_env_var: str
    file_env_var: str
    model_path_env_var: str
    default_repo: str
    default_file: str
    create_agent: Callable[[HFGenerationBackendProtocol], AgentProtocol]
    max_new_tokens: int = 1024
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    supports_tools: bool = True

    def repo(self) -> str:
        return os.environ.get(self.repo_env_var, self.default_repo)

    def filename(self) -> str:
        return os.environ.get(self.file_env_var, self.default_file)


class GgufModelServer:
    def __init__(
        self,
        spec: GgufModelSpec,
        *,
        llama_cls: Any | None = None,
        hf_hub_download: Callable[..., str] | None = None,
    ) -> None:
        self.spec = spec
        self._llama_cls = llama_cls
        self._hf_hub_download = hf_hub_download
        self._model_path: str | None = None
        self._llm: Any | None = None
        self._backend: LlamaCppChatTemplateBackend | None = None
        self._agent: AgentProtocol | None = None

    def predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        cmd = request.get("cmd", "next_action")
        try:
            if cmd == "next_action":
                return self._next_action(request)
            if cmd == "reset_state":
                self.reset_state()
                return {"ack": True}
            if cmd == "snapshot_state":
                return {"snapshot": self.snapshot_state()}
            if cmd == "restore_state":
                self.restore_state(request.get("snapshot"))
                return {"ack": True}
            if cmd == "unload":
                self.unload()
                return {"ack": True}
        except Exception as err:
            return {
                "error": str(err),
                "error_type": type(err).__name__,
            }
        return {"error": f"Unknown command: {cmd}"}

    def reset_state(self) -> None:
        if self._agent is not None:
            self._agent.reset_state()

    def snapshot_state(self) -> AgentStateSnapshot:
        if self._agent is None:
            return _initial_hf_agent_snapshot(self.spec.backend_label)
        return self._agent.snapshot_state()

    def restore_state(self, snapshot: object) -> None:
        if not isinstance(snapshot, Mapping):
            raise ValueError("restore_state requires a snapshot object")
        self._load_agent().restore_state(snapshot)  # type: ignore[arg-type]

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.close()
        elif self._llm is not None:
            close = getattr(self._llm, "close", None)
            if callable(close):
                close()

        self._agent = None
        self._backend = None
        self._llm = None
        self._model_path = None
        gc.collect()

    def load_model(self) -> Any:
        return self._load_llm()

    def _next_action(self, request: Mapping[str, Any]) -> dict[str, Any]:
        history = ra.deserialize_history(
            _require_mapping(
                request.get("history", {"instructions": [], "events": []}),
                field="history",
            )
        )
        tools = ra.deserialize_tools(_require_tool_list(request.get("tools", [])))
        decision = self._load_agent().next_action(history=history, tools=tools)
        return ra.serialize_decision(decision)

    def _load_agent(self) -> AgentProtocol:
        if self._agent is None:
            backend = self._load_backend()
            self._agent = self.spec.create_agent(backend)
        return self._agent

    def _load_backend(self) -> LlamaCppChatTemplateBackend:
        if self._backend is None:
            model_path = self._resolve_model_path()
            config = HFBackendConfig(
                model_id=self.spec.model_id,
                model_path=model_path,
                max_new_tokens=self.spec.max_new_tokens,
            )
            self._backend = LlamaCppChatTemplateBackend.from_model_path(
                model_path=model_path,
                config=config,
                n_ctx=self.spec.n_ctx,
                n_gpu_layers=self.spec.n_gpu_layers,
                supports_tools=self.spec.supports_tools,
                llama_cls=self._llama_cls,
            )
            self._llm = self._backend.llm
        return self._backend

    def _load_llm(self) -> Any:
        return self._load_backend().llm

    def _resolve_model_path(self) -> str:
        if self._model_path is not None:
            return self._model_path

        local_path = os.environ.get(self.spec.model_path_env_var, "").strip()
        if local_path:
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"{self.spec.model_path_env_var} points to missing file: {local_path}"
                )
            print(f"Using pre-downloaded GGUF at: {local_path}")
            self._model_path = local_path
            return local_path

        download = self._hf_hub_download
        if download is None:
            from huggingface_hub import hf_hub_download

            download = hf_hub_download

        repo = self.spec.repo()
        filename = self.spec.filename()
        print(f"Downloading {self.spec.model_label} GGUF: {repo}/{filename}")
        self._model_path = download(repo_id=repo, filename=filename)
        print(f"Model downloaded to: {self._model_path}")
        return self._model_path


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return dict(value)


def _require_tool_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("tools must be a list")
    tools: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("tools entries must be objects")
        tools.append(dict(item))
    return tools


def _initial_hf_agent_snapshot(backend_label: str) -> AgentStateSnapshot:
    return {
        "version": AGENT_STATE_VERSION,
        "backend": backend_label,
        "data": {
            "next_generated_call_index": 1,
            "next_debug_turn_index": 1,
        },
    }
