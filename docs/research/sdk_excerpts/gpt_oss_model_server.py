"""GPT-OSS GGUF model server backed by SDK agent semantics."""

from __future__ import annotations

import os
from typing import Any

from aicomp_sdk.agents.gpt_oss_agent import DEFAULT_GPT_OSS_MODEL_ID, GPTOSSAgent
from kaggle_evaluation.jed_attack_134815.gguf_model_server import (
    GgufModelServer,
    GgufModelSpec,
)


SPEC = GgufModelSpec(
    model_label="GPT-OSS",
    backend_label="gpt_oss",
    model_id=DEFAULT_GPT_OSS_MODEL_ID,
    repo_env_var="GPT_OSS_GGUF_REPO",
    file_env_var="GPT_OSS_GGUF_FILE",
    model_path_env_var="GPT_OSS_MODEL_PATH",
    default_repo="unsloth/gpt-oss-20b-GGUF",
    default_file="gpt-oss-20b-Q4_K_M.gguf",
    create_agent=lambda backend: GPTOSSAgent(backend),
)

_server = GgufModelServer(SPEC)


def predict(request: dict[str, Any]) -> dict[str, Any]:
    return _server.predict(request)


def _load_model() -> Any:
    return _server.load_model()


def _unload_model() -> None:
    _server.unload()


if __name__ == "__main__":
    import kaggle_evaluation.core.templates

    class GPTOSSModelInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
        def __init__(self):
            super().__init__(predict)

    server = GPTOSSModelInferenceServer()
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        server.serve()
    else:
        _load_model()
        print("GPT-OSS model server ready for local testing.")
