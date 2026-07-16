"""Build real GGUF-backed agents matching the Kaggle llama.cpp scoring path.

The competition scores against GGUF models served via llama.cpp. The SDK's
built-in ``agent_selection='gpt_oss'`` uses the *transformers* backend, so for
fidelity we build a custom agent factory that wraps ``LlamaCppChatTemplateBackend``
around the same Q4_K_M GGUF files, matching
``kaggle_evaluation/jed_attack_134815/gguf_model_server.py``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Kaggle-matching GGUF model coordinates."""

    repo: str
    filename: str
    agent_key: str  # "gpt_oss" | "gemma_4"


MODEL_SPECS: dict[str, ModelSpec] = {
    "gpt_oss": ModelSpec(
        repo=os.getenv("GPT_OSS_GGUF_REPO", "unsloth/gpt-oss-20b-GGUF"),
        filename=os.getenv("GPT_OSS_GGUF_FILE", "gpt-oss-20b-Q4_K_M.gguf"),
        agent_key="gpt_oss",
    ),
    "gemma_4": ModelSpec(
        repo=os.getenv("GEMMA_GGUF_REPO", "unsloth/gemma-4-26B-A4B-it-GGUF"),
        filename=os.getenv("GEMMA_GGUF_FILE", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
        agent_key="gemma_4",
    ),
}


def gguf_target_path(model_key: str, models_dir: Path) -> Path:
    """Return the local path where a model's GGUF is stored.

    Args:
        model_key: Key into ``MODEL_SPECS``.
        models_dir: Directory holding downloaded GGUF files.

    Returns:
        The expected local GGUF path.
    """
    return models_dir / MODEL_SPECS[model_key].filename


def download_gguf(model_key: str, models_dir: Path) -> Path:
    """Download the model's GGUF from Hugging Face if not already present.

    Args:
        model_key: Key into ``MODEL_SPECS``.
        models_dir: Target directory.

    Returns:
        The local GGUF path.
    """
    from huggingface_hub import hf_hub_download

    spec = MODEL_SPECS[model_key]
    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=spec.repo,
        filename=spec.filename,
        local_dir=models_dir,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    return Path(downloaded)


def gguf_agent_factory(
    model_key: str,
    model_path: Path,
    *,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    max_new_tokens: int = 1024,
) -> Callable[[], Any]:
    """Return a zero-arg factory building GGUF-backed agents over ONE shared backend.

    Loads the llama.cpp GGUF model exactly once and returns a factory that builds a
    fresh (cheap) SDK agent over that shared backend on each call. This is critical:
    ``run_attack`` invokes the factory for the generation env AND every replay env,
    so building a new backend per call would reload the multi-GB GGUF each time.
    Sharing one backend is safe because the SDK evaluates sequentially and the
    llama.cpp completion call is stateless per turn. Wiring matches Kaggle's
    ``gguf_model_server.py`` (``HFBackendConfig`` built directly, ``max_new_tokens``
    1024, ``supports_tools=True``).

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.
        model_path: Local path to the GGUF file.
        n_ctx: llama.cpp context window (Kaggle uses 8192).
        n_gpu_layers: GPU offload layer count (-1 = all).
        max_new_tokens: Generation cap (Kaggle uses 1024).

    Returns:
        A zero-arg callable returning a fresh SDK agent bound to the shared backend.

    Raises:
        ValueError: If ``model_key`` is unknown.
    """
    from aicomp_sdk.agents.hf_chat_template.backends.llama_cpp import (
        LlamaCppChatTemplateBackend,
    )
    from aicomp_sdk.agents.hf_chat_template.types import HFBackendConfig

    make_agent: Callable[[Any], Any]
    if model_key == "gpt_oss":
        from aicomp_sdk.agents.gpt_oss_agent import DEFAULT_GPT_OSS_MODEL_ID, GPTOSSAgent

        model_id: str = DEFAULT_GPT_OSS_MODEL_ID
        make_agent = GPTOSSAgent
    elif model_key == "gemma_4":
        from aicomp_sdk.agents.gemma4_agent import DEFAULT_GEMMA4_MODEL_ID, Gemma4Agent

        model_id = DEFAULT_GEMMA4_MODEL_ID
        make_agent = Gemma4Agent
    else:
        raise ValueError(f"unknown model_key: {model_key}")

    config = HFBackendConfig(
        model_id=model_id,
        model_path=str(model_path),
        max_new_tokens=max_new_tokens,
    )
    # Load the GGUF once; share this backend across all agent instances.
    backend = LlamaCppChatTemplateBackend.from_model_path(
        model_path=str(model_path),
        config=config,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        supports_tools=True,
    )

    def factory() -> Any:
        """Build a fresh agent over the shared, already-loaded GGUF backend."""
        return make_agent(backend)

    return factory
