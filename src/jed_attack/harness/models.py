"""Build real GGUF-backed agents matching the Kaggle llama.cpp scoring path.

The competition scores against GGUF models served via llama.cpp. The SDK's
built-in ``agent_selection='gpt_oss'`` uses the *transformers* backend, so for
fidelity we build a custom agent factory that wraps ``LlamaCppChatTemplateBackend``
around the same Q4_K_M GGUF files.
"""

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from aicomp_sdk.agents.protocol import AgentProtocol


class _Resettable(Protocol):
    """A llama.cpp handle whose KV cache can be cleared between replays."""

    def reset(self) -> None: ...


class ResettableBackend(Protocol):
    """A resident generation backend exposing a resettable llama.cpp handle."""

    llm: _Resettable


_MAX_RETRIES = 4  # transient 5xx under concurrent load; total wait ~ 1+2+4 = 7s
_RETRY_BACKOFF_S = 1.0


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


class ResidentAgentFactory:
    """Builds fresh agents over ONE shared, already-loaded GGUF backend.

    The backend wraps a single llama.cpp context whose KV cache PERSISTS across
    completion calls (llama.cpp prefix-caching). ``SandboxEnv.reset`` only clears the
    *agent's* conversation, not that KV cache, so consecutive logically-independent
    replays on the same resident backend leak context into each other and later
    replays degrade to non-firing (verified: an identical exfil message fires
    severity 16 on the first replay then 0 on every subsequent one; a ``reset``
    between them restores a stable 16 each time). Callers reusing the backend across
    independent replays MUST call :meth:`reset` between them; see
    :func:`jed_attack.campaign.submission_score.replay_trace`.
    """

    def __init__(
        self, backend: ResettableBackend, make_agent: Callable[[Any], AgentProtocol]
    ) -> None:
        self._backend = backend
        self._make_agent = make_agent

    def reset(self) -> None:
        """Clear the shared backend's KV cache so the next replay starts clean.

        Only the llama.cpp backend keeps a persistent prefix cache across generate
        calls; the transformers backend is stateless per generate, so its reset is a
        no-op (``llm`` is absent).
        """
        llm = getattr(self._backend, "llm", None)
        if llm is not None:
            llm.reset()

    def __call__(self) -> AgentProtocol:
        """Build a fresh agent over the shared, already-loaded GGUF backend."""
        return self._make_agent(self._backend)


def gguf_agent_factory(
    model_key: str,
    model_path: Path,
    *,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    max_new_tokens: int = 1024,
    main_gpu: int | None = None,
) -> ResidentAgentFactory:
    """Return a zero-arg factory building GGUF-backed agents over ONE shared backend.

    Loads the llama.cpp GGUF model exactly once and returns a factory that builds a
    fresh (cheap) SDK agent over that shared backend on each call. This is critical:
    ``run_attack`` invokes the factory for the generation env AND every replay env,
    so building a new backend per call would reload the multi-GB GGUF each time.
    Sharing one backend is only correct if the caller resets the backend's KV cache
    between logically-independent replays -- the llama.cpp completion call is NOT
    stateless per turn (see :class:`ResidentAgentFactory`).

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.
        model_path: Local path to the GGUF file.
        n_ctx: llama.cpp context window.
        n_gpu_layers: GPU offload layer count (-1 = all).
        max_new_tokens: Generation cap (256 and 1024 give identical results -- each
            agent turn stays under 256 tokens, so the cap never binds).
        main_gpu: If set, loads the whole model onto this GPU (single-process
            placement, e.g. gpt_oss on GPU 0 and gemma on GPU 1 in one process).
            ``None`` keeps the SDK's default placement.

    Returns:
        A :class:`ResidentAgentFactory` over the shared backend: call it to build a
        fresh agent, and call its ``reset()`` to clear the backend KV cache between
        independent replays.

    Raises:
        ValueError: If ``model_key`` is unknown.
    """
    from aicomp_sdk.agents.hf_chat_template.backends.llama_cpp import (
        LlamaCppChatTemplateBackend,
    )
    from aicomp_sdk.agents.hf_chat_template.types import HFBackendConfig

    make_agent: Callable[[Any], Any]
    if model_key == "gpt_oss":
        from aicomp_sdk.agents.gpt_oss_agent import (
            DEFAULT_GPT_OSS_MODEL_ID,
            GPTOSSAgent,
        )

        model_id: str = DEFAULT_GPT_OSS_MODEL_ID
        make_agent = GPTOSSAgent
    elif model_key == "gemma_4":
        from functools import partial

        from aicomp_sdk.agents.gemma4_agent import DEFAULT_GEMMA4_MODEL_ID, Gemma4Agent

        from jed_attack.harness.kaggle_parsers import KaggleGemma4ToolCallParser

        model_id = DEFAULT_GEMMA4_MODEL_ID
        # Match the grader: it builds the gemma agent with KaggleGemma4ToolCallParser
        # (gemma_model_server.py), not the SDK default. partial injects the parser so
        # ResidentAgentFactory's make_agent(backend) call reproduces the grader exactly.
        make_agent = partial(Gemma4Agent, parser=KaggleGemma4ToolCallParser())
    else:
        raise ValueError(f"unknown model_key: {model_key}")

    config = HFBackendConfig(
        model_id=model_id,
        model_path=str(model_path),
        max_new_tokens=max_new_tokens,
    )
    # Load the GGUF once; share this backend across all agent instances.
    # Whole model on a specific GPU (split_mode 0 == LLAMA_SPLIT_MODE_NONE) when
    # main_gpu is set; default (None) keeps the SDK's placement. Uses the integer
    # directly so this module stays importable without llama-cpp-python installed.
    llama_kwargs = (
        {"main_gpu": main_gpu, "split_mode": 0} if main_gpu is not None else None
    )
    backend = LlamaCppChatTemplateBackend.from_model_path(
        model_path=str(model_path),
        config=config,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        supports_tools=True,
        llama_kwargs=llama_kwargs,
    )

    # The freshly built backend's ``llm`` is non-None (only ``close()`` nulls it),
    # so it satisfies ResettableBackend; the SDK types ``llm`` as ``Any | None``.
    return ResidentAgentFactory(cast(ResettableBackend, backend), make_agent)


def transformers_gpt_oss_factory(main_gpu: int | None = None) -> ResidentAgentFactory:
    """Return a resident factory for gpt_oss using the SDK's TRANSFORMERS backend.

    The Kaggle grader serves gpt_oss via the SDK's ``build_gpt_oss_backend`` -- the
    transformers backend loading the native MXFP4 ``openai/gpt-oss-20b`` with
    ``max_new_tokens=256`` -- NOT a llama.cpp GGUF. A 4-bit-requantized GGUF is a
    DIFFERENT model: it mis-ranks shapes on the reasoning row (measured: the forge
    champion is 22 gen-tokens on MXFP4 vs 28 on our Q4_K_M GGUF, and no-forge shapes
    diverge in the opposite direction). This factory runs the grader's exact backend so
    the objective ranks shapes on the model the grader actually replays.

    The transformers backend is stateless per ``generate`` (no persistent prefix cache),
    so :meth:`ResidentAgentFactory.reset` is a no-op for it. Metering reads
    ``HFGenerationResponse.raw_text`` exactly as for the GGUF backend.

    Args:
        main_gpu: CUDA device index to pin the model to (``device_map='cuda:N'``);
            ``None`` uses the SDK default ``device_map='auto'``.

    Returns:
        A :class:`ResidentAgentFactory` over the shared transformers backend.
    """
    from aicomp_sdk.agents.gpt_oss_agent import GPTOSSAgent, build_gpt_oss_backend

    device_map = "auto" if main_gpu is None else f"cuda:{main_gpu}"
    backend = build_gpt_oss_backend(device_map=device_map)
    return ResidentAgentFactory(cast(ResettableBackend, backend), GPTOSSAgent)


def _resolve_agent(model_key: str) -> tuple[str, Callable[[Any], AgentProtocol]]:
    """Return the SDK model id and agent constructor for a model key.

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.

    Returns:
        A ``(model_id, agent_constructor)`` pair from the SDK.

    Raises:
        ValueError: If ``model_key`` is unknown.
    """
    if model_key == "gpt_oss":
        from aicomp_sdk.agents.gpt_oss_agent import (
            DEFAULT_GPT_OSS_MODEL_ID,
            GPTOSSAgent,
        )

        return DEFAULT_GPT_OSS_MODEL_ID, GPTOSSAgent
    if model_key == "gemma_4":
        from aicomp_sdk.agents.gemma4_agent import DEFAULT_GEMMA4_MODEL_ID, Gemma4Agent

        return DEFAULT_GEMMA4_MODEL_ID, Gemma4Agent
    raise ValueError(f"unknown model_key: {model_key}")


class _LlamaServerClient:
    """Minimal OpenAI-compatible client for a running ``llama-server`` instance.

    Exposes ``create_chat_completion`` with the same signature and return shape as
    ``llama_cpp.Llama.create_chat_completion``, so it drops straight into the SDK's
    ``LlamaCppChatTemplateBackend`` — the SDK does all message/tool conversion and
    response parsing, keeping behavior identical to the in-process llama.cpp path.
    """

    def __init__(
        self, base_url: str, *, model: str = "local", timeout: float = 600.0
    ) -> None:
        """Store the endpoint.

        Args:
            base_url: The server's OpenAI base, e.g. ``http://localhost:8080/v1``.
            model: The ``model`` field to send (llama-server serves one model).
            timeout: Per-request timeout in seconds.
        """
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = timeout

    def create_chat_completion(self, **kwargs: object) -> dict[str, Any]:
        """POST a chat-completion request to the server and return the JSON dict.

        Retries transient server errors (HTTP 5xx / connection resets) with backoff:
        under heavy concurrent load (gate + producers + scorer on one server) the
        server intermittently 500s, which must not crash a long replay/gate run.

        Args:
            **kwargs: The OpenAI chat-completion body (messages, tools, max_tokens...).

        Returns:
            The parsed OpenAI-shaped completion dict.
        """
        body = json.dumps({"model": self._model, **kwargs}).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as err:
                if err.code < 500:
                    raise  # client error (bad request) — retrying won't help
                last_error = err
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                last_error = err
            time.sleep(_RETRY_BACKOFF_S * (2**attempt))
        raise RuntimeError(
            f"llama-server request failed after {_MAX_RETRIES} retries"
        ) from last_error

    def close(self) -> None:
        """No persistent resources to release."""


def llama_server_agent_factory(
    model_key: str,
    base_url: str,
    *,
    model: str = "local",
    max_new_tokens: int = 1024,
) -> Callable[[], AgentProtocol]:
    """Build agents backed by a running ``llama-server`` (llama.cpp over HTTP).

    Same fidelity as the in-process GGUF factory — it reuses the SDK's
    ``LlamaCppChatTemplateBackend`` and real ``GPTOSSAgent``/``Gemma4Agent``, only
    swapping the inference call to an HTTP request. The server holds the model
    resident, so nothing reloads per candidate. Point ``base_url`` at a server
    launched with ``--jinja`` on the exact Q4_K_M GGUF (see ``docs/serving.md``).

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.
        base_url: The server's OpenAI base, e.g. ``http://localhost:8080/v1``.
        model: The ``model`` field to send in requests.
        max_new_tokens: Generation cap.

    Returns:
        A zero-arg callable returning a fresh SDK agent bound to the server.
    """
    from aicomp_sdk.agents.hf_chat_template.backends.llama_cpp import (
        LlamaCppChatTemplateBackend,
    )
    from aicomp_sdk.agents.hf_chat_template.types import HFBackendConfig

    model_id, make_agent = _resolve_agent(model_key)
    config = HFBackendConfig(
        model_id=model_id, model_path="", max_new_tokens=max_new_tokens
    )
    backend = LlamaCppChatTemplateBackend(
        llm=_LlamaServerClient(base_url, model=model),
        config=config,
        supports_tools=True,
    )

    def factory() -> AgentProtocol:
        """Build a fresh agent over the shared llama-server backend."""
        return make_agent(backend)

    return factory


def llama_server_chat_client(
    model_key: str, override: str | None = None, *, timeout: float = 600.0
) -> _LlamaServerClient:
    """Return a raw OpenAI-compatible chat client for a served model (no agent/tools).

    Unlike the agent factories, this exposes plain ``create_chat_completion`` so
    callers can prompt the model directly — used by the adversarial attacker to
    generate its next user message from the target's observed behavior.

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.
        override: Explicit base URL, if given.
        timeout: Per-request timeout in seconds.

    Returns:
        A client whose ``create_chat_completion(**body)`` returns the OpenAI dict.
    """
    return _LlamaServerClient(resolve_base_url(model_key, override), timeout=timeout)


def resolve_base_url(model_key: str, override: str | None = None) -> str:
    """Resolve a llama-server base URL from an override or env var.

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.
        override: Explicit base URL, if given.

    Returns:
        The resolved base URL (env default: gpt_oss=:8080, gemma_4=:8081).
    """
    if override:
        return override
    env_var = "GPT_OSS_BASE_URL" if model_key == "gpt_oss" else "GEMMA_BASE_URL"
    default_port = "8080" if model_key == "gpt_oss" else "8081"
    return os.environ.get(env_var, f"http://localhost:{default_port}/v1")


def resolve_endpoints(model_key: str) -> list[str]:
    """All served base URLs for a model: the ``*_BASE_URLS`` list, else default.

    green sets ``GPT_OSS_BASE_URLS`` / ``GEMMA_BASE_URLS`` to a comma-separated
    list (localhost first, then dylan) so scoring fans out across hosts; unset
    falls back to the one :func:`resolve_base_url` default, so single-host runs
    are unchanged.

    Args:
        model_key: ``"gpt_oss"`` or ``"gemma_4"``.

    Returns:
        Ordered, non-empty list of base URLs (localhost/default is failover).
    """
    env_var = "GPT_OSS_BASE_URLS" if model_key == "gpt_oss" else "GEMMA_BASE_URLS"
    raw = os.environ.get(env_var, "")
    endpoints = [u.strip() for u in raw.split(",") if u.strip()]
    return endpoints or [resolve_base_url(model_key)]
