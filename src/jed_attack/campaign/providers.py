"""Proposer provider registry.

Non-secret proposer config — backend kind, model, and endpoint — lives here in code and
is selected by name. Only the API token comes from the environment, looked up by each
provider's ``key_env`` at call time. So switching providers is a name change (no sprawl
of ``PROPOSER_API_BASE``/``_MODEL`` env vars), and secrets never live in code or config
files. ``config.TEAM_PROPOSERS`` names the lanes the async team runs, one worker per
usable entry (see :func:`optimize_prompts.optimize_team`).
"""

import os
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI, OpenAI

from jed_attack.harness.models import resolve_base_url


@dataclass(frozen=True)
class Provider:
    """A proposer backend definition.

    Attributes:
        kind: ``"local"`` (a served target model), ``"api"`` (OpenAI-compatible chat
            endpoint), ``"codex"`` (the codex CLI), or ``"codex_responses"`` (the codex
            ChatGPT-account Responses backend; see
            :mod:`jed_attack.campaign.codex_proposer`).
        model: Model name — the served model key for ``local``, or the API model id.
        base_url: ``api`` only — the OpenAI-compatible base URL.
        key_env: ``api`` only — the env var holding the bearer token (never the token).
        max_tokens: The model's maximum tokens (default is a high default; a model whose
            API rejects it gets its real max in this field).
        agentic: Route this lane through the native function-calling tool-loop proposer
            (:func:`jed_attack.campaign.agentic_proposer.propose_batch_agentic`) so it
            can live-test candidates before submitting, not the one-shot streamer.
    """

    kind: str
    model: str = ""
    base_url: str = ""
    key_env: str = ""
    max_tokens: int = 65536
    agentic: bool = False


# z.ai GLM Coding Plan endpoint (subscription weekly quota), NOT the pay-as-you-go
# /api/paas/v4 (which needs a prepaid balance and 429s with code 1113 when empty).
_ZAI = "https://api.z.ai/api/coding/paas/v4"
CHEAPEST_BASE_URL = "https://api.cheapestinference.com/v1"
CHEAPEST_KEY_ENV = "CHEAPEST_API_KEY"
_CHEAP = CHEAPEST_BASE_URL
# The codex ChatGPT-account Responses lane's kind (dispatched in optimize_prompts to
# jed_attack.campaign.codex_proposer, which can't be imported here -- it imports this).
CODEX_RESPONSES_KIND = "codex_responses"

PROVIDERS: dict[str, Provider] = {
    # Local served target models: free, no provider block, but the weakest proposer.
    "gpt_oss": Provider("local", model="gpt_oss"),
    "gemma_4": Provider("local", model="gemma_4"),
    # z.ai GLM Coding Plan (subscription weekly quota via _ZAI; token in ZAI_API_KEY).
    # The glm-5 family reliably honors OpenAI structured outputs (probed 2026-07-23 with
    # a 32768 budget); glm-4.6 only loosely adheres (salvage) and glm-4.7 is flaky, so
    # they are kept for reference but excluded from the team's z.ai rotation.
    "zai-glm4.6": Provider(
        "api", model="glm-4.6", base_url=_ZAI, key_env="ZAI_API_KEY"
    ),
    "zai-glm5": Provider("api", model="glm-5", base_url=_ZAI, key_env="ZAI_API_KEY"),
    "zai-glm5-turbo": Provider(
        "api", model="glm-5-turbo", base_url=_ZAI, key_env="ZAI_API_KEY"
    ),
    "zai-glm5.1": Provider(
        "api", model="glm-5.1", base_url=_ZAI, key_env="ZAI_API_KEY"
    ),
    "zai-glm5.2": Provider(
        "api", model="glm-5.2", base_url=_ZAI, key_env="ZAI_API_KEY"
    ),
    # cheapestinference.com (token in CHEAPEST_API_KEY). The optimizer replaces the
    # static fallback roster with /v1/models at startup, so these entries are aliases
    # for known current ids and for explicit operator pins.
    "cheapest-deepseek": Provider(
        "api", model="deepseek-v4-flash", base_url=_CHEAP, key_env=CHEAPEST_KEY_ENV
    ),
    "cheapest-mimo": Provider(
        "api", model="mimo-v2.5", base_url=_CHEAP, key_env=CHEAPEST_KEY_ENV
    ),
    # kimi-k2.7 is the team's top exploit-discoverer, so it runs the agentic tool-loop
    # proposer that live-tests candidates before submitting (see agentic_proposer). The
    # CI cycle rebuilds providers via cheapest_provider_for_model, which returns this
    # very entry for "kimi-k2.7", so the agentic flag survives live-model resolution.
    "cheapest-kimi": Provider(
        "api",
        model="kimi-k2.7",
        base_url=_CHEAP,
        key_env=CHEAPEST_KEY_ENV,
        agentic=True,
    ),
    "cheapest-kimi2.6": Provider(
        "api", model="kimi-k2.6", base_url=_CHEAP, key_env=CHEAPEST_KEY_ENV
    ),
    "cheapest-glm5.2": Provider(
        "api", model="glm-5.2", base_url=_CHEAP, key_env=CHEAPEST_KEY_ENV
    ),
    "cheapest-minimax": Provider(
        "api", model="minimax-m3", base_url=_CHEAP, key_env=CHEAPEST_KEY_ENV
    ),
    # codex CLI: provider-blocked on these red-team prompts here, kept for other envs.
    "codex": Provider("codex"),
    # codex ChatGPT-account Responses backend (token from ~/.codex/auth.json, no
    # key_env). gpt-5.5 COMPLIES with the red-team proposer prompt and hard-enforces the
    # two-pool schema via Responses text.format; the gpt-5.6 family is
    # moderation-blocked.
    # daybreak ("gpt-daybreak-blue-latest", the cyber model) is now ENTITLED on the
    # ChatGPT plan and speaks this backend + strict structured output for benign inputs,
    # but its input moderation returns invalid_prompt ("Request blocked") for BOTH the
    # SubmissionBatch schema's attack Field descriptions AND the red-team
    # proposer prompt
    # itself (probed 2026-08-12). So it authors nothing here -- deliberately NOT a lane.
    # See codex_proposer.
    "codex-gpt55": Provider(CODEX_RESPONSES_KIND, model="gpt-5.5"),
    "codex-gpt54": Provider(CODEX_RESPONSES_KIND, model="gpt-5.4"),
}


def get(name: str) -> Provider:
    """Return the named provider, or raise with the list of valid names.

    Args:
        name: A registry key.

    Returns:
        The :class:`Provider`.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return PROVIDERS[name]
    except KeyError:
        raise KeyError(
            f"unknown proposer '{name}'; valid: {sorted(PROVIDERS)}"
        ) from None


def is_cheapest(provider: Provider) -> bool:
    """Return whether ``provider`` is backed by the CheapestInference key/window."""
    return (
        provider.base_url == CHEAPEST_BASE_URL and provider.key_env == CHEAPEST_KEY_ENV
    )


def cheapest_provider_for_model(model: str) -> Provider:
    """Return a CheapestInference provider for ``model``.

    Known ids reuse their registry entry, preserving any per-model knobs. Unknown ids
    returned by /v1/models are still usable because CheapestInference speaks the same
    OpenAI-compatible chat API for all listed models.
    """
    for provider in PROVIDERS.values():
        if is_cheapest(provider) and provider.model == model:
            return provider
    return Provider(
        "api",
        model=model,
        base_url=CHEAPEST_BASE_URL,
        key_env=CHEAPEST_KEY_ENV,
    )


def _model_ids_from_response(payload: object) -> tuple[str, ...]:
    """Parse an OpenAI-compatible models response into ordered model ids."""
    if isinstance(payload, dict):
        raw_items = payload.get("data", [])
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []
    if not isinstance(raw_items, list):
        raw_items = []

    ids: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            model_id = item.get("id")
        else:
            continue
        if isinstance(model_id, str) and model_id and model_id not in seen:
            seen.add(model_id)
            ids.append(model_id)
    return tuple(ids)


def fetch_cheapest_model_ids() -> tuple[str, ...]:
    """Fetch the live CheapestInference model ids visible to ``CHEAPEST_API_KEY``.

    Raises:
        RuntimeError: If the key is missing, the endpoint fails, or the response is
            not usable. Callers decide whether to fall back to a static roster.
    """
    key = os.environ.get(CHEAPEST_KEY_ENV)
    if not key:
        raise RuntimeError(f"{CHEAPEST_KEY_ENV} unset")
    try:
        response = httpx.get(
            f"{CHEAPEST_BASE_URL}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10.0,
        )
        response.raise_for_status()
        model_ids = _model_ids_from_response(response.json())
    except Exception as exc:
        raise RuntimeError("CheapestInference /v1/models lookup failed") from exc
    if not model_ids:
        raise RuntimeError("CheapestInference /v1/models returned no model ids")
    return model_ids


def openai_client(provider: Provider) -> OpenAI:
    """Build an OpenAI-SDK client for a provider (api or local llama-server).

    Both cheapest and llama-server speak the OpenAI chat API, so one client type drives
    both via base_url. The bearer token is read from the provider's key_env at call time
    (local providers use a placeholder — llama-server ignores auth).

    Args:
        provider: The ``api`` or ``local`` provider to build a client for.

    Returns:
        An ``openai.OpenAI`` client pointed at the provider's endpoint.
    """
    key = (
        os.environ.get(provider.key_env, "sk-local") if provider.key_env else "sk-local"
    )
    base = provider.base_url or resolve_base_url(provider.model)  # local -> served base
    return OpenAI(base_url=base.rstrip("/"), api_key=key)


def async_openai_client(provider: Provider) -> AsyncOpenAI:
    """Async OpenAI-SDK client for a provider (api or local). Mirrors openai_client.

    The read timeout is disabled (``read=None``) because the proposer STREAMS the
    completion and enforces its own idle cutoff (abort only if no token arrives for a
    while); a fixed read timeout would kill a slow but actively streaming model.

    Args:
        provider: The ``api`` or ``local`` provider to build a client for.

    Returns:
        An ``openai.AsyncOpenAI`` client pointed at the provider's endpoint.
    """
    key = (
        os.environ.get(provider.key_env, "sk-local") if provider.key_env else "sk-local"
    )
    base = provider.base_url or resolve_base_url(provider.model)
    return AsyncOpenAI(
        base_url=base.rstrip("/"),
        api_key=key,
        timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0),
    )
