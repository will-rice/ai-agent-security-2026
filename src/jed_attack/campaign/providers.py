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

from openai import AsyncOpenAI, OpenAI

from jed_attack.harness.models import resolve_base_url


@dataclass(frozen=True)
class Provider:
    """A proposer backend definition.

    Attributes:
        kind: ``"local"`` (a served target model), ``"api"`` (OpenAI-compatible chat
            endpoint), or ``"codex"`` (the codex CLI).
        model: Model name — the served model key for ``local``, or the API model id.
        base_url: ``api`` only — the OpenAI-compatible base URL.
        key_env: ``api`` only — the env var holding the bearer token (never the token).
    """

    kind: str
    model: str = ""
    base_url: str = ""
    key_env: str = ""


# z.ai GLM Coding Plan endpoint (subscription weekly quota), NOT the pay-as-you-go
# /api/paas/v4 (which needs a prepaid balance and 429s with code 1113 when empty).
_ZAI = "https://api.z.ai/api/coding/paas/v4"
_CHEAP = "https://api.cheapestinference.com/v1"

PROVIDERS: dict[str, Provider] = {
    # Local served target models: free, no provider block, but the weakest proposer.
    "gpt_oss": Provider("local", model="gpt_oss"),
    "gemma_4": Provider("local", model="gemma_4"),
    # z.ai GLM Coding Plan (subscription weekly quota via _ZAI; token in ZAI_API_KEY).
    "zai-glm4.6": Provider(
        "api", model="glm-4.6", base_url=_ZAI, key_env="ZAI_API_KEY"
    ),
    "zai-glm5.2": Provider(
        "api", model="glm-5.2", base_url=_ZAI, key_env="ZAI_API_KEY"
    ),
    # cheapestinference.com flat-rate pools (token in CHEAPEST_API_KEY). Model ids
    # verified from their docs (/docs/getting-started/models). Core = deepseek/mimo;
    # Frontier = kimi/glm/minimax (stronger, pricier flat rate).
    "cheapest-deepseek": Provider(
        "api", model="deepseek-v4-flash", base_url=_CHEAP, key_env="CHEAPEST_API_KEY"
    ),
    "cheapest-mimo": Provider(
        "api", model="mimo-v2.5", base_url=_CHEAP, key_env="CHEAPEST_API_KEY"
    ),
    "cheapest-kimi": Provider(
        "api", model="kimi-k2.7", base_url=_CHEAP, key_env="CHEAPEST_API_KEY"
    ),
    "cheapest-glm5.2": Provider(
        "api", model="glm-5.2", base_url=_CHEAP, key_env="CHEAPEST_API_KEY"
    ),
    "cheapest-minimax": Provider(
        "api", model="minimax-m3", base_url=_CHEAP, key_env="CHEAPEST_API_KEY"
    ),
    # codex CLI: provider-blocked on these red-team prompts here, kept for other envs.
    "codex": Provider("codex"),
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

    Args:
        provider: The ``api`` or ``local`` provider to build a client for.

    Returns:
        An ``openai.AsyncOpenAI`` client pointed at the provider's endpoint.
    """
    key = (
        os.environ.get(provider.key_env, "sk-local") if provider.key_env else "sk-local"
    )
    base = provider.base_url or resolve_base_url(provider.model)
    return AsyncOpenAI(base_url=base.rstrip("/"), api_key=key)
