"""Proposer provider registry.

Non-secret proposer config — backend kind, model, and endpoint — lives here in code and
is selected by name. Only the API token comes from the environment, looked up by each
provider's ``key_env`` at call time. So switching providers is a name change (no sprawl
of ``PROPOSER_API_BASE``/``_MODEL`` env vars), and secrets never live in code or config
files. ``jed-optimize --proposer <name>`` and the live ``proposer.json`` both use these
names; see :func:`optimize_prompts.current_provider`.
"""

from dataclasses import dataclass


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


PROVIDERS: dict[str, Provider] = {
    # Local served target models: free, no provider block, but the weakest proposer.
    "gpt_oss": Provider("local", model="gpt_oss"),
    "gemma_4": Provider("local", model="gemma_4"),
    # z.ai GLM (metered; token in ZAI_API_KEY).
    "zai-glm4.6": Provider(
        "api",
        model="glm-4.6",
        base_url="https://api.z.ai/api/paas/v4",
        key_env="ZAI_API_KEY",
    ),
    "zai-glm5.2": Provider(
        "api",
        model="glm-5.2",
        base_url="https://api.z.ai/api/paas/v4",
        key_env="ZAI_API_KEY",
    ),
    # cheapestinference.com flat-rate pool (token in CHEAPEST_API_KEY). Model ids may
    # need tweaking to match their catalog — the sanity check catches a wrong name.
    "cheapest-deepseek": Provider(
        "api",
        model="deepseek-v4-flash",
        base_url="https://api.cheapestinference.com/v1",
        key_env="CHEAPEST_API_KEY",
    ),
    "cheapest-glm5.2": Provider(
        "api",
        model="glm-5.2",
        base_url="https://api.cheapestinference.com/v1",
        key_env="CHEAPEST_API_KEY",
    ),
    # codex CLI: provider-blocked on these red-team prompts here, kept for other envs.
    "codex": Provider("codex"),
}

# Default proposer when nothing is selected (free, always available on green).
DEFAULT = "gpt_oss"


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
