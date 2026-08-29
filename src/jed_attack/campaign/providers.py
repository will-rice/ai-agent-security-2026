"""Proposer provider registry.

The proposer is a single keyless lane: the agentic codex CLI
(:mod:`jed_attack.campaign.codex_agentic_proposer`). This module just names it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    """A proposer backend definition.

    Attributes:
        kind: The proposer backend kind (only ``CODEX_AGENTIC_KIND`` is registered).
        model: The codex model id (e.g. ``gpt-5.5``).
    """

    kind: str
    model: str = ""


# The AGENTIC codex lane's kind: codex `exec` with repo + oracle tool access, dispatched
# to jed_attack.campaign.codex_agentic_proposer. It READS source and PROBES the victim
# before authoring, so its batch is research-informed and pre-verified, not blind.
CODEX_AGENTIC_KIND = "codex_agentic"

PROVIDERS: dict[str, Provider] = {
    # Agentic lane: codex exec (gpt-5.5) with repo + oracle tool access.
    "codex-agentic": Provider(CODEX_AGENTIC_KIND, model="gpt-5.5"),
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
