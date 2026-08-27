"""Differential sweep: score every single-mutation forge on the real oracle."""

from jed_attack.adversarial.alphabet import MutationForge, single_token_forges
from jed_attack.adversarial.oracle import OracleResult, evaluate


def differential_sweep(
    base_intent: str, model: str
) -> list[tuple[MutationForge, OracleResult]]:
    """Score each single-token forge of ``base_intent`` on ``model``, leanest first.

    Baseline (no forge) is included as ``MutationForge("baseline", base_intent)`` so the
    sweep shows which mutation lowers the token cost while keeping ``fires``.
    """
    candidates = [MutationForge("baseline", base_intent)] + single_token_forges(
        base_intent, model
    )
    scored = [(f, evaluate(f.text, model)) for f in candidates]
    scored.sort(key=lambda fr: fr[1].token_cost)
    return scored
