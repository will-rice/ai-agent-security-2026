"""Gate guardrails — the leaderboard scoring oracles.

Producers author attacks against the SDK's permissive ``OptimalGuardrail`` (the public
leaderboard defense, which they can read). Scoring also re-evaluates every message under
the stricter SDK ``RulesGuardrail`` as a private-leaderboard proxy the producers do not
target. The real private guardrail is held out and undisclosed; ``RulesGuardrail`` is the
best deterministic proxy we have until an LLM-judge proxy is added.
"""

from collections.abc import Callable

from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail as RulesGuardrail
from aicomp_sdk.guardrails.base import GuardrailBase
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail

# Gate oracle set: "optimal" is the public guardrail (visible to producers); "rules" is
# the stricter private-LB proxy.
GATE_GUARDRAILS: dict[str, Callable[[], GuardrailBase]] = {
    "optimal": OptimalGuardrail,
    "rules": RulesGuardrail,
}
