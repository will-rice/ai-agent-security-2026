"""Gate guardrail — the leaderboard scoring oracle.

Producers author attacks against the SDK's permissive ``OptimalGuardrail`` (the public
leaderboard defense, which they can read), but the gate the optimizer scores against is
now the private-proxy ``rules`` guardrail (SDK ``RulesGuardrail`` baseline): the real
private guardrail is held out and undisclosed (host-confirmed; the forum speculates an
LLM-judge), and ``rules`` is the closest available stand-in — it inspects payload data
(denying EXFILTRATION) rather than the permissive public ``optimal`` guardrail. Public
(``optimal``) is retired from the scoring path.
"""

from collections.abc import Callable

from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail as RulesGuardrail
from aicomp_sdk.guardrails.base import GuardrailBase

# Gate oracle set: only the private-proxy guardrail ("rules") now scores candidates.
GATE_GUARDRAILS: dict[str, Callable[[], GuardrailBase]] = {
    "rules": RulesGuardrail,
}
