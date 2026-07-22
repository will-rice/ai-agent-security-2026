"""Gate guardrail — the leaderboard scoring oracle.

Producers author attacks against the SDK's permissive ``OptimalGuardrail`` (the public
leaderboard defense, which they can read), and scoring currently uses ONLY this public
guardrail — the ``private`` board is dormant (0.0). The real private guardrail is held out
and undisclosed (host-confirmed; the forum speculates an LLM-judge); a private proxy (an
LLM-judge, or the SDK ``RulesGuardrail`` baseline) can be added to ``GATE_GUARDRAILS`` here
when the private board is re-enabled.
"""

from collections.abc import Callable

from aicomp_sdk.guardrails.base import GuardrailBase
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail

# Gate oracle set: only the public guardrail ("optimal") for now. Add private proxies
# (e.g. an LLM-judge) here when the private board is re-enabled.
GATE_GUARDRAILS: dict[str, Callable[[], GuardrailBase]] = {
    "optimal": OptimalGuardrail,
}
