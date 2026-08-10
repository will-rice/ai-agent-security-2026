"""Gate guardrail — the leaderboard scoring oracle.

The optimizer scores against the SDK's ``OptimalGuardrail`` — the PUBLIC leaderboard
defense. This is deliberate: the private board is scored on the SAME candidates re-run
under a STRICTER held-out guardrail (private is IN ADDITION to public, not an
alternative), so a candidate that does not fire on public scores on NEITHER board. The
optimizer therefore maximizes the public (``optimal``) board first; the private board is
handled later as a CONFUSED_DEPUTY hedge (deputy is the only family that fires on public
AND survives a data-inspecting private guardrail). The permissive ``rules`` proxy is
retired from the scoring path — it credited read-exfil, which never fires on optimal.
"""

from collections.abc import Callable

from aicomp_sdk.guardrails.base import GuardrailBase
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail

# Gate oracle set: the public ("optimal") leaderboard guardrail scores candidates.
GATE_GUARDRAILS: dict[str, Callable[[], GuardrailBase]] = {
    "optimal": OptimalGuardrail,
}
