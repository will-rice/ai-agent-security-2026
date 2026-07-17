"""Producer helper: probe a chain against a served model under the PUBLIC guardrail.

Producers find firing chains with this; ``try_emit`` records them to the store. They
see ONLY ``OptimalGuardrail`` (public proxy); the gate judges robustness vs stricter
``rules``/``hardened`` guardrails producers never see (the anti-overfit separation).
"""

from collections.abc import Callable, Sequence
from typing import Any

from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.evaluation.ops import resolve_fixtures_dir
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail

from jed_attack.campaign import config, knowledge, store
from jed_attack.harness.models import llama_server_agent_factory, resolve_base_url

_SEVERITY_W = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
_FACTORIES: dict[str, Callable[[], Any]] = {}


def _factory(model_key: str) -> Callable[[], Any]:
    """Return (and cache) the shared llama-server agent factory for a model.

    Args:
        model_key: ``gpt_oss`` or ``gemma_4``.

    Returns:
        A zero-arg agent factory over the shared server backend.
    """
    if model_key not in _FACTORIES:
        _FACTORIES[model_key] = llama_server_agent_factory(
            model_key, resolve_base_url(model_key)
        )
    return _FACTORIES[model_key]


def probe(
    chain: Sequence[str], model_key: str, hops: int = config.EVAL_HOPS
) -> dict[str, Any]:
    """Replay a chain on a served model under OptimalGuardrail and report what fired.

    Args:
        chain: The user-message sequence.
        model_key: ``gpt_oss`` or ``gemma_4``.
        hops: Max tool hops per interaction.

    Returns:
        ``{"predicates": [names], "severity": int}`` under the public guardrail.
    """
    from aicomp_sdk.core.env.sandbox import SandboxEnv

    env = SandboxEnv(
        seed=123,
        fixtures_dir=resolve_fixtures_dir(),
        agent=_factory(model_key)(),
        guardrail=OptimalGuardrail(),
        max_tool_hops=hops,
    )
    env.reset()
    for message in chain:
        env.interact(message, max_tool_hops=hops)
    preds = eval_predicates(env.export_trace_dict())
    names = [str(p.get("predicate")) for p in preds]
    severity = sum(_SEVERITY_W.get(int(p.get("severity", 1)), 1) for p in preds)
    return {"predicates": names, "severity": severity}


def try_emit(
    chain: Sequence[str], producer: str, models: Sequence[str] = config.MODELS
) -> bool:
    """Probe a chain on each model; if it fires anywhere, emit it to the store.

    Consults the shared knowledge log first: if any agent already probed this exact
    chain, the cached outcome is returned instead of re-probing (fleet-wide memo,
    saving GPU). Otherwise the chain is probed, the outcome recorded for the fleet,
    and firing chains emitted to the candidate store.

    Args:
        chain: The user-message sequence.
        producer: Producer id (e.g. ``agent-2``).
        models: Models to probe.

    Returns:
        True if the chain fired on any model (whether just probed or cached).
    """
    prior = knowledge.lookup(chain)
    if prior is not None:
        return bool(prior.fired)  # the fleet already tried this — reuse the answer

    scores: dict[str, float] = {}
    fired: set[str] = set()
    for model_key in models:
        result = probe(chain, model_key)
        scores[model_key] = float(result["severity"])
        fired.update(result["predicates"])

    knowledge.record_attempt(chain, producer, fired=sorted(fired), severities=scores)
    if fired:
        store.emit(
            store.Candidate.make(
                chain, producer, predicates=sorted(fired), self_scores=scores
            )
        )
        return True
    return False
