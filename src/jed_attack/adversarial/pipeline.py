"""Hybrid search: GCG proposes -> oracle validates -> GA together; stop on plateau.

The oracle decides everything scored. A negative result (nothing beats ``floor_tokens``)
is returned as ``None`` and logged -- proof the message space is tapped, not hidden.
"""

import logging
from dataclasses import dataclass

from jed_attack.adversarial import telemetry
from jed_attack.adversarial.ga import evolve
from jed_attack.adversarial.gcg_driver import run_gcg
from jed_attack.adversarial.oracle import OracleResult, evaluate

_log = logging.getLogger(__name__)


@dataclass
class Best:
    """The best firing message found and its oracle result."""

    message: str
    result: OracleResult


def beats_floor(tokens: float, floor: float) -> bool:
    """True only when ``tokens`` STRICTLY UNDERCUTS the champion ``floor``.

    A tie is not a win -- it avoids shipping a lateral (no-leaner) shape.
    """
    return tokens < floor


def search(
    base_template: str,
    model: str,
    gcg_steps: int = 250,
    ga_rounds: int = 5,
    floor_tokens: float = float("inf"),
    gcg_optim_len: int = 12,
) -> Best | None:
    """Run GCG (gpt only) -> oracle-validate -> GA; return the best firing shape.

    Args:
        base_template: Message template. Must contain ``{optim_str}`` for the GCG
            lane; the GA seed is this template with ``{optim_str}`` stripped.
        model: ``"gpt_oss"`` or ``"gemma_4"``.
        gcg_steps: GCG optimization steps (gpt_oss only).
        ga_rounds: Number of GA evolution rounds.
        floor_tokens: The champion's token_cost; the result must strictly undercut it.
        gcg_optim_len: GCG optim_str token length; free on the board (shared cached
            prefix), so a longer one adds target-forcing capacity for free.

    Returns:
        The best firing :class:`Best` if it strictly undercuts ``floor_tokens``, else
        ``None`` (logged as a negative result).
    """
    seeds: list[str] = []
    if model == "gpt_oss":
        # optim_str lives in the message's shared, prefix-cached region, so its input
        # tokens are ~free on the board (see oracle.token_cost) -- spend a longer one
        # for more capacity to force the target without paying for it.
        optim_init = " ".join(["x"] * gcg_optim_len)
        for cand in run_gcg(
            base_template, num_steps=gcg_steps, optim_str_init=optim_init
        ):
            res = evaluate(cand.message, model)  # oracle is the judge, not GCG loss
            _log.info("GCG candidate fires=%s tokens=%.2f", res.fires, res.token_cost)
            telemetry.log(
                {
                    "gcg_candidate/tokens": res.token_cost,
                    "gcg_candidate/fires": float(res.fires),
                }
            )
            if res.fires:
                seeds.append(cand.message)
    ga_seed = base_template.replace("{optim_str}", "").strip()
    seeds.append(ga_seed)
    survivors = evolve(seeds, model, rounds=ga_rounds)
    firing = [
        (s.message, s.result)
        for s in survivors
        if s.result is not None and s.result.fires
    ]
    if not firing:
        _log.warning("search(%s): no firing shape found", model)
        telemetry.log({"result/best_tokens": float("inf"), "result/beats_floor": 0.0})
        return None
    best_message, best_result = min(firing, key=lambda pair: pair[1].token_cost)
    if not beats_floor(best_result.token_cost, floor_tokens):
        _log.warning(
            "search(%s): best tokens %.2f did not undercut floor %.2f -- "
            "message space tapped",
            model,
            best_result.token_cost,
            floor_tokens,
        )
        telemetry.log(
            {
                "result/best_tokens": best_result.token_cost,
                "result/beats_floor": 0.0,
            }
        )
        return None
    telemetry.log(
        {"result/best_tokens": best_result.token_cost, "result/beats_floor": 1.0}
    )
    return Best(message=best_message, result=best_result)
