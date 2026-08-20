"""Hybrid search: GCG proposes -> oracle validates -> GA together; stop on plateau.

The oracle decides everything scored. A negative result (nothing beats ``floor_board``)
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


def beats_floor(board: float, floor: float) -> bool:
    """True only when ``board`` STRICTLY exceeds the champion ``floor``.

    A tie is not a win -- it avoids shipping a lateral (no-better) shape.
    """
    return board > floor


def search(
    base_template: str,
    model: str,
    gcg_steps: int = 250,
    ga_rounds: int = 5,
    floor_board: float = 0.0,
) -> Best | None:
    """Run GCG (gpt only) -> oracle-validate -> GA; return the best firing shape.

    Args:
        base_template: Message template. Must contain ``{optim_str}`` for the GCG
            lane; the GA seed is this template with ``{optim_str}`` stripped.
        model: ``"gpt_oss"`` or ``"gemma_4"``.
        gcg_steps: GCG optimization steps (gpt_oss only).
        ga_rounds: Number of GA evolution rounds.
        floor_board: The champion's board_density; the result must strictly beat it.

    Returns:
        The best firing :class:`Best` if it strictly beats ``floor_board``, else
        ``None`` (logged as a negative result).
    """
    seeds: list[str] = []
    if model == "gpt_oss":
        for cand in run_gcg(base_template, num_steps=gcg_steps):
            res = evaluate(cand.message, model)  # oracle is the judge, not GCG loss
            _log.info("GCG candidate fires=%s board=%.2f", res.fires, res.board_density)
            telemetry.log(
                {
                    "gcg_candidate/board": res.board_density,
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
        telemetry.log({"result/best_board": 0.0, "result/beats_floor": 0.0})
        return None
    best_message, best_result = max(firing, key=lambda pair: pair[1].board_density)
    if not beats_floor(best_result.board_density, floor_board):
        _log.warning(
            "search(%s): best board %.2f did not beat floor %.2f -- "
            "message space tapped",
            model,
            best_result.board_density,
            floor_board,
        )
        telemetry.log(
            {
                "result/best_board": best_result.board_density,
                "result/beats_floor": 0.0,
            }
        )
        return None
    telemetry.log(
        {"result/best_board": best_result.board_density, "result/beats_floor": 1.0}
    )
    return Best(message=best_message, result=best_result)
