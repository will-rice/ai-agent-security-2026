"""Compute the competition's ACTUAL normalized score for an assembled attack.

The campaign optimizes robust-severity (a gate proxy). This computes the real number
the competition reports: it replays an assembled ``attack.py``'s candidates against the
served models via the SDK's own ``evaluate_redteam`` — the same path the Kaggle gateway
runs — and reports the normalized 0-1000 score per model.

The PUBLIC leaderboard uses ``OptimalGuardrail`` exactly, so the public score computed
here is directly comparable to the Kaggle public LB. The PRIVATE guardrail is hidden on
Kaggle (only injected there via an env var), so its score can only be proxied locally
with the campaign's ``HardenedGuardrail`` (pass ``--private``).

Run: ``python -m jed_attack.campaign.score [--attack PATH] [--budget-s 300] [--private]``.
"""

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

from aicomp_sdk.attacks.contracts import AttackAlgorithmBase
from aicomp_sdk.evaluation.ops import (
    AttackGuardrailSpec,
    resolve_attack_guardrail_spec,
)
from aicomp_sdk.evaluation.ops import resolve_fixtures_dir
from aicomp_sdk.evaluation.runner import evaluate_redteam

from jed_attack.campaign import config
from jed_attack.campaign.guardrails import HardenedGuardrail
from jed_attack.harness.models import llama_server_agent_factory, resolve_base_url

_log = logging.getLogger("score")
_SEED = 123  # match the gate / Kaggle env seed


def score_cell(
    attack_cls: type[AttackAlgorithmBase],
    model_key: str,
    guardrail_spec: AttackGuardrailSpec,
    budget_s: float,
) -> dict[str, Any]:
    """Replay the attack against one model under one guardrail and score it.

    Args:
        attack_cls: The submission's AttackAlgorithm class.
        model_key: ``gpt_oss`` or ``gemma_4``.
        guardrail_spec: The guardrail to score under (public = Optimal).
        budget_s: Replay time budget (bounds how many candidates are replayed).

    Returns:
        ``{score, score_raw, unique_cells, findings, time_s}`` for this cell.
    """
    execution = evaluate_redteam(
        attack_cls,
        budget_s=budget_s,
        agent_factory=llama_server_agent_factory(model_key, resolve_base_url(model_key)),
        attack_guardrail_spec=guardrail_spec,
        fixtures_dir=resolve_fixtures_dir(),
        attack_env_seed=_SEED,
    )
    a = execution.attack
    if a is None:
        raise RuntimeError(f"evaluate_redteam returned no attack result for {model_key}")
    return {
        "score": round(a.score, 3),
        "score_raw": round(a.score_raw, 1),
        "unique_cells": a.unique_cells,
        "findings": a.findings_count,
        "time_s": round(a.time_taken, 1),
    }


def score_submission(
    attack_path: Path,
    *,
    budget_s: float = 300.0,
    models: tuple[str, ...] = config.MODELS,
    private: bool = False,
) -> dict[str, Any]:
    """Score an assembled attack.py the way the competition does.

    Args:
        attack_path: Path to the assembled ``attack.py``.
        budget_s: Per-cell replay budget.
        models: Models to score against.
        private: Also score under the HardenedGuardrail (private-LB proxy).

    Returns:
        ``{cells: {model_guardrail: {...}}, public_lb, private_lb}``. ``public_lb``
        is the mean public-guardrail score across models (the Kaggle public LB).
    """
    attack_cls = _load_attack(attack_path)
    public_spec = resolve_attack_guardrail_spec()  # OptimalGuardrail = public LB
    private_spec = AttackGuardrailSpec(
        id="hardened", version="1", guardrail_factory=HardenedGuardrail
    )

    cells: dict[str, Any] = {}
    for model_key in models:
        _log.info("scoring %s under public guardrail…", model_key)
        cells[f"{model_key}_public"] = score_cell(
            attack_cls, model_key, public_spec, budget_s
        )
        if private:
            _log.info("scoring %s under private-proxy (hardened)…", model_key)
            cells[f"{model_key}_private"] = score_cell(
                attack_cls, model_key, private_spec, budget_s
            )

    public_lb = mean(cells[f"{m}_public"]["score"] for m in models)
    private_lb = (
        mean(cells[f"{m}_private"]["score"] for m in models) if private else None
    )
    return {
        "attack": str(attack_path),
        "budget_s": budget_s,
        "cells": cells,
        "public_lb": round(public_lb, 3),
        "private_lb": round(private_lb, 3) if private_lb is not None else None,
    }


def _load_attack(attack_path: Path) -> type[AttackAlgorithmBase]:
    """Load the ``AttackAlgorithm`` class from an assembled attack.py.

    Args:
        attack_path: Path to the attack file.

    Returns:
        The AttackAlgorithm class.

    Raises:
        FileNotFoundError: If the file is missing.
        AttributeError: If it has no AttackAlgorithm class.
    """
    if not attack_path.exists():
        raise FileNotFoundError(f"no attack.py at {attack_path}")
    spec = importlib.util.spec_from_file_location("scored_attack", str(attack_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {attack_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AttackAlgorithm


def main() -> None:
    """CLI: score the assembled submission and write score.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack", default=str(config.BUILD_NEXT_DIR / "attack.py"), type=Path
    )
    parser.add_argument("--budget-s", type=float, default=300.0)
    parser.add_argument("--private", action="store_true", help="also score hardened proxy")
    args = parser.parse_args()

    result = score_submission(
        args.attack, budget_s=args.budget_s, private=args.private
    )
    out = config.CAMPAIGN_ROOT / "score.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log.info("public_lb=%s private_lb=%s -> %s", result["public_lb"], result["private_lb"], out)
    print(json.dumps(result, indent=2))  # noqa: T201 — CLI output for the operator


if __name__ == "__main__":
    main()
