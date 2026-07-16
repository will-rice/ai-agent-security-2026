"""Submission entry point: AttackAlgorithm for the JED red-team benchmark.

This module is inlined into a single self-contained file at build time and must
import only ``aicomp_sdk`` and the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aicomp_sdk.attacks.contracts import (
    AttackAlgorithmBase,
    AttackCandidate,
    AttackRunConfig,
)
from aicomp_sdk.core.env.api import AttackEnvProtocol
from aicomp_sdk.utils.timebox import Timebox

from jed_attack.submission.engine import Cell, GoExplore, score_hash
from jed_attack.submission.selector import select_candidates
from jed_attack.submission.templates import DIVERSIFIERS, INJECTION_PAGES, open_page

_BUDGET_SAFETY = (
    0.9  # return before the hard generation deadline (see Global Constraints)
)


class AttackAlgorithm(AttackAlgorithmBase):
    """Hybrid discover-and-farm attack.

    Runs a Go-Explore archive search seeded with injection-page opens and benign
    tool-signature diversifiers to discover breaching prompt shapes, then farms
    tool-signature variants around each confirmed breach to mint additional
    distinct scoring cells. The final candidate set is capped and ordered by
    ``select_candidates``.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        """Store the optional evaluator-provided config mapping.

        Args:
            config: Optional configuration mapping supplied by the evaluator.
        """
        super().__init__(config)

    def run(
        self, env: AttackEnvProtocol, config: AttackRunConfig
    ) -> list[AttackCandidate]:
        """Discover breaching prompt shapes, then farm tool-signature variants.

        Args:
            env: The opaque attack environment.
            config: Runtime budget and caps.

        Returns:
            The selected replayable attack candidates.
        """
        timebox = Timebox(float(config.time_budget_s) * _BUDGET_SAFETY)
        hops = int(config.max_tool_hops)

        engine = GoExplore(
            mutations=_discovery_mutations(),
            max_hops=hops,
            rng_seed=int(env.seed),
        )
        engine.seed_from(env)

        # Discovery gets the first half of the (safety-adjusted) budget.
        discovery_box = Timebox(float(config.time_budget_s) * _BUDGET_SAFETY * 0.5)
        while not discovery_box.expired() and not timebox.expired():
            engine.step(env)

        # Only cells that actually fired a predicate are worth replaying (a
        # non-firing candidate costs replay budget for zero score), so filter here.
        farmed: list[Cell] = [
            cell for cell in engine.archive.values() if cell.predicates
        ]
        for breach in engine.breached_cells():
            if timebox.expired():
                break
            farmed.extend(_farm_variants(env, breach, hops, timebox))

        chains = select_candidates(farmed, max_candidates=2000, max_messages=32)
        return [AttackCandidate.from_messages(chain) for chain in chains]


def _discovery_mutations() -> list[str]:
    """Return the injection-driving follow-up messages the engine branches with.

    Returns:
        Injection-page opens (which drive dangerous tool use via injected page
        content) plus benign tool-varying moves that diversify the tool-call
        signature into new scoring cells.
    """
    return [open_page(page_id) for page_id in INJECTION_PAGES] + list(DIVERSIFIERS)


def _farm_variants(
    env: AttackEnvProtocol,
    breach: Cell,
    hops: int,
    timebox: Timebox,
) -> list[Cell]:
    """Mint tool-signature variants around a confirmed breach cell.

    Replays the breach's own message chain with each benign diversifier appended,
    from a clean reset, keeping variants that still fire a predicate with a novel
    scored cell hash.

    Args:
        env: The attack environment.
        breach: A breached archive cell to diversify around.
        hops: Max tool hops per interaction.
        timebox: The overall run budget.

    Returns:
        Newly farmed cells with novel scored hashes.
    """
    from aicomp_sdk.core.predicates import eval_predicates, is_breach

    out: list[Cell] = []
    seen: set[str] = {breach.score_hash}
    base_chain = list(breach.user_messages)
    for diversifier in DIVERSIFIERS:
        if timebox.expired():
            return out
        env.reset()
        for message in [*base_chain, diversifier]:
            env.interact(message, max_tool_hops=hops)
        trace = env.export_trace_dict()
        key = score_hash(trace)
        if key in seen:
            continue
        predicates = eval_predicates(trace)
        if not predicates:
            continue
        seen.add(key)
        out.append(
            Cell(
                snapshot=None,
                user_messages=list(trace["user_messages"]),
                score_hash=key,
                depth=breach.depth + 1,
                breached=is_breach(trace, predicates),
                predicates=predicates,
            )
        )
    return out
