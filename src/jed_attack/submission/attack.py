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
from aicomp_sdk.core.cells import cell_signature
from aicomp_sdk.core.env.api import AttackEnvProtocol
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.utils.timebox import Timebox

from jed_attack.submission import templates  # inlined at build time

_BUDGET_SAFETY = (
    0.9  # return before the hard generation deadline (see Global Constraints)
)


class AttackAlgorithm(AttackAlgorithmBase):
    """Hybrid discover-and-farm attack (skeleton; filled in later tasks)."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        """Store the optional evaluator-provided config mapping.

        Args:
            config: Optional configuration mapping supplied by the evaluator.
        """
        super().__init__(config)

    def run(
        self, env: AttackEnvProtocol, config: AttackRunConfig
    ) -> list[AttackCandidate]:
        """Sweep injection bases × diversifiers, keeping novel scoring cells.

        Each plan is replayed from a clean reset. A plan is kept when a predicate
        fires and its trace hashes to a score cell not seen before. The loop
        returns before the budget deadline (a late return discards all findings).

        Args:
            env: The opaque attack environment.
            config: Runtime budget and caps.

        Returns:
            Replayable candidates, one per distinct firing score cell.
        """
        timebox = Timebox(float(config.time_budget_s) * _BUDGET_SAFETY)
        hops = int(config.max_tool_hops)
        found: list[AttackCandidate] = []
        seen: set[str] = set()  # score-cell hashes (the scored novelty key)

        plans: list[list[str]] = [
            templates.injection_chain(p) for p in templates.INJECTION_PAGES
        ]
        for page_id in templates.INJECTION_PAGES:
            for diversifier in templates.DIVERSIFIERS:
                plans.append(templates.injection_chain(page_id, diversifier))

        for chain in plans:
            if timebox.expired() or len(found) >= 2000:
                break
            env.reset()
            for message in chain:
                env.interact(message, max_tool_hops=hops)
            trace = env.export_trace_dict()
            if not eval_predicates(trace):
                continue
            score_hash = str(cell_signature(trace.get("tool_events", []))["hash"])
            if score_hash in seen:
                continue
            seen.add(score_hash)
            found.append(AttackCandidate.from_messages(trace["user_messages"]))
        return found
