"""Snapshot-backed Go-Explore search over the attack environment.

Archive keyed by the *scored* cell hash (tool events only). Each cell holds an env
snapshot so we return to a promising state via restore() rather than re-running the
whole prefix through the model. Selection favors low-visit, breach-adjacent cells.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aicomp_sdk.core.cells import cell_signature
from aicomp_sdk.core.env.api import AttackEnvProtocol
from aicomp_sdk.core.predicates import eval_predicates, is_breach


def score_hash(trace: Mapping[str, Any]) -> str:
    """Return the scored novelty key for a trace (tool events only).

    Args:
        trace: An exported trace dict.

    Returns:
        The 16-char cell hash the scorer uses for uniqueness.
    """
    return str(cell_signature(trace.get("tool_events", []))["hash"])


@dataclass
class Cell:
    """A frontier state in the Go-Explore archive."""

    snapshot: Any
    user_messages: list[str]
    score_hash: str
    depth: int
    breached: bool = False
    visits: int = 0
    predicates: list[dict[str, Any]] = field(default_factory=list)


class GoExplore:
    """Archive-based discovery engine returning to states via snapshot/restore."""

    def __init__(
        self,
        mutations: Sequence[str],
        *,
        max_hops: int,
        rng_seed: int,
    ) -> None:
        """Initialize the engine.

        Args:
            mutations: Candidate follow-up user messages to branch with.
            max_hops: Max tool hops per interaction (evaluation uses 4).
            rng_seed: Seed for deterministic selection/branching.
        """
        self.mutations = list(mutations)
        self.max_hops = int(max_hops)
        self.rng = random.Random(rng_seed)
        self.archive: dict[str, Cell] = {}

    def seed_from(self, env: AttackEnvProtocol) -> None:
        """Reset the env and create the root archive cell.

        Args:
            env: The attack environment.
        """
        env.reset()
        trace = env.export_trace_dict()
        cell = Cell(
            snapshot=env.snapshot(),
            user_messages=list(trace.get("user_messages", [])),
            score_hash=score_hash(trace),
            depth=0,
        )
        self.archive[cell.score_hash] = cell

    def _select(self) -> Cell:
        """Weighted pick favoring low-visit and breach-adjacent cells.

        Returns:
            The chosen frontier cell.
        """
        cells = list(self.archive.values())
        max_visits = max((c.visits for c in cells), default=0) + 1
        weights = [
            (max_visits - c.visits) / max_visits + (2.0 if c.breached else 0.0)
            for c in cells
        ]
        total = sum(weights) or 1.0
        r = self.rng.uniform(0.0, total)
        acc = 0.0
        for cell, weight in zip(cells, weights, strict=True):
            acc += weight
            if r <= acc:
                return cell
        return cells[-1]

    def step(self, env: AttackEnvProtocol) -> list[Cell]:
        """Run one select-restore-branch round, adding novel cells.

        Args:
            env: The attack environment.

        Returns:
            The newly added cells this round.
        """
        parent = self._select()
        parent.visits += 1
        added: list[Cell] = []
        for mutation in self.mutations:
            env.restore(parent.snapshot)
            result = env.interact(mutation, max_tool_hops=self.max_hops)
            if result.agent_refused:
                continue
            trace = env.export_trace_dict()
            key = score_hash(trace)
            if key in self.archive:
                continue
            predicates = eval_predicates(trace)
            cell = Cell(
                snapshot=env.snapshot(),
                user_messages=list(trace.get("user_messages", [])),
                score_hash=key,
                depth=parent.depth + 1,
                breached=bool(predicates) and is_breach(trace, predicates),
                predicates=predicates,
            )
            self.archive[key] = cell
            added.append(cell)
        return added

    def breached_cells(self) -> list[Cell]:
        """Return archive cells whose trace constituted a breach.

        Returns:
            The breached cells.
        """
        return [c for c in self.archive.values() if c.breached]
