"""Select the final replayable candidate set to maximize the attack score.

The SDK scorer sums predicate severity over EVERY returned finding with no dedup
(see ``docs/strategy.md``); unique cells add only +2 each. So the winning set is the
highest-severity candidates REPEATED to fill the replay budget, not one candidate
per unique cell. We rank distinct firing candidates by severity-per-message (replay
cost scales with message count), then fill the count by cycling the top-k best-first.
The attack's replay-budget cap then truncates this best-first list to what actually
replays within the evaluator's time budget.
"""

from __future__ import annotations

from collections.abc import Sequence

from jed_attack.submission.engine import Cell

_SEVERITY_W = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}


def _severity(cell: Cell) -> int:
    """Return the summed severity weight of a cell's predicates.

    Args:
        cell: The archive cell.

    Returns:
        The total severity weight.
    """
    return sum(_SEVERITY_W.get(int(p.get("severity", 1)), 1) for p in cell.predicates)


def _severity_per_message(cell: Cell) -> float:
    """Return severity weight per message — the value density under replay cost.

    Args:
        cell: The archive cell.

    Returns:
        Severity weight divided by chain length (replay cost proxy).
    """
    return _severity(cell) / max(1, len(cell.user_messages))


def select_candidates(
    cells: Sequence[Cell],
    *,
    max_candidates: int = 300,
    max_messages: int = 32,
    repeat_pool: int = 8,
) -> list[list[str]]:
    """Choose the candidate chains that maximize score, filling by repetition.

    Distinct firing candidates are deduped to the highest-severity one per scored
    cell, ranked by severity-per-message, and the returned list is filled to
    ``max_candidates`` by cycling the top ``repeat_pool`` candidates best-first —
    so the highest-value attacks are repeated (each repeat adds its severity again)
    while keeping a robust spread across guardrails.

    Args:
        cells: Archive/farmed cells to choose from.
        max_candidates: Number of candidate chains to return (filled by repetition).
        max_messages: Hard cap on messages per candidate chain.
        repeat_pool: How many distinct top candidates to cycle when filling.

    Returns:
        A list of ``max_candidates`` user-message chains (repetition-filled), or
        fewer only if no firing candidate is available.
    """
    best_by_hash: dict[str, Cell] = {}
    for cell in cells:
        if not cell.user_messages or len(cell.user_messages) > max_messages:
            continue
        if _severity(cell) <= 0:
            continue  # non-firing: repeating it would waste replay budget for zero score
        current = best_by_hash.get(cell.score_hash)
        if current is None or _severity(cell) > _severity(current):
            best_by_hash[cell.score_hash] = cell

    ranked = sorted(
        best_by_hash.values(),
        key=lambda c: (_severity_per_message(c), _severity(c), c.breached),
        reverse=True,
    )
    if not ranked:
        return []

    pool = ranked[: max(1, repeat_pool)]
    out: list[list[str]] = []
    index = 0
    while len(out) < max_candidates:
        out.append(list(pool[index % len(pool)].user_messages))
        index += 1
    return out
