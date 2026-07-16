"""Candidate selector fills the budget by repeating the highest-severity attacks."""

from jed_attack.submission.engine import Cell
from jed_attack.submission.selector import select_candidates


def _cell(h: str, msgs: list[str], breached: bool = False, sev: int = 0) -> Cell:
    cell = Cell(
        snapshot=None, user_messages=msgs, score_hash=h, depth=1, breached=breached
    )
    if sev:
        cell.predicates = [{"predicate": "EXFILTRATION", "severity": sev}]
    return cell


def test_fills_to_max_by_repetition() -> None:
    """Distinct firing cells are cycled to fill max_candidates (severity repeats)."""
    cells = [_cell("a", ["m1"], sev=5), _cell("b", ["n1"], sev=5)]
    out = select_candidates(cells, max_candidates=10)
    assert len(out) == 10  # filled by repetition, not one-per-cell
    assert {tuple(chain) for chain in out} == {("m1",), ("n1",)}


def test_ranks_by_severity_per_message() -> None:
    """The highest severity-per-message candidate wins the single slot."""
    low = _cell("lo", ["a"], sev=3)  # weight 4 over 1 msg
    high = _cell("hi", ["b"], sev=5)  # weight 16 over 1 msg
    out = select_candidates([low, high], max_candidates=1)
    assert out == [["b"]]


def test_respects_message_and_candidate_caps() -> None:
    """Over-long chains are excluded; output length equals max_candidates."""
    long_chain = _cell("c", [f"m{i}" for i in range(40)], sev=5)
    ok = [_cell(str(i), [f"x{i}"], sev=5) for i in range(3)]
    out = select_candidates([long_chain, *ok], max_candidates=5, max_messages=32)
    assert len(out) == 5
    assert all(len(chain) <= 32 for chain in out)
    assert [f"m{i}" for i in range(40)] not in out


def test_empty_when_no_firing_candidates() -> None:
    """Non-firing cells (severity 0) are skipped; nothing to repeat."""
    assert select_candidates([_cell("a", ["m"], sev=0)], max_candidates=5) == []
