"""Candidate selector maximizes unique scored cells under the caps."""

from jed_attack.submission.engine import Cell
from jed_attack.submission.selector import select_candidates


def _cell(h: str, msgs: list[str], breached: bool = False, sev: int = 0) -> Cell:
    cell = Cell(
        snapshot=None, user_messages=msgs, score_hash=h, depth=1, breached=breached
    )
    if sev:
        cell.predicates = [{"predicate": "EXFILTRATION", "severity": sev}]
    return cell


def test_dedupes_by_score_hash() -> None:
    """Two cells with the same scored hash collapse to one candidate."""
    cells = [_cell("a", ["m1"]), _cell("a", ["m1", "m2"]), _cell("b", ["n1"])]
    out = select_candidates(cells, max_candidates=10)
    assert len(out) == 2


def test_respects_message_cap_and_candidate_cap() -> None:
    """Over-long chains are dropped and the candidate cap is enforced."""
    long_chain = _cell("c", [f"m{i}" for i in range(40)])
    ok = [_cell(str(i), [f"x{i}"]) for i in range(5)]
    out = select_candidates([long_chain, *ok], max_candidates=3, max_messages=32)
    assert len(out) == 3
    assert all(len(chain) <= 32 for chain in out)


def test_prioritizes_breaches_when_capped() -> None:
    """Under a tight cap, breached/high-severity cells are kept first."""
    plain = [_cell(f"p{i}", [f"p{i}"]) for i in range(5)]
    breach = _cell("hit", ["hit"], breached=True, sev=5)
    out = select_candidates([*plain, breach], max_candidates=1)
    assert out == [["hit"]]
