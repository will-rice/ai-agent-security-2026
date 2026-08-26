"""Tests for the parallel island archive set (GPU-free)."""

from pathlib import Path

import pytest

from jed_attack.campaign import archive
from jed_attack.campaign.islands import IslandSet


def _elite(family: str, density_gpt: float, bucket: int = 5) -> archive.Elite:
    """A minimal firing gpt_oss elite whose throughput sets its board density."""
    return archive.Elite(
        text=f"http.post data=SECRET_MARKER url={{u}} {family}{bucket}",
        mtype="exfil",
        throughput={"gpt_oss": density_gpt, "gemma_4": 0.0},
        severity={"gpt_oss": 16.0, "gemma_4": 0.0},
        diagnosis="",
        family=family,
        bucket=bucket,
        url_scheme="s://{h}",
        turns={"gpt_oss": 2.0, "gemma_4": 0.0},
        input_chars=40,
    )


def test_islandset_isolates_inserts_and_finds_global_best() -> None:
    """Inserts land only in their target island; global best is the densest."""
    islands = IslandSet.for_count(3)
    islands.insert(1, _elite("plain", 0.02))
    islands.insert(2, _elite("forge", 0.05))
    # each insert lands only in its island
    assert len(islands.archives[1].frontier()) == 1
    assert len(islands.archives[0].frontier()) == 0
    # global best is the densest across islands (island 2)
    best = islands.global_best_elite()
    assert best is not None and best.family == "forge"
    assert islands.best_island() == 2


def test_islandset_persist_round_trip(tmp_path: Path) -> None:
    """to_jsonl/from_jsonl round-trips island frontiers through per-island files."""
    islands = IslandSet.for_count(2)
    islands.insert(1, _elite("plain", 0.02))
    islands.to_jsonl(tmp_path)
    restored = IslandSet.from_jsonl(tmp_path, 2)
    assert len(restored.archives[1].frontier()) == 1
    best = restored.global_best_elite()
    assert best is not None and best.family == "plain"


def test_novelty_island_keeps_the_more_novel_of_a_contested_cell() -> None:
    """Island 0 keeps the structurally-distinct elite over the higher-throughput one."""
    from jed_attack.campaign.islands import novelty

    islands = IslandSet.for_count(2)
    # island 0 (novelty) already holds two 'plain' shapes -> a third 'plain' is FAMILIAR
    islands.insert(0, _elite("plain", 0.09, bucket=5))
    islands.insert(0, _elite("verb_variant", 0.09, bucket=9))
    familiar = _elite(
        "plain", 0.20, bucket=5
    )  # high throughput but same as an existing cell
    novel = _elite(
        "injection_variant", 0.01, bucket=2
    )  # low throughput but structurally new
    others = islands.archives[0].frontier()
    assert novelty(novel, others) > novelty(familiar, others)
    islands.insert(0, novel)
    texts = {e.family for e in islands.archives[0].frontier()}
    assert (
        "injection_variant" in texts
    )  # the novel shape is kept despite low throughput


def test_novelty_insert_still_requires_firing() -> None:
    """A non-firing elite is never kept, even by the novelty island."""
    islands = IslandSet.for_count(1)
    dud = archive.Elite(
        text="no marker here",
        mtype="exfil",
        throughput={"gpt_oss": 0.0, "gemma_4": 0.0},
        severity={"gpt_oss": 0.0, "gemma_4": 0.0},
        diagnosis="",
        family="dud",
        bucket=1,
    )
    assert islands.insert(0, dud) is False
    assert islands.archives[0].frontier() == []


def _elite_2d(
    family: str,
    bucket: int,
    gpt_throughput: float = 0.0,
    gemma_throughput: float = 0.0,
) -> archive.Elite:
    """A firing elite whose severity is fixed at 16 on whichever model has throughput.

    Lets a test build genuinely Pareto-INCOMPARABLE elites (one strong on gpt_oss,
    another strong on gemma_4), unlike ``_elite`` (gemma_4 always 0), which is needed
    to grow island 0's frontier past a single member.
    """
    return archive.Elite(
        text=f"http.post data=SECRET_MARKER url={{u}} {family}{bucket}",
        mtype="exfil",
        throughput={"gpt_oss": gpt_throughput, "gemma_4": gemma_throughput},
        severity={
            "gpt_oss": 16.0 if gpt_throughput > 0.0 else 0.0,
            "gemma_4": 16.0 if gemma_throughput > 0.0 else 0.0,
        },
        diagnosis="",
        family=family,
        bucket=bucket,
    )


def test_novelty_insert_purges_a_stale_same_cell_residue_on_a_win() -> None:
    """A novelty win must not be silently vetoed by a stale off-frontier cellmate.

    Regression for: ``_novelty_insert`` computed its ``blockers`` from
    ``archive0.frontier()`` only, then evicted just those blockers before delegating to
    ``archive0.insert(elite)``. But ``Archive.insert`` also rejects on ANY same-cell
    member that dominates the newcomer -- including a member that fell off the
    frontier earlier (dominated by a THIRD, cross-cell elite) and was therefore never
    even considered a blocker. That stale residue silently vetoed a legitimate novelty
    winner. Repro: seed a same-cell elite (``incumbent``) that a cross-cell elite
    (``dominator``) later pushes off the frontier (so it becomes invisible residue),
    plus an unrelated Pareto-incomparable elite (``rival``) so the archive has >1
    frontier member. A same-cell newcomer (``winner``) that legitimately out-novels
    ``dominator`` (island 0's only real frontier blocker) must still be kept, with the
    stale ``incumbent`` purged from the cell rather than vetoing it.
    """
    islands = IslandSet.for_count(1)
    incumbent = _elite_2d("f", 10, gpt_throughput=0.05)
    dominator = _elite_2d("g", 1, gpt_throughput=0.09)  # dominates incumbent
    rival = _elite_2d("g", 3, gemma_throughput=0.05)  # Pareto-incomparable to dominator
    winner = _elite_2d("f", 10, gpt_throughput=0.001)  # incumbent's cell; wins novelty

    assert islands.insert(0, incumbent) is True
    assert islands.insert(0, dominator) is True
    assert islands.insert(0, rival) is True
    # incumbent is now dominated (off-frontier) but still squats its (family, bucket)
    # cell -- exactly the invisible-residue setup the bug needs.
    assert incumbent not in islands.archives[0].frontier()

    assert islands.insert(0, winner) is True
    families = {(e.family, e.bucket) for e in islands.archives[0].frontier()}
    assert ("f", 10) in families  # winner kept, not silently dropped
    assert ("g", 3) in families  # rival untouched
    assert ("g", 1) not in families  # dominator lost the novelty contest to winner


def test_stall_triggers_reset_and_migration_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flat generations after the priming call stall; reset clears; migration ticks."""
    from jed_attack.campaign import config
    from jed_attack.campaign import islands as isl

    monkeypatch.setattr(config, "ISLAND_STAGNATION_GENERATIONS", 3)
    monkeypatch.setattr(config, "ISLAND_MIGRATION_GENERATIONS", 4)
    islands = IslandSet.for_count(2)
    islands.insert(1, _elite("plain", 0.05))
    d = islands.local_best_density(1)
    assert islands.note_generation(1, d) is False  # priming call: baseline only
    flags = [islands.note_generation(1, d) for _ in range(3)]  # flat density
    assert flags == [False, False, True]  # 3rd flat generation after baseline stalls
    islands.reset_island(1, _elite("forge", 0.02))
    assert islands.stall[1] == 0 and len(islands.archives[1].frontier()) == 1
    assert not isl.should_migrate(3) and isl.should_migrate(4)


def test_improving_island_never_stalls() -> None:
    """An island that improves every generation must never register a stall.

    Regression guard for a design where ``insert()`` pre-applies the exact
    ``best_seen`` bump ``note_generation`` exists to detect: that makes
    ``note_generation``'s improvement branch unreachable in a real interleaved
    insert -> note_generation loop, so an island improving every generation would
    still stall and get hard-reset. ``note_generation`` must be the SOLE owner of
    ``best_seen`` for this to pass.
    """
    islands = IslandSet.for_count(2)
    for gen in range(5):
        # each insert strictly Pareto-dominates the last (same cell, higher gpt
        # throughput) -> local_best_density genuinely improves every generation.
        islands.insert(1, _elite("plain", 0.01 * (gen + 1)))
        flagged = islands.note_generation(1, islands.local_best_density(1))
        assert flagged is False
    assert islands.stall[1] == 0
