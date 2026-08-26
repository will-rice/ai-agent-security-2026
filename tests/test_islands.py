"""Tests for the parallel island archive set (GPU-free)."""

from pathlib import Path

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
