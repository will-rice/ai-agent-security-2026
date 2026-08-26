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
