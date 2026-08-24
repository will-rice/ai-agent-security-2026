"""Tests for the pre-submission firing gate (campaign.verify)."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from jed_attack.campaign import verify
from jed_attack.campaign.assemble import build_permodel

_Replay = Callable[[str, str, object], tuple[dict, list, float]]


def _fake_replay(fire_if: Callable[[str], bool]) -> _Replay:
    """A replay_chain stub firing one EXFIL per message where ``fire_if`` is true.

    A K-message chain fires K posts, matching the grader's per-message replay. The trace
    carries ``raw_gen_tokens`` = total message length, so ordering-by-cost is testable.
    """

    def replay(
        messages: Sequence[str], model: str, guard: object
    ) -> tuple[dict, list, float]:
        preds = [
            {"predicate": "EXFILTRATION", "severity": 5} for m in messages if fire_if(m)
        ]
        trace = {"raw_gen_tokens": float(sum(len(m) for m in messages))}
        return trace, preds, 0.1

    return replay


def test_firing_only_drops_non_firing(monkeypatch: pytest.MonkeyPatch) -> None:
    """firing_only keeps only candidates that fire on the model."""
    monkeypatch.setattr(verify, "replay_chain", _fake_replay(lambda c: "GOOD" in c))
    chains = [["GOOD a"], ["DUD b"], ["GOOD c"]]
    assert verify.firing_only(chains, "gpt_oss") == [["GOOD a"], ["GOOD c"]]


def test_firing_only_gates_on_the_given_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """firing_only(predicate=CONFUSED_DEPUTY) keeps deputy-firing candidates only."""

    def replay(
        messages: Sequence[str], model: str, guard: object
    ) -> tuple[dict, list, float]:
        preds = []
        for m in messages:
            if "NOTIFY" in m:
                preds.append({"predicate": "CONFUSED_DEPUTY", "severity": 3})
            elif "EXFIL" in m:
                preds.append({"predicate": "EXFILTRATION", "severity": 5})
        return {"raw_gen_tokens": 1.0}, preds, 0.1

    monkeypatch.setattr(verify, "replay_chain", replay)
    chains = [["NOTIFY a"], ["EXFIL b"], ["NOTIFY c"]]
    kept = verify.firing_only(
        chains, "gpt_oss", min_severity=3, predicate="CONFUSED_DEPUTY"
    )
    assert kept == [["NOTIFY a"], ["NOTIFY c"]]  # EXFIL one is not a deputy fire


def test_firing_only_orders_leanest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Firing chains come back ordered by ascending gen tokens (shortest-job-first).

    The grader replays in list order until its deadline, so the cheapest firing
    candidates must ship first to maximize how many complete.
    """
    monkeypatch.setattr(verify, "replay_chain", _fake_replay(lambda m: "GOOD" in m))
    fat = ["GOOD " + "x" * 40]  # more chars -> more gen tokens -> ships last
    lean = ["GOOD y"]  # fewest chars -> ships first
    mid = ["GOOD " + "z" * 15]
    assert verify.firing_only([fat, lean, mid], "gpt_oss") == [lean, mid, fat]


def test_firing_only_token_margin_drops_ballooners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With token_margin, a firing-but-token-heavy host (the balloon tail) is dropped.

    A host-dependent lean shape fires lean on most hosts but balloons on a few; the gate
    ships only the lean-firing hosts so the cut is not diluted by the ballooners.
    """
    monkeypatch.setattr(verify, "replay_chain", _fake_replay(lambda m: "GOOD" in m))
    lean = [["GOOD a"], ["GOOD b"], ["GOOD c"], ["GOOD d"]]  # ~6 tokens each
    balloon = ["GOOD " + "x" * 50]  # ~55 tokens -> above lean floor + margin
    kept = verify.firing_only(lean + [balloon], "gpt_oss", token_margin=4.0)
    assert balloon not in kept
    assert all(chain in kept for chain in lean)


def test_firing_only_default_keeps_token_heavy_firing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without token_margin (default), even a token-heavy firing candidate is kept."""
    monkeypatch.setattr(verify, "replay_chain", _fake_replay(lambda m: "GOOD" in m))
    balloon = ["GOOD " + "x" * 50]
    kept = verify.firing_only([["GOOD a"], balloon], "gpt_oss")  # no token filter
    assert balloon in kept


def _replay_with_hops(hops_for: Callable[[str], int]) -> _Replay:
    """Replay stub: fires EXFIL where SECRET_MARKER is present, hops_for events/msg."""

    def replay(
        messages: Sequence[str], model: str, guard: object
    ) -> tuple[dict, list, float]:
        preds = [
            {"predicate": "EXFILTRATION", "severity": 5}
            for m in messages
            if "SECRET_MARKER" in m
        ]
        events = [{} for m in messages for _ in range(hops_for(m))]
        return {"raw_gen_tokens": 10.0, "tool_events": events}, preds, 0.1

    return replay


def test_firing_only_check_hops_drops_off_spec_toolcall_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_hops drops a firing candidate whose tool-call count != its declared calls.

    A single-post shape declares one call via its one SECRET_MARKER; a host that makes
    the model emit extra tool calls (an 8-turn ballooner) is off-spec and dropped even
    though it fires.
    """
    monkeypatch.setattr(
        verify, "replay_chain", _replay_with_hops(lambda m: 3 if "BALLOON" in m else 1)
    )
    clean = ["SECRET_MARKER a"]  # 1 marker, 1 tool call -> on-spec
    balloon = ["SECRET_MARKER BALLOON b"]  # 1 marker, 3 tool calls -> off-spec
    kept = verify.firing_only([clean, balloon], "gpt_oss", check_hops=True)
    assert kept == [clean]


def test_firing_only_default_ignores_hops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without check_hops (default), an off-spec-hops firing candidate is kept."""
    monkeypatch.setattr(
        verify, "replay_chain", _replay_with_hops(lambda m: 3 if "BALLOON" in m else 1)
    )
    balloon = ["SECRET_MARKER BALLOON b"]
    assert verify.firing_only([balloon], "gpt_oss") == [balloon]


def test_firing_only_checks_all_messages_full_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multipost chain must fire EVERY message under full_fire.

    A later-post dud drops the whole chain, which chain[0]-only checking would ship.
    """
    monkeypatch.setattr(verify, "replay_chain", _fake_replay(lambda m: "GOOD" in m))
    kept = ["GOOD a SECRET_MARKER", "GOOD b SECRET_MARKER"]  # both fire -> 2*5=10
    dud = ["GOOD c SECRET_MARKER", "DUD d SECRET_MARKER"]  # 2nd duds -> 5 < 10 -> drop
    assert verify.firing_only([kept, dud], "gpt_oss", full_fire=True) == [kept]


def test_verify_and_filter_drops_duds_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duds are dropped, the cut is rewritten firing-only, and the manifest passes."""
    monkeypatch.setattr(verify, "replay_chain", _fake_replay(lambda c: "GOOD" in c))
    build_permodel(
        [["GOOD f1"], ["DUD f2"], ["GOOD f3"]], [["GOOD p1"], ["DUD p2"]], tmp_path
    )
    manifest = verify.verify_and_filter_cut(tmp_path / "attack.py")

    assert manifest["all_fire"] is True
    assert manifest["pools"]["_FORGE"]["input"] == 3
    assert manifest["pools"]["_FORGE"]["firing"] == 2
    assert manifest["pools"]["_PLAIN"]["dropped"] == 1
    # the rewritten cut contains ONLY firing candidates
    assert "DUD" not in (tmp_path / "attack.py").read_text()
    # and the gate passes on the cleaned cut
    ok, reason = verify.is_verified(tmp_path / "attack.py")
    assert ok, reason


def test_verify_refuses_a_pool_with_zero_firing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pool with no firing candidate raises rather than shipping an empty pool."""
    monkeypatch.setattr(verify, "replay_chain", _fake_replay(lambda c: "GOOD" in c))
    build_permodel([["GOOD f"]], [["DUD p"]], tmp_path)  # plain pool all-dud
    with pytest.raises(ValueError, match="ZERO firing"):
        verify.verify_and_filter_cut(tmp_path / "attack.py")


def test_gate_fails_without_manifest(tmp_path: Path) -> None:
    """The gate refuses a cut with no verified.json beside it."""
    build_permodel([["x"]], [["y"]], tmp_path)
    ok, reason = verify.is_verified(tmp_path / "attack.py")
    assert not ok and "verified.json" in reason


def test_gate_fails_when_cut_changed_after_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing the cut after verification breaks the hash bind, so the gate refuses."""
    monkeypatch.setattr(verify, "replay_chain", _fake_replay(lambda c: True))
    build_permodel([["GOOD f"]], [["GOOD p"]], tmp_path)
    verify.verify_and_filter_cut(tmp_path / "attack.py")
    assert verify.is_verified(tmp_path / "attack.py")[0]
    # tamper after verification -> hash mismatch -> gate must refuse
    p = tmp_path / "attack.py"
    p.write_text(p.read_text() + "\n# tampered\n")
    ok, reason = verify.is_verified(p)
    assert not ok and "hash" in reason
