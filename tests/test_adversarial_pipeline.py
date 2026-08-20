"""Tests for the hybrid search pipeline (jed_attack.adversarial.pipeline)."""

from jed_attack.adversarial.pipeline import beats_floor


def test_beats_floor_true_when_strictly_above() -> None:
    """A board strictly above the floor beats it."""
    assert beats_floor(96.0, 95.9)


def test_beats_floor_false_when_equal_or_below() -> None:
    """A tie or a loss does not beat the floor."""
    # a tie with the champion is NOT a win -- avoids shipping a lateral shape.
    assert not beats_floor(95.9, 95.9)
    assert not beats_floor(90.0, 95.9)
