"""Tests for the single-token host pool (campaign.fill)."""

import pytest
from aicomp_sdk.evaluation.ops import MAX_REPLAY_FINDINGS

from jed_attack.campaign import fill


def test_single_token_host_set_exceeds_pool_cap() -> None:
    """The committed pool covers a full replay-capped submission with room to spare."""
    assert len(fill.SINGLE_TOKEN_HOSTS) > MAX_REPLAY_FINDINGS


def test_host_draws_unique_single_token_hosts_in_order() -> None:
    """host(index) is deterministic and index-ordered into SINGLE_TOKEN_HOSTS."""
    hs = [fill.host(i) for i in range(2000)]
    assert len(set(hs)) == 2000  # unique
    assert hs == fill.SINGLE_TOKEN_HOSTS[:2000]  # deterministic, index-ordered


def test_host_raises_past_the_set() -> None:
    """Indexing past the pool raises rather than wrapping or growing silently."""
    with pytest.raises(IndexError):
        fill.host(len(fill.SINGLE_TOKEN_HOSTS))
