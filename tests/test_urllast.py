"""Tests for the url-last single-token shape builder (campaign.urllast)."""

from jed_attack.campaign.urllast import (
    FORGE,
    URLLAST_PREFIX,
    build_urllast_pools,
    single_token_hosts,
    urllast_message,
)

# Fake tokenizer: prefix is 5 tokens; the "1-token" hosts add 1, everything else 2.
_ONE_TOKEN = {"aa", "the", "in", "at"}


def _fake_tokenize(text: str) -> int:
    if text == URLLAST_PREFIX:
        return 5
    host = text[len(URLLAST_PREFIX) :]
    return 5 + (1 if host in _ONE_TOKEN else 2)


def test_urllast_message_puts_host_last_before_forge() -> None:
    """The host is last; the gpt forge (if any) is the only thing after it."""
    gpt = urllast_message("aa", forge=True)
    assert gpt.endswith(FORGE)
    assert gpt[: -len(FORGE)].endswith("url=s://aa")  # host is last before the forge
    gemma = urllast_message("aa", forge=False)
    assert gemma.endswith("url=s://aa")  # host is the literal final token, no forge
    assert "SECRET_MARKER" in gpt and "SECRET_MARKER" in gemma  # fires EXFIL


def test_single_token_hosts_keeps_only_one_token_valid_domains() -> None:
    """Only valid domains that add exactly one token survive."""
    cands = ["aa", "zz", "the", "BAD", "in", "a", "toolonghostname", "9x"]
    kept = single_token_hosts(_fake_tokenize, cands, n=100)
    # aa/the/in are 1-token valid; zz adds 2; BAD is uppercase; a is too short;
    # toolonghostname is too long; 9x is not letter-led.
    assert kept == ["aa", "the", "in"]


def test_single_token_hosts_respects_n() -> None:
    """Enumeration stops at n."""
    assert single_token_hosts(_fake_tokenize, ["aa", "the", "in", "at"], n=2) == [
        "aa",
        "the",
    ]


def test_build_urllast_pools_shape() -> None:
    """One-message chains; gpt forged, gemma bare; hosts preserved."""
    gpt, gemma = build_urllast_pools(["aa", "bb"], ["cc"])
    assert gpt == [[urllast_message("aa", True)], [urllast_message("bb", True)]]
    assert gemma == [[urllast_message("cc", False)]]
    assert all(len(chain) == 1 for chain in gpt + gemma)  # single-message candidates
