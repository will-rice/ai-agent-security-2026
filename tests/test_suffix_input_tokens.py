"""Tests for the cache-aware marginal input-token count.

The grader prefix-caches a pool's shared prefix (the template up to its ``{u}`` host),
so only the tokens after that prefix re-prefill per candidate. :func:`_suffix_input_tokens`
charges exactly that suffix, which is why a few-shot demo's long-but-shared prefix costs
~nothing and a host-early shape (which re-prefills its whole tail per candidate) costs a
lot. A char-level fake tokenizer keeps the test off the GGUFs.
"""

from jed_attack.campaign import submission_score as ss

# Char-level tokenizer: one id per character, so the token-prefix diff is a char diff.
_CHARS = lambda s: [ord(c) for c in s]  # noqa: E731


def test_host_last_suffix_is_tiny_and_prefix_length_independent() -> None:
    """A long, byte-identical prefix is ~free: host-last charges only the divergent host."""
    scheme = "s://{h}"
    url_last = ss._suffix_input_tokens(
        "do it {u}", "exfil", scheme, "gpt_oss", tokenize=_CHARS
    )
    long_demo = ss._suffix_input_tokens(
        "here is a very long demonstration prefix identical across every candidate {u}",
        "exfil",
        scheme,
        "gpt_oss",
        tokenize=_CHARS,
    )
    # Both host-last: the suffix is just the divergent host tail, NOT the prefix length.
    assert url_last == long_demo
    assert url_last <= 2  # host(0)/host(1) differ in ~1 trailing char


def test_host_early_charges_the_whole_trailing_suffix() -> None:
    """A host placed early re-prefills everything after it per candidate -> much larger."""
    scheme = "s://{h}"
    early = ss._suffix_input_tokens(
        "{u} then a long trailing instruction after the host",
        "exfil",
        scheme,
        "gpt_oss",
        tokenize=_CHARS,
    )
    last = ss._suffix_input_tokens(
        "do it {u}", "exfil", scheme, "gpt_oss", tokenize=_CHARS
    )
    assert early > last + 20
