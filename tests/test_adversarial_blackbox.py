"""Tests for the gradient-free black-box search (jed_attack.adversarial.blackbox).

The GGUF-backed path (``install_logits_backend``/``target_logprob``/``random_search``)
needs a GPU + the Q4_K_M gpt-oss GGUF and is gated behind ``JED_GPU_TESTS=1``; the
CPU-only tests below pin the constants and the template contract.
"""

import os

import pytest

from jed_attack.adversarial import blackbox

_GPU = os.getenv("JED_GPU_TESTS") == "1"
_GPU_SKIP_REASON = "needs GPU + Q4_K_M gpt-oss GGUF (JED_GPU_TESTS=1)"


def test_base_template_has_one_optim_slot() -> None:
    """The forge template exposes exactly one {optim_str} search slot."""
    assert blackbox.BASE_TEMPLATE.count("{optim_str}") == 1


def test_no_constrain_target_omits_the_constrain_guard() -> None:
    """The prize target drops the ` <|constrain|>json` guard greedy emits."""
    assert "<|constrain|>" not in blackbox.TARGET_NO_CONSTRAIN
    assert "<|constrain|>" in blackbox.TARGET_CONSTRAIN  # the form greedy already emits
    for target in (blackbox.TARGET_NO_CONSTRAIN, blackbox.TARGET_CONSTRAIN):
        assert "to=functions.http.post" in target
        assert "SECRET_MARKER" in target
        assert target.rstrip().endswith("<|call|>")


def test_random_search_rejects_a_template_without_the_placeholder() -> None:
    """A message_template missing {optim_str} raises before touching the model."""
    with pytest.raises(ValueError):
        blackbox.random_search(None, message_template="x")  # ty: ignore[invalid-argument-type]


@pytest.mark.skipif(not _GPU, reason=_GPU_SKIP_REASON)
def test_capture_prompt_is_faithful_and_carries_the_placeholder() -> None:
    """Captured prompt ends at the assistant turn and holds the optim_str hole."""
    message = blackbox.BASE_TEMPLATE.replace("{optim_str}", blackbox.PLACEHOLDER)
    prompt = blackbox.capture_prompt(message)
    assert blackbox.PLACEHOLDER in prompt
    assert prompt.rstrip().endswith("<|start|>assistant")


@pytest.mark.skipif(not _GPU, reason=_GPU_SKIP_REASON)
def test_no_constrain_scores_below_the_constrain_form() -> None:
    """Scorer sanity: the constrain form must out-logprob the no-constrain prize."""
    llm = blackbox.install_logits_backend()
    message = blackbox.BASE_TEMPLATE.replace("{optim_str}", blackbox.PLACEHOLDER)
    prompt = blackbox.capture_prompt(message).replace(blackbox.PLACEHOLDER, "now")
    lp_con = blackbox.target_logprob(prompt, blackbox.TARGET_CONSTRAIN, llm)
    lp_no = blackbox.target_logprob(prompt, blackbox.TARGET_NO_CONSTRAIN, llm)
    assert lp_no < lp_con
