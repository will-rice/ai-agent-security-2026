"""Control-token alphabet + single-mutation forge construction (TEMPLATEFUZZ M4/M5)."""

from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer

MODEL_HF = {"gpt_oss": "openai/gpt-oss-20b", "gemma_4": "google/gemma-4-26B-A4B-it"}


@dataclass(frozen=True)
class MutationForge:
    """A single structural mutation: a control-token injection onto a base intent."""

    name: str
    text: str


@lru_cache(maxsize=4)
def special_tokens(model: str) -> tuple[str, ...]:
    """Every control/added token in ``model``'s tokenizer, longest-first.

    Uses ``get_added_vocab`` (added-token table); the harmony ``<|channel|>``/
    ``<|call|>`` and gemma ``<|tool_call>`` families live there.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_HF[model])
    assert tok is not None  # from_pretrained only returns None on an invalid repo id
    added = sorted(tok.get_added_vocab(), key=len, reverse=True)
    return tuple(added)


def single_token_forges(base: str, model: str) -> list[MutationForge]:
    """M4/M5 sweep: append each control token as a forged trailing fragment to ``base``.

    The differential sweep (Task 2) scores each; the analysis-forge is the member that
    keeps firing while collapsing generated tokens, so this rediscovers it.
    """
    forges: list[MutationForge] = []
    for tokstr in special_tokens(model):
        forges.append(MutationForge(name=f"append:{tokstr}", text=f"{base}{tokstr}"))
    return forges
