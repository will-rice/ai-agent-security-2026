"""Token-ablation minimizer: shrink a firing shape to its local token floor.

Exploits exact, deterministic GGUF replay (greedy decode): greedily delete each FREE
span of the input template, keep any edit that STILL fires single-post AND lowers the
TOTAL token count (input + generated), and iterate to a fixed point. This is the
black-box analog of a GCG coordinate step -- on our prompt tokens, with exact reward
instead of gradients. The proposer approximates this by hand; this does it exhaustively
and provably to a local minimum, so it is run as a post-pass on the champion.

Correctness rests on the firing check, not the protected-literal mask: any edit that
breaks firing (deletes ``SECRET_MARKER``, ``http.post``, a harmony control token, ...)
scores non-firing and is rejected. The mask only skips those doomed trials to save
replays. ``{u}`` must survive and stay LAST (the host-last throughput invariant), and
severity must stay EXACTLY one post -- a runaway multi-post edit is rejected like a dud.
"""

import functools
import logging
import re
from collections.abc import Callable

from jed_attack.campaign import config
from jed_attack.campaign.fill import render_message
from jed_attack.campaign.guardrails import GATE_GUARDRAILS
from jed_attack.campaign.submission_score import (
    _fillable_severity,
    _severity,
    _trace_gen_tokens,
    replay_trace,
)
from jed_attack.harness.models import gguf_target_path

_log = logging.getLogger(__name__)

_GATE = GATE_GUARDRAILS[config.GATE_GUARDRAIL_NAME]
# Required literals: skip deletion trials inside them (the firing check would reject the
# edit anyway). NOT a correctness boundary -- just a replay-saving heuristic.
_PROTECTED = re.compile(r"<\|[^|]*\|>|SECRET_MARKER|\{u\}|http\.post|data=|url=")
_ONE_POST = int(config.EXFIL_FIRE_WEIGHT)


@functools.lru_cache(maxsize=None)
def _tokenizer(model: str) -> Callable[[str], int]:
    """A cached vocab-only token counter (harmony specials count as one token)."""
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(gguf_target_path(model, config.MODELS_DIR)),
        vocab_only=True,
        verbose=False,
    )
    return lambda s: len(llm.tokenize(s.encode(), add_bos=False, special=True))


def _total_tokens(text: str, url_scheme: str, model: str) -> tuple[int, int, int]:
    """Return ``(severity, gen_tokens, input_tokens)`` for the template on host 0."""
    rendered = render_message(text, "exfil", url_scheme, 0)
    trace, predicates, _ = replay_trace(rendered, model, _GATE)
    return _severity(predicates), _trace_gen_tokens(trace), _tokenizer(model)(rendered)


def _robust_severity(text: str, url_scheme: str, model: str) -> float:
    """Fill-ability severity across the word-host stress probes (else 0 if too fragile).

    The primary-host firing check is fast but can bank a host-fragile minimum (fires on
    ``s://in`` but not ``s://an``). Every ACCEPTED edit is re-checked here so the token
    floor we return still passes the same robustness gate the shipped pool uses.
    """
    return _fillable_severity(text, "exfil", url_scheme, model, _GATE, _ONE_POST)


def _free_spans(text: str) -> list[tuple[int, int]]:
    """Deletable spans: maximal free word-runs first, then single free chars."""
    protected = bytearray(len(text))
    for m in _PROTECTED.finditer(text):
        for i in range(m.start(), m.end()):
            protected[i] = 1
    words: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        if not protected[i] and not text[i].isspace():
            j = i
            while j < len(text) and not protected[j] and not text[j].isspace():
                j += 1
            words.append((i, j))
            i = j
        else:
            i += 1
    chars = [(i, i + 1) for i in range(len(text)) if not protected[i]]
    return words + chars


def _valid(cand: str) -> bool:
    """Structural guards preserved through every edit (firing decides the rest)."""
    return (
        "{u}" in cand
        and "SECRET_MARKER" in cand
        and cand.rstrip().endswith("{u}")  # host stays LAST (throughput invariant)
    )


def minimize_shape(text: str, url_scheme: str, model: str) -> tuple[str, int, int]:
    """Greedy first-improvement deletion to a fixed point.

    Args:
        text: The firing shape template (contains ``{u}``, ends at the host).
        url_scheme: The ``{h}``-slotted URL template (e.g. ``s://{h}``).
        model: The victim the shape is scored on (``gpt_oss`` / ``gemma_4``).

    Returns:
        ``(minimized_text, gen_tokens, input_tokens)`` for the local token floor. Raises
        if the seed does not fire single-post (nothing to minimize).
    """
    sev, gen, inp = _total_tokens(text, url_scheme, model)
    if sev != _ONE_POST:
        raise ValueError(f"seed does not fire single-post on {model} (severity={sev})")
    best, best_total = text, gen + inp
    improved = True
    while improved:
        improved = False
        for s, e in _free_spans(best):
            cand = best[:s] + best[e:]
            if not _valid(cand):
                continue
            csev, cgen, cinp = _total_tokens(cand, url_scheme, model)
            # Accept only a leaner, single-post, AND still-ROBUST edit. The robust check
            # (extra word-host replays) runs only on a candidate improvement, not every
            # trial, so it costs little while keeping the returned floor un-fragile.
            if (
                csev == _ONE_POST
                and cgen + cinp < best_total
                and _robust_severity(cand, url_scheme, model) >= _ONE_POST
            ):
                _log.info(
                    "%s: -%d tok -> %d (in=%d gen=%d) %r",
                    model,
                    best_total - (cgen + cinp),
                    cgen + cinp,
                    cinp,
                    cgen,
                    cand,
                )
                best, best_total = cand, cgen + cinp
                improved = True
                break
    _sev, fgen, finp = _total_tokens(best, url_scheme, model)
    return best, fgen, finp
