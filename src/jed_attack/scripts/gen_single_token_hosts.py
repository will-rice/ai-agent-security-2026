"""Regenerate the single-token host set from the two victim GGUF tokenizers.

A host is kept iff it tokenizes to exactly one token under BOTH gpt_oss and gemma
(vocab-only). Writes the sorted 2-char-then-3-char lowercase a-z list to
campaign/single_token_hosts.json. Re-run whenever a victim model changes.

Usage: uv run python -m jed_attack.scripts.gen_single_token_hosts
"""

import itertools
import json
import logging
import string
from pathlib import Path

from llama_cpp import Llama

from jed_attack.campaign import config

_OUT = Path(config.__file__).parent / "single_token_hosts.json"
_MODELS = ("models/gpt-oss-20b-Q4_K_M.gguf", "models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    toks = [Llama(model_path=m, vocab_only=True, verbose=False) for m in _MODELS]
    hosts: list[str] = []
    for length in (2, 3):
        for combo in itertools.product(string.ascii_lowercase, repeat=length):
            h = "".join(combo)
            if all(len(t.tokenize(h.encode(), add_bos=False)) == 1 for t in toks):
                hosts.append(h)
    _OUT.write_text(json.dumps(hosts), encoding="utf-8")
    logging.info("wrote %d single-token hosts -> %s", len(hosts), _OUT)


if __name__ == "__main__":
    main()
