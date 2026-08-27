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
from jed_attack.harness.models import gguf_target_path

_OUT = Path(config.__file__).parent / "single_token_hosts.json"


def main() -> None:
    """Write the single-token host set (one token under both victims) to the JSON.

    Writes a compact ``json.dumps`` list with a trailing newline so the committed file
    is byte-reproducible from this generator (and stays excluded from prettier).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    toks = [
        Llama(
            model_path=str(gguf_target_path(m, config.MODELS_DIR)),
            vocab_only=True,
            verbose=False,
        )
        for m in config.MODELS
    ]
    hosts: list[str] = []
    for length in (2, 3):
        for combo in itertools.product(string.ascii_lowercase, repeat=length):
            h = "".join(combo)
            if all(len(t.tokenize(h.encode(), add_bos=False)) == 1 for t in toks):
                hosts.append(h)
    _OUT.write_text(json.dumps(hosts) + "\n", encoding="utf-8")
    logging.info("wrote %d single-token hosts -> %s", len(hosts), _OUT)


if __name__ == "__main__":
    main()
