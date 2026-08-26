#!/usr/bin/env python3
"""Build the single-token url-last submission cut -- our board-confirmed best shape.

Settings are ALL-CAPS CONSTANTS (no CLI flags): the shape is solved, so there is nothing
to tune per run. Enumerates single-token hosts from each model's GGUF vocab, builds the
url-last per-model pools, and writes the router ``attack.py``. Firing is NOT checked
(it needs the GPU-resident backend); run this, then run the firing gate
(``python -m jed_attack.scripts.verify_cut``) on the written cut: it drops non-firing
hosts and stamps the hash-bound manifest the submit gate requires. ~99.8% of
single-token hosts fire, so the shipped pool is essentially the full enumeration.
"""

from jed_attack.campaign import config
from jed_attack.campaign.assemble import build_permodel
from jed_attack.campaign.urllast import (
    build_urllast_pools,
    enumerate_single_token_hosts,
)
from jed_attack.harness.models import gguf_target_path

# --- Fixed build config ---
N_HOSTS = 2000  # candidates per model (grader replays up to MAX_REPLAY_FINDINGS=2000)
OUT_DIR = config.SUBMISSION_CUTS_DIR / "urllast_single_token"
# --------------------------


def main() -> None:
    """Enumerate single-token hosts, build the url-last cut, print the path."""
    gpt_hosts = enumerate_single_token_hosts(
        str(gguf_target_path("gpt_oss", config.MODELS_DIR)), N_HOSTS
    )
    gemma_hosts = enumerate_single_token_hosts(
        str(gguf_target_path("gemma_4", config.MODELS_DIR)), N_HOSTS
    )
    print(f"single-token hosts: gpt_oss={len(gpt_hosts)} gemma_4={len(gemma_hosts)}")
    gpt_pool, gemma_pool = build_urllast_pools(gpt_hosts, gemma_hosts)
    out = build_permodel(gpt_pool, gemma_pool, OUT_DIR)
    print(f"wrote {out} ({out.stat().st_size // 1024}K)")
    print(f"next: uv run python -m jed_attack.scripts.verify_cut {out}")


if __name__ == "__main__":
    main()
