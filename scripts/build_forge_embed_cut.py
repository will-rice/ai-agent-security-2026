#!/usr/bin/env python3
"""Build the forge-embedded url-last cut -- a board-only A/B vs the url-last champion.

Identical to ``build_urllast_cut.py`` except the gpt_oss pool embeds the host in the
forge (host truly last) so the ~15-token forge tail no longer re-prefills per candidate.
The gemma pool is unchanged (plain url-last), so a board score isolates the gpt-column
effect. Firing is NOT checked here (needs the GPU backend); run
``scripts/verify_cut.py``
on the written cut to drop non-firing hosts and stamp the gate manifest.
"""

from jed_attack.campaign import config
from jed_attack.campaign.assemble import build_permodel
from jed_attack.campaign.urllast import (
    build_forge_embed_pools,
    enumerate_single_token_hosts,
)
from jed_attack.harness.models import gguf_target_path

# --- Fixed build config ---
N_HOSTS = 2000  # candidates per model (grader replays up to MAX_REPLAY_FINDINGS=2000)
OUT_DIR = config.SUBMISSION_CUTS_DIR / "forge_embed_urllast"
# --------------------------


def main() -> None:
    """Enumerate single-token hosts, build the forge-embedded cut, print the path."""
    gpt_hosts = enumerate_single_token_hosts(
        str(gguf_target_path("gpt_oss", config.MODELS_DIR)), N_HOSTS
    )
    gemma_hosts = enumerate_single_token_hosts(
        str(gguf_target_path("gemma_4", config.MODELS_DIR)), N_HOSTS
    )
    print(f"single-token hosts: gpt_oss={len(gpt_hosts)} gemma_4={len(gemma_hosts)}")
    gpt_pool, gemma_pool = build_forge_embed_pools(gpt_hosts, gemma_hosts)
    out = build_permodel(gpt_pool, gemma_pool, OUT_DIR)
    print(f"wrote {out} ({out.stat().st_size // 1024}K)")
    print(f"next: uv run python scripts/verify_cut.py {out}")


if __name__ == "__main__":
    main()
