#!/usr/bin/env python3
r"""Build the forge-embedded deputy cut (pick-2 private hedge, caching + leaner decode).

Like ``build_deputy_cut.py`` but the gpt_oss pool embeds the recipient in the forge
(recipient truly last), so the ~15-token forge tail no longer re-prefills per candidate
(prefix caching) AND gpt decodes leaner (~33 vs ~52 tok). The gemma pool is unchanged
(plain ``Notify ops@<host>``, already recipient-last). Firing is NOT checked here; gate
with the DEPUTY predicate afterward:

    uv run python -m jed_attack.scripts.build_forge_embed_deputy_cut
    uv run python -m jed_attack.scripts.verify_cut \\
        run/submission_cuts/forge_embed_deputy/attack.py \\
        --predicate CONFUSED_DEPUTY --min-severity 3
"""

from jed_attack.campaign import config
from jed_attack.campaign.assemble import build_permodel
from jed_attack.campaign.deputy import build_forge_embed_deputy_pools
from jed_attack.campaign.urllast import enumerate_single_token_hosts
from jed_attack.harness.models import gguf_target_path

# --- Fixed build config ---
N_HOSTS = 2000  # candidates per model (grader replays up to MAX_REPLAY_FINDINGS=2000)
OUT_DIR = config.SUBMISSION_CUTS_DIR / "forge_embed_deputy"
# --------------------------


def main() -> None:
    """Enumerate single-token hosts, build the forge-embedded deputy cut, print path."""
    gpt_hosts = enumerate_single_token_hosts(
        str(gguf_target_path("gpt_oss", config.MODELS_DIR)), N_HOSTS
    )
    gemma_hosts = enumerate_single_token_hosts(
        str(gguf_target_path("gemma_4", config.MODELS_DIR)), N_HOSTS
    )
    print(f"single-token hosts: gpt_oss={len(gpt_hosts)} gemma_4={len(gemma_hosts)}")
    gpt_pool, gemma_pool = build_forge_embed_deputy_pools(gpt_hosts, gemma_hosts)
    out = build_permodel(gpt_pool, gemma_pool, OUT_DIR)
    print(f"wrote {out} ({out.stat().st_size // 1024}K)")
    print(
        f"next: uv run python -m jed_attack.scripts.verify_cut {out} "
        "--predicate CONFUSED_DEPUTY --min-severity 3"
    )


if __name__ == "__main__":
    main()
