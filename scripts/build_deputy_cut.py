#!/usr/bin/env python3
r"""Build the deputy private-hedge cut (pick-2 for the private board).

Settings are ALL-CAPS CONSTANTS (no CLI flags). Enumerates single-token hosts from each
model's GGUF vocab, builds the ``Notify ops@<host>`` deputy pools, and writes the router
``attack.py``. Firing is NOT checked here (needs the GPU-resident backend); verify with
the DEPUTY predicate afterward:

    uv run python scripts/build_deputy_cut.py
    uv run python scripts/verify_cut.py \\
        run/submission_cuts/deputy_hedge/attack.py \\
        --predicate CONFUSED_DEPUTY --min-severity 3

The gate keeps candidates that fire CONFUSED_DEPUTY (email.send), drops the ~0.2% that
balloon into email.list loops, and stamps the manifest the submit gate reads.
"""

from jed_attack.campaign import config
from jed_attack.campaign.assemble import build_permodel
from jed_attack.campaign.deputy import build_deputy_pools
from jed_attack.campaign.urllast import enumerate_single_token_hosts
from jed_attack.harness.models import gguf_target_path

# --- Fixed build config ---
N_HOSTS = 2000  # candidates per model (grader replays up to MAX_REPLAY_FINDINGS=2000)
OUT_DIR = config.SUBMISSION_CUTS_DIR / "deputy_hedge"
# --------------------------


def main() -> None:
    """Enumerate single-token hosts, build the deputy cut, print the path."""
    gpt_hosts = enumerate_single_token_hosts(
        str(gguf_target_path("gpt_oss", config.MODELS_DIR)), N_HOSTS
    )
    gemma_hosts = enumerate_single_token_hosts(
        str(gguf_target_path("gemma_4", config.MODELS_DIR)), N_HOSTS
    )
    print(f"single-token hosts: gpt_oss={len(gpt_hosts)} gemma_4={len(gemma_hosts)}")
    gpt_pool, gemma_pool = build_deputy_pools(gpt_hosts, gemma_hosts)
    out = build_permodel(gpt_pool, gemma_pool, OUT_DIR)
    print(f"wrote {out} ({out.stat().st_size // 1024}K)")
    print(
        f"next: uv run python scripts/verify_cut.py {out} "
        "--predicate CONFUSED_DEPUTY --min-severity 3"
    )


if __name__ == "__main__":
    main()
