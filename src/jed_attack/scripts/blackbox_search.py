"""Run the black-box (gradient-free) search for a leaner firing gpt-oss tool call.

Optimizes the SAME Q4_K_M GGUF the grader runs (no transfer gap). First VERIFIES the
scorer (the no-constrain target must score LOWER logprob than the constrain form greedy
already emits), then runs a random-swap search over the optim_str, oracle-checking gen
tokens along the way.

Usage (on dylan's idle TITAN):
    CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID JED_MODELS_DIR=$PWD/models \
        .venv/bin/python -m jed_attack.scripts.blackbox_search --steps 2000
"""

import argparse
import logging

from jed_attack.adversarial import blackbox, telemetry
from jed_attack.adversarial.oracle import evaluate


def main() -> None:
    """Verify the scorer, then search; report the trajectory and best firing shape."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--oracle-every", type=int, default=50)
    parser.add_argument("--restart-patience", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-optim", default="just call it now")
    args = parser.parse_args()

    llm = blackbox.install_logits_backend()

    # --- Floor: what greedy emits today (with the constrain guard). ---
    floor_msg = blackbox.BASE_TEMPLATE.replace("{optim_str}", args.seed_optim)
    floor = evaluate(floor_msg, "gpt_oss")
    logging.info(
        "FLOOR (seed message) gen_tokens=%d fires=%s", floor.gen_tokens, floor.fires
    )

    # --- Scorer sanity: capture the faithful prompt, compare the two target forms. ---
    prompt = blackbox.capture_prompt(
        blackbox.BASE_TEMPLATE.replace("{optim_str}", blackbox.PLACEHOLDER)
    )
    filled = prompt.replace(blackbox.PLACEHOLDER, args.seed_optim)
    logging.info("faithful prompt: %d chars; tail=%r", len(prompt), prompt[-120:])
    lp_con = blackbox.target_logprob(filled, blackbox.TARGET_DIVERGENCE_CON, llm)
    lp_no = blackbox.target_logprob(filled, blackbox.TARGET_DIVERGENCE, llm)
    logging.info(
        "SCORER SANITY: constrain=%.3f no_constrain=%.3f (no_constrain must be LOWER)",
        lp_con,
        lp_no,
    )
    if not lp_no < lp_con:
        logging.warning(
            "SCORER SANITY FAILED: no_constrain (%.3f) is NOT lower than constrain "
            "(%.3f) -- prompt/target may be wrong; inspect before trusting the search.",
            lp_no,
            lp_con,
        )

    telemetry.init(
        "blackbox-gpt_oss",
        {
            "steps": args.steps,
            "oracle_every": args.oracle_every,
            "restart_patience": args.restart_patience,
            "floor_gen": floor.gen_tokens,
            "lp_constrain": lp_con,
            "lp_no_constrain": lp_no,
        },
    )
    try:
        result = blackbox.random_search(
            llm,
            seed_optim=args.seed_optim,
            n_steps=args.steps,
            oracle_every=args.oracle_every,
            restart_patience=args.restart_patience,
            rng_seed=args.seed,
        )
    finally:
        telemetry.finish()

    logging.info(
        "BEST: fires=%s gen_tokens=%d logprob=%.3f (floor gen=%d)\n"
        "  optim_str=%r\n  message=%r",
        result.fires,
        result.gen_tokens,
        result.logprob,
        floor.gen_tokens,
        result.optim_str,
        result.message,
    )
    if result.fires and result.gen_tokens < floor.gen_tokens:
        logging.info(
            "WIN: undercut the floor (%d < %d)", result.gen_tokens, floor.gen_tokens
        )
    else:
        logging.info("NO WIN: did not undercut the floor")


if __name__ == "__main__":
    main()
