"""One-command launcher for the prompt-optimizer swarm.

Sanity-checks the configured proposer, then spawns N detached worker processes each
running the optimizer loop forever. Replaces the bash launch script with a single
``uv run jed-optimize`` entry point; the proposer and model come from flags (or the
env). Run on the host that serves the target models (green)::

    uv run jed-optimize --workers 3 --proposer api --model glm-4.6

The ``api`` backend also needs ``PROPOSER_API_BASE`` + ``PROPOSER_API_KEY`` exported in
the environment. The key is deliberately NOT a flag, so it never lands in argv or shell
history; the launcher passes the ambient environment through to the workers unchanged.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from jed_attack.campaign import config, optimize_prompts

_log = logging.getLogger("launch")


def running_workers() -> list[int]:
    """Return PIDs of optimizer workers already running on this host."""
    found = subprocess.run(
        ["pgrep", "-f", "campaign.optimize_prompts"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(pid) for pid in found.stdout.split() if pid.isdigit()]


def stop_workers(pids: list[int]) -> None:
    """SIGTERM the given worker PIDs and pause briefly for them to exit.

    Args:
        pids: Worker process ids to stop.
    """
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:  # already gone
            pass
    time.sleep(2)


def spawn(workers: int, log_dir: Path) -> list[int]:
    """Spawn ``workers`` detached optimizer processes (worker 1 owns the W&B run).

    Each worker is started in its own session (``start_new_session``) with stdin closed
    and output appended to ``optimizer-<i>.log``, so it survives the launcher exiting.

    Args:
        workers: Number of worker processes to start.
        log_dir: Directory for per-worker log files.

    Returns:
        The spawned worker PIDs.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    pids: list[int] = []
    for i in range(1, workers + 1):
        env = {**os.environ, "JED_WANDB": "1" if i == 1 else "0"}
        log_path = log_dir / f"optimizer-{i}.log"
        with log_path.open("ab") as logf:
            proc = subprocess.Popen(
                [sys.executable, "-m", "jed_attack.campaign.optimize_prompts"],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        pids.append(proc.pid)
        _log.info(
            "worker %d up (pid %d, wandb=%s) -> %s",
            i,
            proc.pid,
            env["JED_WANDB"],
            log_path,
        )
    return pids


def main() -> None:
    """CLI: sanity-check the proposer, then launch the swarm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--proposer", choices=["local", "api", "codex"], help="sets JED_PROPOSER"
    )
    parser.add_argument(
        "--model",
        help="proposer model override (JED_PROPOSER_MODEL or PROPOSER_API_MODEL)",
    )
    parser.add_argument(
        "--proposals", type=int, default=optimize_prompts.DEFAULT_PROPOSALS
    )
    parser.add_argument(
        "--restart", action="store_true", help="stop any running workers first"
    )
    parser.add_argument(
        "--skip-sanity", action="store_true", help="launch without the proposer check"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config.ensure_dirs()

    # Proposer config flows to the workers via the environment; also mirror it onto the
    # in-process module globals so the sanity check below uses the same backend.
    if args.proposer:
        os.environ["JED_PROPOSER"] = args.proposer
        optimize_prompts.PROPOSER_BACKEND = args.proposer
    backend = os.environ.get("JED_PROPOSER", "local")
    if args.model:
        if backend == "api":
            os.environ["PROPOSER_API_MODEL"] = args.model
            optimize_prompts.PROPOSER_API_MODEL = args.model
        else:
            os.environ["JED_PROPOSER_MODEL"] = args.model
            optimize_prompts.PROPOSER_MODEL = args.model

    existing = running_workers()
    if existing and not args.restart:
        _log.error(
            "optimizer already running (pids %s); pass --restart to replace", existing
        )
        raise SystemExit(1)
    if existing:
        _log.info("stopping %d running worker(s): %s", len(existing), existing)
        stop_workers(existing)

    if not args.skip_sanity:
        produced = optimize_prompts.proposer_sanity(args.proposals)
        if not produced:
            _log.error(
                "proposer '%s' produced no valid proposals — not launching. Check the "
                "backend config (api needs PROPOSER_API_BASE + PROPOSER_API_KEY).",
                backend,
            )
            raise SystemExit(2)
        _log.info("sanity ok: proposer '%s' produced %d proposals", backend, produced)

    pids = spawn(args.workers, config.CAMPAIGN_ROOT / "logs")
    model = (
        os.environ.get("PROPOSER_API_MODEL")
        if backend == "api"
        else os.environ.get("JED_PROPOSER_MODEL", "gpt_oss")
    )
    _log.info(
        "swarm running: %d workers | proposer=%s model=%s | logs: %s | wandb: %s",
        len(pids),
        backend,
        model,
        config.CAMPAIGN_ROOT / "logs",
        "will-rice/jed-prompt-opt",
    )


if __name__ == "__main__":
    main()
