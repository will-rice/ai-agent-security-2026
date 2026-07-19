"""One-command launcher for the prompt-optimizer swarm.

Sanity-checks the selected proposer, then spawns N detached worker processes each
running the optimizer loop forever. Replaces the bash launch script with a single ``uv
run jed-optimize`` entry point. The proposer is chosen by NAME from the registry
(:mod:`providers`) — all endpoint/model config lives there in code; only the API token
comes from the environment (each provider names its own ``key_env``). Run on the host
that serves the target models (green)::

    uv run jed-optimize --list                            # show providers
    uv run jed-optimize --restart --proposer zai-glm4.6   # needs ZAI_API_KEY exported
    uv run jed-optimize --switch  --proposer gpt_oss      # live switch, no restart

``--proposer`` persists the choice to ``proposer.json``, which running workers re-read
every generation — so ``--switch`` changes the backend without stopping the swarm. The
API token is never a flag or a config value; it stays in the env the whole time.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from jed_attack.campaign import config, optimize_prompts, providers

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


def _print_providers() -> None:
    """Print the proposer provider registry."""
    for name in sorted(providers.PROVIDERS):
        p = providers.PROVIDERS[name]
        if p.kind == "api":
            base = p.base_url or "(base_url unset)"
            detail = f"api  {p.model:20} {base}  key=${p.key_env}"
        else:
            detail = f"{p.kind:4} {p.model}"
        print(f"  {name:20} {detail}")


def _preflight_api_model(provider: providers.Provider) -> None:
    """Warn (do not fail) if an api provider's model isn't in its live catalog.

    Queries the provider's ``/v1/models`` endpoint and checks ``provider.model`` is
    advertised. A mismatch or unreachable endpoint only warns, because the swarm falls
    back to the local model per-generation when an api provider is unavailable — so a
    wrong id or a closed usage window degrades to local rather than blocking the launch.

    Args:
        provider: The ``api`` provider to check.
    """
    try:
        catalog = optimize_prompts.fetch_api_models(provider)
    except Exception as exc:
        _log.warning(
            "catalog unreachable at %s (%s); swarm uses local '%s' until '%s' is "
            "reachable (check base_url and that %s is exported)",
            provider.base_url,
            exc,
            providers.DEFAULT,
            provider.model,
            provider.key_env,
        )
        return
    if provider.model not in catalog:
        _log.warning(
            "model '%s' is not in the provider catalog; the swarm will use local '%s'. "
            "Available: %s",
            provider.model,
            providers.DEFAULT,
            sorted(catalog),
        )
        return
    _log.info(
        "model '%s' validated against the provider catalog (%d models)",
        provider.model,
        len(catalog),
    )


def _launch_swarm(args: argparse.Namespace) -> None:
    """Stop-if-restart, sanity-check the proposer, and spawn the worker swarm.

    Args:
        args: Parsed CLI args (restart, skip_sanity, proposals, workers).

    Raises:
        SystemExit: If workers already run without ``--restart``, or the proposer
            produces no valid proposals.
    """
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
                "proposer produced no valid proposals — not launching. For an api "
                "provider, check its base_url in providers.py and that its key_env is "
                "exported in this shell."
            )
            raise SystemExit(2)
        _log.info("sanity ok: proposer produced %d proposals", produced)

    pids = spawn(args.workers, config.CAMPAIGN_ROOT / "logs")
    provider = optimize_prompts.current_provider()
    _log.info(
        "swarm running: %d workers | proposer=%s %s | logs: %s | wandb: %s",
        len(pids),
        provider.kind,
        provider.model,
        config.CAMPAIGN_ROOT / "logs",
        "will-rice/jed-prompt-opt",
    )


def main() -> None:
    """CLI: select the proposer, sanity-check it, then launch (or switch) the swarm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--proposer",
        choices=sorted(providers.PROVIDERS),
        help="provider name (see --list); persisted to proposer.json for live switch",
    )
    parser.add_argument(
        "--proposals", type=int, default=optimize_prompts.DEFAULT_PROPOSALS
    )
    parser.add_argument(
        "--restart", action="store_true", help="stop any running workers first"
    )
    parser.add_argument(
        "--switch",
        action="store_true",
        help="only update the live proposer (needs --proposer); do not launch",
    )
    parser.add_argument(
        "--skip-sanity", action="store_true", help="launch without the proposer check"
    )
    parser.add_argument(
        "--list", action="store_true", help="list proposer providers and exit"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="list the selected api provider's model catalog and exit",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.list:
        _print_providers()
        return

    config.ensure_dirs()

    # The provider we are about to select/use (validated before we commit to it).
    provider = (
        providers.get(args.proposer)
        if args.proposer
        else optimize_prompts.current_provider()
    )

    if args.models:
        if provider.kind != "api":
            parser.error("--models only applies to api providers")
        for model in sorted(optimize_prompts.fetch_api_models(provider)):
            print(f"  {model}")
        return

    # Pre-flight the api model against its live catalog (warn-only: the swarm falls back
    # to the local model per-generation when an api provider is unavailable).
    if provider.kind == "api" and not args.skip_sanity:
        _preflight_api_model(provider)

    # Selecting a provider writes proposer.json; the sanity check and every worker read
    # it via current_provider(), so no proposer state is passed through the environment.
    if args.proposer:
        optimize_prompts.set_provider(args.proposer)
        _log.info("proposer set to '%s'", args.proposer)

    if args.switch:
        if not args.proposer:
            parser.error("--switch requires --proposer")
        _log.info(
            "live proposer -> '%s' (running workers pick it up next generation)",
            args.proposer,
        )
        return

    _launch_swarm(args)


if __name__ == "__main__":
    main()
