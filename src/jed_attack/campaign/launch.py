"""One-command launcher for the prompt-optimizer swarm.

Sanity-checks the selected proposer, then spawns N detached worker processes each
running the optimizer loop forever. Replaces the bash launch script with a single ``uv
run jed-optimize`` entry point. The proposer is chosen by NAME from the registry
(:mod:`providers`) — all endpoint/model config lives there in code; only the API token
comes from the environment (each provider names its own ``key_env``). Run on the host
that serves the target models (green)::

    uv run jed-optimize                     # all you need: replace/start on the default
                                            # preference chain (keys read from .env)
    uv run jed-optimize --list              # show providers
    uv run jed-optimize --switch --proposer gpt_oss   # live switch, no restart

With no flags it replaces any running swarm and uses ``providers.PREFERENCE`` (skipping
api providers whose key is absent), so the common case needs nothing memorised. Tokens
come from a gitignored ``.env`` (:func:`load_dotenv`), never a flag; ``--proposer``
persists a chain to ``proposer.json`` that running workers re-read every generation.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

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
        SystemExit: If workers are already running and ``--no-restart`` was passed.
    """
    existing = running_workers()
    if existing and args.no_restart:
        _log.error(
            "optimizer already running (pids %s); omit --no-restart to replace",
            existing,
        )
        raise SystemExit(1)
    if existing:
        _log.info("replacing %d running worker(s): %s", len(existing), existing)
        stop_workers(existing)

    if not args.skip_sanity:
        produced = optimize_prompts.proposer_sanity(args.proposals)
        if produced:
            _log.info("sanity ok: proposer produced %d proposals", produced)
        else:
            # Non-blocking: a one-shot 0 is usually transient (api rate-limited or a
            # busy single-request server). The workers retry the whole chain (and fall
            # to parametric) every generation, so launch anyway rather than abort.
            _log.warning(
                "sanity: proposer produced nothing right now (api 429 or server busy); "
                "launching anyway — workers retry the chain + parametric each gen"
            )

    pids = spawn(args.workers, config.CAMPAIGN_ROOT / "logs")
    chain = " -> ".join(p.model or p.kind for p in optimize_prompts.current_providers())
    _log.info(
        "swarm running: %d workers | proposer chain: %s | logs: %s | wandb: %s",
        len(pids),
        chain,
        config.CAMPAIGN_ROOT / "logs",
        "will-rice/jed-prompt-opt",
    )


def _validated_selection(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[str]:
    """Parse ``--proposer`` into validated registry names (empty if not given).

    Args:
        args: Parsed CLI args.
        parser: The parser (used to ``error`` on an unknown name).

    Returns:
        The ordered, validated provider names.
    """
    names = (
        [name.strip() for name in args.proposer.split(",") if name.strip()]
        if args.proposer
        else []
    )
    for name in names:
        if name not in providers.PROVIDERS:
            parser.error(f"unknown proposer '{name}'; see --list")
    return names


def _print_catalog(
    chain: list[providers.Provider], parser: argparse.ArgumentParser
) -> None:
    """Print the first api provider's live model catalog (for ``--models``).

    Args:
        chain: The resolved provider chain.
        parser: The parser (used to ``error`` if the chain has no api provider).
    """
    api = [provider for provider in chain if provider.kind == "api"]
    if not api:
        parser.error("--models needs an api provider (pass --proposer <api-name>)")
    for model in sorted(optimize_prompts.fetch_api_models(api[0])):
        print(f"  {model}")


def _preflight_chain(chain: list[providers.Provider]) -> None:
    """Warn-only pre-flight of each api provider's model against its catalog.

    Args:
        chain: The resolved provider chain.
    """
    for provider in chain:
        if provider.kind == "api":
            _preflight_api_model(provider)


def main() -> None:
    """CLI: select the proposer, sanity-check it, then launch (or switch) the swarm."""
    # Load API tokens from a gitignored .env (e.g. ZAI_API_KEY, CHEAPEST_API_KEY) so
    # they need not be exported by hand; workers inherit them via spawn()'s env.
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--proposer",
        help="provider name(s), comma-separated preference chain (see --list); "
        "persisted to proposer.json for live switching",
    )
    parser.add_argument(
        "--proposals", type=int, default=optimize_prompts.DEFAULT_PROPOSALS
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="(default, now implied) replace a running swarm; kept for compatibility",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="refuse to launch if a swarm is already running (instead of replacing it)",
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

    # Selecting writes proposer.json; the sanity check and every worker read the chain
    # via current_providers(), so no proposer state passes through the environment.
    selected = _validated_selection(args, parser)
    if selected:
        optimize_prompts.set_providers(selected)
        _log.info("proposer chain set to %s", selected)

    chain = optimize_prompts.current_providers()
    if args.models:
        _print_catalog(chain, parser)
        return

    if not args.skip_sanity:
        _preflight_chain(chain)

    if args.switch:
        if not selected:
            parser.error("--switch requires --proposer")
        _log.info(
            "live proposer chain -> %s (workers pick it up next generation)", selected
        )
        return

    _launch_swarm(args)


if __name__ == "__main__":
    main()
