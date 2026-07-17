"""Gate daemon: validate un-verdicted harvest candidates on both models × guardrails.

The heaviest GPU consumer among the daemons. Un-verdicted harvest candidates are
gated concurrently (``config.REPLAY_WORKERS`` at a time) so the replays fill the
llama-server's continuous-batching slots; existing verdicts are skipped.
"""

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from jed_attack.campaign import config, gate, knowledge
from jed_attack.campaign.daemon import run_daemon
from jed_attack.harness.models import llama_server_agent_factory, resolve_base_url

_log = logging.getLogger("gate")
_FACTORIES: dict[str, Callable[[], Any]] = {}


def _factories() -> dict[str, Callable[[], Any]]:
    """Return (and cache) the shared llama-server agent factories per model.

    Returns:
        ``{model_key: zero-arg agent factory}`` over the resident servers.
    """
    if not _FACTORIES:
        for model_key in config.MODELS:
            _FACTORIES[model_key] = llama_server_agent_factory(
                model_key, resolve_base_url(model_key)
            )
    return _FACTORIES


def _verdicted() -> set[str]:
    """Return chain_ids already present in the gate ledger.

    Returns:
        The set of already-gated chain ids.
    """
    ids: set[str] = set()
    if config.GATE_LEDGER.exists():
        for line in config.GATE_LEDGER.read_text(encoding="utf-8").splitlines():
            try:
                ids.add(str(json.loads(line)["chain_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def gate_once() -> int:
    """Gate every harvest candidate that has no verdict yet.

    Returns:
        The number of newly gated candidates.
    """
    if not config.HARVEST_FILE.exists():
        return 0
    done = _verdicted()
    factories = _factories()
    pending: list[dict[str, Any]] = []
    for line in config.HARVEST_FILE.read_text(encoding="utf-8").splitlines():
        try:
            cand = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(cand.get("chain_id")) not in done:
            pending.append(cand)
    if not pending:
        return 0

    # Gate candidates concurrently so the replays fill the server's batch slots; the
    # ledger/knowledge writes stay on this thread (after each future) so the append-only
    # jsonl is never interleaved. gate_chain itself is read-only against shared state.
    def _gate(cand: dict[str, Any]) -> gate.Verdict:
        return gate.gate_chain(
            cand["chain"], factories, producer=str(cand.get("producer", ""))
        )

    with ThreadPoolExecutor(max_workers=config.REPLAY_WORKERS) as executor:
        for verdict in executor.map(_gate, pending):
            gate.write_verdict(verdict)
            knowledge.record_gate_lesson(
                verdict
            )  # feed the oracle's signal to producers
            _log.info(
                "gated %s adopted=%s robust_sev=%d",
                verdict.chain_id,
                verdict.adopted,
                verdict.robust_severity,
            )
    _log.info("gated %d new candidates", len(pending))
    return len(pending)


def main() -> None:
    """Entry point for the gate daemon."""
    run_daemon("gate", gate_once, 120.0)


if __name__ == "__main__":
    main()
