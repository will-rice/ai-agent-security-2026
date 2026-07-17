"""Assemble the gate-adopted pool into a self-contained submission attack.py.

Because every candidate is already gate-validated, the submission needs no
live search — it returns the adopted chains directly, severity-ranked and
repeated to fill the candidate budget (severity-repetition; see
docs/strategy.md). The emitted attack.py imports only ``aicomp_sdk`` +
stdlib, so it satisfies the submission-isolation contract with no inlining.
"""

import json
from pathlib import Path

from jed_attack.campaign import config
from jed_attack.campaign.gate import Verdict, read_adopted

_TEMPLATE = '''"""JED red-team submission — assembled from gate-adopted candidates.

Do not edit — auto-generated from gate-adopted attack chains.
"""

from aicomp_sdk.attacks.contracts import (
    AttackAlgorithmBase,
    AttackCandidate,
    AttackRunConfig,
)
from aicomp_sdk.core.env.api import AttackEnvProtocol

# Gate-adopted attack chains, highest robust severity first. Each survived the stricter
# `rules` guardrail on at least one model (anti-overfit; see docs/campaign.md).
_POOL: tuple[tuple[str, ...], ...] = (
{pool}
)
_MAX_CANDIDATES = {max_candidates}


class AttackAlgorithm(AttackAlgorithmBase):
    """Return the gate-validated candidate pool, repeated to fill the replay budget."""

    def run(
        self, env: AttackEnvProtocol, config: AttackRunConfig
    ) -> list[AttackCandidate]:
        """Return adopted chains, cycled best-first to fill the candidate budget.

        Args:
            env: The opaque attack environment (unused; the pool is pre-validated).
            config: Runtime budget and caps (unused; count is capped internally).

        Returns:
            The replayable attack candidates.
        """
        del env, config
        if not _POOL:
            return []
        out: list[AttackCandidate] = []
        index = 0
        while len(out) < _MAX_CANDIDATES:
            out.append(AttackCandidate.from_messages(_POOL[index % len(_POOL)]))
            index += 1
        return out
'''


def ship_pool_chains(cap: int, ledger: Path | None = None) -> list[tuple[str, ...]]:
    """Return the single-message ship-pool candidate chains (best-severity first).

    The exact candidate set the submission would return, for the calibrated scorer
    to replay. Same selection as ``assemble(single_message=True, cap=cap)``.

    Args:
        cap: Max candidates.
        ledger: Override gate-ledger path.

    Returns:
        The candidate chains (each a 1-tuple message).
    """
    return [chain for chain, _ in _pool_chains(read_adopted(ledger), True, cap)]


def _pool_chains(
    adopted: list[Verdict], single_message: bool, cap: int
) -> list[tuple[tuple[str, ...], str]]:
    """Select the (chain, tag) pairs to render into the submission pool.

    Args:
        adopted: Adopted verdicts (severity-ranked, best first).
        single_message: If True, flatten every chain into one-message candidates
            (each fires independently and replays far faster, so many more fit the
            gateway's replay budget — the multi-message pool overran it). If False,
            keep whole chains.
        cap: Maximum number of candidates to include (budget cap).

    Returns:
        Up to ``cap`` ``(chain, tag)`` pairs, best-severity first, deduped.
    """
    pairs: list[tuple[tuple[str, ...], str]] = []
    seen: set[str] = set()
    for verdict in adopted:
        chains = _flatten(verdict) if single_message else [verdict.chain]
        for chain in chains:
            key = " ".join(chain)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((chain, f"sev={verdict.robust_severity}"))
            if len(pairs) >= cap:
                return pairs
    return pairs


_INDEPENDENT_PREDICATES = frozenset({"CONFUSED_DEPUTY", "EXFILTRATION"})


def _flatten(verdict: Verdict) -> list[tuple[str, ...]]:
    """Split a verdict's chain into budget-efficient single-message candidates.

    Confused-deputy sends and ``SECRET_MARKER`` http.post exfil each fire on their
    own (verified: a single POST-marker message fires EXFILTRATION on both models
    under both guardrails), so a chain of them flattens into one fast candidate per
    message. Chains with dependency-based predicates (UNTRUSTED_TO_ACTION /
    DESTRUCTIVE_WRITE, or read-then-post exfil) don't split cleanly, so they are
    skipped rather than seeding non-firing candidates.

    Args:
        verdict: An adopted verdict (its ``fires`` identify the predicates).

    Returns:
        The single-message candidates to contribute (possibly empty).
    """
    if len(verdict.chain) == 1:
        return [verdict.chain]  # already single-message (any predicate, incl. exfil)
    predicates = {
        p for m in verdict.fires.values() for preds in m.values() for p in preds
    }
    if predicates <= _INDEPENDENT_PREDICATES:
        return [(msg,) for msg in verdict.chain]  # independent fires — safe to split
    return []  # dependency-based predicate — can't split without losing the fire


def _render_pool(pairs: list[tuple[tuple[str, ...], str]]) -> str:
    """Render (chain, tag) pairs as Python tuple literals.

    Args:
        pairs: The chains + severity tags to render.

    Returns:
        The indented pool body for the template.
    """
    lines = []
    for chain, tag in pairs:
        chain_literal = ", ".join(repr(msg) for msg in chain)
        lines.append(f"    ({chain_literal},),  # {tag}")
    return "\n".join(lines)


def assemble(
    out_dir: Path | None = None,
    ledger: Path | None = None,
    *,
    single_message: bool = False,
    cap: int | None = None,
) -> Path:
    """Build ``attack.py`` from the gate ledger's adopted candidates.

    Args:
        out_dir: Output dir for the built submission (defaults to config build_next).
        ledger: Override gate-ledger path.
        single_message: Flatten chains into one-message candidates (budget-safe;
            the multi-message pool overran the ~9000s gateway replay budget).
        cap: Max candidates (defaults to ``config.MAX_CANDIDATES``). With
            ``single_message`` this is the count of distinct one-message candidates
            returned (no repetition), sized to fit the replay budget.

    Returns:
        The path to the written ``attack.py``.
    """
    adopted = read_adopted(ledger)
    limit = cap if cap is not None else config.MAX_CANDIDATES
    pairs = _pool_chains(adopted, single_message, limit)
    target = out_dir or config.BUILD_NEXT_DIR
    target.mkdir(parents=True, exist_ok=True)
    # Single-message pools are returned once each (no repetition); whole-chain pools
    # cycle to fill the candidate budget.
    max_candidates = len(pairs) if single_message else limit
    source = _TEMPLATE.format(
        pool=_render_pool(pairs),
        max_candidates=max_candidates,
    )
    attack_path = target / "attack.py"
    attack_path.write_text(source, encoding="utf-8")
    status = {
        "adopted_count": len(adopted),
        "total_robust_severity": sum(v.robust_severity for v in adopted),
        "top": [
            {
                "producer": v.producer,
                "robust_severity": v.robust_severity,
                "chain": list(v.chain),
            }
            for v in adopted[:10]
        ],
    }
    (target / "build_next_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    return attack_path
