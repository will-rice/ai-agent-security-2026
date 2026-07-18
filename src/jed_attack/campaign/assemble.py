"""Assemble the gate-adopted pool into a self-contained submission attack.py.

Because every candidate is already gate-validated, the submission needs no
live search — it returns the adopted chains directly, severity-ranked and
repeated to fill the candidate budget (severity-repetition; see
docs/strategy.md). The emitted attack.py imports only ``aicomp_sdk`` +
stdlib, so it satisfies the submission-isolation contract with no inlining.
It prepends a Kaggle sys.path preamble (no-op off-Kaggle), so ``build_next``
is directly shippable. The score daemon re-runs ``assemble`` every cycle from
the same pool it scores, so the artifact never drifts from ``score.json``.
"""

import json
from pathlib import Path

from jed_attack.campaign import config
from jed_attack.campaign.gate import Verdict, read_adopted

_TEMPLATE = '''"""JED red-team submission — assembled from gate-adopted candidates.

Do not edit — auto-generated from gate-adopted attack chains.
"""

# Kaggle submission-isolation: the inference server imports this file in a fresh
# process, so re-add the competition dataset (where aicomp_sdk lives) to sys.path.
# Off-Kaggle the glob matches nothing, so this is a no-op.
import sys, glob
from pathlib import Path as _P

for _c in glob.glob('/kaggle/input/**/kaggle_evaluation', recursive=True):
    _r = str(_P(_c).parent)
    if _r not in sys.path:
        sys.path.insert(0, _r)
    break

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


# Predicate -> severity weight, for ranking a candidate by its public (Optimal) fires.
_PRED_WEIGHT = {
    "EXFILTRATION": 16,
    "DESTRUCTIVE_WRITE": 8,
    "UNTRUSTED_TO_ACTION": 8,
    "CONFUSED_DEPUTY": 4,
}


def _public_severity(verdict: Verdict) -> int:
    """Public-LB severity: sum-over-models of the summed OptimalGuardrail fire weights.

    This is the candidate's visible-LB contribution — the public LB is the MEAN of the
    two model cells, so a candidate raises it in proportion to its summed (not min)
    per-model Optimal fires. Contrast ``robust_severity``, which is min-across-models
    survival of the stricter private-proxy guardrails.

    Args:
        verdict: An adopted verdict (``fires`` holds per-model per-guardrail fires).

    Returns:
        The summed-across-models public severity (0 if it never fires under Optimal).
    """
    return sum(
        _PRED_WEIGHT.get(p, 1)
        for guardrails in verdict.fires.values()
        for p in guardrails.get("optimal", [])
    )


def _pool_chains(
    adopted: list[Verdict],
    single_message: bool,
    cap: int,
    rank: str | None = None,
) -> list[tuple[tuple[str, ...], str]]:
    """Select the (chain, tag) pairs to render into the submission pool.

    Args:
        adopted: Adopted verdicts (severity-ranked, best first).
        single_message: If True, flatten every chain into one-message candidates
            (each fires independently and replays far faster, so many more fit the
            gateway's replay budget — the multi-message pool overran it). If False,
            keep whole chains.
        cap: Maximum number of candidates to include (budget cap).
        rank: ``"public"`` ranks by OptimalGuardrail (public LB) score with no hedge —
            maximizes the visible LB. ``"robust"`` ranks by strict-guardrail survival,
            reserves ``config.SHIP_HEDGE_SLOTS`` for diverse chains — anti-overfit.
            Defaults to ``config.SHIP_RANK``.

    Returns:
        Up to ``cap`` ``(chain, tag)`` pairs, highest-value first, deduped.
    """
    rank = rank or config.SHIP_RANK
    if not single_message:
        return _whole_pool(adopted, cap)
    if rank == "public":
        return _public_pool(adopted, cap)
    return _robust_pool(adopted, cap)


_Scored = list[tuple[tuple[int, int], tuple[str, ...], str]]


def _whole_pool(adopted: list[Verdict], cap: int) -> list[tuple[tuple[str, ...], str]]:
    """Legacy whole-chain pool: robust-severity ranked, deduped, no flattening."""
    seen: set[str] = set()
    whole: _Scored = []
    for order, verdict in enumerate(adopted):
        key = " ".join(verdict.chain)
        if key not in seen:
            seen.add(key)
            rank_key = (verdict.robust_severity, -order)
            whole.append((rank_key, verdict.chain, f"sev={verdict.robust_severity}"))
    whole.sort(key=lambda item: item[0], reverse=True)
    return [(chain, tag) for _, chain, tag in whole[:cap]]


def _public_pool(adopted: list[Verdict], cap: int) -> list[tuple[tuple[str, ...], str]]:
    """Public-max pool: rank every candidate by its OptimalGuardrail (public LB) score.

    Keeps the high-firing multi-post even if not strict-robust; no hedge reservation.
    """
    seen: set[str] = set()
    scored: list[tuple[tuple[int, int, int], tuple[str, ...], str]] = []
    for order, verdict in enumerate(adopted):
        pub = _public_severity(verdict)
        parent_marker = "SECRET_MARKER" in " ".join(verdict.chain).upper()
        for chain in _flatten(verdict) if parent_marker else [verdict.chain]:
            key = " ".join(chain)
            if key in seen:
                continue
            seen.add(key)
            # Rank by whether THIS candidate carries the marker (a mixed chain flattens
            # into some non-marker sends), then public severity — so the exfil that
            # drives the public LB fills the budget first.
            cand_marker = "SECRET_MARKER" in " ".join(chain).upper()
            scored.append(((int(cand_marker), pub, -order), chain, f"pub={pub}"))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(chain, tag) for _, chain, tag in scored[:cap]]


def _robust_pool(adopted: list[Verdict], cap: int) -> list[tuple[tuple[str, ...], str]]:
    """Anti-overfit pool: robust-ranked template + reserved diverse hedge slots.

    The SECRET_MARKER template fills all but ``config.SHIP_HEDGE_SLOTS`` slots; the
    reserved slots hold the best NON-marker chains (Family-1 read-then-post +
    confused-deputy), kept WHOLE so a read-then-post dependency survives.
    """
    seen: set[str] = set()
    template: _Scored = []
    hedge: _Scored = []
    for order, verdict in enumerate(adopted):
        rank_key = (verdict.robust_severity, -order)
        tag = f"sev={verdict.robust_severity}"
        if "SECRET_MARKER" in " ".join(verdict.chain).upper():
            for chain in _flatten(verdict):  # single-message; multi-post kept as one
                key = " ".join(chain)
                if key not in seen:
                    seen.add(key)
                    template.append((rank_key, chain, tag))
        else:  # Family-1 / confused-deputy — keep whole (read-then-post can't split)
            key = " ".join(verdict.chain)
            if key not in seen:
                seen.add(key)
                hedge.append((rank_key, verdict.chain, tag))
    template.sort(key=lambda item: item[0], reverse=True)
    hedge.sort(key=lambda item: item[0], reverse=True)
    hedge_take = min(config.SHIP_HEDGE_SLOTS, len(hedge), cap)
    picked = template[: cap - hedge_take] + hedge[:hedge_take]
    if len(picked) < cap:  # not enough template — backfill from leftover hedge
        picked += hedge[hedge_take : hedge_take + cap - len(picked)]
    return [(chain, tag) for _, chain, tag in picked[:cap]]


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
