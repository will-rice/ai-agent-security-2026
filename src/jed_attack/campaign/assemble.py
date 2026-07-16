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


def _render_pool(adopted: list[Verdict]) -> str:
    """Render adopted chains as Python tuple literals.

    Args:
        adopted: Adopted verdicts (severity-ranked).

    Returns:
        The indented pool body for the template.
    """
    lines = []
    for verdict in adopted:
        chain_literal = ", ".join(repr(msg) for msg in verdict.chain)
        tag = f"sev={verdict.robust_severity}"
        lines.append(f"    ({chain_literal},),  # {tag}")
    return "\n".join(lines)


def assemble(out_dir: Path | None = None, ledger: Path | None = None) -> Path:
    """Build ``attack.py`` from the gate ledger's adopted candidates.

    Args:
        out_dir: Output dir for the built submission (defaults to config build_next).
        ledger: Override gate-ledger path.

    Returns:
        The path to the written ``attack.py``.
    """
    adopted = read_adopted(ledger)
    target = out_dir or config.BUILD_NEXT_DIR
    target.mkdir(parents=True, exist_ok=True)
    source = _TEMPLATE.format(
        pool=_render_pool(adopted),
        max_candidates=config.MAX_CANDIDATES,
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
