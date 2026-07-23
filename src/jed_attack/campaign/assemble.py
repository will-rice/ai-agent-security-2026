"""Isolated ``attack.py`` template + pool renderer for the shipped submission.

The submission returns a pre-scored candidate pool directly, so it needs no live search.
:func:`build` wraps literal messages (the global-best :class:`~jed_attack.campaign.
blackboard.Record`'s messages) into that pool and renders it into ``attack.py`` via
:data:`_TEMPLATE` and :func:`_render_pool`. The emitted attack.py imports only
``aicomp_sdk`` + stdlib, so it satisfies the submission-isolation contract with no
inlining. It prepends a Kaggle sys.path preamble (no-op off-Kaggle), so ``build_next``
is directly shippable.
"""

import json
import logging
from pathlib import Path

_log = logging.getLogger("assemble")

_TEMPLATE = '''"""JED red-team submission — assembled from pre-scored candidates.

Do not edit — auto-generated from the composed attack-chain pool.
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

# Pre-scored attack chains, highest value first (see docs/campaign.md).
_POOL: tuple[tuple[str, ...], ...] = (
{pool}
)


class AttackAlgorithm(AttackAlgorithmBase):
    """Return the authored attack candidates directly — one per message."""

    def run(
        self, env: AttackEnvProtocol, config: AttackRunConfig
    ) -> list[AttackCandidate]:
        """Return the authored candidate pool as-is (one candidate per message).

        The pool is authored, not searched, and each message ships exactly once (no
        repetition to fill a budget), so the candidate count equals the message count.

        Args:
            env: The opaque attack environment (unused; messages are pre-authored).
            config: Runtime budget and caps (unused).

        Returns:
            One :class:`AttackCandidate` per authored message chain.
        """
        del env, config
        return [AttackCandidate.from_messages(chain) for chain in _POOL]
'''


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


def build(messages: list[str], out_dir: Path) -> Path:
    """Write the global-best authored submission's ``attack.py``.

    Wraps each literal message as a single-message chain tagged ``"authored"`` and
    renders it via :data:`_TEMPLATE` / :func:`_render_pool`, so the shipped submission
    carries no duplicated template.

    Args:
        messages: The global-best submission's messages (typically the
            blackboard's ``best().messages``).
        out_dir: Output directory for the built submission (typically
            ``config.BUILD_NEXT_DIR``).

    Returns:
        The path to the written ``attack.py``.
    """
    pairs: list[tuple[tuple[str, ...], str]] = [
        ((message,), "authored") for message in messages
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    source = _TEMPLATE.format(pool=_render_pool(pairs))
    attack_path = out_dir / "attack.py"
    attack_path.write_text(source, encoding="utf-8")
    status = {"candidate_count": len(pairs), "source": "authored"}
    (out_dir / "build_next_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    _log.info(
        "built authored submission (%d candidates) -> %s", len(pairs), attack_path
    )
    return attack_path
