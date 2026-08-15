"""Embed a pre-filled candidate list into a fixed, isolated ``attack.py``.

The submission ships the candidate list itself: :func:`build` takes the materialized
JSON (``Submission.to_shipped_json``) and writes it into :data:`_TEMPLATE`, a thin,
submission-agnostic skeleton whose only logic is to return the pre-validated candidates
(capped at the SDK's ``MAX_REPLAY_FINDINGS``) -- NO live probe and NO self-sizing, so
the victim is never run during generation. The candidates are validated (100% firing) at
dump time and local replay is byte-identical to the T4 gateway, so re-running the victim
here would only duplicate the replay the gateway does. The emitted ``attack.py`` imports
only ``aicomp_sdk`` + stdlib (``json`` is stdlib and the list is an embedded string
literal), so it satisfies the isolation contract with no first-party import. It prepends
a Kaggle sys.path preamble (no-op off-Kaggle), so ``build_next`` is directly shippable.
"""

import json
from pathlib import Path

_TEMPLATE = '''"""JED red-team submission -- ships a pre-filled candidate list.

Do not edit -- auto-generated. The candidate list was materialized at dump time; this
file returns it as-is (NO live probe, no self-sizing). Imports only aicomp_sdk + stdlib.
"""

import glob
import json
import sys
from collections.abc import Mapping
from pathlib import Path as _P
from typing import Any

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
from aicomp_sdk.evaluation.ops import MAX_REPLAY_FINDINGS

_CANDIDATES = json.loads(r"""__CANDIDATES_JSON__""")


class AttackAlgorithm(AttackAlgorithmBase):
    """Return the pre-validated candidates as-is (no probe, no self-sizing)."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.last_telemetry: dict[str, Any] = {}
        try:
            super().__init__(config)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass

    def run(
        self, env: AttackEnvProtocol, config: AttackRunConfig
    ) -> list[AttackCandidate]:
        # Probe-free, no self-sizing: NEVER run the victim here, and NEVER trim to
        # a latency estimate. The gateway replays each returned candidate once and
        # scores what completes before its replay deadline (partial credit -- a
        # replay timeout no longer voids the submission), so oversizing is safe and
        # trimming could only cap us below the completable ceiling. Return every
        # candidate up to the gateway's own replay cap (MAX_REPLAY_FINDINGS); it
        # truncates past that anyway. env and config are unused.
        return [
            AttackCandidate.from_messages(list(c))
            for c in _CANDIDATES[:MAX_REPLAY_FINDINGS]
        ]
'''


def build(candidates_json: str, out_dir: Path) -> Path:
    """Write the shipped ``attack.py`` with the pre-filled candidate list.

    The skeleton returns the whole embedded list (capped at the SDK's
    ``MAX_REPLAY_FINDINGS``) with no live probe and no self-sizing -- the gateway scores
    what it completes before its replay deadline (partial credit), so oversizing is safe
    and a latency-based trim could only cap us below the completable ceiling.

    Args:
        candidates_json: A JSON array of message chains. Each chain is one candidate --
            a list of K user messages (K=1 single-post, K>1 a multi-message bundle the
            gateway replays in one env). Embedded as ``_CANDIDATES``; the opaque grading
            env never reveals which victim is graded, so the artifact serves one shared
            list to every victim.
        out_dir: Output directory (typically ``config.BUILD_NEXT_DIR``).

    Returns:
        The path to the written ``attack.py``.

    Raises:
        ValueError: If the embedded JSON contains a ``\"\"\"`` sequence that would break
            out of the embedded string literal.
    """
    candidates_json = json.dumps(json.loads(candidates_json), separators=(",", ":"))
    if '"""' in candidates_json:
        raise ValueError("candidates_json must not contain a triple-quote sequence")
    out_dir.mkdir(parents=True, exist_ok=True)
    source = _TEMPLATE.replace("__CANDIDATES_JSON__", candidates_json)
    attack_path = out_dir / "attack.py"
    attack_path.write_text(source, encoding="utf-8")
    return attack_path
