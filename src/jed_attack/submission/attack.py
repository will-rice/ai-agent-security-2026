"""Submission entry point: AttackAlgorithm for the JED red-team benchmark.

This module is inlined into a single self-contained file at build time and must
import only ``aicomp_sdk`` and the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aicomp_sdk.attacks.contracts import (
    AttackAlgorithmBase,
    AttackCandidate,
    AttackRunConfig,
)
from aicomp_sdk.core.env.api import AttackEnvProtocol


class AttackAlgorithm(AttackAlgorithmBase):
    """Hybrid discover-and-farm attack (skeleton; filled in later tasks)."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        """Store the optional evaluator-provided config mapping.

        Args:
            config: Optional configuration mapping supplied by the evaluator.
        """
        super().__init__(config)

    def run(
        self, env: AttackEnvProtocol, config: AttackRunConfig
    ) -> list[AttackCandidate]:
        """Search for replayable attack candidates.

        Args:
            env: The opaque attack environment exposing interact/snapshot/restore.
            config: Runtime budget and step/hop caps for this run.

        Returns:
            A list of replayable attack candidates.
        """
        return []
