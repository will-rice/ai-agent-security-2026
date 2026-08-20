"""Thin, dependency-tolerant W&B wrapper for the adversarial search.

Mirrors the gating/failure-handling pattern of
:mod:`jed_attack.campaign.optimize_prompts` (``JED_WANDB`` env gate, warn-and-continue
on any failure) but keeps the run on the wandb module global (``wandb.run``) instead of
threading a handle through call sites, since the adversarial search's call sites
(``gcg_driver``, ``ga``, ``pipeline``) are plain functions with no run object to pass.
Project is ``jed-adversarial-search`` -- kept separate from the optimizer's
``jed-prompt-opt`` so a long search run is watchable on its own board.
"""

import logging
import os
from typing import Any

_log = logging.getLogger(__name__)


def init(run_name: str, config: dict[str, Any]) -> None:
    """Start a W&B run for this search, or no-op when disabled/unavailable.

    Gated behind ``JED_WANDB`` (set to ``0`` to disable). Any import/init failure
    (not installed, offline, backend error) degrades to no telemetry rather than
    crashing the search.

    Args:
        run_name: The run's display name.
        config: The run's config dict (e.g. model, step/round counts, floor board).
    """
    if os.environ.get("JED_WANDB", "1") == "0":
        return
    try:
        import wandb

        wandb.init(
            entity="will-rice",
            project="jed-adversarial-search",
            name=run_name,
            config=config,
        )
    except Exception:  # not installed / offline / init error
        _log.warning("wandb init failed; running without it", exc_info=True)


def log(metrics: dict[str, Any]) -> None:
    """Log metrics to the active W&B run; a no-op if no run is active.

    Never raises: this is called from inside search loops that must keep running
    whether or not telemetry was initialized.

    Args:
        metrics: The metrics dict to log.
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.run.log(metrics)
    except Exception:  # wandb offline / backend hiccup -- keep searching
        _log.warning("wandb logging failed; continuing without it", exc_info=True)


def finish() -> None:
    """Close the active W&B run, if one is live."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:  # nothing actionable on shutdown
        _log.warning("wandb finish failed", exc_info=True)
